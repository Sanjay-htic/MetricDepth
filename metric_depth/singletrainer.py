import argparse
import logging
import os
import pprint
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log
from dataloader import SUNRGBDDataset, RealsenseDataset

parser = argparse.ArgumentParser(
    description='Depth Anything V2 - Single GPU Metric Depth Estimation'
)

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='sunrgbd', choices=['sunrgbd', 'realsense'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)


def main():
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    #warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('train', logging.INFO)
    logger.propagate = 0

    logger.info(pprint.pformat(vars(args)))

    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------
    size = (args.img_size, args.img_size)

    if args.dataset == 'sunrgbd':
        trainset = SUNRGBDDataset(r'E:\Sanjay\Dataset\SUNRGBD\SUNRGBD\realsense\sa', 'train', img_size=size)
        valset   = SUNRGBDDataset (r'E:\Sanjay\Dataset\SUNRGBD\SUNRGBD\realsense\sh', 'val', img_size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
        valset   = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    else:
        raise NotImplementedError

    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(
        **model_configs[args.encoder],
        max_depth=args.max_depth
    )

    if args.pretrained_from:
        ckpt = torch.load(args.pretrained_from, map_location='cpu')
        model.load_state_dict(
            {k: v for k, v in ckpt.items() if 'pretrained' in k},
            strict=False
        )

    model = model.to(device)

    # -------------------------------------------------------------
    # Loss & Optimizer
    # -------------------------------------------------------------
    criterion = SiLogLoss().to(device)

    optimizer = AdamW(
        [
            {
                'params': [p for n, p in model.named_parameters() if 'pretrained' in n],
                'lr': args.lr
            },
            {
                'params': [p for n, p in model.named_parameters() if 'pretrained' not in n],
                'lr': args.lr * 10.0
            }
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    total_iters = args.epochs * len(trainloader)

    previous_best = {
        'd1': 0, 'd2': 0, 'd3': 0,
        'abs_rel': 100, 'sq_rel': 100,
        'rmse': 100, 'rmse_log': 100,
        'log10': 100, 'silog': 100
    }

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    for epoch in range(args.epochs):

        logger.info(
            f'Epoch [{epoch}/{args.epochs}] '
            f'd1={previous_best["d1"]:.3f}, '
            f'abs_rel={previous_best["abs_rel"]:.3f}'
        )

        model.train()
        total_loss = 0.0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            iters = epoch * len(trainloader) + i


            img   = sample['image'].to(device)
            depth = sample['depth'].to(device)
            mask  = sample['valid_mask'].to(device)

            if random.random() < 0.5:
                img   = img.flip(-1)
                depth = depth.flip(-1)
                mask  = mask.flip(-1)

            pred = model(img)

            loss = criterion(
                pred,
                depth,
                (mask == 1) &
                (depth >= args.min_depth) &
                (depth <= args.max_depth)
            )
            if i % 25 == 0:
                with torch.no_grad():
                    # Take first sample in batch
                    rgb_vis = prepare_rgb_for_vis(img[0])

                    gt_depth_vis = normalize_depth_for_vis(
                        depth[0], args.min_depth, args.max_depth
                    )

                    pred_depth_vis = depth_to_uint8(
                        pred[0], args.min_depth, args.max_depth
                    )

                    writer.add_image('train/rgb', rgb_vis, iters)
                    writer.add_image('train/depth_gt', gt_depth_vis, iters)
                    writer.add_image('train/depth_pred', pred_depth_vis, iters)


            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            #iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0
            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info(
                    f'Iter [{i}/{len(trainloader)}] '
                    f'LR={lr:.7f}, Loss={loss.item():.4f}'
                )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        model.eval()

        results = {k: 0.0 for k in previous_best}
        nsamples = 0

        with torch.no_grad():
            for sample in valloader:
                img   = sample['image'].to(device).float()
                depth = sample['depth'][0].to(device)        # [1,1,H,W] or [H,W]
                mask  = sample['valid_mask'][0].to(device)   # [1,1,H,W] or [H,W]
                #print("Shape of the img for validation:", img.shape)
                #print("Shape of the depth for validation:", depth.shape)
                #print("Shape of the mask for validation:", mask.shape)

                # ---------- Prediction ----------
                pred = model(img)
                pred = F.interpolate(
                    pred[:, None],
                    depth.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )[0, 0]                                       # [H,W]
                #print("shape of the pred :",pred.shape)

                # ---------- Squeeze GT ----------
                if depth.ndim == 3:
                    depth = depth.squeeze(0)       # [H,W]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)         # [H,W]
                    
                #print("shape of the depth after squeezing:", depth)
                #print("shape of the mask after squeezing:", mask)

                # ---------- Valid mask ----------
                valid = (
                    (mask == 1) &
                    (depth >= args.min_depth) &
                    (depth <= args.max_depth)
                )

                if valid.sum() < 10:
                    continue

                # ---------- Sanity check ----------
                assert pred.shape == depth.shape == valid.shape, \
                    f"Eval mismatch: pred {pred.shape}, depth {depth.shape}, valid {valid.shape}"

                # ---------- Vectorize ----------
                pred_valid  = pred[valid]      # [N]
                depth_valid = depth[valid]     # [N]

                # Optional debug (keep once)
                #print("pred_valid:", pred_valid.shape, "depth_valid:", depth_valid.shape)

                # ---------- Metrics ----------
                cur = eval_depth(pred_valid, depth_valid)

                for k in results:
                    results[k] += cur[k]

                nsamples += 1
                
        if nsamples == 0:
            with torch.no_grad():
                rgb_vis = prepare_rgb_for_vis(img[0])

                gt_depth_vis = normalize_depth_for_vis(
                    depth, args.min_depth, args.max_depth
                )

                pred_depth_vis = depth_to_uint8(
                    pred, args.min_depth, args.max_depth
                )

                writer.add_image('val/rgb', rgb_vis, epoch)
                writer.add_image('val/depth_gt', gt_depth_vis, epoch)
                writer.add_image('val/depth_pred', pred_depth_vis, epoch)

                        
        for k in results:
            results[k] /= max(nsamples, 1)
            writer.add_scalar(f'eval/{k}', results[k], epoch)

            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], results[k])
            else:
                previous_best[k] = min(previous_best[k], results[k])

        logger.info(f'Validation results: {results}')

        # ---------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            },
            os.path.join(args.save_path, 'latest.pth')
        )


if __name__ == '__main__':
    def normalize_depth_for_vis(depth, min_d, max_d):
        """
        depth: [H, W] tensor
        returns: [1, H, W] tensor in [0,1]
        """
        depth = depth.detach().cpu()
        if depth.ndim == 3:
            depth = depth.squeeze(0)   # [H, W]
        depth = depth.clamp(min_d, max_d)
        depth = (depth - min_d) / (max_d - min_d)
        return depth.unsqueeze(0)


    def prepare_rgb_for_vis(img):
        """
        img: [3,H,W] tensor, possibly ImageNet normalized
        returns: [3,H,W] in [0,1]
        """
        img = img.detach().cpu()

        # ImageNet mean/std (DepthAnything uses this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        # If image looks normalized, denormalize
        if img.min() < 0 or img.max() > 1.5:
            img = img * std + mean

        return img
    
    def depth_to_uint8(depth, min_d, max_d):
        """
        depth: [H,W] tensor
        returns: [1,H,W] uint8 tensor in [0,255]
        """
        depth = depth.detach().cpu()
        if depth.ndim == 3:
            depth = depth.squeeze(0)

        depth = depth.clamp(min_d, max_d)
        depth = (depth - min_d) / (max_d - min_d)
        depth = (depth * 255.0).to(torch.uint8)

        return depth.unsqueeze(0)


    main()
