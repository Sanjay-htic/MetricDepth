"""
Training script for DepthAnything-V2 + ZoeDepth combined model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ZoeDepth'))

# Import model
from da2_zoe_model import DepthAnythingV2ZoeDepth

# Import dataset loaders
from dataloader import IBimsDataset, SUNRGBDDataset, NYUDataset, DA2KRelativeDataset

# Import losses
from zoedepth.trainers.loss import SILogLoss, GradL1Loss
from metric_depth.util.loss import SiLogLoss as MetricSiLogLoss
from metric_depth.util.metric import eval_depth


class CombinedLoss(nn.Module):
    """Combined loss function from both ZoeDepth and DepthAnything."""
    
    def __init__(self, silog_weight=1.0, grad_weight=0.5, metric_silog_weight=0.5):
        super().__init__()
        self.silog_loss = SILogLoss(beta=0.15)
        self.grad_loss = GradL1Loss()
        self.metric_silog_loss = MetricSiLogLoss(lambd=0.5)
        
        self.silog_weight = silog_weight
        self.grad_weight = grad_weight
        self.metric_silog_weight = metric_silog_weight
    
    def forward(self, prediction, target, mask=None):
        """
        Args:
            prediction: Model output dict with 'metric_depth' key
            target: Ground truth depth (B, 1, H, W) or (B, H, W)
            mask: Valid mask (B, 1, H, W) or (B, H, W)
        """
        pred = prediction['metric_depth']
        
        # Ensure correct dimensions
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        
        # Ensure same spatial size
        if pred.shape[-2:] != target.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode='bilinear', align_corners=True
            )
        
        # Compute losses
        loss_silog = self.silog_loss(pred, target, mask)
        loss_grad = self.grad_loss(pred, target, mask)
        
        # Metric SiLog loss
        pred_squeezed = pred.squeeze(1) if pred.ndim == 4 else pred
        target_squeezed = target.squeeze(1) if target.ndim == 4 else target
        mask_squeezed = mask.squeeze(1) if mask.ndim == 4 else mask
        loss_metric_silog = self.metric_silog_loss(pred_squeezed, target_squeezed, mask_squeezed)
        
        # Total loss
        total_loss = (
            self.silog_weight * loss_silog +
            self.grad_weight * loss_grad +
            self.metric_silog_weight * loss_metric_silog
        )
        
        return {
            'total': total_loss,
            'silog': loss_silog,
            'grad': loss_grad,
            'metric_silog': loss_metric_silog
        }


def create_dataloader(config, split='train'):
    """Create data loader from configuration."""
    datasets = []
    
    if 'ibims' in config.datasets and split == 'train':
        if os.path.exists(config.ibims_root):
            ibims = IBimsDataset(config.ibims_root, target_size=config.target_size)
            datasets.append(ibims)
            print(f"Added IBIMS dataset: {len(ibims)} samples")
    
    if 'sunrgbd' in config.datasets:
        if os.path.exists(config.sunrgbd_json) and os.path.exists(config.sunrgbd_root):
            sunrgbd = SUNRGBDDataset(
                config.sunrgbd_json,
                config.sunrgbd_root,
                target_size=config.target_size,
                use_bfx_depth=config.use_bfx_depth
            )
            datasets.append(sunrgbd)
            print(f"Added SUNRGBD dataset: {len(sunrgbd)} samples")
    
    if 'nyu' in config.datasets:
        if os.path.exists(config.nyu_csv) and os.path.exists(config.nyu_root):
            nyu = NYUDataset(
                config.nyu_csv,
                config.nyu_root,
                target_size=config.target_size,
                is_test=(split == 'test')
            )
            datasets.append(nyu)
            print(f"Added NYU dataset: {len(nyu)} samples")
    
    if 'da2k' in config.datasets and split == 'train':
        if os.path.exists(config.da2k_root) and os.path.exists(config.da2k_annotation):
            da2k = DA2KRelativeDataset(
                config.da2k_root,
                config.da2k_annotation,
                target_size=config.target_size
            )
            datasets.append(da2k)
            print(f"Added DA2K dataset: {len(da2k)} samples")
    
    if len(datasets) == 0:
        raise ValueError(f"No valid datasets found for split '{split}'")
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    # Create data loader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=(split == 'train')
    )
    
    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, config):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_dict = {'silog': 0.0, 'grad': 0.0, 'metric_silog': 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        image = batch['image'].to(device)
        depth = batch['depth'].to(device)
        mask = batch.get('mask', torch.ones_like(depth, dtype=torch.bool)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(image)
        
        # Compute loss
        losses = criterion(output, depth, mask)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        loss_dict['silog'] += losses['silog'].item()
        loss_dict['grad'] += losses['grad'].item()
        loss_dict['metric_silog'] += losses['metric_silog'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'silog': f"{losses['silog'].item():.4f}",
            'grad': f"{losses['grad'].item():.4f}"
        })
        
        # Log to tensorboard (every N batches)
        if batch_idx % config.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', losses['total'].item(), global_step)
            writer.add_scalar('Train/BatchSILog', losses['silog'].item(), global_step)
            writer.add_scalar('Train/BatchGrad', losses['grad'].item(), global_step)
            writer.add_scalar('Train/BatchMetricSILog', losses['metric_silog'].item(), global_step)
    
    # Average losses
    avg_loss = total_loss / num_batches
    for key in loss_dict:
        loss_dict[key] /= num_batches
    
    return avg_loss, loss_dict

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer, config):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    loss_dict = {'silog': 0.0, 'grad': 0.0, 'metric_silog': 0.0}
    num_batches = 0
    
    # Limit validation to 20% of dataset
    total_val_samples = len(dataloader.dataset)
    max_val_samples = int(0.2 * total_val_samples)
    max_val_batches = max(1, max_val_samples // config.batch_size)
    
    # Metrics - accumulate properly for pixel-wise calculation
    all_preds = []
    all_targets = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc=f"Val {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Stop after processing 20% of dataset
        if batch_idx >= max_val_batches:
            break
            
        image = batch['image'].to(device)
        depth = batch['depth'].to(device)
        mask = batch.get('mask', torch.ones_like(depth, dtype=torch.bool)).to(device)
        
        # Forward pass
        output = model(image)
        pred = output['metric_depth']
        
        # Ensure same size
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode='bilinear', align_corners=True
            )
        
        # Compute loss
        losses = criterion(output, depth, mask)
        
        # Accumulate losses
        total_loss += losses['total'].item()
        loss_dict['silog'] += losses['silog'].item()
        loss_dict['grad'] += losses['grad'].item()
        loss_dict['metric_silog'] += losses['metric_silog'].item()
        num_batches += 1
        
        # Store predictions and targets for metric computation
        # Apply proper masking: valid pixels = (mask == True) AND (depth > 0)
        valid_mask = mask.bool() & (depth > 0)
        
        # Clamp predictions to avoid invalid values
        eps = 1e-6
        pred_clamped = torch.clamp(pred, min=eps)
        depth_clamped = torch.clamp(depth, min=eps)
        
        # Store for batch-wise metric computation (keep spatial structure)
        all_preds.append(pred_clamped.cpu())
        all_targets.append(depth_clamped.cpu())
        all_masks.append(valid_mask.cpu())
        
        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 
                         'batches': f"{batch_idx+1}/{max_val_batches}"})
    
    # Average losses
    avg_loss = total_loss / num_batches
    for key in loss_dict:
        loss_dict[key] /= num_batches
    
    # Compute metrics on all collected batches properly
    metrics_accum = {
        'd1': 0.0, 'd2': 0.0, 'd3': 0.0,
        'abs_rel': 0.0, 'sq_rel': 0.0,
        'rmse': 0.0, 'rmse_log': 0.0,
        'log10': 0.0, 'silog': 0.0
    }
    
    if len(all_preds) > 0:
        # Concatenate all batches
        pred_all = torch.cat(all_preds, dim=0)  # (N, 1, H, W)
        target_all = torch.cat(all_targets, dim=0)  # (N, 1, H, W)
        mask_all = torch.cat(all_masks, dim=0)  # (N, 1, H, W)
        
        # Apply mask: only compute metrics on valid pixels
        # Convert to (N, H, W) for easier masking
        pred_flat = pred_all.squeeze(1)  # (N, H, W)
        target_flat = target_all.squeeze(1)  # (N, H, W)
        mask_flat = mask_all.squeeze(1)  # (N, H, W)
        
        # Get valid pixels only
        pred_valid = pred_flat[mask_flat]  # (M,) - flattened valid pixels
        target_valid = target_flat[mask_flat]  # (M,) - flattened valid pixels
        
        if len(pred_valid) > 0:
            # Reshape to (1, 1, M) for eval_depth function
            # This maintains compatibility with the existing metric function
            pred_for_metrics = pred_valid.unsqueeze(0).unsqueeze(0)  # (1, 1, M)
            target_for_metrics = target_valid.unsqueeze(0).unsqueeze(0)  # (1, 1, M)
            
            # Compute metrics
            metrics = eval_depth(pred_for_metrics, target_for_metrics)
            
            # Store metrics
            for key in metrics_accum:
                metrics_accum[key] = metrics[key]
        else:
            print("Warning: No valid pixels found for metric computation")
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/SILog', loss_dict['silog'], epoch)
    writer.add_scalar('Val/Grad', loss_dict['grad'], epoch)
    writer.add_scalar('Val/MetricSILog', loss_dict['metric_silog'], epoch)
    
    for key, value in metrics_accum.items():
        writer.add_scalar(f'Val/{key.upper()}', value, epoch)
    
    return avg_loss, loss_dict, metrics_accum


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save latest
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    
    # Save best
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', float('inf'))
    return epoch, loss


def main(config):
    """Main training function."""
    # Set device
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Create model
    print("Creating model...")
    model = DepthAnythingV2ZoeDepth(
        encoder=config.encoder,
        n_bins=config.n_bins,
        bin_centers_type=config.bin_centers_type,
        bin_embedding_dim=config.bin_embedding_dim,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        n_attractors=config.n_attractors,
        attractor_alpha=config.attractor_alpha,
        attractor_gamma=config.attractor_gamma,
        attractor_kind=config.attractor_kind,
        attractor_type=config.attractor_type,
        features=256,
        out_channels=[256, 256, 256, 256],  
        train_encoder=config.train_encoder,
        encoder_lr_factor=config.encoder_lr_factor
    ).to(device)
    
    # Load pretrained checkpoint if specified
    if config.pretrained_checkpoint:     
        print(f"Loading pretrained checkpoint from {config.pretrained_checkpoint}")
        checkpoint = torch.load(config.pretrained_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(config, split='train')
    
    val_loader = None
    if config.validate:
        try:
            val_loader = create_dataloader(config, split='val')
            print(f"Validation dataset size: {len(val_loader.dataset)} samples")
            print(f"Validation will use ~20% of dataset (~{int(0.2 * len(val_loader.dataset))} samples)")
        except Exception as e:
            print(f"Warning: Could not create validation loader: {e}")
            print("Skipping validation")
            config.validate = False
    
    # Create loss function
    criterion = CombinedLoss(
        silog_weight=config.silog_weight,
        grad_weight=config.grad_weight,
        metric_silog_weight=config.metric_silog_weight
    ).to(device)
    
    # Create optimizer
    if config.use_lr_params:
        params = model.get_lr_params(config.lr)
    else:
        params = model.parameters()
    
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.lr_step_size, 
        gamma=config.lr_gamma
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume:
        print(f"Resuming from checkpoint: {config.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, config.resume, device)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        # Log epoch losses
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/EpochSILog', train_loss_dict['silog'], epoch)
        writer.add_scalar('Train/EpochGrad', train_loss_dict['grad'], epoch)
        writer.add_scalar('Train/EpochMetricSILog', train_loss_dict['metric_silog'], epoch)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f} | SILog: {train_loss_dict['silog']:.4f} | "
              f"Grad: {train_loss_dict['grad']:.4f} | MetricSILog: {train_loss_dict['metric_silog']:.4f}")
        
        # Validate
        if config.validate and val_loader is not None:
            val_loss, val_loss_dict, val_metrics = validate(
                model, val_loader, criterion, device, epoch, writer, config
            )
            print(f"Val Loss: {val_loss:.4f} | SILog: {val_loss_dict['silog']:.4f}")
            print(f"Val Metrics - D1: {val_metrics['d1']:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
                  f"AbsRel: {val_metrics['abs_rel']:.4f}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, config.save_dir, is_best=is_best)
        print(f"Saved checkpoint for epoch {epoch}")
    
    print("\nTraining completed!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DepthAnything-V2 + ZoeDepth')
    
    # Model configuration
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--n-bins', type=int, default=64)
    parser.add_argument('--bin-centers-type', type=str, default='softplus', 
                       choices=['normed', 'softplus', 'hybrid1', 'hybrid2'])
    parser.add_argument('--bin-embedding-dim', type=int, default=128)
    parser.add_argument('--min-depth', type=float, default=1e-3)
    parser.add_argument('--max-depth', type=float, default=10.0)
    parser.add_argument('--n-attractors', type=int, nargs='+', default=[16, 8, 4, 1])
    parser.add_argument('--attractor-alpha', type=float, default=300)
    parser.add_argument('--attractor-gamma', type=int, default=2)
    parser.add_argument('--attractor-kind', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--attractor-type', type=str, default='inv', choices=['inv', 'exp'])
    parser.add_argument('--train-encoder', action='store_true')
    parser.add_argument('--encoder-lr-factor', type=float, default=10.0)
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-step-size', type=int, default=15)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--use-lr-params', action='store_true', default=True)
    
    # Loss configuration
    parser.add_argument('--silog-weight', type=float, default=1.0)
    parser.add_argument('--grad-weight', type=float, default=0.5)
    parser.add_argument('--metric-silog-weight', type=float, default=0.5)
    
    # Data configuration
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['nyu', 'sunrgbd'], 
                       choices=['nyu', 'sunrgbd', 'ibims', 'da2k'])
    parser.add_argument('--target-size', type=int, default=384)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true', default=True)
    
    # Dataset paths
    parser.add_argument('--nyu-csv', type=str, default='')
    parser.add_argument('--nyu-root', type=str, default='')
    parser.add_argument('--sunrgbd-json', type=str, default='')
    parser.add_argument('--sunrgbd-root', type=str, default='')
    parser.add_argument('--use-bfx-depth', action='store_true', default=False)
    parser.add_argument('--ibims-root', type=str, default='')
    parser.add_argument('--da2k-root', type=str, default='')
    parser.add_argument('--da2k-annotation', type=str, default='')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--pretrained-checkpoint', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    
    # Other
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--validate', action='store_true', default=True)
    
    args = parser.parse_args()
    main(args)