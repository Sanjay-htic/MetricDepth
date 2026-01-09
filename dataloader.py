import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import cv2
import csv
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as TF
from glob import glob
import torch.nn.functional as F

class SUNRGBDDataset(Dataset):
    def __init__(self, root_dir, img_size=384):
        self.root_dir = root_dir
        self.img_size = img_size

        self.rgb_paths = []
        self.depth_paths = []
        self.intrinsic_paths = []

        print(f"Scanning root directory: {root_dir}")

        # Check if root_dir exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")

        scene_count = 0
        for scene in sorted(os.listdir(root_dir)):
            scene_path = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_path):
                print(f"Skipping non-directory: {scene}")
                continue

            rgb_dir = os.path.join(scene_path, "image")
            depth_dir = os.path.join(scene_path, "depth_bfx")
            intrinsic_path = os.path.join(scene_path, "intrinsics.txt")
            # Use more flexible glob patterns
            rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")) + 
                            glob(os.path.join(rgb_dir, "*.png")) + 
                            glob(os.path.join(rgb_dir, "*.jpeg")))
            
            depth_files = sorted(glob(os.path.join(depth_dir, "*.png")))

            if len(rgb_files) == 0:
                print(f"  → No RGB images found in {rgb_dir}")
                continue
            if len(depth_files) == 0:
                print(f"  → No depth images found in {depth_dir}")
                continue

            if len(rgb_files) != len(depth_files):
                print(f"  → Mismatch: {len(rgb_files)} RGB vs {len(depth_files)} depth → skipping scene")
                continue

            self.rgb_paths.extend(rgb_files)
            self.depth_paths.extend(depth_files)
            self.intrinsic_paths.extend([intrinsic_path] * len(rgb_files))

            scene_count += 1

        print(f"Found {scene_count} valid scenes")
        
        # RGB → Tensor
        self.rgb_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


        # HARD SAFETY CHECK
        if len(self.rgb_paths) == 0:
            raise AssertionError("SUNRGBD dataset is empty - no valid scenes found. Check paths and file extensions.")

        assert len(self.rgb_paths) == len(self.depth_paths) == len(self.intrinsic_paths), \
            "RGB / Depth / Intrinsic length mismatch"

        print(f"[SUNRGBD] Successfully loaded {len(self.rgb_paths)} samples")
    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # ---------- RGB ----------
        rgb_pil = Image.open(self.rgb_paths[idx]).convert("RGB")
        orig_w, orig_h = rgb_pil.size
        rgb = self.rgb_transform(rgb_pil)

        # ---------- DEPTH ----------
        depth = np.array(Image.open(self.depth_paths[idx]), dtype=np.float32)
        depth = depth / 1000.0  # mm → meters

        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze(0)

        mask = depth > 0
        depth = torch.clamp(depth, 0.1, 80.0)

        # ---------- INTRINSICS ----------
        K = np.loadtxt(self.intrinsic_paths[idx]).astype(np.float32)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        intrinsics = torch.tensor(
            [fx * sx, fy * sy, cx * sx, cy * sy],
            #[fx, fy, cx, cy],
            dtype=torch.float32,
        )

        return {
            "image": rgb,            # [3,384,384]
            "depth": depth,          # [1,384,384]
            "mask": mask,            # [1,384,384]
            "intrinsics": intrinsics,# [4]
            "dataset": "sunrgbd",
        }
                                            
class NYUDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=384, is_test=False):
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_test = is_test

        self.rgb_paths = []
        self.depth_paths = []

        missing = 0

        # ---- Correct CSV parsing ----
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(self.rgb_paths) >= 100:
                    break

                if len(row) < 2:
                    continue

                rgb_rel, depth_rel = row[0].strip(), row[1].strip()
                rgb_path = os.path.join(root_dir, rgb_rel)
                depth_path = os.path.join(root_dir, depth_rel)

                if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                    missing += 1
                    continue

                self.rgb_paths.append(rgb_path)
                self.depth_paths.append(depth_path)

        print(f"NYU: {len(self.rgb_paths)} valid samples, {missing} missing")

        assert len(self.rgb_paths) > 0, "NYU dataset is empty — check CSV paths"

        self.rgb_transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        # ---- RGB ----
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb)

        # ---- Depth (uint16 PNG, millimeters → meters) ----
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth / 1000.0  # mm → meters

        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # Resize depth safely (NO PIL)
        depth = TF.resize(
            depth,
            size=[self.target_size, self.target_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # Valid depth mask
        mask = depth > 0

        depth = torch.clamp(depth, 0.1, 10.0)

        return {
            "image": rgb,
            "depth": depth,
            "mask": mask,
            "dataset": "nyu",
            "rgb_path": os.path.basename(rgb_path),
        }

class RealsenseDataset(Dataset):
    def __init__(self, csv_file, root_dir, img_size=384):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_size = img_size

        self.rgb_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def load_depth(self, path, depth_scale):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)

        depth = depth.astype(np.float32)

        # Proper scaling using CSV
        depth *= float(depth_scale)

        depth = cv2.resize(
            depth,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        mask = depth > 0

        depth = torch.clamp(depth, 0.1, 80.0)

        return depth, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        rgb_path = os.path.join(self.root_dir, row["rgb_path"].lstrip("\\/"))
        depth_path = os.path.join(self.root_dir, row["depth_raw_path"].lstrip("\\/"))

        # ---- RGB ----
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb)

        # ---- Depth ----
        depth, mask = self.load_depth(
            depth_path,
            depth_scale=row["depth_scale"]
        )

        # ---- Intrinsics (scaled correctly) ----
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])

        orig_w = float(row["width"])
        orig_h = float(row["height"])

        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

        return {
            "image": rgb,
            "depth": depth,
            "mask": mask,
            "intrinsics": intrinsics,
            "dataset": "realsense",
        }
