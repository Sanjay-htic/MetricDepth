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



class IBimsDataset(Dataset):
    def __init__(self, root, target_size=384):
        self.root = root
        self.target_size = target_size

        all_files = sorted([f for f in os.listdir(root) if f.endswith(".mat")])

        self.mat_files = []
        for f in all_files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                self.mat_files.append(f)
            else:
                print(f"Warning: Missing MAT file → {fp}")

        print(f"IBIMS: {len(self.mat_files)} valid samples")

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.root, self.mat_files[idx])
        data = loadmat(file_path)
        s = data['data'][0, 0]

        # --- load RGB ---
        rgb = s['rgb']   # H×W×3 numpy
        rgb = cv2.resize(rgb, (self.target_size, self.target_size))

        # Convert to tensor CHW
        rgb = self.transform(rgb)   # will NOT resize

        # --- load Depth ---
        depth = s['depth'].astype(np.float32)
        depth = cv2.resize(depth, (self.target_size, self.target_size), cv2.INTER_NEAREST)

        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = torch.clamp(depth, 1e-4, 20.0)

        # Apply ImageNet normalization to match model expectations
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb = normalize(rgb)
        
        return {
            "image": rgb.float(),
            "depth": depth.float(),
            "name": self.mat_files[idx],
            # training helpers to align with ZoeDepth trainer API
            "mask": torch.ones_like(depth, dtype=torch.bool),
            "dataset": "ibims",
        }



class DA2KRelativeDataset(Dataset):
    def __init__(self, root_dir, annotation_file="annotations.json", transform=None, target_size=384, verbose=True):
        self.root_dir = root_dir
        self.annotation_path = annotation_file  # now full path or just filename
        self.target_size = target_size
        self.verbose = verbose

        # Load annotations
        with open(self.annotation_path, 'r') as f:
            self.raw_annotations = json.load(f)

        # Build valid image list
        self.samples = []
        print("Scanning and validating images...")
        for rel_path in tqdm(sorted(self.raw_annotations.keys()), disable=not verbose):
            # Try different possible root combinations
            candidate_paths = [
                os.path.join(self.root_dir, rel_path.replace("/", os.sep)),
                os.path.join(self.root_dir, os.path.basename(rel_path)),  # just filename
                os.path.join(self.root_dir, *rel_path.split("/")[1:]),    # strip first folder if duplicated
                rel_path,  # absolute? (unlikely)
            ]

            img_path = None
            for cand in candidate_paths:
                if os.path.exists(cand):
                    img_path = cand
                    break

            if img_path is None:
                if self.verbose:
                    print(f"Warning: Image not found, skipping: {rel_path}")
                continue

            self.samples.append({
                "image_path": img_path,
                "rel_path": rel_path,  # key in JSON
            })

        print(f"Found {len(self.samples)} / {len(self.raw_annotations)} valid images.")

        self.transform = transform or T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_path = sample_info["image_path"]
        rel_path = sample_info["rel_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image + empty annotations as fallback
            dummy = torch.zeros(3, self.target_size, self.target_size)
            return {
                "image": dummy,
                "image_path": rel_path,
                "points1": torch.zeros(0, 2, dtype=torch.long),
                "points2": torch.zeros(0, 2, dtype=torch.long),
                "closer_is_point1": torch.zeros(0, dtype=torch.bool),
                "num_pairs": 0
            }

        orig_w, orig_h = image.size
        image_tensor = self.transform(image)

        # Get annotations
        raw_annotations = self.raw_annotations[rel_path]

        points1 = []
        points2 = []
        closer_is_point1 = []

        h_ratio = self.target_size / orig_h
        w_ratio = self.target_size / orig_w

        for ann in raw_annotations:
            p1 = ann["point1"]
            p2 = ann["point2"]

            p1_resized = [int(round(p1[0] * h_ratio)), int(round(p1[1] * w_ratio))]
            p2_resized = [int(round(p2[0] * h_ratio)), int(round(p2[1] * w_ratio))]

            points1.append(p1_resized)
            points2.append(p2_resized)
            closer_is_point1.append(ann["closer_point"] == "point1")

        return {
            "image": image_tensor
        }
        
class SUNRGBDDataset(Dataset):
    def __init__(self, json_file, root_dir, target_size=392,use_bfx_depth=True, transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.use_bfx_depth = use_bfx_depth

        with open(json_file, 'r') as f:
            raw = json.load(f)

        self.data = []
        missing = 0

        for item in raw:
            rgb_path = os.path.join(root_dir, item["image"])
            depth_key = "depth_bfx" if use_bfx_depth and "depth_bfx" in item else "depth"
            depth_path = os.path.join(root_dir, item[depth_key])

            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                missing += 1
                continue

            self.data.append(item)

        print(f"SUNRGBD: {len(self.data)} valid samples, {missing} missing")
        
        self.rgb_transform = transform or T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load RGB
        rgb_path = os.path.join(self.root_dir, item["image"])
        try:
            rgb = Image.open(rgb_path).convert("RGB")
        except Exception as e:
            print(f"Error loading RGB {rgb_path}: {e}")
            rgb = Image.new("RGB", (640, 480), (0, 0, 0))

        # Load Depth (use depth_bfx if requested and exists)
        depth_key = "depth_bfx" if self.use_bfx_depth and "depth_bfx" in item else "depth"
        depth_path = os.path.join(self.root_dir, item[depth_key])

        try:
            depth_img = Image.open(depth_path)  # 16-bit PNG
            depth = np.array(depth_img, dtype=np.float32)
            # SUN RGB-D depth is in millimeters → convert to meters
            depth = depth / 1000.0
        except Exception as e:
            print(f"Error loading depth {depth_path}: {e}")
            depth = np.zeros((480, 640), dtype=np.float32)

        # Resize & tensor
        rgb_tensor = self.rgb_transform(rgb)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # 1×H×W
        depth_tensor = self.depth_transform(depth_tensor)
        depth_tensor = torch.clamp(depth_tensor, 0.1, 10.0)  # indoor range

        return {
            "image": rgb_tensor,           # (3, H, W)
            "depth": depth_tensor,         # (1, H, W)
            "rgb_path": item["image"],
            "depth_path": item[depth_key],
            "dataset": "sunrgbd",
            "mask": torch.ones_like(depth_tensor, dtype=torch.bool),
            #"annotations": item.get("annotations", [])  # optional: for future use
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

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)

        depth = depth.astype(np.float32)

        # mm → meters (adjust if your depths are already in meters)
        if depth.max() > 100:
            depth /= 1000.0

        depth = cv2.resize(
            depth, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = depth > 0

        depth = torch.clamp(depth, 0.1, 80.0)  # Adjust max if needed for your phantom data

        return depth, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Use raw strings for safety with backslashes
        rgb_path = os.path.join(self.root_dir, row["rgb_path"].lstrip("\\/"))
        rgb_path = r'{}'.format(rgb_path)  # Or simply rf"{os.path.join(...)}" in Python 3.12+

        depth_path = os.path.join(self.root_dir, row["depth_raw_path"].lstrip("\\/"))
        depth_path = r'{}'.format(depth_path)

        rgb = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb)

        depth, mask = self.load_depth(depth_path)

        return {
            "image": rgb,
            "depth": depth,
            "mask": mask,
            "dataset": "realsense",
        }