"""
datasets/ir_dataset.py
-----------------------
Dataset loaders for Swin2SR super-resolution training and evaluation.

Training  : DIV2K — 800 high-resolution images; LR generated on-the-fly
            via bicubic downsampling (×2 or ×4).
Evaluation : Set5, Set14, BSD100 — pre-downsampled LR + HR pairs.

Expected directory layout:
    data/ir/
        DIV2K/
            DIV2K_train_HR/          ← 800 HR images (0001.png … 0800.png)
        Set5/
            HR/                      ← 5 HR images
            LR_bicubic/
                X2/                  ← 5 LR images (×2)
                X4/                  ← 5 LR images (×4)
        Set14/                       ← same structure
        BSD100/                      ← same structure
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def load_image(path: str) -> torch.Tensor:
    """Load image as float tensor in [0, 1], shape (C, H, W)."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)   # (3, H, W), float32 in [0,1]


def random_crop(hr: torch.Tensor, lr: torch.Tensor,
                patch_size: int, scale: int) -> tuple:
    """
    Randomly crop aligned HR/LR patches.

    Args:
        hr         : (C, H, W) float tensor
        lr         : (C, h, w) float tensor where h = H//scale
        patch_size : LR patch size (HR patch = patch_size * scale)
        scale      : upscale factor

    Returns:
        (lr_patch, hr_patch)
    """
    _, lh, lw = lr.shape
    lr_h = random.randint(0, lh - patch_size)
    lr_w = random.randint(0, lw - patch_size)
    hr_h = lr_h * scale
    hr_w = lr_w * scale
    hp   = patch_size * scale

    lr_patch = lr[:, lr_h:lr_h + patch_size, lr_w:lr_w + patch_size]
    hr_patch = hr[:, hr_h:hr_h + hp,         hr_w:hr_w + hp]
    return lr_patch, hr_patch


def augment(lr: torch.Tensor, hr: torch.Tensor) -> tuple:
    """Random horizontal flip + 90° rotation (applied identically to LR and HR)."""
    # Horizontal flip
    if random.random() > 0.5:
        lr = TF.hflip(lr)
        hr = TF.hflip(hr)
    # Random 90° rotation
    k = random.randint(0, 3)
    if k > 0:
        lr = torch.rot90(lr, k, dims=[1, 2])
        hr = torch.rot90(hr, k, dims=[1, 2])
    return lr, hr


# ------------------------------------------------------------------ #
#  Training Dataset (DIV2K)                                           #
# ------------------------------------------------------------------ #

class DIV2KDataset(Dataset):
    """
    DIV2K HR-only dataset; generates LR on-the-fly with bicubic downsampling.

    Args:
        hr_dir     : path to DIV2K_train_HR folder
        scale      : downscale factor (2 or 4)
        patch_size : LR patch size for random crop
    """

    def __init__(self, hr_dir: str, scale: int = 4, patch_size: int = 48):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size

        hr_path = Path(hr_dir)
        if not hr_path.exists():
            raise FileNotFoundError(f"DIV2K HR directory not found: {hr_dir}")

        self.hr_paths = sorted([p for p in hr_path.iterdir() if is_image(p)])
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No images found in {hr_dir}")

        print(f"[Dataset] DIV2K | HR images: {len(self.hr_paths)} | scale: ×{scale}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = load_image(str(self.hr_paths[idx]))

        # Generate LR via bicubic downsampling
        _, H, W = hr.shape
        lh, lw = H // self.scale, W // self.scale
        lr = TF.resize(hr, [lh, lw],
                       interpolation=TF.InterpolationMode.BICUBIC,
                       antialias=True)
        lr = lr.clamp(0, 1)

        # Random aligned crop
        if min(lh, lw) >= self.patch_size:
            lr, hr = random_crop(hr, lr, self.patch_size, self.scale)

        # Augmentation
        lr, hr = augment(lr, hr)

        return {"lq": lr, "gt": hr}


# ------------------------------------------------------------------ #
#  Test / Evaluation Dataset (Set5, Set14, BSD100 …)                 #
# ------------------------------------------------------------------ #

class SRTestDataset(Dataset):
    """
    Paired LR-HR dataset for evaluation.

    Supports two modes:
      1. folder_lq + folder_gt : pre-downsampled LR/HR pairs
      2. folder_gt only        : HR-only; LR generated on-the-fly

    Args:
        folder_lq  : path to LR images (or None)
        folder_gt  : path to HR images
        scale      : upscale factor
    """

    def __init__(self, folder_gt: str, scale: int = 4,
                 folder_lq: Optional[str] = None):
        super().__init__()
        self.scale = scale
        self.generate_lr = (folder_lq is None)

        gt_path = Path(folder_gt)
        self.gt_paths = sorted([p for p in gt_path.iterdir() if is_image(p)])

        if folder_lq is not None:
            lq_path = Path(folder_lq)
            self.lq_paths = sorted([p for p in lq_path.iterdir() if is_image(p)])
            assert len(self.lq_paths) == len(self.gt_paths), (
                f"LQ ({len(self.lq_paths)}) and GT ({len(self.gt_paths)}) "
                "image counts do not match."
            )
        else:
            self.lq_paths = None

        print(
            f"[Dataset] Test | GT: {folder_gt} | "
            f"Images: {len(self.gt_paths)} | scale: ×{scale}"
        )

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        hr = load_image(str(self.gt_paths[idx]))

        if self.lq_paths is not None:
            lr = load_image(str(self.lq_paths[idx]))
        else:
            _, H, W = hr.shape
            lr = TF.resize(hr, [H // self.scale, W // self.scale],
                           interpolation=TF.InterpolationMode.BICUBIC,
                           antialias=True).clamp(0, 1)

        return {
            "lq":   lr,
            "gt":   hr,
            "name": self.gt_paths[idx].stem,
        }


# ------------------------------------------------------------------ #
#  Builders                                                           #
# ------------------------------------------------------------------ #

def build_ir_train_loader(args) -> DataLoader:
    """Build DIV2K training DataLoader."""
    dataset = DIV2KDataset(
        hr_dir=args.ir_data_dir,
        scale=args.scale,
        patch_size=args.patch_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )


def build_ir_test_loader(folder_gt: str, scale: int,
                          folder_lq: Optional[str] = None,
                          num_workers: int = 4) -> DataLoader:
    """Build evaluation DataLoader for a single test set (Set5, Set14, …)."""
    dataset = SRTestDataset(
        folder_gt=folder_gt,
        scale=scale,
        folder_lq=folder_lq,
    )
    # batch_size=1 for evaluation (images may differ in size)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
