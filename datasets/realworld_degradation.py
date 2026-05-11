"""
datasets/realworld_degradation.py
----------------------------------
Real-world image degradation pipeline for training robust SR models.

Instead of clean bicubic downsampling, applies a realistic degradation chain:
    HR → Gaussian blur → Bicubic downsample → Gaussian noise → JPEG compression

This simulates real camera/compression artifacts that classical SR models struggle with.
"""

import random
import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from datasets.ir_dataset import IMG_EXTENSIONS, is_image, random_crop, augment


class RealWorldDegradation:
    """
    Configurable real-world degradation pipeline.

    Args:
        scale        : downscale factor (2 or 4)
        blur_sigma   : Gaussian blur sigma range (min, max)
        noise_sigma  : Gaussian noise sigma range (min, max), in [0, 255] scale
        jpeg_quality : JPEG compression quality range (min, max)
    """

    def __init__(
        self,
        scale: int = 4,
        blur_sigma: tuple = (0.5, 3.0),
        noise_sigma: tuple = (5, 30),
        jpeg_quality: tuple = (30, 95),
    ):
        self.scale = scale
        self.blur_sigma = blur_sigma
        self.noise_sigma = noise_sigma
        self.jpeg_quality = jpeg_quality

    def __call__(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply degradation pipeline to an HR tensor.

        Args:
            hr_tensor : (C, H, W) float tensor in [0, 1]

        Returns:
            lr_tensor : (C, H//scale, W//scale) float tensor in [0, 1]
        """
        _, H, W = hr_tensor.shape
        lh, lw = H // self.scale, W // self.scale

        # 1. Gaussian blur
        sigma = random.uniform(*self.blur_sigma)
        hr_pil = TF.to_pil_image(hr_tensor)
        hr_blurred = hr_pil.filter(ImageFilter.GaussianBlur(radius=int(2 * sigma + 1)))

        # 2. Bicubic downsample
        lr_pil = hr_blurred.resize((lw, lh), Image.BICUBIC)

        # 3. Gaussian noise
        lr_np = np.array(lr_pil).astype(np.float32)
        noise_std = random.uniform(*self.noise_sigma)
        noise = np.random.randn(*lr_np.shape).astype(np.float32) * noise_std
        lr_np = np.clip(lr_np + noise, 0, 255)

        # 4. JPEG compression
        quality = random.randint(*self.jpeg_quality)
        lr_pil = Image.fromarray(lr_np.astype(np.uint8))
        buffer = io.BytesIO()
        lr_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        lr_pil = Image.open(buffer).convert("RGB")

        lr_tensor = TF.to_tensor(lr_pil)
        return lr_tensor


class DIV2KRealWorldDataset(Dataset):
    """
    DIV2K training dataset with real-world degradation.

    Same HR images as classical training, but LR is generated with
    blur + noise + JPEG instead of clean bicubic.

    Args:
        hr_dir       : path to DIV2K_train_HR folder
        scale        : downscale factor
        patch_size   : LR patch size for random crop
        degradation  : RealWorldDegradation instance
    """

    def __init__(self, hr_dir: str, scale: int = 4, patch_size: int = 48,
                 degradation: RealWorldDegradation = None):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.degradation = degradation or RealWorldDegradation(scale=scale)

        hr_path = Path(hr_dir)
        self.hr_paths = sorted([p for p in hr_path.iterdir() if is_image(p)])
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No images found in {hr_dir}")

        print(
            f"[Dataset] DIV2K Real-World | HR images: {len(self.hr_paths)} | "
            f"scale: ×{scale} | degradation: blur+noise+jpeg"
        )

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = TF.to_tensor(Image.open(str(self.hr_paths[idx])).convert("RGB"))

        # Apply real-world degradation
        lr = self.degradation(hr)

        # Random aligned crop
        _, lh, lw = lr.shape
        if min(lh, lw) >= self.patch_size:
            lr, hr = random_crop(hr, lr, self.patch_size, self.scale)

        # Augmentation
        lr, hr = augment(lr, hr)

        return {"lq": lr, "gt": hr}


def build_realworld_train_loader(args) -> DataLoader:
    """Build DIV2K training DataLoader with real-world degradation."""
    degradation = RealWorldDegradation(
        scale=args.scale,
        blur_sigma=(0.5, 3.0),
        noise_sigma=(5, 30),
        jpeg_quality=(30, 95),
    )
    dataset = DIV2KRealWorldDataset(
        hr_dir=args.ir_data_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        degradation=degradation,
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


def apply_fixed_degradation(hr_tensor: torch.Tensor, scale: int = 4,
                             blur_sigma: float = 1.5, noise_sigma: float = 15.0,
                             jpeg_quality: int = 50) -> torch.Tensor:
    """
    Apply a fixed (deterministic) degradation for evaluation purposes.
    Same pipeline but with fixed parameters instead of random ranges.
    """
    _, H, W = hr_tensor.shape
    lh, lw = H // scale, W // scale

    hr_pil = TF.to_pil_image(hr_tensor)
    hr_blurred = hr_pil.filter(ImageFilter.GaussianBlur(radius=int(2 * blur_sigma + 1)))
    lr_pil = hr_blurred.resize((lw, lh), Image.BICUBIC)

    lr_np = np.array(lr_pil).astype(np.float32)
    np.random.seed(42)
    noise = np.random.randn(*lr_np.shape).astype(np.float32) * noise_sigma
    lr_np = np.clip(lr_np + noise, 0, 255)

    lr_pil = Image.fromarray(lr_np.astype(np.uint8))
    buffer = io.BytesIO()
    lr_pil.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    lr_pil = Image.open(buffer).convert("RGB")

    return TF.to_tensor(lr_pil)
