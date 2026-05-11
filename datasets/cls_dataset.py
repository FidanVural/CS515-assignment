"""
datasets/cls_dataset.py
------------------------
ImageFolder DataLoader for SwinV2 classification fine-tuning.

Supports any ImageFolder-structured dataset (ImageNet-1K, Tiny ImageNet-200, etc.)

Expected directory structure:
    data_dir/
        train/
            <class_folder>/  (e.g. n01440764/)
            ...
        val/
            <class_folder>/
            ...

Augmentation pipeline follows the SwinV2 paper recipe:
  Train : RandomResizedCrop(256) + RandomHorizontalFlip + RandAugment(m=9,n=2)
          + Normalize(ImageNet mean/std)
  Val   : Resize(292) + CenterCrop(256) + Normalize
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# ImageNet normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 256) -> transforms.Compose:
    """
    Training augmentation pipeline (SwinV2 paper recipe):
      1. RandomResizedCrop  (scale [0.08, 1.0])
      2. RandomHorizontalFlip
      3. RandAugment (magnitude=9, num_ops=2)
      4. ToTensor + Normalize
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform(img_size: int = 256) -> transforms.Compose:
    """
    Validation transform:
      1. Resize to 292 (shortest edge)  — standard 1.14× ratio
      2. CenterCrop to img_size
      3. ToTensor + Normalize
    """
    resize_size = int(img_size * 292 / 256)   # ~292 for 256
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_cls_dataloaders(args) -> tuple:
    """
    Build train and validation DataLoaders for classification.

    Args:
        args : parsed arguments (data_dir, img_size, batch_size, num_workers)

    Returns:
        (train_loader, val_loader)
    """
    data_root = Path(args.data_dir)
    train_dir = data_root / "train"
    val_dir   = data_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}\n"
            "Expected structure: data_dir/train/<class_folders>/"
        )
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=build_train_transform(args.img_size),
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=build_val_transform(args.img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print(
        f"[Dataset] {len(train_dataset.classes)}-class ImageFolder | "
        f"Train: {len(train_dataset):,} | "
        f"Val: {len(val_dataset):,} | "
        f"Classes: {len(train_dataset.classes)}"
    )
    return train_loader, val_loader
