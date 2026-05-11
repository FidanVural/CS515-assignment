"""
parameters.py
-------------
All argument definitions for the Swin2 multi-task pipeline.
Shared arguments apply to both tasks; task-specific groups are
activated based on --task classification | IR.

Usage examples:
  python main.py --task classification --mode train --data_dir data/imagenet
  python main.py --task IR --mode test --scale 4 --folder_lq data/ir/Set5/LR/X4
"""

import argparse
import yaml
import os


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Swin2 Multi-Task Pipeline: Classification & Image Super-Resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    #  Shared / Global Arguments                                          #
    # ------------------------------------------------------------------ #
    shared = parser.add_argument_group("Shared")
    shared.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "IR"],
        help="Task to run: 'classification' or 'IR' (super-resolution)",
    )
    shared.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Pipeline mode",
    )
    shared.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda | cpu",
    )
    shared.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (classification: 128, IR: 32 recommended)",
    )
    shared.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    shared.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate",
    )
    shared.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (AdamW)",
    )
    shared.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    shared.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    shared.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    shared.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    shared.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log every N batches",
    )
    shared.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args with YAML values)",
    )

    # ------------------------------------------------------------------ #
    #  Classification-Specific Arguments                                  #
    # ------------------------------------------------------------------ #
    cls_group = parser.add_argument_group(
        "Classification",
        "Arguments specific to the image classification task",
    )
    cls_group.add_argument(
        "--data_dir",
        type=str,
        default="data/imagenet",
        help="Root directory of the dataset (must contain train/ and val/ in ImageFolder format)",
    )
    cls_group.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of output classes",
    )
    cls_group.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="SwinV2 model size variant",
    )
    cls_group.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (square) for SwinV2",
    )
    cls_group.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Path to SimMIM / custom pretrained .pth checkpoint to fine-tune from",
    )
    cls_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing epsilon for cross-entropy loss",
    )
    cls_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Linear LR warm-up epochs before cosine decay",
    )
    cls_group.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (fp16) training",
    )

    # ------------------------------------------------------------------ #
    #  IR (Super-Resolution) Specific Arguments                           #
    # ------------------------------------------------------------------ #
    ir_group = parser.add_argument_group(
        "IR (Super-Resolution)",
        "Arguments specific to the image super-resolution task",
    )
    ir_group.add_argument(
        "--ir_data_dir",
        type=str,
        default="data/ir/DIV2K",
        help="Root directory of DIV2K high-resolution training images",
    )
    ir_group.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="Super-resolution upscale factor",
    )
    ir_group.add_argument(
        "--patch_size",
        type=int,
        default=48,
        help="LR patch size for training crops (HR patch = patch_size * scale)",
    )
    ir_group.add_argument(
        "--ir_model_path",
        type=str,
        default=None,
        help="Path to pretrained Swin2SR .pth weights",
    )
    ir_group.add_argument(
        "--folder_lq",
        type=str,
        default=None,
        help="[TEST] Folder containing low-quality (LR) input images",
    )
    ir_group.add_argument(
        "--folder_gt",
        type=str,
        default=None,
        help="[TEST] Folder containing ground-truth (HR) reference images",
    )
    ir_group.add_argument(
        "--save_imgs",
        action="store_true",
        default=False,
        help="[TEST] Save SR output images to results directory",
    )
    ir_group.add_argument(
        "--realworld_degradation",
        action="store_true",
        default=False,
        help="[TRAIN] Use real-world degradation (blur+noise+JPEG) instead of bicubic",
    )
    ir_group.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store test outputs and metrics",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    #  YAML Config Override (optional)                                    #
    # ------------------------------------------------------------------ #
    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)

    return args


def print_args(args):
    """Pretty-print all parsed arguments."""
    print("\n" + "=" * 60)
    print(f"  Task     : {args.task.upper()}")
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  Device   : {args.device}")
    print("-" * 60)
    for k, v in sorted(vars(args).items()):
        if k not in ("task", "mode", "device"):
            print(f"  {k:<25} {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = get_args()
    print_args(args)
