"""
main.py
--------
Pipeline entry-point for the CS515 Swin2 assignment.

Routes to classification or image super-resolution (IR) based on --task.

Usage:
  # Classification — train
  python main.py --task classification --mode train \
      --pretrained_weights checkpoints/simmim_swinv2_small.pth \
      --config configs/classification.yaml

  # Classification — test
  python main.py --task classification --mode test \
      --resume checkpoints/classification_best.pth \
      --config configs/classification.yaml

  # IR — train (×4)
  python main.py --task IR --mode train \
      --ir_data_dir data/ir/DIV2K/DIV2K_train_HR \
      --scale 4 --batch_size 32 --epochs 500 --lr 2e-4

  # IR — test (×4, auto-detect Set5/Set14/BSD100)
  python main.py --task IR --mode test \
      --ir_data_dir data/ir/DIV2K/DIV2K_train_HR \
      --scale 4 --resume checkpoints/ir_x4_best.pth --save_imgs
"""

import os
import random
import logging
import numpy as np
import torch

from parameters import get_args, print_args
from models.classification.swinv2_cls import build_classification_model
from models.ir.swin2sr import build_ir_model
from train import run_train_classification, run_train_ir
from test import run_test_classification, run_test_ir


# ------------------------------------------------------------------ #
#  Logging setup                                                      #
# ------------------------------------------------------------------ #

def setup_logging(log_dir: str = "checkpoints", task: str = "run"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


# ------------------------------------------------------------------ #
#  Reproducibility                                                    #
# ------------------------------------------------------------------ #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ------------------------------------------------------------------ #
#  Device                                                             #
# ------------------------------------------------------------------ #

def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    # ---- Parse arguments ----------------------------------------- #
    args = get_args()

    setup_logging(args.checkpoint_dir, task=f"{args.task}_{args.mode}")
    logger = logging.getLogger("main")

    print_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.device = str(device)

    if torch.cuda.is_available():
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}  |  "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ---- Build model --------------------------------------------- #
    if args.task == "classification":
        logger.info("Task: IMAGE CLASSIFICATION (SwinV2)")
        model = build_classification_model(args)

        if args.mode == "train":
            run_train_classification(model, args)

        elif args.mode == "test":
            run_test_classification(model, args)

    elif args.task == "IR":
        logger.info("Image Super-Resolution (Swin2SR)")
        model = build_ir_model(args)

        if args.mode == "train":
            run_train_ir(model, args)

        elif args.mode == "test":
            run_test_ir(model, args)

    else:
        raise ValueError(
            f"Unknown task: '{args.task}'. Choose 'classification' or 'IR'."
        )


if __name__ == "__main__":
    main()
