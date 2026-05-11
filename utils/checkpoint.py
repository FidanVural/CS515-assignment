"""
utils/checkpoint.py
-------------------
Save, load and resume utilities for model checkpoints.
Supports both classification and IR tasks.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> str:
    """
    Save a training checkpoint to disk.

    Args:
        state          : dict containing model state, optimizer state, epoch, etc.
        checkpoint_dir : directory where checkpoints are stored
        filename       : checkpoint filename (e.g., 'cls_epoch_10.pth')

    Returns:
        path : full path to the saved file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")
    return path


def save_best_checkpoint(state: dict, checkpoint_dir: str, task: str) -> str:
    """Save the best model checkpoint, overwriting the previous best."""
    filename = f"{task}_best.pth"
    return save_checkpoint(state, checkpoint_dir, filename)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, scheduler=None,
                    device: str = "cuda") -> dict:
    """
    Load a full training checkpoint (model + optimizer + metadata).

    Args:
        path      : path to the .pth checkpoint file
        model     : model to load weights into
        optimizer : optimizer to restore state (optional)
        scheduler : LR scheduler to restore state (optional)
        device    : target device

    Returns:
        meta : dict with 'epoch', 'best_metric', etc.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    meta = {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", None),
        "task": checkpoint.get("task", "unknown"),
    }
    logger.info(
        f"Resumed from epoch {meta['epoch']} | "
        f"best metric: {meta['best_metric']}"
    )
    return meta


def load_pretrained_weights(path: str, model: torch.nn.Module,
                             strict: bool = False,
                             device: str = "cuda") -> tuple:
    """
    Load pretrained weights (e.g., SimMIM checkpoint) into a model,
    ignoring shape-mismatched keys (e.g., classification head).

    Args:
        path    : path to pretrained .pth file
        model   : target model
        strict  : if False, missing/unexpected keys are ignored
        device  : target device

    Returns:
        (missing_keys, unexpected_keys) : lists of unmatched parameter names
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained weights not found: {path}")

    logger.info(f"Loading pretrained weights from {path}")
    checkpoint = torch.load(path, map_location=device)

    # Support multiple checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  # raw state dict

    # Filter keys that don't match model (e.g., classification head from SimMIM)
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        # Strip 'encoder.' prefix that some SimMIM checkpoints add
        clean_k = k.replace("encoder.", "")
        if clean_k in model_state and model_state[clean_k].shape == v.shape:
            filtered[clean_k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    logger.info(
        f"Loaded {len(filtered)} / {len(model_state)} layers. "
        f"Skipped (shape mismatch): {len(skipped)}. "
        f"Missing: {len(missing)}."
    )
    return missing, unexpected
