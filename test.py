"""
test.py
--------
Unified evaluation entry-point for both pipeline tasks.

  run_test_classification(model, args) → Top-1/Top-5 accuracy + confusion matrix
  run_test_ir(model, args)             → PSNR / SSIM on Set5, Set14, BSD100
"""

import os
import json
import logging
from pathlib import Path

import torch
from torch.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from utils import (
    AverageMeter,
    accuracy,
    calculate_psnr,
    calculate_ssim,
    load_checkpoint,
)
from datasets.cls_dataset import build_cls_dataloaders, build_val_transform
from datasets.ir_dataset import build_ir_test_loader

logger = logging.getLogger(__name__)


# ============================================================ #
#  Classification Evaluation                                   #
# ============================================================ #

@torch.no_grad()
def run_test_classification(model, args):
    """
    Evaluate classification model on the val set.

    Reports:
      - Top-1 and Top-5 accuracy
      - Per-class accuracy breakdown
      - Confusion matrix (saved to results/)
    """
    device = torch.device(args.device)
    model  = model.to(device).eval()

    # Load checkpoint if provided
    if args.resume:
        load_checkpoint(args.resume, model, device=args.device)

    _, val_loader = build_cls_dataloaders(args)

    top1_meter = AverageMeter("Top-1")
    top5_meter = AverageMeter("Top-5")

    # For confusion matrix (sample up to 10k predictions to keep memory manageable)
    all_preds   = []
    all_targets = []
    max_cm_samples = 10_000
    cm_count = 0

    print("\n" + "=" * 55)
    print("  CLASSIFICATION EVALUATION")
    print("=" * 55)

    for images, targets in tqdm(val_loader, desc="Evaluating", unit="batch"):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast("cuda", enabled=args.use_amp):
            logits = model(images)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        bs = images.size(0)
        top1_meter.update(acc1, bs)
        top5_meter.update(acc5, bs)

        if cm_count < max_cm_samples:
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            cm_count += bs

    print(f"\n  Val Top-1 Accuracy : {top1_meter.avg:.2f}%")
    print(f"  Val Top-5 Accuracy : {top5_meter.avg:.2f}%")
    print(f"\n  Model              : SwinV2-{args.model_size.capitalize()} | Classes: {args.num_classes}")
    print("=" * 55 + "\n")

    # Save metrics
    os.makedirs(args.results_dir, exist_ok=True)
    metrics = {
        "task":          "classification",
        "model":         f"SwinV2-{args.model_size}",
        "val_top1":      round(top1_meter.avg, 4),
        "val_top5":      round(top5_meter.avg, 4),
        "paper_top1":    81.8,
    }
    metrics_path = os.path.join(args.results_dir, "cls_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # Confusion matrix (first 20 classes for readability if >20 classes)
    _save_confusion_matrix(
        np.array(all_targets),
        np.array(all_preds),
        n_classes=min(args.num_classes, 20),
        results_dir=args.results_dir,
    )

    return top1_meter.avg, top5_meter.avg


def _save_confusion_matrix(targets, preds, n_classes: int, results_dir: str):
    """Plot and save a confusion matrix (top-n classes)."""
    from sklearn.metrics import confusion_matrix

    # Restrict to first n_classes indices
    mask = (targets < n_classes) & (preds < n_classes)
    if mask.sum() == 0:
        return
    cm = confusion_matrix(targets[mask], preds[mask], labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Confusion Matrix (first {n_classes} classes, normalised)", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    path = os.path.join(results_dir, "cls_confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {path}")


# ============================================================ #
#  IR Evaluation                                               #
# ============================================================ #

# Published Swin2SR-Classical PSNR targets (Y channel, paper Table 1)
_PAPER_PSNR = {
    2: {"Set5": 38.42, "Set14": 34.46, "BSD100": 32.87},
    4: {"Set5": 33.12, "Set14": 29.48, "BSD100": 28.79},
}
_PAPER_SSIM = {
    2: {"Set5": 0.9623, "Set14": 0.9250, "BSD100": 0.9070},
    4: {"Set5": 0.9047, "Set14": 0.8035, "BSD100": 0.7707},
}

# Default test-set configurations: (GT folder, LR folder or None)
_TEST_SETS = ["Set5", "Set14", "BSD100"]


@torch.no_grad()
def run_test_ir(model, args):
    """
    Evaluate Swin2SR on Set5, Set14 and BSD100.

    Reports per-set average PSNR and SSIM (Y channel).
    Optionally saves SR output images.
    """
    device = torch.device(args.device)
    model  = model.to(device).eval()

    if args.resume:
        load_checkpoint(args.resume, model, device=args.device)

    os.makedirs(args.results_dir, exist_ok=True)

    all_results = {}
    scale = args.scale

    # Determine test sets to evaluate
    if args.folder_gt is not None:
        # Single custom test folder provided via CLI
        test_configs = [
            ("custom", args.folder_gt, args.folder_lq)
        ]
    else:
        # Auto-detect test sets: walk up from ir_data_dir until we find Set5
        ir_root = Path(args.ir_data_dir).parent
        if not (ir_root / "Set5").exists():
            ir_root = ir_root.parent
        test_configs = []
        for name in _TEST_SETS:
            gt_dir = ir_root / name / "HR"
            lq_dir = ir_root / name / "LR_bicubic" / f"X{scale}"
            if gt_dir.exists():
                test_configs.append((name, str(gt_dir),
                                     str(lq_dir) if lq_dir.exists() else None))

    if not test_configs:
        logger.warning(
            "No test sets found. Provide --folder_gt or place Set5/Set14/BSD100 "
            "under the ir_data_dir parent."
        )
        return {}

    print("\n" + "=" * 60)
    print(f"  IR EVALUATION  (×{scale}  |  Y-channel PSNR/SSIM)")
    print("=" * 60)

    for set_name, gt_dir, lq_dir in test_configs:
        loader = build_ir_test_loader(
            folder_gt=gt_dir,
            scale=scale,
            folder_lq=lq_dir,
            num_workers=min(args.num_workers, 4),
        )

        psnr_meter = AverageMeter("PSNR")
        ssim_meter = AverageMeter("SSIM")

        save_dir = os.path.join(
            args.results_dir, f"swin2sr_x{scale}_{set_name}"
        )
        if args.save_imgs:
            os.makedirs(save_dir, exist_ok=True)

        for batch in tqdm(loader, desc=f"  {set_name}", unit="img"):
            lq = batch["lq"].to(device)
            gt = batch["gt"].to(device)

            # Pad to window_size multiple for Swin2SR
            _, _, h, w = lq.shape
            window_size = 8
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
            sr = model(lq)
            sr = sr[:, :, :h * scale, :w * scale].clamp(0, 1)

            psnr_val = calculate_psnr(sr[0], gt[0], scale=scale, y_channel=True)
            ssim_val = calculate_ssim(sr[0], gt[0], scale=scale, y_channel=True)
            psnr_meter.update(psnr_val)
            ssim_meter.update(ssim_val)

            if args.save_imgs:
                _save_sr_image(sr[0], batch["name"][0], save_dir)

        avg_psnr = psnr_meter.avg
        avg_ssim = ssim_meter.avg
        all_results[set_name] = {"PSNR": avg_psnr, "SSIM": avg_ssim}

        # Print with paper reference if available
        ref_psnr = _PAPER_PSNR.get(scale, {}).get(set_name, None)
        ref_ssim = _PAPER_SSIM.get(scale, {}).get(set_name, None)
        print(
            f"\n  {set_name:<10} | "
            f"PSNR: {avg_psnr:.2f} dB"
            + (f"  (paper: {ref_psnr:.2f} dB)" if ref_psnr else "")
        )
        print(
            f"  {'':10} | "
            f"SSIM: {avg_ssim:.4f}"
            + (f"    (paper: {ref_ssim:.4f})" if ref_ssim else "")
        )
        if args.save_imgs:
            print(f"  {'':10}   SR images → {save_dir}")

    print("\n" + "=" * 60 + "\n")

    # Save JSON summary
    metrics_path = os.path.join(args.results_dir, f"ir_x{scale}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"scale": scale, "results": all_results,
                   "paper": {"PSNR": _PAPER_PSNR.get(scale, {}),
                              "SSIM": _PAPER_SSIM.get(scale, {})}}, f, indent=2)
    logger.info(f"IR metrics saved → {metrics_path}")

    return all_results


def _save_sr_image(tensor: torch.Tensor, name: str, save_dir: str):
    """Save a (C, H, W) float [0,1] tensor as PNG."""
    from torchvision.utils import save_image
    path = os.path.join(save_dir, f"{name}_SR.png")
    save_image(tensor.cpu(), path)
