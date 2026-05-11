"""
train.py
---------
Unified training entry-point for both pipeline tasks.

  run_train_classification(model, args) → fine-tunes SwinV2 on ImageFolder dataset
  run_train_ir(model, args)            → trains Swin2SR on DIV2K

Called by main.py after model and dataset construction.
"""

import os
import math
import logging
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from utils import (
    AverageMeter,
    accuracy,
    LabelSmoothingCrossEntropy,
    CharbonnierLoss,
    save_checkpoint,
    save_best_checkpoint,
    load_checkpoint,
)
from datasets.cls_dataset import build_cls_dataloaders
from datasets.ir_dataset import build_ir_train_loader
from datasets.realworld_degradation import build_realworld_train_loader

logger = logging.getLogger(__name__)


# ============================================================ #
#  Learning-rate utilities                                     #
# ============================================================ #

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int,
                                    total_epochs: int, min_lr: float = 1e-6):
    """Linear warmup + cosine annealing scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================ #
#  Classification Training                                     #
# ============================================================ #

def train_one_epoch_cls(model, loader, criterion, optimizer,
                         scaler, device, epoch, args):
    """Run one epoch of classification training."""
    model.train()
    loss_meter = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    top5_meter = AverageMeter("top5")

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast("cuda", enabled=args.use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.detach(), targets, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1, bs)
        top5_meter.update(acc5, bs)

        if batch_idx % args.log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] [{batch_idx}/{len(loader)}]  "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Top-1: {top1_meter.avg:.2f}%  "
                f"Top-5: {top5_meter.avg:.2f}%  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate_cls(model, loader, criterion, device, args):
    """Validate classification model on the val split."""
    model.eval()
    loss_meter = AverageMeter("val_loss")
    top1_meter = AverageMeter("val_top1")
    top5_meter = AverageMeter("val_top5")

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda", enabled=args.use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1, bs)
        top5_meter.update(acc5, bs)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def run_train_classification(model, args):
    """Full classification training loop."""
    device = torch.device(args.device)
    model  = model.to(device)

    # Data
    train_loader, val_loader = build_cls_dataloaders(args)

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Optimizer — separate LR groups (no weight decay on biases/LayerNorm)
    param_groups = model.get_parameter_groups(lr=args.lr, weight_decay=args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)
    scaler       = GradScaler("cuda", enabled=args.use_amp)

    # Resume
    start_epoch  = 0
    best_top1    = 0.0
    if args.resume:
        meta        = load_checkpoint(args.resume, model, optimizer, scheduler, args.device)
        start_epoch = meta["epoch"] + 1
        best_top1   = meta.get("best_metric") or 0.0

    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "tb_cls"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger.info(f"Starting classification training | epochs: {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_top1, train_top5 = train_one_epoch_cls(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        val_loss, val_top1, val_top5 = validate_cls(
            model, val_loader, criterion, device, args
        )
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        logger.info(
            f"Epoch {epoch} | {elapsed:.1f} min | "
            f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | "
            f"Val Top-5: {val_top5:.2f}%"
        )

        # TensorBoard
        writer.add_scalars("Loss",  {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Top1",  {"train": train_top1, "val": val_top1}, epoch)
        writer.add_scalar("LR",    optimizer.param_groups[0]["lr"], epoch)

        # Save checkpoint every epoch
        state = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "best_metric":    best_top1,
            "task":           "classification",
        }
        save_checkpoint(state, args.checkpoint_dir, f"cls_epoch_{epoch:03d}.pth")

        # Save best
        if val_top1 > best_top1:
            best_top1 = val_top1
            state["best_metric"] = best_top1
            save_best_checkpoint(state, args.checkpoint_dir, "classification")
            logger.info(f"  ★ New best Top-1: {best_top1:.2f}%")

    writer.close()
    logger.info(f"Training complete. Best Val Top-1: {best_top1:.2f}%")
    return best_top1


# ============================================================ #
#  IR Training                                                 #
# ============================================================ #

def run_train_ir(model, args):
    """Full Swin2SR training loop on DIV2K."""
    device = torch.device(args.device)
    model  = model.to(device)

    use_realworld = getattr(args, "realworld_degradation", False)
    if use_realworld:
        train_loader = build_realworld_train_loader(args)
    else:
        train_loader = build_ir_train_loader(args)
    criterion    = CharbonnierLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99)
    )
    # MultiStepLR: halve LR at 250k, 400k, 450k, 475k iterations
    # We convert to epoch-based milestones approximation
    total_iters = args.epochs * len(train_loader)
    milestones   = [
        int(total_iters * r) // len(train_loader)
        for r in (0.5, 0.8, 0.9, 0.95)
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    start_epoch = 0
    best_psnr   = 0.0
    if args.resume:
        meta        = load_checkpoint(args.resume, model, optimizer, scheduler, args.device)
        start_epoch = meta["epoch"] + 1
        best_psnr   = meta.get("best_metric") or 0.0

    writer = SummaryWriter(
        log_dir=os.path.join(args.checkpoint_dir, f"tb_ir_x{args.scale}")
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger.info(
        f"Starting IR training | scale: ×{args.scale} | epochs: {args.epochs}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = AverageMeter("loss")
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            pred = model(lq)
            loss = criterion(pred, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()

            loss_meter.update(loss.item(), lq.size(0))

            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"[IR ×{args.scale}] Epoch [{epoch}] "
                    f"[{batch_idx}/{len(train_loader)}]  "
                    f"Loss: {loss_meter.avg:.6f}  "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        elapsed = (time.time() - t0) / 60
        logger.info(
            f"IR Epoch {epoch} | {elapsed:.1f} min | "
            f"Avg Loss: {loss_meter.avg:.6f}"
        )
        writer.add_scalar(f"Loss/IR_x{args.scale}", loss_meter.avg, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        state = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "best_metric":    best_psnr,
            "task":           f"IR_x{args.scale}",
        }
        save_checkpoint(state, args.checkpoint_dir,
                        f"ir_x{args.scale}_epoch_{epoch:03d}.pth")

    writer.close()
    logger.info("IR training complete.")
