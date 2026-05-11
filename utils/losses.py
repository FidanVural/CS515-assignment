"""
utils/losses.py
---------------
Loss functions for both tasks:
  - Classification : LabelSmoothingCrossEntropy
  - IR             : CharbonnierLoss (robust L1 variant used in SR literature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Replaces hard one-hot labels y ∈ {0,1} with soft labels:
        y_smooth = (1 - eps) * y_hard + eps / num_classes

    This prevents overconfidence and improves generalisation.

    Args:
        smoothing : label smoothing factor ε  (default 0.1, as in paper)
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw model outputs
            targets : (B,)  integer class indices

        Returns:
            loss : scalar tensor
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth labels: (1 - ε) on the correct class, ε/(C-1) elsewhere
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1, a.k.a. pseudo-Huber loss):

        L(x) = sqrt(x^2 + eps^2)

    Used as the default reconstruction loss in Swin2SR and SwinIR.
    More robust to outliers than MSE; smoother gradient than plain L1.

    Args:
        eps       : small constant for numerical stability  (default 1e-3)
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : SR output tensor  (B, C, H, W)
            target : HR ground truth   (B, C, H, W)

        Returns:
            loss : scalar tensor
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps ** 2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
