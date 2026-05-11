"""
utils/metrics.py
----------------
Evaluation metrics for both tasks:
  - Classification : Top-1 / Top-5 accuracy, AverageMeter
  - IR             : PSNR and SSIM (on Y channel, as per SR convention)
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ============================================================ #
#  General Utility                                             #
# ============================================================ #

class AverageMeter:
    """Running average tracker (loss, accuracy, PSNR, …)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ============================================================ #
#  Classification Metrics                                      #
# ============================================================ #

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Compute Top-k accuracy for the given batch.

    Args:
        output : model logits  (B, num_classes)
        target : ground-truth labels (B,)
        topk   : tuple of k values to compute

    Returns:
        list of float accuracies in [0, 100]
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()                          # (maxk, B)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size).item())
    return results


# ============================================================ #
#  IR / Super-Resolution Metrics                               #
# ============================================================ #

def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a float [0,1] tensor (C, H, W) to uint8 numpy (H, W, C).
    Clamps to [0, 1] before conversion.
    """
    img = tensor.detach().cpu().float().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()        # (H, W, C)
    img = (img * 255.0).round().astype(np.uint8)
    return img


def rgb_to_ycbcr_y(img_uint8: np.ndarray) -> np.ndarray:
    """
    Extract Y channel from an RGB uint8 image using the standard
    BT.601 formula (matches paper convention for PSNR/SSIM on Y).

    Args:
        img_uint8 : (H, W, 3) uint8 RGB image

    Returns:
        y_channel : (H, W) float64 in [16, 235]
    """
    img_float = img_uint8.astype(np.float64)
    y = (16.0
         + 65.481  * img_float[:, :, 0] / 255.0
         + 128.553 * img_float[:, :, 1] / 255.0
         + 24.966  * img_float[:, :, 2] / 255.0)
    return y


def calculate_psnr(sr: torch.Tensor, hr: torch.Tensor,
                   scale: int = 4, y_channel: bool = True) -> float:
    """
    Compute PSNR between SR and HR tensors.

    Args:
        sr, hr     : float tensors (C, H, W) in [0, 1]
        scale      : upscale factor — used to crop border pixels (paper standard)
        y_channel  : if True, compute on Y channel only (PSNR_Y)

    Returns:
        psnr : float in dB
    """
    sr_np = tensor_to_uint8(sr)
    hr_np = tensor_to_uint8(hr)

    # Crop border to match paper evaluation
    if scale > 0:
        sr_np = sr_np[scale:-scale, scale:-scale]
        hr_np = hr_np[scale:-scale, scale:-scale]

    if y_channel:
        sr_y = rgb_to_ycbcr_y(sr_np)
        hr_y = rgb_to_ycbcr_y(hr_np)
        mse = np.mean((sr_y - hr_y) ** 2)
    else:
        mse = np.mean((sr_np.astype(np.float64) - hr_np.astype(np.float64)) ** 2)

    if mse < 1e-10:
        return float("inf")
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return float(psnr)


def calculate_ssim(sr: torch.Tensor, hr: torch.Tensor,
                   scale: int = 4, y_channel: bool = True) -> float:
    """
    Compute SSIM between SR and HR tensors.

    Args:
        sr, hr     : float tensors (C, H, W) in [0, 1]
        scale      : upscale factor — border crop
        y_channel  : if True, compute on Y channel only (SSIM_Y)

    Returns:
        ssim : float in [0, 1]
    """
    sr_np = tensor_to_uint8(sr)
    hr_np = tensor_to_uint8(hr)

    if scale > 0:
        sr_np = sr_np[scale:-scale, scale:-scale]
        hr_np = hr_np[scale:-scale, scale:-scale]

    if y_channel:
        sr_y = rgb_to_ycbcr_y(sr_np)
        hr_y = rgb_to_ycbcr_y(hr_np)
        result = ssim_fn(sr_y, hr_y, data_range=255.0)
    else:
        result = ssim_fn(
            sr_np, hr_np,
            data_range=255,
            channel_axis=2,
        )
    return float(result)
