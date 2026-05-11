"""
models/classification/swinv2_cls.py
-------------------------------------
SwinV2 classification wrapper built on top of the `timm` library.

Supports:
  - SwinV2-Tiny / Small / Base variants
  - Loading SimMIM pretrained weights (self-supervised initialisation)
  - Standard timm pretrained ImageNet weights (optional)

Paper reference:
  "Swin Transformer V2: Scaling Up Capacity and Resolution"
  Liu et al., CVPR 2022.  https://arxiv.org/abs/2111.09883

timm model names used:
  tiny  → swinv2_tiny_window8_256
  small → swinv2_small_window8_256
  base  → swinv2_base_window8_256
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for SwinV2 classification. "
        "Install with: pip install timm"
    ) from e

logger = logging.getLogger(__name__)

# Mapping from human-readable size name to timm model identifier
_TIMM_MODEL_MAP = {
    "tiny":  "swinv2_tiny_window8_256",
    "small": "swinv2_small_window8_256",
    "base":  "swinv2_base_window8_256",
}

# Published paper Top-1 accuracies on ImageNet-1K (for reference)
_PAPER_SCORES = {
    "tiny":  81.8,
    "small": 83.7,
    "base":  84.2,
}


class SwinV2Classifier(nn.Module):
    """
    SwinV2 image classifier.

    Args:
        model_size   : 'tiny' | 'small' | 'base'
        num_classes  : number of output classes (default 1000 for ImageNet)
        img_size     : input image resolution (default 256)
        timm_pretrained : if True, load timm's ImageNet pretrained weights
                          (use False when loading SimMIM / custom weights)

    Example:
        >>> model = SwinV2Classifier(model_size='tiny', num_classes=1000)
        >>> model.load_simmim_weights('simmim_swinv2_tiny.pth')
        >>> out = model(torch.randn(2, 3, 256, 256))  # (2, 1000)
    """

    def __init__(
        self,
        model_size: str = "tiny",
        num_classes: int = 1000,
        img_size: int = 256,
        timm_pretrained: bool = False,
    ):
        super().__init__()

        if model_size not in _TIMM_MODEL_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from: {list(_TIMM_MODEL_MAP.keys())}"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        model_name = _TIMM_MODEL_MAP[model_size]

        logger.info(
            f"Building SwinV2-{model_size.capitalize()} "
            f"(timm: {model_name}, pretrained={timm_pretrained})"
        )

        self.backbone = timm.create_model(
            model_name,
            pretrained=timm_pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SwinV2-{model_size.capitalize()} | "
            f"Params: {n_params / 1e6:.1f}M | "
            f"Paper Top-1 (ImageNet-1K): {_PAPER_SCORES[model_size]}%"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W) float tensor

        Returns:
            logits : (B, num_classes)
        """
        return self.backbone(x)

    def load_simmim_weights(self, checkpoint_path: str, device: str = "cpu") -> tuple:
        """
        Load SimMIM pretrained backbone weights.

        SimMIM checkpoints contain encoder weights trained with masked image
        modelling (no labels). The classification head is NOT present, so we
        load only matching layers and re-initialise the head.

        Args:
            checkpoint_path : path to SimMIM .pth file
            device          : map location for loading

        Returns:
            (missing_keys, unexpected_keys)
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"SimMIM checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading SimMIM weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Handle various checkpoint formats
        state_dict = (
            ckpt.get("model")
            or ckpt.get("state_dict")
            or ckpt.get("model_state")
            or ckpt
        )

        # Strip common prefixes added by SimMIM training wrappers
        clean_sd = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("encoder.", "module.", "backbone."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            clean_sd[new_k] = v

        # Filter keys: only load layers whose shapes match
        model_sd = self.backbone.state_dict()
        matched = {
            k: v
            for k, v in clean_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        skipped = [k for k in clean_sd if k not in matched]

        missing, unexpected = self.backbone.load_state_dict(matched, strict=False)
        logger.info(
            f"SimMIM load: matched {len(matched)}/{len(model_sd)} layers | "
            f"skipped (shape mismatch): {len(skipped)} | "
            f"missing: {len(missing)}"
        )
        return missing, unexpected

    def freeze_backbone(self):
        """Freeze all layers except the final classification head."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen — only classification head will be updated.")

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen — full fine-tuning enabled.")

    def get_parameter_groups(self, lr: float, weight_decay: float = 0.05):
        """
        Return parameter groups with:
          - No weight decay for biases and LayerNorm parameters
          - Standard weight decay for everything else

        This follows the AdamW fine-tuning recipe from the SwinV2 paper.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay,    "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]


def build_classification_model(args) -> SwinV2Classifier:
    """
    Factory function called by main.py.

    Builds the model and optionally loads SimMIM / custom pretrained weights.

    Args:
        args : parsed arguments from parameters.py

    Returns:
        model : SwinV2Classifier ready for training or evaluation
    """
    model = SwinV2Classifier(
        model_size=args.model_size,
        num_classes=args.num_classes,
        img_size=args.img_size,
        timm_pretrained=False,   # We manage weights ourselves
    )

    if args.pretrained_weights is not None:
        model.load_simmim_weights(args.pretrained_weights, device=args.device)
    else:
        logger.warning(
            "No pretrained_weights provided — training from random initialisation."
        )

    return model
