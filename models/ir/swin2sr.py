"""
models/ir/swin2sr.py
---------------------
Factory function for building the Swin2SR super-resolution model.

The model architecture is from the official Swin2SR implementation:
  "Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution
   and Restoration", Conde et al., ECCV 2022.
  https://github.com/mv-lab/swin2sr

Supports:
  - Loading official pretrained weights
  - Classical SR (×2, ×4) with PixelShuffle upsampler
"""

import logging
import torch

logger = logging.getLogger(__name__)


def build_ir_model(args):
    """
    Build Swin2SR Classical SR model for the requested scale.
    Optionally loads pretrained weights.

    Args:
        args : parsed arguments from parameters.py

    Returns:
        model : Swin2SR ready for training or evaluation
    """
    from models.ir.network_swin2sr import Swin2SR

    model = Swin2SR(
        upscale=args.scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Swin2SR-Classical ×{args.scale} | Params: {n_params / 1e6:.1f}M")

    if args.ir_model_path is not None:
        logger.info(f"Loading Swin2SR weights from {args.ir_model_path}")
        ckpt = torch.load(args.ir_model_path, map_location=args.device)
        state_dict = ckpt.get("params") or ckpt.get("params_ema") or ckpt.get("model_state") or ckpt
        model.load_state_dict(state_dict, strict=False)
        logger.info("Swin2SR weights loaded successfully.")
    else:
        logger.info("No ir_model_path provided — training Swin2SR from scratch.")

    return model
