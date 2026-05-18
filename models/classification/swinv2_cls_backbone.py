"""
models/classification/swinv2_cls_backbone.py
----------------------------------------------
SwinV2 classification model with all Swin Transformer blocks implemented
explicitly — no timm model wrappers.

Architecture (SwinV2 Paper Figure 1):
    Input → PatchEmbed → [Stage1 → Stage2 → Stage3 → Stage4] → AvgPool → Head

Each stage is a BasicLayer containing SwinTransformerBlocks that alternate
between W-MSA (regular window) and SW-MSA (shifted window) attention.
PatchMerging between stages halves spatial resolution and doubles channels.

Core SwinV2 innovations (Liu et al., CVPR 2022):
  1. Cosine attention      — replaces dot-product with cosine similarity
  2. Log-spaced CPB        — continuous relative position bias via small MLP
  3. Res-post-norm         — LayerNorm after attention/MLP (not before)

Paper reference:
  "Swin Transformer V2: Scaling Up Capacity and Resolution"
  Liu et al., CVPR 2022.  https://arxiv.org/abs/2111.09883

Model configurations (Paper Table 1):
  Tiny  → embed_dim=96,  depths=[2,2, 6,2], num_heads=[3,6,12,24]  ~28M
  Small → embed_dim=96,  depths=[2,2,18,2], num_heads=[3,6,12,24]  ~50M
  Base  → embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]  ~88M
"""

import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Model variant configurations (Paper Table 1)
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "tiny": {
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
    },
    "small": {
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
    },
    "base": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
    },
}


# =========================================================================
#  Mlp — Feed-Forward Network
# =========================================================================

class Mlp(nn.Module):
    """Two-layer MLP with GELU activation (used inside every Swin block).

    Paper: standard FFN component — dim → hidden_dim → dim.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =========================================================================
#  Window utilities
# =========================================================================

def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition — merge windows back into feature map.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: original spatial dimensions
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =========================================================================
#  WindowAttention — W-MSA / SW-MSA  (SwinV2 Sec. 3.1–3.2)
# =========================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with SwinV2 improvements.

    Key differences from original Swin (V1):
      1. Cosine attention: uses normalized Q·K cosine similarity instead of
         dot-product, scaled by a learnable temperature τ (logit_scale).
         Prevents attention score explosion in large models.
         (Paper Sec. 3.1)

      2. Continuous relative position bias (CPB): instead of a fixed learned
         bias table, feeds log-spaced relative coordinates through a small
         MLP (2→512→num_heads) to produce position bias. Enables transfer
         across different window sizes and resolutions.
         (Paper Sec. 3.2, Eq. 2)

    Args:
        dim: input channel dimension
        window_size: (Wh, Ww) window height and width
        num_heads: number of attention heads
        qkv_bias: add learnable bias to Q, K, V projections
        attn_drop: dropout on attention weights
        proj_drop: dropout on output projection
        pretrained_window_size: window size used during pre-training (for CPB scaling)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Learnable temperature for cosine attention (one per head)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # CPB MLP: maps 2D relative coordinates → per-head bias values
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Build log-spaced relative coordinate table (Paper Eq. 2)
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Pairwise relative position index for each token in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Q, K, V projection (single linear, split into 3)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size^2
            mask: attention mask for SW-MSA, shape (num_windows, N, N) or None
        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # QKV projection with asymmetric bias (bias on Q and V, not K)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention: normalize Q and K, then compute similarity
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device),
        ).exp()
        attn = attn * logit_scale

        # Add continuous relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply shifted-window mask (if SW-MSA)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'pretrained_window_size={self.pretrained_window_size}, '
                f'num_heads={self.num_heads}')


# =========================================================================
#  SwinTransformerBlock  (Paper Figure 3b)
# =========================================================================

class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block.

    Two consecutive blocks form a pair:
      - Block with shift_size=0            → W-MSA  (regular window attention)
      - Block with shift_size=window//2    → SW-MSA (shifted window attention)

    Internal flow:
        x → LN₁ → [Cyclic Shift] → Window Partition → Attention → Window Merge
          → [Reverse Shift] → + shortcut → LN₂ → MLP → + shortcut → out

    Args:
        dim: input channel count
        input_resolution: (H, W) spatial resolution at this stage
        num_heads: attention heads
        window_size: local window size (M in the paper)
        shift_size: 0 for W-MSA, window_size//2 for SW-MSA
        mlp_ratio: MLP hidden dim = dim × mlp_ratio
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """Build attention mask for shifted-window MSA (Paper Fig. 4)."""
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        # Cyclic shift (for SW-MSA)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")


# =========================================================================
#  PatchMerging  (Swin Paper Sec. 3.1 — downsampling between stages)
# =========================================================================

class PatchMerging(nn.Module):
    """Merge 2×2 neighboring patches → halve resolution, double channels.

    (B, H×W, C) → concat 4 sub-patches → (B, H/2×W/2, 4C) → Linear → (B, H/2×W/2, 2C)

    Used between stages to build the hierarchical feature pyramid.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# =========================================================================
#  PatchEmbed  (Paper: "Patch Partition" + "Linear Embedding")
# =========================================================================

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim.

    Uses Conv2d(in_chans, embed_dim, kernel=patch_size, stride=patch_size).
    For patch_size=4: (B, 3, 256, 256) → (B, 64×64, 96) = (B, 4096, 96).
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, C)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =========================================================================
#  BasicLayer — one stage of the Swin hierarchy
# =========================================================================

class BasicLayer(nn.Module):
    """A stage containing N SwinTransformerBlocks + optional PatchMerging.

    Blocks alternate between W-MSA (shift=0) and SW-MSA (shift=window//2).

    Args:
        dim: channel dimension for this stage
        input_resolution: (H, W) spatial resolution at stage input
        depth: number of SwinTransformerBlocks
        num_heads: attention heads
        window_size: local window size M
        downsample: PatchMerging class or None (last stage has no downsampling)
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# =========================================================================
#  SwinV2Backbone — full hierarchical model
# =========================================================================

class SwinV2Backbone(nn.Module):
    """
    Hierarchical SwinV2 backbone for image classification.

    Architecture (Paper Figure 1, SwinV2-Small example with 256×256 input):

        Input (B, 3, 256, 256)
          │
          ▼  PatchEmbed: Conv2d(3, 96, 4, stride=4)
        (B, 4096, 96)          ← 64×64 patches, 96-dim
          │
          ▼  Stage 1: 2× SwinTransformerBlock(dim=96, heads=3)
          ▼  PatchMerging: 64×64 → 32×32, channels 96 → 192
        (B, 1024, 192)
          │
          ▼  Stage 2: 2× SwinTransformerBlock(dim=192, heads=6)
          ▼  PatchMerging: 32×32 → 16×16, channels 192 → 384
        (B, 256, 384)
          │
          ▼  Stage 3: 18× SwinTransformerBlock(dim=384, heads=12)
          ▼  PatchMerging: 16×16 → 8×8, channels 384 → 768
        (B, 64, 768)
          │
          ▼  Stage 4: 2× SwinTransformerBlock(dim=768, heads=24)
        (B, 64, 768)
          │
          ▼  LayerNorm → AdaptiveAvgPool1d → Linear(768, num_classes)
        (B, num_classes)
    """

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=200,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- 4 Hierarchical Stages ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer),
            )
            layer = BasicLayer(
                dim=dim,
                input_resolution=resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # ---- Classification Head ----
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """PatchEmbed → 4 stages → LayerNorm → global average pooling."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, layer.input_resolution)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """(B, 3, H, W) → (B, num_classes)"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# =========================================================================
#  SwinV2Classifier — wrapper with training utility methods
# =========================================================================

class SwinV2Classifier(nn.Module):
    """Classification model wrapping SwinV2Backbone.

    Provides the same interface as the timm-based version (swinv2_cls.py):
      - forward(x)               → logits
      - freeze_backbone()        → freeze all except head
      - unfreeze_all()           → unfreeze everything
      - get_parameter_groups()   → AdamW groups (decay / no-decay)
    """

    def __init__(self, model_size="small", num_classes=200, img_size=256):
        super().__init__()

        if model_size not in _MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from: {list(_MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        cfg = _MODEL_CONFIGS[model_size]

        self.backbone = SwinV2Backbone(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SwinV2-{model_size.capitalize()} (custom backbone) | "
            f"Params: {n_params / 1e6:.1f}M"
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze everything except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen — only classification head will be updated.")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen — full fine-tuning enabled.")

    def get_parameter_groups(self, lr, weight_decay=0.05):
        """AdamW groups: weight decay on matrices, none on bias/LayerNorm."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]


# ---------------------------------------------------------------------------
#  Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for size in ("tiny", "small", "base"):
        model = SwinV2Classifier(model_size=size, num_classes=200, img_size=256)
        x = torch.randn(2, 3, 256, 256)
        logits = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"SwinV2-{size.capitalize():5s} | params: {n / 1e6:.1f}M | "
              f"input: {tuple(x.shape)} → output: {tuple(logits.shape)}")
