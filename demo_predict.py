"""
demo_predict.py
---------------
Randomly samples images from the validation set and displays
Top-5 predictions from the trained SwinV2-Small classifier.

Usage:
    python demo_predict.py --num_samples 8
    python demo_predict.py --num_samples 6 --seed 42 --save_fig results/demo_predictions.png
"""

import argparse
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

from models.classification.swinv2_cls import SwinV2Classifier

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #
CHECKPOINT = "checkpoints/classification_best.pth"
DATA_DIR = "data/tiny-imagenet-200"
IMG_SIZE = 256
NUM_CLASSES = 200
MODEL_SIZE = "small"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_class_names(data_dir: str) -> dict:
    """Build class_idx -> human-readable name mapping."""
    val_dir = Path(data_dir) / "val"
    dataset = ImageFolder(str(val_dir))
    idx_to_wnid = {v: k for k, v in dataset.class_to_idx.items()}

    words_file = Path(data_dir) / "words.txt"
    wnid_to_name = {}
    if words_file.exists():
        with open(words_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    wnid_to_name[parts[0]] = parts[1].split(",")[0].strip()

    idx_to_name = {}
    for idx, wnid in idx_to_wnid.items():
        idx_to_name[idx] = wnid_to_name.get(wnid, wnid)

    return idx_to_name


def load_model(checkpoint_path: str, device: str) -> SwinV2Classifier:
    """Load trained model from checkpoint."""
    model = SwinV2Classifier(
        model_size=MODEL_SIZE,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        timm_pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def get_val_transform() -> transforms.Compose:
    """Same validation transform used during training."""
    resize_size = int(IMG_SIZE * 292 / 256)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: str, top_k: int = 5):
    """Run inference and return top-k predictions."""
    x = image_tensor.unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)
    return top_indices[0].cpu().tolist(), top_probs[0].cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Demo: predict on val images")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--save_fig", type=str, default="results/demo_predictions.png")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading model from {CHECKPOINT}...")
    model = load_model(CHECKPOINT, device)

    print("Building class names...")
    idx_to_name = get_class_names(DATA_DIR)

    transform = get_val_transform()

    val_dir = Path(DATA_DIR) / "val"
    val_dataset = ImageFolder(str(val_dir))
    sampled = random.sample(range(len(val_dataset)), min(args.num_samples, len(val_dataset)))

    n = len(sampled)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5.5 * cols, 4.5 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.6, wspace=0.3)

    correct = 0

    for i, idx in enumerate(sampled):
        img_path, true_label = val_dataset.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil)

        top_indices, top_probs = predict(model, img_tensor, device, top_k=5)
        pred_label = top_indices[0]
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        row, col = divmod(i, cols)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, col],
                                                 height_ratios=[3, 2], hspace=0.05)

        # Image
        ax_img = fig.add_subplot(inner[0])
        ax_img.imshow(img_pil.resize((128, 128)), aspect="equal")
        ax_img.axis("off")

        border_color = "#2ecc71" if is_correct else "#e74c3c"
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        true_name = idx_to_name.get(true_label, f"class_{true_label}")
        ax_img.set_title(f"GT: {true_name}", fontsize=10, fontweight="bold",
                         pad=4, color="#2c3e50")

        # Predictions text
        ax_txt = fig.add_subplot(inner[1])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        pred_lines = []
        for rank, (ci, cp) in enumerate(zip(top_indices, top_probs), 1):
            name = idx_to_name.get(ci, f"class_{ci}")
            check = " ✓" if ci == true_label else ""
            pred_lines.append(f"{rank}. {name} ({cp:.1%}){check}")

        text_color = "#27ae60" if is_correct else "#c0392b"
        ax_txt.text(0.5, 0.95, "\n".join(pred_lines),
                    transform=ax_txt.transAxes,
                    fontsize=8.5, fontfamily="monospace",
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                              edgecolor=text_color, alpha=0.9))

    # Hide unused grid cells
    fig.suptitle(
        f"SwinV2-Small Predictions on Tiny ImageNet-200  |  "
        f"{correct}/{n} correct ({correct/n:.0%})",
        fontsize=13, fontweight="bold", y=0.98, color="#2c3e50"
    )

    Path(args.save_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_fig, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nResults: {correct}/{n} correct ({correct/n:.0%})")
    print(f"Figure saved → {args.save_fig}")
    plt.close()


if __name__ == "__main__":
    main()
