"""
demo.py
--------
Interactive Gradio demo for both tasks:
  Tab 1 — Image Classification  (SwinV2-Small on Tiny ImageNet-200)
  Tab 2 — Image Super-Resolution (Swin2SR ×4)

Launch:
    python demo.py                      # default port 7860
    python demo.py --port 7861          # custom port
    python demo.py --share              # public Gradio link

Each tab lets you upload an image **and** tune inference / degradation
hyper-parameters via sliders so you can explore their effect live.
"""

import argparse
import io
import logging
import random

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFilter
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.datasets import ImageFolder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

# ================================================================== #
#  Paths                                                              #
# ================================================================== #
CLS_CHECKPOINT = "checkpoints/classification_best.pth"
SR_PRETRAINED  = "checkpoints/swin2sr_classical_x4.pth"
SR_FINETUNED   = "checkpoints/ir_x4_epoch_099.pth"
DATA_DIR       = "data/tiny-imagenet-200"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================== #
#  Lazy-loaded global models (loaded once on first call)              #
# ================================================================== #
_cls_model = None
_sr_pretrained_model = None
_sr_finetuned_model = None
_idx_to_name = None


def _load_cls_model():
    global _cls_model, _idx_to_name
    if _cls_model is not None:
        return _cls_model, _idx_to_name

    from models.classification.swinv2_cls import SwinV2Classifier

    _cls_model = SwinV2Classifier(
        model_size="small", num_classes=200, img_size=256, timm_pretrained=False,
    )
    ckpt = torch.load(CLS_CHECKPOINT, map_location=DEVICE)
    _cls_model.load_state_dict(ckpt["model_state"])
    _cls_model.to(DEVICE).eval()
    logger.info("Classification model loaded.")

    # Class names
    val_dir = Path(DATA_DIR) / "val"
    dataset = ImageFolder(str(val_dir))
    idx_to_wnid = {v: k for k, v in dataset.class_to_idx.items()}

    words_file = Path(DATA_DIR) / "words.txt"
    wnid_to_name = {}
    if words_file.exists():
        with open(words_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    wnid_to_name[parts[0]] = parts[1].split(",")[0].strip()

    _idx_to_name = {i: wnid_to_name.get(w, w) for i, w in idx_to_wnid.items()}
    return _cls_model, _idx_to_name


_val_dataset = None

def _get_val_dataset():
    global _val_dataset
    if _val_dataset is None:
        _val_dataset = ImageFolder(str(Path(DATA_DIR) / "val"))
    return _val_dataset


def random_val_sample():
    """Pick a random image from the validation set."""
    ds = _get_val_dataset()
    idx = random.randint(0, len(ds) - 1)
    img_path, _ = ds.samples[idx]
    return Image.open(img_path).convert("RGB")


def _load_sr_model(variant: str = "pretrained"):
    global _sr_pretrained_model, _sr_finetuned_model

    if variant == "pretrained" and _sr_pretrained_model is not None:
        return _sr_pretrained_model
    if variant == "finetuned" and _sr_finetuned_model is not None:
        return _sr_finetuned_model

    from models.ir.network_swin2sr import Swin2SR

    model = Swin2SR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2.0, upsampler="pixelshuffle", resi_connection="1conv",
    )

    ckpt_path = SR_PRETRAINED if variant == "pretrained" else SR_FINETUNED
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("params") or ckpt.get("params_ema") or ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    logger.info(f"SR model ({variant}) loaded from {ckpt_path}")

    if variant == "pretrained":
        _sr_pretrained_model = model
    else:
        _sr_finetuned_model = model
    return model


# ================================================================== #
#  Classification inference                                           #
# ================================================================== #

_cls_transform = transforms.Compose([
    transforms.Resize(292, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def classify_image(image: Image.Image, top_k: int):
    """Run classification on a single PIL image."""
    if image is None:
        return "Upload an image or click 'Random Val Sample'."

    model, idx_to_name = _load_cls_model()

    img = image.convert("RGB")
    x = _cls_transform(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = probs.topk(int(top_k), dim=1)

    results = {}
    for prob, idx in zip(top_probs[0].cpu().tolist(), top_indices[0].cpu().tolist()):
        name = idx_to_name.get(idx, f"class_{idx}")
        results[name] = float(prob)

    return results


# ================================================================== #
#  Super-Resolution inference                                         #
# ================================================================== #

def _apply_degradation(hr_pil: Image.Image, scale: int,
                       blur_sigma: float, noise_sigma: float,
                       jpeg_quality: int) -> Image.Image:
    """Apply real-world degradation to an HR PIL image."""
    w, h = hr_pil.size
    lw, lh = w // scale, h // scale

    if blur_sigma > 0:
        radius = max(1, int(2 * blur_sigma + 1))
        if radius % 2 == 0:
            radius += 1
        hr_pil = hr_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    lr_pil = hr_pil.resize((lw, lh), Image.BICUBIC)

    if noise_sigma > 0:
        lr_np = np.array(lr_pil).astype(np.float32)
        noise = np.random.randn(*lr_np.shape).astype(np.float32) * noise_sigma
        lr_np = np.clip(lr_np + noise, 0, 255)
        lr_pil = Image.fromarray(lr_np.astype(np.uint8))

    if jpeg_quality < 100:
        buf = io.BytesIO()
        lr_pil.save(buf, format="JPEG", quality=int(jpeg_quality))
        buf.seek(0)
        lr_pil = Image.open(buf).convert("RGB")

    return lr_pil


@torch.no_grad()
def super_resolve(image: Image.Image, model_choice: str,
                  blur_sigma: float, noise_sigma: float,
                  jpeg_quality: int):
    """Degrade input HR image, then super-resolve it."""
    if image is None:
        return None, None, "Bir gorsel yukleyin."

    variant = "pretrained" if model_choice == "Pretrained (Bicubic)" else "finetuned"
    model = _load_sr_model(variant)
    scale = 4

    hr_pil = image.convert("RGB")

    lr_pil = _apply_degradation(hr_pil, scale, blur_sigma, noise_sigma, jpeg_quality)

    lr_tensor = TF.to_tensor(lr_pil).unsqueeze(0).to(DEVICE)

    _, _, h, w = lr_tensor.shape
    window_size = 8
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        lr_tensor = torch.nn.functional.pad(lr_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    sr_tensor = model(lr_tensor)
    sr_tensor = sr_tensor[:, :, :h * scale, :w * scale].clamp(0, 1)

    sr_pil = TF.to_pil_image(sr_tensor[0].cpu())

    info = (
        f"Model: {model_choice}\n"
        f"Degradation: blur={blur_sigma:.1f}, noise={noise_sigma:.0f}, jpeg={jpeg_quality}\n"
        f"LR size: {lr_pil.size[0]}x{lr_pil.size[1]}  →  "
        f"SR size: {sr_pil.size[0]}x{sr_pil.size[1]}"
    )

    return lr_pil, sr_pil, info


# ================================================================== #
#  Gradio UI                                                          #
# ================================================================== #

def build_app():
    with gr.Blocks(title="CS515 — Swin2 Multi-Task Demo") as app:

        gr.Markdown(
            "# CS515 Deep Learning — Swin2 Multi-Task Demo\n"
            "**Image Classification** (SwinV2-Small) &nbsp;|&nbsp; "
            "**Image Super-Resolution** (Swin2SR ×4)"
        )

        # ---------------------------------------------------------- #
        #  Tab 1: Classification                                      #
        # ---------------------------------------------------------- #
        with gr.Tab("Image Classification"):
            gr.Markdown(
                "Upload an image or click **Random Val Sample** to get "
                "**Top-K predictions** from SwinV2-Small (Tiny ImageNet-200)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cls_input = gr.Image(type="pil", label="Input Image", sources=["upload"])
                    cls_random_btn = gr.Button(
                        "🎲 Random Val Sample", variant="secondary",
                    )
                    cls_top_k = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Top-K",
                        info="Number of top predictions to show",
                    )
                    cls_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    cls_output = gr.Label(label="Predictions", num_top_classes=10)

            cls_random_btn.click(
                fn=random_val_sample,
                inputs=[],
                outputs=cls_input,
            )
            cls_btn.click(
                fn=classify_image,
                inputs=[cls_input, cls_top_k],
                outputs=cls_output,
            )

        # ---------------------------------------------------------- #
        #  Tab 2: Super-Resolution                                    #
        # ---------------------------------------------------------- #
        with gr.Tab("Image Super-Resolution (×4)"):
            gr.Markdown(
                "Upload an HR image or select an example below. "
                "Compare **Pretrained** (bicubic-only) vs "
                "**Fine-tuned** (real-world degradation) models."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sr_input = gr.Image(type="pil", label="Input HR Image", sources=["upload"])
                    sr_examples = gr.Examples(
                        examples=[
                            "examples/butterfly_HR.png",
                            "examples/baby_HR.png",
                            "examples/bridge_HR.png",
                        ],
                        inputs=sr_input,
                        label="Example HR Images",
                    )
                    with gr.Accordion("Model & Degradation Params", open=True):
                        sr_model_choice = gr.Radio(
                            choices=["Pretrained (Bicubic)", "Fine-tuned (Real-World)"],
                            value="Fine-tuned (Real-World)",
                            label="Model",
                        )
                        sr_blur = gr.Slider(
                            0.0, 5.0, value=1.5, step=0.1,
                            label="Blur Sigma",
                            info="Gaussian blur strength (0 = no blur)",
                        )
                        sr_noise = gr.Slider(
                            0, 50, value=15, step=1,
                            label="Noise Sigma",
                            info="Gaussian noise std (0 = no noise)",
                        )
                        sr_jpeg = gr.Slider(
                            10, 100, value=50, step=5,
                            label="JPEG Quality",
                            info="JPEG compression (100 = no compression)",
                        )
                    sr_btn = gr.Button("Super-Resolve", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        sr_lr_out = gr.Image(type="pil", label="Degraded LR")
                        sr_sr_out = gr.Image(type="pil", label="Super-Resolved (×4)")
                    sr_info = gr.Textbox(label="Info", lines=3, interactive=False)

            sr_btn.click(
                fn=super_resolve,
                inputs=[sr_input, sr_model_choice, sr_blur, sr_noise, sr_jpeg],
                outputs=[sr_lr_out, sr_sr_out, sr_info],
            )

    return app


# ================================================================== #
#  Main                                                               #
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio demo for CS515 Swin2 pipeline")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
