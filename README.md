# CS515 — Swin2 Multi-Task Pipeline

A unified pipeline for **Image Classification** and **Image Super-Resolution (IR)** using **Swin Transformer V2** architecture.

| Task | Model | Dataset | Metric |
|---|---|---|---|
| Classification | SwinV2-Tiny | ImageNet-1K | Top-1 / Top-5 Accuracy |
| Super-Resolution | Swin2SR-Classical | DIV2K → Set5 / Set14 / BSD100 | PSNR / SSIM (Y) |

---

## References

- **SwinV2**: [Liu et al., CVPR 2022](https://arxiv.org/abs/2111.09883) — [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- **Swin2SR**: [Conde et al., ECCV 2022](https://arxiv.org/abs/2209.11345) — [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr)
- **SimMIM**: [Xie et al., CVPR 2022](https://arxiv.org/abs/2111.09886) — pretrained weights source

---

## Installation

```bash
git clone <this-repo>
cd cs515_assignment
pip install -r requirements.txt
```

---

## Project Structure

```
cs515_assignment/
├── main.py               # Pipeline entry point  (--task classification | IR)
├── train.py              # Training loops for both tasks
├── test.py               # Evaluation for both tasks
├── parameters.py         # All argparse definitions
├── requirements.txt
│
├── models/
│   ├── classification/
│   │   └── swinv2_cls.py     # SwinV2-Tiny/Small/Base via timm + SimMIM loader
│   └── ir/
│       └── swin2sr.py        # Swin2SR Classical SR (adapted from mv-lab/swin2sr)
│
├── datasets/
│   ├── cls_dataset.py        # ImageNet-1K DataLoader
│   └── ir_dataset.py         # DIV2K (train) + Set5/Set14/BSD100 (test)
│
├── utils/
│   ├── metrics.py            # Accuracy, PSNR, SSIM
│   ├── losses.py             # LabelSmoothingCE, CharbonnierLoss
│   └── checkpoint.py         # save/load/resume
│
└── configs/
    ├── classification.yaml
    └── ir.yaml
```

---

## Data Preparation

### Classification — ImageNet-1K
```
data/imagenet/
    train/
        n01440764/
        ...
    val/
        n01440764/
        ...
```
Download from [image-net.org](https://image-net.org/) (account required).

### IR — DIV2K + Test Sets
```
data/ir/
    DIV2K/
        DIV2K_train_HR/      # 800 .png files
    Set5/
        HR/
        LR_bicubic/X2/
        LR_bicubic/X4/
    Set14/                   # same structure
    BSD100/                  # same structure
```
- **DIV2K**: [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Set5 / Set14 / BSD100**: [SwinIR releases](https://github.com/JingyunLiang/SwinIR/releases)

---

## SimMIM Pretrained Weights

Download SimMIM pretrained SwinV2-Tiny weights from the official repo:
```
https://github.com/microsoft/Swin-Transformer (Releases section)
```
Save as `checkpoints/simmim_swinv2_tiny.pth`.

---

## Usage

### Classification

**Train** (SimMIM → ImageNet-1K full fine-tuning):
```bash
python main.py --task classification --mode train \
    --data_dir data/imagenet \
    --model_size tiny \
    --pretrained_weights checkpoints/simmim_swinv2_tiny.pth \
    --epochs 30 --batch_size 128 --lr 1e-4 --use_amp
```

Or use the YAML config:
```bash
python main.py --task classification --config configs/classification.yaml \
    --pretrained_weights checkpoints/simmim_swinv2_tiny.pth
```

**Test**:
```bash
python main.py --task classification --mode test \
    --data_dir data/imagenet \
    --model_size tiny \
    --resume checkpoints/classification_best.pth
```

---

### IR — Super-Resolution

**Train (×4)**:
```bash
python main.py --task IR --mode train \
    --ir_data_dir data/ir/DIV2K/DIV2K_train_HR \
    --scale 4 --batch_size 32 --epochs 500 --lr 2e-4
```

**Train (×2)**:
```bash
python main.py --task IR --mode train \
    --ir_data_dir data/ir/DIV2K/DIV2K_train_HR \
    --scale 2 --batch_size 32 --epochs 500 --lr 2e-4
```

**Test (auto Set5/Set14/BSD100)**:
```bash
python main.py --task IR --mode test \
    --ir_data_dir data/ir/DIV2K/DIV2K_train_HR \
    --scale 4 --resume checkpoints/ir_x4_best.pth --save_imgs
```

**Test (single custom folder)**:
```bash
python main.py --task IR --mode test \
    --scale 4 \
    --folder_gt data/ir/Set5/HR \
    --folder_lq data/ir/Set5/LR_bicubic/X4 \
    --resume checkpoints/ir_x4_best.pth --save_imgs
```

---

## Paper Benchmark Targets

### Classification — SwinV2 (ImageNet-1K)
| Model | Top-1 |
|---|---|
| SwinV2-Tiny | 81.8% |
| SwinV2-Small | 83.7% |
| SwinV2-Base | 84.2% |

### IR — Swin2SR Classical SR
| Scale | Set5 PSNR | Set14 PSNR | BSD100 PSNR |
|---|---|---|---|
| ×2 | 38.42 dB | 34.46 dB | 32.87 dB |
| ×4 | 33.12 dB | 29.48 dB | 28.79 dB |

---

## Monitoring

TensorBoard logs are saved under `checkpoints/`:
```bash
tensorboard --logdir checkpoints/
```

---

## License

Code adapted from:
- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) (MIT)
- [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr) (Apache 2.0)
