# Modular Image Classifier

A modular, extensible image classification framework with support for multiple backbones, datasets, and training features.

---

## 📋 Table of Contents

- [Features](#-features)
- [Setup](#-setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Artifacts](#-artifacts)
- [Available Backbones](#-available-backbones)
- [Available Datasets](#-available-datasets)
- [Upcoming Features](#-upcoming-features)

---

## ✨ Features

| Feature | Description | Status |
|:---|:---|:---|
| **Swappable Backbones** | ViT, ConvNeXt, Faster R-CNN, Custom TinyCNN | ✅ Implemented |
| **Swappable Datasets** | CIFAR-10, CIFAR-100, STL-10 | ✅ Implemented |
| **Reproducibility** | Fixed seed with `--seed` | ✅ Implemented |
| **Transfer Learning** | Freeze backbone with `--freeze-backbone` | ✅ Implemented |
| **Evaluation** | Class-wise + global metrics, confusion matrix plots | ✅ Implemented |
| **Mixed Precision (AMP)** | Faster training with `--use-amp` | ✅ Implemented |
| **LR Scheduler** | Cosine annealing with `--use-scheduler` | ✅ Implemented |
| **Gradient Clipping** | Prevent gradient explosion with `--use-grad-clip` | ✅ Implemented |
| **Weight Decay** | L2 regularization via AdamW with `--use-weight-decay` | ✅ Implemented |
| **Deployment** | ONNX export + PTQ (dynamic/static) with comparison report | 🚧 In Progress |

---

## 🔧 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/modular-image-classifier.git
cd modular-image-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
```
matplotlib
pandas
pyyaml
scikit-learn
seaborn
torch
torchvision
```

---

## 🚀 Usage

### Basic Training

```bash
python3 model_trainer.py \
  --arch custom \
  --dataset cifar10 \
  --batch-size 256 \
  --epochs 50
```

### Full Command

```bash
python3 model_trainer.py \
  --config configs/base.yaml \
  --arch vit \
  --dataset cifar100 \
  --batch-size 256 \
  --epochs 50 \
  --learning-rate 1e-3 \
  --seed 42 \
  --freeze-backbone \
  --reports-dir ./artifacts
```

### Using a Config File

```bash
python3 model_trainer.py --config configs/my_config.yaml
```

### Run in Background (Colab/Linux)

```bash
nohup python3 model_trainer.py --config configs/base.yaml > results/output.txt &
```

---

## ⚙️ Configuration

### Command-Line Arguments

| Argument | Type | Default | Description | Status |
|:---|:---|:---|:---|:---|
| `--config` | str | `configs/base.yaml` | Path to YAML config file | ✅ |
| `--arch` | str | `custom` | Model architecture | ✅ |
| `--dataset` | str | `cifar10` | Dataset name | ✅ |
| `--batch-size` | int | `32` | Batch size | ✅ |
| `--epochs` | int | `10` | Number of epochs | ✅ |
| `--learning-rate` | float | `1e-3` | Learning rate | ✅ |
| `--seed` | int | `42` | Random seed | ✅ |
| `--num-workers` | int | `4` | DataLoader workers | ✅ |
| `--data-dir` | str | `./data` | Data directory | ✅ |
| `--reports-dir` | str | `./artifacts` | Output directory | ✅ |
| `--pre-trained` | flag | `False` | Use pretrained weights | ✅ |
| `--freeze-backbone` | flag | `False` | Freeze backbone layers | ✅ |
| `--use-amp` | flag | `False` | Enable mixed precision | ✅ |
| `--use-scheduler` | flag | `False` | Use cosine scheduler | ✅ |
| `--use-grad-clip` | float | `None` | Gradient clipping value | ✅ |
| `--use-weight-decay` | float | `None` | Weight decay value | ✅ |

### YAML Config Example

```yaml
# configs/base.yaml
arch: vit
dataset: cifar100
batch_size: 256
epochs: 50
learning_rate: 1e-3
seed: 42
num_workers: 4
freeze_backbone: true
```

---

## 📁 Project Structure

```
modular-image-classifier/
├── model_trainer.py          # Main training script
├── model_factory.py          # Model creation factory
├── data_loader.py            # Dataset + DataLoader factory
├── eval.py                   # Evaluation utilities
├── utils.py                  # Utilities (seed, args, etc.)
├── cnn_architectures.py      # Custom CNN architectures
├── configs/
│   ├── base.yaml             # Base configuration
│   └── *.yaml                # Additional configs
├── data/                     # Dataset storage
├── artifacts/                # Output artifacts
│   ├── cm_*.png              # Confusion matrix plots
│   ├── summary.txt           # Metrics summary report
│   └── checkpoints/          # Model checkpoints
├── requirements.txt
└── README.md
```

---

## 📊 Artifacts

| File | Description |
|:---|:---|
| `cm_{arch}_{dataset}.png` | Confusion matrix heatmap |
| `summary.txt` | Class-wise + global metrics report |
| `checkpoints/{arch}_{dataset}_best.pth` | Best model checkpoint |

### Artifacts Directory Structure

```
artifacts/
├── cm_custom_cifar10.png
├── cm_vit_cifar100.png
├── summary.txt
├── checkpoints/
│   ├── custom_cifar10_best.pth
│   └── vit_cifar100_best.pth
```

---

## 🧠 Available Backbones

| Backbone | Flag | Description | Status |
|:---|:---|:---|:---|
| **Custom TinyCNN** | `custom` | Small CNN from scratch | ✅ |
| **ViT** | `vit` | Vision Transformer | ✅ |
| **ConvNeXt** | `convnext` | Modern CNN | ✅ |
| **Faster R-CNN** | `frcnn` | Detection backbone (ResNet+FPN) | ✅ |

---

## 📦 Available Datasets

| Dataset | Flag | Classes | Image Size | Status |
|:---|:---|:---|:---|:---|
| **CIFAR-10** | `cifar10` | 10 | 32×32 | ✅ |
| **CIFAR-100** | `cifar100` | 100 | 32×32 | ✅ |
| **STL-10** | `stl10` | 10 | 96×96 | ✅ |

---

## 🔬 Evaluation

After training, the script automatically generates:

1. **Class-wise metrics** (Precision, Recall, F1-Score, Support)
2. **Global metrics** (Macro F1, Weighted Accuracy)
3. **Confusion Matrix** (visual plot)

### Sample Summary Output

```
Experiment: vit | Dataset: cifar100
------------------------------------------------------------

Class-wise Metrics:
Class          Precision  Recall  F1-Score  Support
apple          0.85       0.82    0.83      100
aquarium_fish  0.79       0.81    0.80      100
...

Global Metrics:
Metric              Score
Macro F1            0.8234
Weighted Precision  0.8256
Weighted Recall     0.8234
Weighted F1         0.8245
============================================================
```

---

## 🚧 Upcoming Features

The following features are currently **in progress** or **planned** for future releases:

### In Progress 🚧

| **ONNX Export** | Export trained models to ONNX format | v1.2.0 |
| **Post-Training Quantization (PTQ)** | Dynamic and static quantization | v1.2.0 |

### Feature Status Legend

| Icon | Meaning |
|:---|:---|
| ✅ | Implemented and tested |
| 🚧 | In progress / partially implemented |
| 📅 | Planned for future release |

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- PyTorch & torchvision
- scikit-learn for metrics

---

**Happy Training! 🚀**