# 🧬 Embryo Development Stage Classification

Automated classification of human embryo development stages from time-lapse microscopy images using deep learning. This project trains and compares four CNN architectures — **MobileNetV2, VGG-16, VGG-19, and InceptionV3** — on a 16-class embryo stage dataset.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Development Stages](#development-stages)
- [Model Architecture](#model-architecture)
- [Custom Loss Function](#custom-loss-function)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Overview

In IVF (In Vitro Fertilization) clinics, embryologists manually monitor embryo development using time-lapse imaging systems. This project automates that process by training deep learning models to classify each frame into one of **16 biological development phases** — from the second polar body stage all the way to the hatched blastocyst.

Key features:
- **Embryo-level train/val/test split** to prevent data leakage
- **Custom combined loss function** handling class imbalance and ordinal relationships
- **WeightedRandomSampler** for balanced training across rare phases
- **Early stopping** and **cosine annealing LR scheduler**
- Comparison of 4 pretrained CNN architectures

---

## Dataset

- **Source**: [Embryo Dataset on Kaggle]
- Time-lapse JPEG frames of human embryos
- Per-embryo CSV annotation files specifying the frame range for each development phase
- Split at the **embryo level**: 70% train / 15% val / 15% test

---

## Development Stages

The model classifies each frame into one of **16 ordered biological phases**:

| Index | Label | Description             |
|-------|-------|-------------------------|
| 0     | pPB2  | Second polar body       |
| 1     | pPNa  | Pro-nuclei appearance   |
| 2     | pPNf  | Pro-nuclei disappearance|
| 3     | p2    | 2 cells                 |
| 4     | p3    | 3 cells                 |
| 5     | p4    | 4 cells                 |
| 6     | p5    | 5 cells                 |
| 7     | p6    | 6 cells                 |
| 8     | p7    | 7 cells                 |
| 9     | p8    | 8 cells                 |
| 10    | p9+   | 9+ cells                |
| 11    | pM    | Morula / compaction     |
| 12    | pSB   | Start blastulation      |
| 13    | pB    | Full blastocyst         |
| 14    | pEB   | Expanded blastocyst     |
| 15    | pHB   | Hatched blastocyst      |

---

## Model Architecture

All four models use **ImageNet pretrained weights** with only the final classification head replaced for 16-class output:

| Model       | Input Size | Final Layer Replaced         |
|-------------|------------|------------------------------|
| MobileNetV2 | 224×224    | `classifier[1]`              |
| VGG-16      | 224×224    | `classifier[6]`              |
| VGG-19      | 224×224    | `classifier[6]`              |
| InceptionV3 | 299×299    | `fc` + `AuxLogits.fc`        |

**Training configuration:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Early stopping: patience=6
- Batch size: 32
- Data augmentation: random crop, horizontal/vertical flip, rotation, color jitter

---

## Custom Loss Function

A custom `EmbryoStageLoss` was designed combining three terms:

```
Loss = 0.4 × Label-Smoothed CE  +  0.4 × Focal Loss  +  0.2 × Ordinal Penalty
```

### 1. Label-Smoothed Cross-Entropy (weight: 0.4)
Softens hard targets to prevent overconfidence and improve calibration:
```
q_i = (1 - ε) · 𝟙[i=y] + ε/K
```

### 2. Focal Loss (weight: 0.4)
Scales loss by `(1 - p_t)^γ` to focus on hard/rare phases:
```
L_focal = (1 - p_t)^γ × L_CE
```

### 3. Ordinal Penalty (weight: 0.2)
Penalises biologically distant misclassifications more severely:
```
L_ord = Σ_j p(j) · |j - y| / (K - 1)
```
Confusing p4 with p5 is a minor error; confusing p2 with pHB is severely penalised.

**Desirable properties satisfied:**
- ✅ Non-negative
- ✅ Zero at perfect prediction
- ✅ Differentiable everywhere
- ✅ Imbalance-aware (focal term)
- ✅ Ordinal-aware (ordinal penalty)
- ✅ Calibrated (label smoothing)

---

## Results

> ⚠️ Note: Models were trained for limited epochs due to GPU session constraints (Kaggle P100/T4 12-hour limit).

| Architecture | Epochs | Test Accuracy | Macro F1 | Weighted F1 |
|--------------|--------|--------------|----------|-------------|
| InceptionV3  | 3      | **68.81%**   | **63.09%**| **70.11%** |
| VGG-19       | 3      | 37.74%       | 38.86%   | 39.56%      |
| MobileNetV2  | 10     | 9.49%        | 2.19%    | 5.71%       |
| VGG-16       | 5      | 4.69%        | 1.81%    | 1.32%       |

**InceptionV3 performed best** despite fewest epochs, likely due to its multi-scale feature extraction and auxiliary classifier providing stronger gradient signal during early training.

> Full training (30 epochs each) is expected to significantly improve all models, particularly MobileNetV2 and VGG-16 which were still in early convergence.

---

## Project Structure

```
embryo-classification/
│
├── embryo-classification.ipynb   # Main Kaggle notebook
├── README.md                     # This file
│
└── checkpoints/                  # Saved model weights (not tracked in git)
    ├── mobilenetv2_best.pt
    ├── vgg16_best.pt
    ├── vgg19_best.pt
    └── inceptionv3_best.pt
```

---

## How to Run

### On Kaggle
1. Upload the notebook to Kaggle
2. Add the [Embryo Dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset) as a data source
3. Enable **GPU accelerator** (P100 or T4)
4. Run cells sequentially (Cell 1 → Cell 16)

### Resuming after session timeout
If your Kaggle session times out mid-training, checkpoints are preserved in `/kaggle/working/checkpoints/`. On a new session:
```python
# Check which models are already trained
for arch in ["mobilenetv2", "vgg16", "vgg19", "inceptionv3"]:
    path = f"{CHECKPOINT_DIR}/{arch}_best.pt"
    print(f"{arch:15s} → {'✅ EXISTS' if os.path.exists(path) else '❌ MISSING'}")
```

---

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
tqdm
Pillow
numpy
```

All dependencies are pre-installed in the Kaggle Python environment.

---

## Notes

- InceptionV3 requires **299×299** input — data loaders are automatically rebuilt at this resolution in Cell 15
- VGG models are memory-intensive; reduce `BATCH_SIZE = 16` if you encounter CUDA OOM errors
- The `WeightedRandomSampler` ensures rare phases like `pPB2` and `pHB` are seen during training despite class imbalance
