# 🔬 Glaucoma Detection from Fundus Images

> Binary classification of glaucoma using deep learning on retinal fundus photographs.  
> EfficientNet-B4 fine-tuned on G1020 + ORIGA datasets — AUC-ROC: 0.72

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

Glaucoma is the **second leading cause of blindness** worldwide, affecting over 80 million people. It progresses silently — patients lose vision without symptoms until severe damage has occurred. Early detection through automated fundus image screening can prevent irreversible blindness.

This project trains a deep learning model to classify fundus photographs as **Normal** or **Glaucoma** using transfer learning with EfficientNet-B4.

---

## 🗂️ Dataset

| Dataset | Images | Source |
|---|---|---|
| [G1020](https://arxiv.org/abs/2006.09158) | 1020 | Fundus images + binary labels |
| [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/) | 650 | Fundus images + CDR + labels |
| **Total (usable)** | **~670** | After file availability check |

**Class distribution:** ~70% Normal / ~30% Glaucoma (imbalanced)

Dataset available on Kaggle:  
👉 [arnavjain1/glaucoma-datasets](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets)

---

## 🧠 Model Architecture

```
Input [B, 3, 384, 384]
    ↓
EfficientNet-B4 Backbone (pretrained ImageNet)
    ↓  Global Average Pooling
[B, 1792]
    ↓  Dropout(0.3)
    ↓  Linear(1792 → 512) + ReLU
    ↓  Dropout(0.15)
    ↓  Linear(512 → 1)
Output: raw logit → sigmoid → P(Glaucoma)
```

### Training Strategy: 2-Phase Fine-Tuning

| Phase | Epochs | Backbone | Learning Rate |
|---|---|---|---|
| Warm-up | 1–5 | ❄️ Frozen | `5 × LR` |
| Full fine-tuning | 6+ | 🔓 Unfrozen | `LR = 5e-5` |

---

## 📊 Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.7178 |
| Sensitivity (Recall) | 0.6429 |
| Specificity | 0.7072 |
| F1-Score | 0.5357 |
| Accuracy | 0.6892 |
| Optimal Threshold | 0.511 |

> ⚠️ Performance is limited by dataset size (~670 usable images). See [Future Work](#-future-work) for improvement strategies.

---

## 🗃️ Repository Structure

```
glaucoma-fundus-classification/
│
├── notebook/
│   └── glaucoma_classification_en.ipynb   # Full Kaggle notebook (English)
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/goktani/glaucoma-fundus-classification.git
cd glaucoma-fundus-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Click **+ New Notebook**
3. **File → Import Notebook** → upload `glaucoma_classification_en.ipynb`
4. Add dataset: `arnavjain1/glaucoma-datasets`
5. Settings → Accelerator → **GPU T4 x2**
6. **Run All**

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `IMAGE_SIZE` | 384 | EfficientNet-B4 optimal resolution |
| `BATCH_SIZE` | 16 | Safe for 16GB GPU |
| `LR` | 5e-5 | Fine-tuning learning rate |
| `EPOCHS` | 50 | With early stopping (patience=10) |
| `Loss` | Focal Loss | α=0.75, γ=2.0 — handles class imbalance |
| `Optimizer` | AdamW | weight_decay=1e-4 |
| `Scheduler` | CosineAnnealingLR | eta_min=1e-7 |

---

## 🔍 Features

- ✅ **Transfer Learning** — EfficientNet-B4 pretrained on ImageNet
- ✅ **Focal Loss** — addresses class imbalance
- ✅ **2-phase fine-tuning** — stable warm-up before full training
- ✅ **Augmentation pipeline** — 9 clinically-motivated transforms via albumentations
- ✅ **Youden's J threshold** — optimal sensitivity/specificity balance
- ✅ **Grad-CAM** — visual explanation of model decisions
- ✅ **Clinical metrics** — AUC, Sensitivity, Specificity, PPV, NPV

---

## 🚧 Future Work

| Priority | Improvement |
|---|---|
| 🟢 Quick | Lower threshold to 0.3–0.35 to prioritize sensitivity |
| 🟢 Quick | Test-Time Augmentation (TTA) |
| 🟡 Medium | Add REFUGE dataset (1200 extra images) |
| 🟡 Medium | 5-Fold stratified cross-validation |
| 🟡 Medium | Switch to EfficientNet-B0 (less overfitting on small data) |
| 🔴 Advanced | Multi-task learning (classification + segmentation) |
| 🔴 Advanced | Model ensemble (EfficientNet + ResNet50 + ViT) |
| 🔴 Advanced | Self-supervised pretraining on unlabeled fundus images |

---

## 📚 References

- **G1020 Dataset**: Bajwa et al. (2020) — [arXiv:2006.09158](https://arxiv.org/abs/2006.09158)
- **ORIGA Dataset**: Zhang et al. (2010) — [PubMed](https://pubmed.ncbi.nlm.nih.gov/21095735/)
- **REFUGE Challenge**: Orlando et al. — [IEEE Dataport](https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge)
- **EfficientNet**: Tan & Le (2019) — [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **Focal Loss**: Lin et al. (2017) — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **Grad-CAM**: Selvaraju et al. (2017) — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

---

## ⚠️ Clinical Disclaimer

> This project is for **research and educational purposes only**. The model is NOT validated for clinical use. Glaucoma diagnosis requires comprehensive ophthalmological examination including intraocular pressure measurement, visual field testing, and expert clinical interpretation. **Do not use this model to make medical decisions.**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
