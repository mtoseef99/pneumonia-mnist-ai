# AI Medical Imaging — PneumoniaMNIST

**Dataset:** PneumoniaMNIST (MedMNIST v2) — 28×28 grayscale chest X-rays, binary classification (Normal / Pneumonia)

---

## Repository Structure

```
repository/
├── README.md                          ← you are here
├── requirements.txt
│
├── task1_classification/
│   ├── task1_classification.ipynb
│   ├── README.md                      ← full task report + run instructions
│   ├── models/                        ← saved model weights (.pth)
│   ├── images/                        ← training curves, confusion matrix, ROC, failure cases
│   ├── reports/                       ← generated markdown report
│   └── results/                       ← metrics CSV, predictions
│
└── task2_report_generation/
    ├── task2_report_generation.ipynb
    ├── README.md                      ← full task report + run instructions
    ├── images/                        ← Cell 11 visualisation panels
    ├── reports/                       ← per-image plain-text report bundles
    └── results/                       ← CSV, JSON, saved figures
```

---

## Tasks at a Glance

### Task 1 — CNN Classification
Train a custom CNN on PneumoniaMNIST and perform comprehensive evaluation.

| Metric | Value |
|---|---|
| Test Accuracy | 91.03% |
| Recall (Sensitivity) | 98.46% |
| ROC-AUC | 0.9800 |

**→ Full details, setup, and run instructions:** [`task1_classification/README.md`](task1_classification/README.md)

```
# Quick start (CPU sufficient)
Open task1_classification/task1_classification.ipynb in Colab → Run all
```

---

### Task 2 — Medical Report Generation
Generate structured radiology reports using MedGemma 1.5 4B-IT and evaluate prompting strategies.

| Metric | Value |
|---|---|
| VLM Accuracy (constrained prompt) | 50% (6/12) |
| CNN Test Accuracy (baseline) | 93.75% |
| Hallucination Flags | 26 / 36 reports |

**Requires:** Hugging Face account with MedGemma access + T4 GPU.

**→ Full details, setup, HF token, and run instructions:** [`task2_report_generation/README.md`](task2_report_generation/README.md)

```
# Quick start (T4 GPU required)
1. Add HF_TOKEN to Colab Secrets
2. Open task2_report_generation/task2_report_generation.ipynb in Colab → Run all
```

---

## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Both notebooks also install dependencies automatically when run in Colab.

---

## Contact

**Muhammad Toseef** — mtoseef2-c@my.city.edu.hk
