# AI Medical Imaging — PneumoniaMNIST

`Python 3.x` | `PyTorch` | `MedMNIST v2` | `MedGemma 4B-IT`

This repository is a submission for a 7-day postdoctoral technical challenge at Alfaisal University, exploring AI applications across three areas of medical imaging: CNN-based pneumonia classification, automated radiology report generation using a vision-language model, and a semantic image retrieval framework using embeddings and vector search. The work spans computer vision, natural language processing, and information retrieval — all grounded in the PneumoniaMNIST chest X-ray dataset.

**Dataset:** [PneumoniaMNIST (MedMNIST v2)](https://medmnist.com/) — 28×28 grayscale chest X-rays, binary
classification (Normal / Pneumonia)


## Requirements

|                       | Task 1              | Task 2                                  |
|-----------------------|---------------------|-----------------------------------------|
| GPU                   | CPU sufficient      | T4 GPU required                         |
| Hugging Face account  | Not required        | Required (MedGemma gated model access)  |

## Repository Structure

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pneumonia-mnist-ai/
├── README.md                          ← you are here
├── models/                            ← saved best model weights
├── reports/                           ← detailed markdown tasks' implementation reports
│
├── task1_classification/
│   ├── task1_classification.ipynb
│   ├── task1_classification_report.md
│   ├── README.md                      ← detailed instructions
│   ├── models/                        ← saved model weights (.pth)
│   ├── images/                        ← training curves, confusion matrix, ROC, failure cases
│   └── results/                       ← metrics CSV, predictions
│
├── task2_report_generation/
│   ├── task2_report_generation.ipynb
│   ├── README.md                      ← full task report
│   ├── images/                        ← Cell 11 visualisation panels
│   ├── reports/                       ← per-image plain-text report
│   └── results/                       ← CSV, JSON, saved figures
│
├── notebooks/                         ← Google Colab notebooks
│   ├── task1_classification.ipynb
│   ├── task2_report_generation.ipynb
└── requirements.txt                   ← Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/mtoseef99/pneumonia-mnist-ai.git
cd pneumonia-mnist-ai
pip install -r requirements.txt
```

> The PneumoniaMNIST dataset downloads automatically on first run via the `medmnist` library — no manual download needed.

**2. Task 1 — CNN Classification (CPU sufficient)**

Open `notebooks/task1_classification.ipynb` in Google Colab and select **Runtime → Run all**.

**3. Task 2 — Medical Report Generation (T4 GPU required)**

1. Request access to [MedGemma on Hugging Face](https://huggingface.co/google/medgemma-4b-it)
2. In Colab, go to **Secrets** and add your token as `HF_TOKEN`
3. Open `notebooks/task2_report_generation.ipynb` and select **Runtime → Run all**


## Tasks at a Glance

### Task 1 — CNN Classification

Train a custom CNN on PneumoniaMNIST and perform comprehensive evaluation.

| Metric               | Value  |
|----------------------|--------|
| Test Accuracy        | 91.03% |
| Recall (Sensitivity) | 98.46% |
| ROC-AUC              | 0.9800 |

**→ Full details, setup, and run instructions:**
[task1_classification/README.md](task1_classification/README.md)

**→ Report of Task 1:**
[task1_classification_report.md](task1_classification/task1_classification_report.md)

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quick start (CPU sufficient)
Open task1_classification/task1_classification.ipynb in Colab → Run all
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

### Task 2 — Medical Report Generation

Generate structured radiology reports using MedGemma 1.5 4B-IT and evaluate
prompting strategies.

| Metric                            | Value           |
|-----------------------------------|-----------------|
| VLM Accuracy (constrained prompt) | 50% (6/12)      |
| CNN Test Accuracy (baseline)      | 93.75%          |
| Hallucination Flags               | 26 / 36 reports |

> VLM accuracy is measured on a 12-image subset using a constrained prompt; the high hallucination rate reflects unconstrained free-text generation. See the [full report](task2_report_generation/README.md) for detailed analysis.

**→ Full detailed report including setup, HF token, and run instructions:**
[task2_report_generation/README.md](task2_report_generation/README.md)

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quick start (T4 GPU required)
1. Add HF_TOKEN to Colab Secrets
2. Open task2_report_generation/task2_report_generation.ipynb in Colab → Run all
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

## Future Work


- **Task 3 completion — Semantic Image Retrieval (CBIR)** — complete the content-based image retrieval framework by extracting embeddings from the trained CNN or MedGemma.
- **Improve VLM diagnostic accuracy** — fine-tune MedGemma on radiology data to surpass the current 50% constrained-prompt baseline  
- **Reduce hallucinations** — apply RAG or structured output constraints to reduce the 26/36 hallucination flag rate 
- **Severity grading** — extend classification beyond binary labels to multi-class pneumonia severity scoring
- **Larger VLMs** — benchmark MedGemma 27B or BioViL-T against the current 4B baseline
- **Domain adaptation** — fine-tune on diverse, multi-site datasets to improve          generalisation and support equitable clinical decision making across populations  


## Citation

If you use this work, please also cite the underlying dataset and model:

**MedMNIST v2**
```bibtex
@article{medmnistv2,
    title={MedMNIST v2 - A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group}
}
```

**MedGemma**
```bibtex
@misc{medgemma2025,
    title={MedGemma: A Family of Medical AI Models},
    author={Google DeepMind},
    year={2025},
    url={https://huggingface.co/google/medgemma-4b-it}
}
```

## License

For academic use only.

## Contact

**Muhammad Toseef** — mtoseef2-c\@my.city.edu.hk
