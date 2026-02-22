Task 1 — PneumoniaMNIST Classification
======================================

Custom 3-block CNN for pneumonia detection on 28×28 chest X-rays.  
**AUC: 0.9800 \| F1: 0.9320 \| Accuracy: 91.0%**

Quick Start
-----------

### 1. Environment Setup

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
# Create and activate environment (conda or venv)
conda create -n pneumonia python=3.10 -y
conda activate pneumonia

# Install dependencies
pip install torch torchvision medmnist scikit-learn matplotlib seaborn numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Apple Silicon (M1/M2/M3):** PyTorch automatically uses the MPS backend — no
extra install needed.  
**CUDA GPU:** Install the appropriate PyTorch CUDA build from
[pytorch.org](https://pytorch.org).  
**CPU only:** Works as-is, just slower (\~2–3 min training vs \~1 min on GPU).

### 2. Run the Notebook

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
jupyter notebook task1_classification/task1_classification.ipynb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run all cells top-to-bottom. The notebook: 1. Downloads PneumoniaMNIST
automatically (first run only, \~6 MB) 2. Trains the model (\~26 epochs, early
stopping) 3. Tunes the classification threshold on validation set 4. Evaluates
on the test set 5. Generates all plots and saves artifacts

**Google Colab:** Upload the notebook, uncomment Cell 1 (`!pip install ...`),
and run all.

Where Everything Is Saved
-------------------------

After running the notebook, the following files are generated:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
task1_classification/
├── task1_classification.ipynb   # Main notebook
├── models/
│   └── task1_best_model.pth             # Best model checkpoint
│   └── summary_metrics.json             # Test metrics (JSON)
├── images/                              # Figures referenced in report
├── results/
│   └── test_predictions.csv.            # Test set predictions with gt
├──  README.md                           # how to
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### File Details

| File                             | What It Contains                                            | Used By                                       |
|----------------------------------|-------------------------------------------------------------|-----------------------------------------------|
| `task1_model.pth`                | Model weights (`state_dict`), optimizer state, best val AUC | Task 2 (VLM integration), Task 3 (embeddings) |
| `summary_metrics.json`           | Accuracy, precision, recall, F1, AUC, threshold             | Report generation, results comparison         |
| `task1_classification_report.md` | Full analysis with embedded figures                         | \-                                            |
| `test_predictions.csv`           | Test set predictions with gt                                | \-                                            |

Hardware Tested
---------------

| Platform        | Device  | Training Time |
|-----------------|---------|---------------|
| MacBook Pro M1  | MPS GPU | \~1 min       |
| Google Colab    | T4 GPU  | \~1 min       |
| Laptop (no GPU) | CPU     | \~3 min       |

Reproducibility
---------------

-   Random seed: `42` (set for Python, NumPy, PyTorch)

-   Normalization stats computed from training set only

-   Official MedMNIST v2 splits (no reshuffling)

-   Early stopping on validation AUC with patience=8

-   Threshold tuned on validation set (never test)
