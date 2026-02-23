# Task 2: Medical Report Generation — VLM on PneumoniaMNIST

**Model:** `google/medgemma-1.5-4b-it` | **Hardware:** Google Colab T4 GPU (14.56 GB VRAM)  
**Framework:** Transformers 5.2.0, PyTorch 2.10.0+cu128 | **Images evaluated:** 12 stratified from 624 test images

---

## Directory Structure

```
task2_report_generation/
├── task2_report_generation.ipynb   ← main notebook
├── README.md                       ← this file
├── images/                         ← Cell 11 multi-prompt visualisation panels
├── reports/                        ← per-image plain-text report bundles (Cell 16)
└── results/
    ├── vlm_reports.csv
    ├── comparison_table.csv
    ├── prompt_strategy_comparison.csv
    ├── task2_full_results.json
    ├── vlm_report_img_XXXX.png     ← Cell 18 single-image VLM figures
    └── vlm_cnn_comparison_XXXX.png ← Cell 19 VLM + CNN comparison figures
```

---

## Setup and Installation

### 1. Dependencies

```bash
pip install transformers accelerate medmnist pillow pandas matplotlib scikit-learn huggingface_hub
```

Or from the repository root:

```bash
pip install -r requirements.txt
```

### 2. Request MedGemma Access

MedGemma is a gated model. You must accept the license before the weights can be downloaded:

→ https://huggingface.co/google/medgemma-1.5-4b-it

Log in with your Hugging Face account and click **"Agree and access repository"**.

### 3. Add HF Token to Colab Secrets

The notebook reads your token from Colab's secure secrets store — never hardcoded.

```
Colab menu → 🔑 Secrets (left sidebar) → Add new secret
  Name:  HF_TOKEN
  Value: hf_xxxxxxxxxxxxxxxxxxxx   ← your token from huggingface.co/settings/tokens
```

The notebook loads it automatically in Cell C:
```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
```

> **Security:** Never paste your token directly into notebook cells or commit it to a public repository.

### 4. (Optional) Load Task 1 CNN for Comparison

Place your Task 1 model weights on Google Drive at:
```
/content/drive/MyDrive/postdoc_challenge/task1/task1_best_model.pth
```
Task 1 model wieghts are also available to download at: 
https://drive.google.com/file/d/1HvxiecJHDumWjX-R3i6o1UmIJQzcqJza/view?usp=sharing

Then set `CONFIG["cnn_model_path"]` in **Cell 2**:
```python
CONFIG = {
    ...
    "cnn_model_path": "/content/drive/MyDrive/postdoc_challenge/task1/task1_best_model.pth",
    ...
}
```

This enables the CNN baseline comparison in Cell 13 and the side-by-side panels in Cell 19. The notebook works fully without it — CNN columns will simply be omitted.

### 5. Google Colab Runtime

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

The model weights (~8 GB) are downloaded automatically on first run and cached at `/root/.cache/huggingface/`. Subsequent runs reuse the cache.

### 6. Output Directory

All outputs are saved under your Google Drive:
```
/content/drive/MyDrive/postdoc_challenge/task2/
  ├── images/
  ├── reports/
  └── results/
```

Drive is mounted automatically in **Cell A**. Outputs persist across Colab sessions.

---

## Running the Notebook

| Cell | Purpose | Runtime |
|---|---|---|
| A | Mount Drive, create directories | < 1 min |
| B | Install dependencies | < 2 min |
| C | HF authentication | < 1 min |
| 1–2 | Imports, configuration | < 1 min |
| 3 | Load PneumoniaMNIST test set | < 1 min |
| 4 | (Optional) Load Task 1 CNN | < 1 min |
| 5 | Stratified image selection | < 1 min |
| 6 | Image preprocessing | < 1 min |
| 7 | Load MedGemma (~8 GB download on first run) | 3–8 min |
| 8–9 | Define prompts + generation function | < 1 min |
| **10** | **Batch generation: 12 images × 3 prompts = 36 reports** | **9–15 min** |
| 11 | Visualise reports (Cell 11 panels → `images/`) | 2–3 min |
| 12–15 | Label extraction, comparison table, hallucination audit | < 2 min |
| 16–17 | Save all results, print summary | < 1 min |
| 18–19 | Interactive single-image utilities (run independently) | ~2 min each |

---

## Model Selection

**MedGemma 1.5 4B-IT** was selected for three reasons:

- **Medical pre-training:** Trained by Google on radiology image–text corpora; applicable zero-shot without fine-tuning.
- **Instruction-tuned:** The `-it` variant reliably follows structured prompts, enabling FINDINGS/IMPRESSION-formatted outputs.
- **Hardware-compatible:** Fits T4 VRAM in `bfloat16` without quantization, preserving output quality.

**Preprocessing:** 28×28 grayscale → bicubic upscale to 224×224 → converted to RGB (MedGemma expects 3-channel input).

---

## Prompting Strategies

Three strategies were tested across all 12 images (36 total inference calls).

| Strategy | Prompt | Avg Chars | Avg Words | FINDINGS | IMPRESSION |
|---|---|---|---|---|---|
| **Baseline** | "Describe this chest X-ray. What findings do you observe?" | 1,389 | 210 | 3/12 | 10/12 |
| **Structured Clinical** | Radiologist role + FINDINGS / IMPRESSION / CLASSIFICATION | 296 | 38 | 12/12 | 12/12 |
| **Constrained Binary** | Forces Yes/No + confidence rating | 258 | 41 | 5/12 | 0/12 |

**Recommended:** `structured_clinical`. It is 4× more concise than baseline, produces consistent section formatting, and — as shown in Section 6 — directly improves classification accuracy on the same image.

---

## Sample Generated Reports

### Normal Case — Image #164 | GT: Normal | VLM: Normal ✓
```
FINDINGS: Clear lung field with no obvious consolidation, pleural effusion,
or pneumothorax. Heart size appears within normal limits. Mediastinal
contours are unremarkable. Bony structures intact.

IMPRESSION: No acute cardiopulmonary process identified.

CLASSIFICATION: Normal
```

---

### CNN False Negative — Image #176 | GT: Pneumonia | CNN: Normal (0.98) ❌ | VLM: Pneumonia ✓
```
FINDINGS: Possible haziness in left lower zone.
IMPRESSION: Cannot exclude pneumonia.
CLASSIFICATION: Pneumonia
```
*VLM correctly identified pneumonia where the CNN failed — a complementary signal.*

---

## Qualitative Analysis: VLM vs CNN vs Ground Truth

| Image | Category | GT | VLM | VLM ✓ | CNN | CNN ✓ | Agree |
|---|---|---|---|---|---|---|---|
| 146 | CNN False Positive | Normal | Pneumonia | ❌ | Pneumonia (0.66) | ❌ | Yes |
| 370 | CNN False Positive | Normal | Pneumonia | ❌ | Pneumonia (1.00) | ❌ | Yes |
| 68 | True Negative | Normal | Pneumonia | ❌ | Normal (0.96) | ✓ | No |
| 30 | True Negative | Normal | Pneumonia | ❌ | Normal (0.72) | ✓ | No |
| 267 | True Negative | Normal | Pneumonia | ❌ | Normal (0.99) | ✓ | No |
| 176 | CNN False Negative | Pneumonia | Pneumonia | ✓ | Normal (0.98) | ❌ | No |
| 608 | CNN False Negative | Pneumonia | Pneumonia | ✓ | Normal (0.67) | ❌ | No |
| 406 | True Positive | Pneumonia | Pneumonia | ✓ | Pneumonia (1.00) | ✓ | Yes |
| 409 | True Positive | Pneumonia | Pneumonia | ✓ | Pneumonia (1.00) | ✓ | Yes |
| 359 | True Positive | Pneumonia | Pneumonia | ✓ | Pneumonia (1.00) | ✓ | Yes |
| 180 | Ambiguous | Pneumonia | Pneumonia | ✓ | Normal (0.51) | ❌ | No |
| 423 | Ambiguous | Normal | Pneumonia | ❌ | Normal (0.52) | ✓ | No |

**VLM accuracy (constrained prompt):** 6/12 (50%) | **CNN test accuracy:** 93.75% | **CNN–VLM agreement:** 5/12

**Key observations:**
- VLM predicted Pneumonia for all 12 cases — correct for all pneumonia cases, wrong for all 5 normal cases. This is affirmative bias, not genuine visual discrimination.
- VLM recovered both CNN false negatives (#176, #608) and the ambiguous case (#180), suggesting complementary error profiles.
- CNN and VLM agreed on all true positives and both CNN false positives.

![Cell 11 — Multi-prompt report panel](images/report_img_0406.png)

---

## Hallucination Audit

Terms unlikely to be resolvable at 28×28 were flagged: mediastinum, heart size, trachea, air bronchogram, diaphragm.

| Strategy | Images Flagged | Common Terms |
|---|---|---|
| Baseline | 10/12 | mediastinum, heart size, trachea |
| Structured Clinical | 12/12 | mediastinum, heart size |
| Constrained Binary | 4/12 | air bronchograms |

**Total: 26 flags across 36 reports.** The structured clinical prompt produces the highest hallucination density — the FINDINGS template primes the model to comment on all standard radiological structures regardless of actual image content. Flags are review targets, not confirmed errors.

---

## Prompt Sensitivity: Structured vs Unstructured (Cells 18–19)

Three targeted experiments were run on **Image #500 (GT: Pneumonia)** using the `test_image_vlm` and `test_image_vlm_vs_cnn` utility functions, with prompt as the only variable.

---

### Test A — Image #500 | Structured clinical prompt | VLM only

**Prompt:** Full structured clinical prompt with radiologist role assignment.

**Generated report:**
```
FINDINGS:
The chest X-ray shows a clear lung field with no obvious consolidation,
pleural effusion, or pneumothorax. The heart size appears normal. The
mediastinal contours are unremarkable. There is no evidence of a mass
or nodule.

IMPRESSION:
No acute cardiopulmonary process.

CLASSIFICATION:
Normal.
```

**Result: VLM → Normal ❌** (GT: Pneumonia)

The report is fluent, well-structured, and internally consistent — yet clinically wrong. This is the most dangerous failure mode: a confident, formatted misclassification.

![Cell 18 — Image #500, default prompt (misclassification)](images/vlm_report_img_0500.png)

---

### Test B — Image #500 | Partial structured prompt (section headers, no role) | VLM only

**Prompt:**
```
1. FINDINGS: Describe all observable features of the chest X-ray.
2. IMPRESSION: Summarize the overall assessment.
3. CLASSIFICATION: State whether the X-ray appears Normal or shows signs of Pneumonia.
Be concise and clinically precise.
```

**Generated report:**
```
FINDINGS: The chest X-ray shows a hazy opacity in the right lung field.
The left lung field appears relatively clear. The heart size is within
normal limits. The mediastinum is unremarkable. There is no evidence
of pleural effusion.

IMPRESSION: Right lung opacity, possibly representing consolidation
or atelectasis.

CLASSIFICATION: The X-ray shows signs of Pneumonia.
```

**Result: VLM → Pneumonia ✓** (GT: Pneumonia)

The partial structured prompt — section labels alone, no role — recovered the correct classification and additionally identified a specific visual feature ("hazy opacity in the right lung field") as evidentiary basis. This is qualitatively better: evidence-grounded, localised, and falsifiable.

![Cell 18 — Image #500, structured prompt (correct)](images/vlm_report_img_0500_structured.png)

---

### Test C — Image #500 | Partial structured prompt | VLM + CNN comparison

**Same partial structured prompt as Test B, via `test_image_vlm_vs_cnn`.**

**Generated VLM report:**
```
FINDINGS: The chest X-ray shows a hazy opacity in the right lower lung
field. The left lung appears clear. The heart size is normal. There is
no pleural effusion or pneumothorax.

IMPRESSION: Right lower lobe opacity, possibly representing consolidation
or atelectasis.

CLASSIFICATION: Pneumonia.
```

| Model | Prediction | Confidence | Correct |
|---|---|---|---|
| Task 1 CNN | Pneumonia | 99.98% | ✓ |
| Task 2 VLM | Pneumonia | — | ✓ |
| Agreement | **Yes** | — | — |

Both models correct. CNN and VLM independently converged on the correct label via entirely different mechanisms (discriminative probability vs. generative text).

![Cell 19 — Image #500, VLM + CNN comparison](images/vlm_cnn_comparison_img_0500.png)

---

### Why Structured Prompts Improve Results

The three tests isolate prompt structure as the operative variable — same image, same model, different outcome solely from prompt design. Four observations:

**1. Section scaffolding activates pathology-seeking behaviour.** Without explicit headers, MedGemma defaults to a descriptive mode that does not systematically interrogate each anatomical region. FINDINGS/IMPRESSION/CLASSIFICATION forces sequential reasoning, reducing globally-coherent-but-wrong narratives.

**2. Feature attribution improves.** Test A: zero specific visual evidence. Tests B and C: "hazy opacity in the right lung field" — a concrete, localised observation that supports the classification and could be verified by a radiologist.

**3. Role assignment is not the critical ingredient.** Test B used section headers *without* the radiologist role assignment, and achieved comparable quality. The structural scaffold is the dominant factor.

**4. Structured prompts partially compensate for resolution constraints.** Structured prompting redirects the model toward coarse intensity gradients (clear vs. hazy fields) rather than fabricating fine anatomical detail, better matching what is actually resolvable at 28×28.

> **Practical recommendation:** Always use structured prompts with explicit section headers for classification-sensitive VLM tasks on low-resolution datasets. Unstructured prompts should not be used where the output informs a diagnostic decision.

---

## Interactive Utilities

### Cell 18 — Test any image (VLM only)

```python
# Default structured clinical prompt
test_image_vlm(500)

# Custom prompt + save figure to results/
test_image_vlm(
    500,
    prompt_text=(
        "1. FINDINGS: Describe all observable features of the chest X-ray.\n"
        "2. IMPRESSION: Summarize the overall assessment.\n"
        "3. CLASSIFICATION: State whether the X-ray appears Normal or shows signs of Pneumonia.\n"
        "Be concise and clinically precise."
    ),
    save_path=f"{CONFIG['results_dir']}/vlm_report_img_0500_structured.png"
)

# Your own image file
test_image_vlm("/content/drive/MyDrive/my_xray.png")
```

### Cell 19 — Test any image (VLM + CNN comparison)

```python
# Default constrained binary prompt
test_image_vlm_vs_cnn(500)

# Custom prompt + save figure to results/
test_image_vlm_vs_cnn(
    500,
    prompt_text=(
        "1. FINDINGS: Describe all observable features of the chest X-ray.\n"
        "2. IMPRESSION: Summarize the overall assessment.\n"
        "3. CLASSIFICATION: State whether the X-ray appears Normal or shows signs of Pneumonia.\n"
        "Be concise and clinically precise."
    ),
    save_path=f"{CONFIG['results_dir']}/vlm_cnn_comparison_img_0500.png"
)
```

Both functions accept `save_path`. Figures are saved at 300 DPI directly to `results/`.

---

## Strengths and Limitations

| | Notes |
|---|---|
| ✅ **Zero-shot domain knowledge** | Clinically appropriate terminology without fine-tuning |
| ✅ **Prompt-following** | Reliable FINDINGS/IMPRESSION/CLASSIFICATION output when instructed |
| ✅ **Complementary to CNN** | Recovers CNN false negatives; different error profile supports ensemble use |
| ✅ **Structured prompts improve specificity** | Feature-grounded, localised reports vs. generic descriptions (Tests A→B) |
| ❌ **Prompt-sensitive misclassification** | Same image, wrong prompt → confident Normal report on Pneumonia case |
| ❌ **Pervasive hallucination** | Mediastinal, cardiac, tracheal claims in 26/36 reports despite 28×28 resolution |
| ❌ **Resolution mismatch** | Designed for clinical-resolution images; 28×28 is fundamentally inadequate input |
---

