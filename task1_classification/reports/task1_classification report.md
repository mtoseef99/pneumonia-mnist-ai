Task 1 — PneumoniaMNIST Classification Report
=============================================

1. **Architecture**
---------------

Three-block CNN designed for native 28×28 grayscale input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Block 1: Conv(1→32)×2 → BN → ReLU → MaxPool → SpatialDropout(0.10)
Block 2: Conv(32→64)×2 → BN → ReLU → MaxPool → SpatialDropout(0.15)
Block 3: Conv(64→128)×2 → BN → ReLU → MaxPool → SpatialDropout(0.20)
Head:    AdaptiveAvgPool → FC(128) → BN → Dropout(0.30) → FC(1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**\~240K parameters.** This architecture outperformed both a Vision Transformer
(\~400K params, AUC \~0.93–0.95) and EfficientNetV2-B0 with transfer learning
(\~7.1M params, AUC \~0.92–0.96) on this dataset. The key advantage is
resolution-appropriate design and deeper models built for 224×224+ inputs waste
capacity or require artificial upsampling at 28×28.

 

| Model                        | Parameters | Test AUC    | Test Accuracy | Test F1     |
|------------------------------|------------|-------------|---------------|-------------|
| **Custom CNN (3-block)**     | **\~240K** | **0.9800**  | **0.9103**    | **0.9320**  |
| Vision Transformer (ViT)     | \~400K     | \~0.93–0.95 | \~0.85–0.88   | \~0.88–0.91 |
| EfficientNetV2-B0 (transfer) | \~7.1M     | \~0.92–0.96 | \~0.83–0.87   | \~0.86–0.90 |


2. **Data**
-------

| Split | Total | Normal        | Pneumonia     |
|-------|-------|---------------|---------------|
| Train | 4,708 | 1,214 (25.8%) | 3,494 (74.2%) |
| Val   | 524   | 135 (25.8%)   | 389 (74.2%)   |
| Test  | 624   | 234 (37.5%)   | 390 (62.5%)   |

 

~   Class Distribution
![Sample Images](images/class_distribution.png)

~   Sample Images
![Sample Images](images/sample_images.png)

~   Pixel Statistics
![Pixel Statistics](images/pixel_distribution.png)

3. Training
-----------

| Parameter      | Value                                    | Rationale                                   |
|----------------|------------------------------------------|---------------------------------------------|
| Optimizer      | Adam (lr=1e-3, wd=1e-4)                  | Fast convergence for compact models         |
| Loss           | BCEWithLogitsLoss + pos_weight           | Class-weighted for 3:1 imbalance            |
| Early stopping | Patience=8 on val AUC                    | Threshold-independent selection             |
| Threshold      | 0.05 (tuned on val F1)                   | Low due to class-weighted probability shift |
| Augmentation   | Rotation ±5°, translation ±5%, sharpness | Medically justified only                    |
| Epochs trained | 26/100 (early stopped)                   | Val AUC peaked at 0.9977                    |

**Excluded augmentations:** horizontal flip (cardiac laterality), vertical flip
(implausible), crop (destructive at 28×28).

~   Training Curves
![Sample Images](images/training_curves.png)

4. **Test Results**
---------------

| Metric                   | Value      |
|--------------------------|------------|
| **Accuracy**             | **0.9103** |
| **Precision**            | **0.8848** |
| **Recall (Sensitivity)** | **0.9846** |
| **Specificity**          | **0.7863** |
| **F1 Score**             | **0.9320** |
| **ROC-AUC**              | **0.9800** |

 

~   Confusion Matrix
<img src="images/confusion_matrix.png" width="400">
<!-- ![Sample Images](images/confusion_matrix.png) -->
 
~   ROC Curve
<img src="images/roc_curve.png" width="350">
<!-- ![Sample Images](images/roc_curve.png) -->
 
5. Failure Analysis
-------------------

| Type            | Count | Rate                                 |
|-----------------|-------|--------------------------------------|
| False Positives | 50    | Normal → Pneumonia                   |
| False Negatives | 6     | Pneumonia → Normal (1.54% miss rate) |

 

~   Failure Cases
![Sample Images](images/error_analysis.png)

6. Model Comparison
-------------------

| Model              | Params   | AUC         | F1          | Resolution       |
|--------------------|----------|-------------|-------------|------------------|
| **Custom CNN**     | **240K** | **0.9800**  | **0.9320**  | Native 28×28     |
| Vision Transformer | 400K     | \~0.93–0.95 | \~0.88–0.91 | Native (patches) |
| EfficientNetV2-B0  | 7.1M     | \~0.92–0.96 | \~0.86–0.90 | Requires 32×32   |

 

7. Limitations & Future Work
----------------------------

-   **28×28 resolution** limits diagnostic detail (air bronchograms,
    interstitial patterns invisible)

-   **Moderate specificity** (78.6%) would generate false alarm burden in
    high-volume settings

-   **Single-seed evaluation** — multi-seed runs with CI would strengthen
    conclusions

-   **No interpretability** — adding Grad-CAM would verify clinically relevant
    attention

-   **Probability calibration** (Platt scaling) could normalize the low
    threshold
