AGENTS.md
==========

Project: Multimodal Probabilistic SPEI Forecasting
--------------------------------------------------

This project predicts probabilistic distributions for:

- SPEI_30d
- SPEI_1y
- SPEI_2y

Evaluation Metric: CRPS (Continuous Ranked Probability Score)

The task is a multi-task probabilistic regression problem using multimodal inputs.


1. Problem Definition
---------------------

Inputs per sample:
- Beetle image
- siteID (19 categories)
- scientificName (144 categories)
- collectDate

Outputs per sample:
- μ_30d, σ_30d
- μ_1y,  σ_1y
- μ_2y,  σ_2y

Total model outputs: 6 values


2. Data Preprocessing
---------------------

2.1 Image Preprocessing

- Resize to 224x224 (or 384x384 if larger backbone)
- Convert to RGB
- Normalize with ImageNet stats:

  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

Train augmentations (light only):
- Horizontal flip
- ±10° rotation
- Mild brightness/contrast

Avoid heavy color jitter.


2.2 siteID Encoding

- Map to integer [0–18]
- Embedding(19, 6)
- Output dimension: 6


2.3 scientificName Encoding

- Map to integer [0–143]
- Embedding(144, 16)
- Output dimension: 16


2.4 collectDate Encoding

Extract:
- year (normalized)
- day-of-year (cyclical encoding)

sin_doy = sin(2π * doy / 365)
cos_doy = cos(2π * doy / 365)

Final date vector: 3 values

Pass through MLP:
  Linear(3 → 16)
  ReLU

Output: 16-dim


3. Model Architecture
---------------------

3.1 Image Branch

Backbone: EfficientNet-B0 or ResNet50 (pretrained)

Output embedding: 512-dim
If backbone larger, reduce with:
  Linear(1280 → 512)
  ReLU


3.2 Metadata Branch

Inputs:
- Site embedding (6)
- Species embedding (16)
- Date MLP output (16)

Concatenate → 38-dim

Metadata MLP:
  Linear(38 → 64)
  ReLU
  Dropout(0.1)
  Linear(64 → 64)
  ReLU

Output: 64-dim


3.3 Fusion

Concatenate:
- Image: 512
- Metadata: 64

Total: 576-dim


3.4 Prediction Head

  Linear(576 → 256)
  ReLU
  Dropout(0.2)

  Linear(256 → 128)
  ReLU

  Linear(128 → 6)

Split outputs:
  mu        = output[:, 0:3]
  log_sigma = output[:, 3:6]
  sigma     = exp(log_sigma)
  sigma     = clamp(sigma, min=1e-3, max=10)


4. Target Normalization
-----------------------

Normalize each target separately:

  SPEI_norm = (SPEI - mean_train) / std_train

Do this for:
- 30d
- 1y
- 2y

During inference:
  μ_real = μ_norm * std + mean
  σ_real = σ_norm * std


5. Loss Function
----------------

5.1 Gaussian Negative Log Likelihood

For each target:

  NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)

Total loss:
  Loss = mean_batch(
      NLL_30d +
      NLL_1y +
      NLL_2y
  )


5.2 Optional Hybrid Loss (Recommended)

  Loss = 0.7 * NLL + 0.3 * CRPS


6. Evaluation
-------------

Primary metric: CRPS

Track:
- CRPS_30d
- CRPS_1y
- CRPS_2y
- Average CRPS

Also compute RMSE for sanity.


7. Cross-Validation
-------------------

Use GroupKFold(n_splits=5)
Group variable: siteID

Prevents:
- Site memorization
- Climate leakage


8. Sigma Calibration (Important)
--------------------------------

On validation set:

  σ_30d = α1 * σ_30d
  σ_1y  = α2 * σ_1y
  σ_2y  = α3 * σ_2y

Search α ∈ [0.7, 1.5]
Select α minimizing CRPS.


9. Baseline Model
-----------------

Metadata-only LightGBM

Features:
- siteID
- scientificName
- sin_doy
- cos_doy
- year

Train 3 Gaussian regressors (mean + variance).

Establish baseline before multimodal model.


10. Ensembling
--------------

Train:
- 5-fold models
- Multiple seeds

Ensemble:

  μ_ensemble = mean(μ_k)

  σ²_ensemble =
      mean(σ_k² + μ_k²)
      - μ_ensemble²

Do NOT average σ directly.


11. Model Variants
------------------

Evaluate:
- Single multi-task model
- Three independent models (one per target)


12. Expected Difficulty
-----------------------

SPEI_30d  → Hardest
SPEI_1y   → Medium
SPEI_2y   → Easiest


13. Training Best Practices
---------------------------

- Mixed precision (AMP)
- Early stopping on validation CRPS
- Cosine or OneCycle LR schedule
- Weight decay: 1e-4
- Maximize GPU batch size


14. Final Objective
-------------------

Minimize:

Average CRPS across:
- SPEI_30d
- SPEI_1y
- SPEI_2y

Primary goals:
- Accurate mean (μ)
- Well-calibrated variance (σ)
- Proper cross-validation
- Robust ensembling