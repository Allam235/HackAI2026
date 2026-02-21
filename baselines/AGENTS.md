# AGENTS.md

## Project: Beetles Scientific-Mood ML Challenge  
**Organized by:** Imageomics Informatics  
**Platform:** CodaBench  
**Evaluation Metric:** CRPS (RMS aggregated across categories)  
**Prediction Type:** Event-level probabilistic forecasting  
**Docker Image:** ytchou97/hdr-image:latest  

---

# 1. Problem Overview

This challenge predicts drought conditions (SPEI) using images of carabid beetle specimens collected during sampling events.

This is:

- An **out-of-sample domain generalization problem**
- A **set-based prediction problem**
- A **probabilistic regression task**
- Evaluated using **CRPS**
- Final ranking based on **RMS of CRPS categories**

Training and test eco-domains (NEON domains) are disjoint. The final phase includes a completely unseen eco-domain.

---

# 2. Prediction Target

For each **eventID**, predict a Gaussian distribution for:

- `SPEI_30d`
- `SPEI_1y`
- `SPEI_2y`

Each prediction must include:

- `mu` (mean)
- `sigma` (standard deviation, > 0)

Required output format:

```python
{
    "SPEI_30d": {"mu": float, "sigma": float},
    "SPEI_1y":  {"mu": float, "sigma": float},
    "SPEI_2y":  {"mu": float, "sigma": float},
}
```

Lower CRPS is better (0 is optimal).

---

# 3. Input Interface (CodaBench Requirement)

`predict()` receives:

```python
List[dict]
```

Each dictionary corresponds to **one specimen within the same sampling event**.

Each specimen dictionary contains:

- `relative_img` (PIL.Image) — beetle image  
- `colorpicker_img` (PIL.Image) — color calibration card  
- `scalebar_img` (PIL.Image) — scale image  
- `scientificName` (str)  
- `domainID` (int, anonymized NEON eco-domain)

Important:

- There may be **multiple specimens per event**
- You must aggregate specimen-level information into a single event-level prediction

---

# 4. Modeling Strategy

This is a **Deep Sets problem**.

For each event:

```
Event = {specimen_1, specimen_2, ..., specimen_n}
```

We compute:

1. Specimen-level embeddings  
2. Aggregate across specimens  
3. Produce event-level Gaussian predictions  

---

# 5. Model Architecture

## 5.1 Specimen Encoder

### Image Processing

Baseline recommendation:

- Use `relative_img` only initially
- Resize to 224x224
- Normalize using ImageNet stats:
  - mean = [0.485, 0.456, 0.406]
  - std  = [0.229, 0.224, 0.225]

Optional upgrade:
- Concatenate relative, colorpicker, and scalebar images (9 channels total)
- Modify first CNN layer accordingly

Backbone:

- EfficientNet-B0 or ResNet50 (pretrained)

Output embedding: 512-dim

---

### Metadata Encoding

Scientific Name:
- Map to integer index
- Embedding(144, 16)

Domain ID:
- Embedding(?, 4–6)
- Small dimension recommended (avoid overfitting to domain)

Concatenate:

```
image_embedding (512)
+ species_embedding (16)
+ domain_embedding (4–6)
```

Pass through:

```
Linear → 256
ReLU
```

Output: 256-dim specimen embedding

---

# 6. Event Aggregation

Use permutation-invariant pooling:

### Baseline (recommended):

```
event_embedding = mean(specimen_embeddings)
```

Optional upgrade:
- Attention pooling

---

# 7. Event-Level Prediction Head

```
Linear(256 → 128)
ReLU

Linear(128 → 6)
```

Split outputs:

```
mu        = output[:, 0:3]
log_sigma = output[:, 3:6]
sigma     = exp(log_sigma)
sigma     = clamp(sigma, min=1e-3, max=10)
```

---

# 8. Target Normalization

Normalize each target separately:

```
SPEI_norm = (SPEI - mean_train) / std_train
```

Apply independently for:

- 30d
- 1y
- 2y

During inference:

```
μ_real = μ_norm * std + mean
σ_real = σ_norm * std
```

---

# 9. Loss Function

## 9.1 Gaussian Negative Log Likelihood (Primary)

For each target:

```
NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)
```

Total loss:

```
Loss = mean_batch(
    NLL_30d +
    NLL_1y +
    NLL_2y
)
```

---

## 9.2 Optional Hybrid Loss

To better align with evaluation metric:

```
Loss = 0.7 * NLL + 0.3 * CRPS
```

CRPS computed using Gaussian closed-form.

---

# 10. Evaluation

Primary metric: CRPS

Leaderboard score:
- RMS across:
  - SPEI_30d
  - SPEI_1y
  - SPEI_2y
  - Novel eco-domain category

Lower is better.

Also monitor:
- RMSE (sanity check only)

---

# 11. Cross-Validation Strategy

Use:

```
GroupKFold(n_splits=5)
group = domainID
```

Purpose:
- Prevent domain leakage
- Simulate novel eco-domain scenario

---

# 12. Sigma Calibration (Critical)

After training:

On validation set:

```
σ_30d = α1 * σ_30d
σ_1y  = α2 * σ_1y
σ_2y  = α3 * σ_2y
```

Search:

```
α ∈ [0.7, 1.5]
```

Choose α minimizing CRPS.

Calibration significantly improves leaderboard performance.

---

# 13. Baselines (Mandatory)

## Metadata-Only Baseline

Per event features:

- Species frequency counts
- Species diversity
- Number of specimens
- Domain ID

Train Gaussian LightGBM regressors.

This establishes whether image features add ecological signal.

---

# 14. Ensembling

Train:

- 5-fold models
- Multiple random seeds

Ensemble means and variances correctly:

```
μ_ensemble = mean(μ_k)

σ²_ensemble =
    mean(σ_k² + μ_k²)
    - μ_ensemble²
```

Do NOT average σ directly.

---

# 15. Domain Generalization Considerations

Final phase includes unseen eco-domain.

Recommendations:

- Use small domain embedding
- Try model without domain embedding
- Use regularization:
  - Dropout
  - Weight decay
  - Early stopping
- Avoid memorizing domain-specific drought baselines

---

# 16. Submission Requirements

Zip file must contain:

```
model.py
requirements.txt
weights/
```

`model.py` must define:

```python
class Model:

    def load(self):
        ...

    def predict(self, list_of_dicts):
        ...
```

`predict()` must:

1. Process each specimen
2. Generate embeddings
3. Aggregate to event-level
4. Return properly formatted Gaussian predictions

---

# 17. Final Objective

Minimize:

Average CRPS (RMS aggregated across categories)

Primary goals:

- Accurate mean (μ)
- Well-calibrated uncertainty (σ)
- Strong domain generalization
- Robust ensembling