# AGENTS.md
#This focuses more on the images

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

`List[dict]`

Each dictionary corresponds to one specimen within the same sampling event.

Each specimen dictionary contains:

```python
relative_img       # PIL.Image of the beetle
colorpicker_img    # PIL.Image of the color calibration card
scalebar_img       # PIL.Image of the scalebar
scientificName     # str, scientific name
domainID           # int, anonymized NEON eco-domain
```

Important:

- There may be multiple specimens per event  
- You must aggregate specimen-level information into a single event-level prediction  

---

# 4. Modeling Strategy

This is a Deep Sets problem.

For each event:

```
Event = {specimen_1, specimen_2, ..., specimen_n}
```

We compute:

- Specimen-level embeddings  
- Aggregate across specimens  
- Produce event-level Gaussian predictions  

---

# 5. Model Architecture

## 5.1 Specimen Encoder

### Image Processing

Baseline recommendation:

- Use `relative_img` only initially  
- Resize to 224x224  
- Normalize using ImageNet stats:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Optional upgrade:

- Apply color correction using colorpicker (not concatenation)  
- Ignore scalebar as input; optionally extract scale numerically  
- Concatenate relative, colorpicker, and scalebar images only if consistent (9 channels)  
- If using 9-channel input, modify the first CNN layer accordingly  

Backbone:

- EfficientNet-B0 or ResNet50 (pretrained)  
- Output embedding: 512-dim  

---

### Metadata Encoding

Scientific Name:

- Map to integer index  
- `Embedding(144, 16)`

Domain ID:

- ⚠️ Use cautiously — may harm generalization  
- `Embedding(?, 4–6)`  
- Small dimension recommended if used  

Concatenate:

- image_embedding (512)  
- species_embedding (16)  
- optional domain_embedding (4–6)  

Pass through:

```python
Linear → 256
ReLU
```

Output: 256-dim specimen embedding  

---

# 6. Event Aggregation

Use permutation-invariant pooling.

Baseline:

```python
event_embedding = mean(specimen_embeddings)
```

Optional upgrade:

- Attention pooling: weight specimens by information content  
- Concatenate explicit event-level features:
  - species counts  
  - richness  
  - diversity  
  - specimen count  

---

# 7. Event-Level Prediction Head

```python
Linear(256 → 128)
ReLU
Linear(128 → 6)
```

Split outputs:

```python
mu        = output[:, 0:3]
log_sigma = output[:, 3:6]
sigma     = softplus(log_sigma) + 1e-4  # numerically stable
sigma     = clamp(sigma, max=10)       # prevent runaway predictions
```

---

# 8. Target Normalization

Normalize each target separately:

```python
SPEI_norm = (SPEI - mean_train) / std_train
```

Apply independently for:

- 30d  
- 1y  
- 2y  

During inference:

```python
μ_real = μ_norm * std + mean
σ_real = σ_norm * std
```

---

# 9. Loss Function

## 9.1 Gaussian Negative Log Likelihood (Primary)

For each target:

```python
NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)
```

Total loss:

```python
Loss = mean_batch(NLL_30d + NLL_1y + NLL_2y)
```

---

## 9.2 Optional Hybrid Loss

To better align with evaluation metric:

```python
Loss = 0.7 * NLL + 0.3 * CRPS
```

CRPS computed using Gaussian closed-form.

---

# 10. Evaluation

Primary metric: CRPS

Leaderboard score:

RMS across:

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

```python
GroupKFold(n_splits=5)
group = domainID
```

Purpose:

- Prevent domain leakage  
- Simulate novel eco-domain scenario  

⚠️ Consider removing domain embedding entirely for final out-of-domain evaluation.

---

# 12. Sigma Calibration (Critical)

After training:

On validation set:

```python
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

Metadata-Only Baseline

Per event features:

- Species frequency counts  
- Species diversity (richness, Shannon)  
- Number of specimens  
- Domain ID (if allowed)  

Train Gaussian LightGBM regressors or MLPs.

This establishes whether image features add ecological signal.

---

# 14. Ensembling

Train:

- 5-fold models  
- Multiple random seeds  

Ensemble means and variances correctly:

```python
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

- Remove or minimize domain embedding  
- Use regularization:
  - Dropout  
  - Weight decay  
  - Early stopping  
- Apply strong data augmentation:
  - Color jitter  
  - Random brightness  
  - Random crops / resized crops  

Avoid memorizing domain-specific drought baselines.

Consider attention pooling and/or domain-adversarial training.

---

# 16. Image Handling Notes

- Use colorpicker for white balance / histogram normalization  
- Scalebar may be ignored unless numeric scale can be reliably extracted  
- Background removal or approximate segmentation recommended to reduce imaging artifacts  
- Data augmentation is critical due to inconsistent imaging  

---

# 17. Submission & Inference Requirements

⚠️ **The submitted model is NOT trained on CodaBench. It is only loaded and evaluated.**  
All training must happen locally. The submission contains only pretrained weights and inference code.

## 17.1 Zip File Contents

```
model.py
requirements.txt
weights/           # all saved model weights / checkpoints
```

## 17.2 `model.py` Interface

```python
class Model:

    def load(self):
        """
        Called once at startup.
        Load all model weights, normalization stats, species mappings,
        and calibration parameters from the weights/ directory.
        """
        ...

    def predict(self, list_of_dicts):
        """
        Called once per event.
        Input: List[dict] — one dict per specimen in the event.
        Output: dict with Gaussian predictions for each SPEI target.
        """
        ...
```

## 17.3 Input Format (what `predict()` receives)

Each dict in the list contains:

| Key | Type | Description |
|---|---|---|
| `relative_img` | PIL.Image | Image of the beetle |
| `colorpicker_img` | PIL.Image | Color calibration card |
| `scalebar_img` | PIL.Image | Scalebar image |
| `scientificName` | str | Scientific name of the beetle |
| `domainID` | int | Anonymized NEON eco-domain ID |

The list contains all specimens from a **single collection event**.

## 17.4 Output Format (what `predict()` must return)

```python
{
    "SPEI_30d": {"mu": float, "sigma": float},
    "SPEI_1y":  {"mu": float, "sigma": float},
    "SPEI_2y":  {"mu": float, "sigma": float},
}
```

## 17.5 What Must Be Saved to `weights/`

During local training, save everything needed for inference:

- Model state dicts (e.g. `torch.save(model.state_dict(), "weights/model.pt")`)  
- Target normalization stats (`mean_train`, `std_train` per SPEI target)  
- Species-to-index mapping (so `scientificName` can be encoded at inference)  
- Sigma calibration multipliers (α per target, from Section 12)  
- If ensembling: all fold checkpoints (e.g. `weights/fold_0.pt` ... `weights/fold_4.pt`)  

Example save pattern:

```python
import torch, json

torch.save(model.state_dict(), "weights/model.pt")
json.dump({
    "species_to_idx": species_to_idx,
    "target_means": [mean_30d, mean_1y, mean_2y],
    "target_stds": [std_30d, std_1y, std_2y],
    "sigma_calibration": [alpha_30d, alpha_1y, alpha_2y],
}, open("weights/config.json", "w"))
```

## 17.6 `load()` Implementation Pattern

```python
def load(self):
    config = json.load(open("weights/config.json"))
    self.species_to_idx = config["species_to_idx"]
    self.target_means = config["target_means"]
    self.target_stds = config["target_stds"]
    self.sigma_cal = config["sigma_calibration"]

    self.model = MyNetwork(...)
    self.model.load_state_dict(torch.load("weights/model.pt", map_location="cpu"))
    self.model.eval()
```

## 17.7 `predict()` Implementation Pattern

```python
def predict(self, list_of_dicts):
    # 1. Process each specimen
    # 2. Generate embeddings
    # 3. Aggregate to event-level (mean pooling / attention)
    # 4. Forward pass → mu_norm, sigma_norm
    # 5. Denormalize: mu_real = mu_norm * std + mean
    # 6. Denormalize: sigma_real = sigma_norm * std
    # 7. Apply calibration: sigma_real *= alpha
    # 8. Return formatted dict
    return {
        "SPEI_30d": {"mu": mu_30d, "sigma": sigma_30d},
        "SPEI_1y":  {"mu": mu_1y,  "sigma": sigma_1y},
        "SPEI_2y":  {"mu": mu_2y,  "sigma": sigma_2y},
    }
```

---

# 18. Final Objective

Minimize:

Average CRPS (RMS aggregated across categories)

Primary goals:

- Accurate mean (μ)  
- Well-calibrated uncertainty (σ)  
- Strong domain generalization  
- Robust ensembling  