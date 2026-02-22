import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset


TARGETS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


class EventModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )

    def forward(self, imgs):
        feats = self.encoder(imgs).squeeze(-1).squeeze(-1)  # (N,512)
        event_feat = feats.mean(dim=0, keepdim=True)        # (1,512)
        out = self.head(event_feat)                         # (1,6)
        mu = out[:, 0:3]
        log_sigma = out[:, 3:6]
        sigma = torch.exp(log_sigma).clamp(1e-3, 10.0)
        return mu, sigma


def std_norm_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def std_norm_cdf(z: torch.Tensor) -> torch.Tensor:
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def gaussian_crps(y, mu, sigma):
    """
    CRPS for Normal(mu, sigma) against observation y (all tensors same shape)
    CRPS = sigma * [ z*(2Φ(z)-1) + 2φ(z) - 1/sqrt(pi) ], z=(y-mu)/sigma
    """
    z = (y - mu) / sigma
    Phi = std_norm_cdf(z)
    phi = std_norm_pdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load norm + weights
    norm_path = Path("baselines") / "norm.json"
    weights_path = Path("baselines") / "weights.pt"

    norm = json.loads(norm_path.read_text())
    y_mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    y_std  = torch.tensor(norm["std"], dtype=torch.float32, device=device)

    net = EventModel().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build validation events (eventID -> indices)
    ds = load_dataset("imageomics/sentinel-beetles")["validation"]
    event_to_indices = defaultdict(list)
    for i in range(len(ds)):
        event_to_indices[int(ds[i]["eventID"])].append(i)

    event_ids = list(event_to_indices.keys())
    print("val events:", len(event_ids))

    # Cache predictions per event once (mu, sigma, y) in REAL scale
    max_specimens = 24
    mus = []
    sigmas = []
    ys = []

    for eid in event_ids:
        idxs = event_to_indices[eid]

        imgs = []
        for idx in idxs[:max_specimens]:
            img = ds[idx]["file_path"].convert("RGB")
            imgs.append(tfm(img))
        imgs = torch.stack(imgs).to(device)

        mu_n, sigma_n = net(imgs)  # normalized
        mu_n = mu_n.squeeze(0)
        sigma_n = sigma_n.squeeze(0)

        # unnormalize
        mu = mu_n * y_std + y_mean
        sigma = (sigma_n * y_std).abs().clamp(1e-3, 50.0)

        first = ds[idxs[0]]
        y = torch.tensor([first[t] for t in TARGETS], dtype=torch.float32, device=device)

        mus.append(mu)
        sigmas.append(sigma)
        ys.append(y)

    mus = torch.stack(mus)       # (E,3)
    sigmas = torch.stack(sigmas) # (E,3)
    ys = torch.stack(ys)         # (E,3)

    # Grid search alpha (single alpha for all targets)
    alphas = torch.linspace(0.7, 1.5, steps=33)
    best = None

    for a in alphas:
        crps = gaussian_crps(ys, mus, sigmas * a).mean().item()
        if best is None or crps < best[1]:
            best = (float(a.item()), crps)

    print("BEST alpha (single):", best[0], "CRPS:", best[1])

    # Optional: per-target alphas
    best_t = []
    for j, name in enumerate(TARGETS):
        bestj = None
        for a in alphas:
            crpsj = gaussian_crps(ys[:, j], mus[:, j], sigmas[:, j] * a).mean().item()
            if bestj is None or crpsj < bestj[1]:
                bestj = (float(a.item()), crpsj)
        best_t.append(bestj)
        print(f"BEST alpha for {name}: {bestj[0]}  CRPS:{bestj[1]}")

    # Save calibration results
    out = {
        "alpha_single": best[0],
        "alpha_per_target": {TARGETS[i]: best_t[i][0] for i in range(3)}
    }
    Path("baselines/calibration.json").write_text(json.dumps(out, indent=2))
    print("Saved baselines/calibration.json")


if __name__ == "__main__":
    main()