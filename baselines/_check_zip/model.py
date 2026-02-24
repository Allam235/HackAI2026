import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms


TARGETS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


class _EventModel(nn.Module):
    """
    Same architecture as training:
    ResNet18 encoder -> mean pool across specimens -> MLP -> mu3 + log_sigma3
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)  # IMPORTANT: weights loaded from weights.pt
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # (N,512,1,1)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )

    def forward(self, imgs):  # imgs: (N,3,224,224)
        feats = self.encoder(imgs).squeeze(-1).squeeze(-1)  # (N,512)
        event_feat = feats.mean(dim=0, keepdim=True)        # (1,512)
        out = self.head(event_feat)                         # (1,6)
        mu = out[:, 0:3]
        log_sigma = out[:, 3:6]
        sigma = torch.exp(log_sigma).clamp(1e-3, 10.0)
        return mu, sigma


class Model:
    def load(self):
        # Fallback baseline stats (won't crash even if weights missing)
        self.means = {"SPEI_30d": -0.1, "SPEI_1y": -0.05, "SPEI_2y": 0.02}
        self.stds  = {"SPEI_30d":  1.2, "SPEI_1y":  1.1,  "SPEI_2y": 1.0}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image preprocessing (must match training)
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Try to load trained model weights + normalization
        weights_path = Path(__file__).parent / "weights.pt"
        norm_path = Path(__file__).parent / "norm.json"

        self.use_trained = False
        if weights_path.exists() and norm_path.exists():
            try:
                with open(norm_path, "r") as f:
                    norm = json.load(f)

                # Expect same target ordering as training
                self.targets = norm.get("targets", TARGETS)
                self.y_mean = torch.tensor(norm["mean"], dtype=torch.float32, device=self.device)
                self.y_std  = torch.tensor(norm["std"], dtype=torch.float32, device=self.device)

                self.net = _EventModel().to(self.device)
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                self.net.eval()

                # --- Load sigma calibration (optional) ---
                cal_path = Path(__file__).parent / "calibration.json"
                self.alpha_per = {k: 1.0 for k in TARGETS}
                if cal_path.exists():
                    cal = json.loads(cal_path.read_text())
                    self.alpha_per.update({k: float(v) for k, v in cal.get("alpha_per_target", {}).items()})
                    print(f"Loaded calibration from {cal_path}: {self.alpha_per}")
                else:
                    print("No calibration.json found; using alpha=1.0")

                self.use_trained = True
                print(f"Loaded weights from {weights_path} on {self.device}")

            except Exception as e:
                # If anything goes wrong, stay in fallback mode
                print(f"Warning: failed to load trained weights, using baseline. Error: {e}")

    @torch.inference_mode()
    def predict(self, event):
        """
        event: list[dict] where each dict has:
          - 'relative_img' (PIL.Image)
          - 'colorpicker_img' (PIL.Image)
          - 'scalebar_img' (PIL.Image)
          - 'scientificName' (str)
          - 'domainID' (int)
        """
        if not self.use_trained:
            return {
                "SPEI_30d": {"mu": float(self.means["SPEI_30d"]), "sigma": float(self.stds["SPEI_30d"])},
                "SPEI_1y":  {"mu": float(self.means["SPEI_1y"]),  "sigma": float(self.stds["SPEI_1y"])},
                "SPEI_2y":  {"mu": float(self.means["SPEI_2y"]),  "sigma": float(self.stds["SPEI_2y"])},
            }

        # Build specimen batch (cap for speed)
        max_specimens = 24
        imgs = []
        for specimen in event[:max_specimens]:
            img = specimen["relative_img"].convert("RGB")
            imgs.append(self.tfm(img))

        imgs = torch.stack(imgs, dim=0).to(self.device)

        mu_norm, sigma_norm = self.net(imgs)  # (1,3) each

        # Unnormalize back to real SPEI scale
        mu = (mu_norm.squeeze(0) * self.y_std) + self.y_mean
        sigma = (sigma_norm.squeeze(0) * self.y_std).abs()

        # Apply per-target sigma calibration
        alpha_vec = torch.tensor(
            [self.alpha_per.get(k, 1.0) for k in TARGETS],
            dtype=torch.float32,
            device=self.device
        )
        sigma = sigma * alpha_vec

        # Map into required dict format with Python floats
        out = {}
        for i, key in enumerate(TARGETS):
            out[key] = {
                "mu": float(mu[i].item()),
                "sigma": float(sigma[i].item()),
            }
        return out