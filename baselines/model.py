import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.cuda.amp import autocast

TARGETS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


class _EventModel(nn.Module):
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
        feats = self.encoder(imgs).squeeze(-1).squeeze(-1)
        event_feat = feats.mean(dim=0, keepdim=True)
        out = self.head(event_feat)
        mu = out[:, 0:3]
        log_sigma = out[:, 3:6]
        sigma = torch.exp(log_sigma).clamp(1e-3, 10.0)
        return mu, sigma


class Model:
    def load(self):
        self.means = {"SPEI_30d": -0.1, "SPEI_1y": -0.05, "SPEI_2y": 0.02}
        self.stds  = {"SPEI_30d":  1.2, "SPEI_1y":  1.1,  "SPEI_2y": 1.0}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.USE_TTA = False
        self.max_specimens = 12

        self.alpha_per = {k: 1.0 for k in TARGETS}
        self.alpha_vec = None

        # --- NEW: domain prior ---
        self.prior = None
        self.use_prior_fusion = True   # set False to disable
        self.prior_strength = 1.0      # >1 = trust model more, <1 = trust prior more

        weights_path = Path(__file__).parent / "weights.pt"
        norm_path = Path(__file__).parent / "norm.json"
        prior_path = Path(__file__).parent / "prior.json"

        self.use_trained = False
        if weights_path.exists() and norm_path.exists():
            try:
                norm = json.loads(norm_path.read_text())
                self.targets = norm.get("targets", TARGETS)
                self.y_mean = torch.tensor(norm["mean"], dtype=torch.float32, device=self.device)
                self.y_std  = torch.tensor(norm["std"], dtype=torch.float32, device=self.device)

                self.net = _EventModel().to(self.device)
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                self.net.eval()

                cal_path = Path(__file__).parent / "calibration.json"
                if cal_path.exists():
                    cal = json.loads(cal_path.read_text())
                    self.alpha_per.update({k: float(v) for k, v in cal.get("alpha_per_target", {}).items()})
                    print(f"Loaded calibration from {cal_path}: {self.alpha_per}")
                else:
                    print("No calibration.json found; using alpha=1.0")

                self.alpha_vec = torch.tensor(
                    [self.alpha_per.get(k, 1.0) for k in TARGETS],
                    dtype=torch.float32,
                    device=self.device
                )

                # --- NEW: load prior.json if present ---
                if prior_path.exists():
                    self.prior = json.loads(prior_path.read_text())
                    print(f"Loaded prior from {prior_path} (domain priors={len(self.prior.get('by_domain', {}))})")
                else:
                    print("No prior.json found; prior fusion disabled.")
                    self.use_prior_fusion = False

                self.use_trained = True
                print(f"[DEBUG] max_specimens={self.max_specimens} USE_TTA={self.USE_TTA} alpha={self.alpha_per} prior_fusion={self.use_prior_fusion}")
                print(f"Loaded weights from {weights_path} on {self.device}")
                print(f"model running on device: {self.device}")

            except Exception as e:
                print(f"Warning: failed to load trained weights, using baseline. Error: {e}")

    def _select_specimens(self, event):
        if len(event) <= self.max_specimens:
            return event

        scored = []
        for idx, specimen in enumerate(event):
            img = specimen["relative_img"]
            w, h = img.size
            scored.append((w * h, idx))
        scored.sort(reverse=True)

        pool_size = min(len(scored), max(self.max_specimens * 4, self.max_specimens))
        pool = [event[i] for (_, i) in scored[:pool_size]]

        step = max(1, pool_size // self.max_specimens)
        chosen = pool[0:pool_size:step][:self.max_specimens]

        if len(chosen) < self.max_specimens:
            need = self.max_specimens - len(chosen)
            chosen += pool[:need]

        return chosen[:self.max_specimens]

    def _get_domain_prior(self, domain_id: int):
        """
        Returns (mu_prior, sigma_prior) tensors on device, shape (3,)
        """
        if not self.prior:
            return None

        by_domain = self.prior.get("by_domain", {})
        key = str(int(domain_id))
        if key in by_domain:
            mu = by_domain[key]["mu"]
            std = by_domain[key]["std"]
        else:
            mu = self.prior["global"]["mu"]
            std = self.prior["global"]["std"]

        mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
        sd_t = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-3)
        return mu_t, sd_t

    @staticmethod
    def _fuse_gaussians(mu_a, sig_a, mu_b, sig_b):
        """
        Precision-weighted Gaussian fusion:
        sigma^2 = 1 / (1/sig_a^2 + 1/sig_b^2)
        mu = sigma^2 * (mu_a/sig_a^2 + mu_b/sig_b^2)
        """
        var_a = (sig_a ** 2).clamp_min(1e-6)
        var_b = (sig_b ** 2).clamp_min(1e-6)
        prec_a = 1.0 / var_a
        prec_b = 1.0 / var_b
        var = 1.0 / (prec_a + prec_b)
        mu = var * (mu_a * prec_a + mu_b * prec_b)
        sig = torch.sqrt(var.clamp_min(1e-6))
        return mu, sig

    @torch.inference_mode()
    def predict(self, event):
        if not self.use_trained:
            return {
                "SPEI_30d": {"mu": float(self.means["SPEI_30d"]), "sigma": float(self.stds["SPEI_30d"])},
                "SPEI_1y":  {"mu": float(self.means["SPEI_1y"]),  "sigma": float(self.stds["SPEI_1y"])},
                "SPEI_2y":  {"mu": float(self.means["SPEI_2y"]),  "sigma": float(self.stds["SPEI_2y"])},
            }

        specimens = self._select_specimens(event)

        imgs = []
        for specimen in specimens:
            img = specimen["relative_img"].convert("RGB")
            imgs.append(self.tfm(img))
        imgs = torch.stack(imgs, dim=0).to(self.device)

        if self.device.type == "cuda":
            with autocast():
                mu_norm, sigma_norm = self.net(imgs)
        else:
            mu_norm, sigma_norm = self.net(imgs)

        mu = (mu_norm.squeeze(0) * self.y_std) + self.y_mean
        sigma = (sigma_norm.squeeze(0) * self.y_std).abs()

        if self.alpha_vec is not None:
            sigma = sigma * self.alpha_vec

        # --- NEW: domain prior fusion ---
        if self.use_prior_fusion and len(event) > 0:
            dom = int(event[0].get("domainID", -1))
            prior = self._get_domain_prior(dom)
            if prior is not None:
                mu_p, sd_p = prior

                # Adjust prior_strength: >1 means trust model more (inflate prior sigma)
                sd_p = sd_p * float(self.prior_strength)

                mu, sigma = self._fuse_gaussians(mu, sigma, mu_p, sd_p)

        out = {}
        for i, key in enumerate(TARGETS):
            out[key] = {"mu": float(mu[i].item()), "sigma": float(sigma[i].item())}
        return out