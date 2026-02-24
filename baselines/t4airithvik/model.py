import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import logging
import time
import math

log = logging.getLogger(__name__)

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EMBEDDING_DIM = 1280


def ensure_rgb(pil_img):
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


class BeetleHead(nn.Module):
    """
    Precomputed-embedding -> Gaussian SPEI predictions.

    Architecture (per AGENTS.md):
        1. Specimen encoder
           - concat(beetle, colorpicker, scalebar) -> Linear(3*D -> 512) -> ReLU
           - species Embedding(num_species, 16)
           - concat(img_proj, species_emb) -> Linear(528 -> 256) -> ReLU
        2. Event aggregation - masked mean pooling
        3. Prediction head
           - Linear(256 -> 128) -> ReLU -> Linear(128 -> 6)
           - output[:, :3] = mu,  softplus(output[:, 3:]) + eps = sigma
    """

    def __init__(self, backbone_dim=EMBEDDING_DIM, num_species=145,
                 species_emb_dim=16, specimen_dim=256, dropout=0.3):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(backbone_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.species_emb = nn.Embedding(num_species, species_emb_dim)

        self.specimen_encoder = nn.Sequential(
            nn.Linear(512 + species_emb_dim, specimen_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(specimen_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6),
        )

    def forward(self, beetle_emb, colorpicker_emb, scalebar_emb,
                species_idx, mask):
        img_cat = torch.cat([beetle_emb, colorpicker_emb, scalebar_emb], dim=-1)
        img_feat = self.img_proj(img_cat)

        sp_feat = self.species_emb(species_idx)

        spec_input = torch.cat([img_feat, sp_feat], dim=-1)
        spec_emb = self.specimen_encoder(spec_input)

        mask_f = mask.unsqueeze(-1).float()
        spec_emb = spec_emb * mask_f
        event_emb = spec_emb.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        out = self.pred_head(event_emb)

        mu = out[:, :3]
        log_sigma = out[:, 3:]
        sigma = F.softplus(log_sigma) + 1e-4
        sigma = sigma.clamp(max=10.0)

        return mu, sigma


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        self.species_to_idx = {}
        self.num_species = 0
        self.target_means = [0.0, 0.0, 0.0]
        self.target_stds = [1.0, 1.0, 1.0]
        self.sigma_cal = [1.0, 1.0, 1.0]
        self.backbone = None
        self.backbone_type = None
        self.head = None

    def load(self):
        """Load backbone, trained head, and config from weights/."""
        t0 = time.perf_counter()
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        self.species_to_idx = config["species_to_idx"]
        self.num_species = config["num_species"]
        self.target_means = config["target_means"]
        self.target_stds = config["target_stds"]
        self.sigma_cal = config["sigma_calibration"]
        backbone_dim = config.get("backbone_dim", EMBEDDING_DIM)
        self.backbone_type = config.get("backbone", "efficientnet_b0")

        if self.backbone_type == "bioclip2":
            from open_clip import create_model_and_transforms
            bioclip, _, preprocess = create_model_and_transforms(
                "hf-hub:imageomics/bioclip-2",
                output_dict=True,
                require_pretrained=True,
            )
            self.backbone = bioclip
            self.transform = preprocess
        else:
            import torchvision.transforms as T
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        self.backbone.to(self.device).eval()

        self.head = BeetleHead(
            backbone_dim=backbone_dim,
            num_species=self.num_species,
        )
        state = torch.load(
            os.path.join(base_dir, "model.pt"),
            map_location=self.device,
        )
        self.head.load_state_dict(state)
        self.head.to(self.device).eval()

        log.info("load() done in %.3fs  backbone=%s  dim=%d",
                 time.perf_counter() - t0, self.backbone_type, backbone_dim)

    def _encode_images(self, img_tensor):
        with torch.no_grad():
            if self.backbone_type == "bioclip2":
                return self.backbone(img_tensor)["image_features"]
            else:
                return self.backbone(img_tensor)

    def predict(self, list_of_dicts):
        beetle_list, color_list, scale_list, species_list = [], [], [], []

        for spec in list_of_dicts:
            imgs = torch.stack([
                self.transform(ensure_rgb(spec["relative_img"])),
                self.transform(ensure_rgb(spec["colorpicker_img"])),
                self.transform(ensure_rgb(spec["scalebar_img"])),
            ]).to(self.device)

            embs = self._encode_images(imgs)

            beetle_list.append(embs[0])
            color_list.append(embs[1])
            scale_list.append(embs[2])

            sp = self.species_to_idx.get(spec["scientificName"], 0)
            species_list.append(sp)

        N = len(list_of_dicts)
        beetle_emb = torch.stack(beetle_list).unsqueeze(0)
        color_emb = torch.stack(color_list).unsqueeze(0)
        scale_emb = torch.stack(scale_list).unsqueeze(0)
        species_idx = torch.tensor(species_list, device=self.device).unsqueeze(0)
        mask = torch.ones(1, N, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            mu_norm, sigma_norm = self.head(
                beetle_emb, color_emb, scale_emb, species_idx, mask,
            )

        tm = torch.tensor(self.target_means, device=self.device)
        ts = torch.tensor(self.target_stds, device=self.device)
        sc = torch.tensor(self.sigma_cal, device=self.device)

        mu = (mu_norm[0] * ts + tm)
        sigma = (sigma_norm[0] * ts * sc).clamp(min=1e-4)

        names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
        return {
            name: {"mu": float(mu[i]), "sigma": float(sigma[i])}
            for i, name in enumerate(names)
        }
