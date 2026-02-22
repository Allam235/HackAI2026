import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import random

from torch.optim import AdamW
from torchvision import models, transforms
from datasets import load_dataset


TARGETS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


class EventModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep pretrained weights for better performance
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # -> (N,512,1,1)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # mu3 + log_sigma3
        )

    def forward(self, imgs):
        feats = self.encoder(imgs).squeeze(-1).squeeze(-1)  # (N,512)
        event_feat = feats.mean(dim=0, keepdim=True)        # (1,512)
        out = self.head(event_feat)                         # (1,6)
        mu = out[:, 0:3]
        log_sigma = out[:, 3:6]
        sigma = torch.exp(log_sigma).clamp(1e-3, 10.0)
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return (0.5 * torch.log(sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)).mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = load_dataset("imageomics/sentinel-beetles")["train"]

    # Build event -> dataset indices mapping (safe; no pandas)
    event_to_indices = defaultdict(list)
    for i in range(len(ds)):
        event_to_indices[int(ds[i]["eventID"])].append(i)

    event_ids = list(event_to_indices.keys())
    print("train events:", len(event_ids))

    # Event-level target stats (one label per event from first specimen)
    y_mat = []
    for eid in event_ids:
        first_row = ds[event_to_indices[eid][0]]
        y_mat.append([first_row[t] for t in TARGETS])
    y_mat = np.array(y_mat, dtype=np.float32)
    y_mean = y_mat.mean(axis=0)
    y_std = y_mat.std(axis=0) + 1e-6

    # Create tensors once (speed)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

    Path("baselines").mkdir(exist_ok=True)
    with open(Path("baselines") / "norm.json", "w") as f:
        json.dump({"targets": TARGETS, "mean": y_mean.tolist(), "std": y_std.tolist()}, f, indent=2)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = EventModel().to(device)
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    model.train()
    epochs = 5
    max_specimens = 24  # CPU-friendly cap

    for ep in range(1, epochs + 1):
        random.shuffle(event_ids)
        total = 0.0
        for step, eid in enumerate(event_ids, start=1):
            idxs = event_to_indices[eid]

            imgs = []
            cap = min(max_specimens, len(idxs))  # dynamic cap per event
            for idx in idxs[:cap]:
                row = ds[idx]
                img = row["file_path"]  # already a PIL image in this dataset
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")
                imgs.append(tfm(img))

            imgs = torch.stack(imgs, dim=0).to(device)

            first_row = ds[idxs[0]]
            y = torch.tensor([[first_row[t] for t in TARGETS]], dtype=torch.float32, device=device)
            y = (y - y_mean_t) / y_std_t

            mu, sigma = model(imgs)
            loss = gaussian_nll(y, mu, sigma)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{len(event_ids)} loss {total/step:.4f}")

        print(f"epoch {ep} avg loss {total/len(event_ids):.4f}")

    torch.save(model.state_dict(), Path("baselines") / "weights.pt")
    print("saved baselines/weights.pt and baselines/norm.json")


if __name__ == "__main__":
    main()