"""
Full training pipeline using precomputed backbone embeddings.

GPU-optimized for NVIDIA RTX 2500 (AMP, torch.compile, cudnn.benchmark).

Prerequisites:
    python precompute_embeddings.py   ->   train_embeddings.pt, validation_embeddings.pt

Usage:
    python train.py
    python train.py --epochs 100 --lr 5e-4 --batch 64
"""

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from model import BeetleHead, EMBEDDING_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_NAMES = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
LOG_EVERY = 5000


def require_gpu():
    """Verify CUDA GPU is available and print diagnostics. Abort if not."""
    if not torch.cuda.is_available():
        log.error("=" * 60)
        log.error("  CUDA IS NOT AVAILABLE")
        log.error("  This script requires an NVIDIA GPU (RTX 2050).")
        log.error("  Running on CPU would take far too long.")
        log.error("")
        log.error("  Troubleshooting:")
        log.error("    1. Check nvidia-smi runs in your terminal")
        log.error("    2. Reinstall PyTorch with CUDA support:")
        log.error("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
        log.error("    3. Make sure your NVIDIA drivers are up to date")
        log.error("=" * 60)
        raise SystemExit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    cuda_ver = torch.version.cuda
    cap = torch.cuda.get_device_capability(0)

    log.info("=" * 60)
    log.info("  GPU CHECK PASSED")
    log.info("  Device:      %s", gpu_name)
    log.info("  VRAM:        %.1f GB", vram_gb)
    log.info("  CUDA:        %s", cuda_ver)
    log.info("  Compute cap: %d.%d", cap[0], cap[1])
    log.info("  PyTorch:     %s", torch.__version__)
    log.info("=" * 60)

    return torch.device("cuda")


# ===================================================================
# Dataset
# ===================================================================

class PrecomputedEventDataset(Dataset):
    """Loads a .pt file and yields one event per __getitem__."""

    def __init__(self, pt_path, target_means=None, target_stds=None):
        t0 = time.perf_counter()
        log.info("Loading %s …", pt_path)
        data = torch.load(pt_path, weights_only=False)

        self.beetle_emb = data["beetle_emb"]
        self.colorpicker_emb = data["colorpicker_emb"]
        self.scalebar_emb = data["scalebar_emb"]
        self.species_idx = data["species_idx"]
        self.domain_id = data["domain_id"]
        self.event_id = data["event_id"]
        self.targets = data["targets"]

        self.embedding_dim = data.get("embedding_dim", self.beetle_emb.shape[1])
        self.num_species = data.get("num_species", 145)
        self.species_to_idx = data.get("species_to_idx", {})
        self.backbone_name = data.get("backbone", "efficientnet_b0")

        tm = target_means or data.get("target_means", [0.0, 0.0, 0.0])
        ts = target_stds or data.get("target_stds", [1.0, 1.0, 1.0])
        self.target_means = torch.tensor(tm, dtype=torch.float32)
        self.target_stds = torch.tensor(ts, dtype=torch.float32)

        self.event_groups = defaultdict(list)
        for idx in range(len(self.event_id)):
            self.event_groups[int(self.event_id[idx])].append(idx)
        self.event_ids = sorted(self.event_groups.keys())

        self.total_rows = len(self.event_id)
        self._rows_processed = 0

        log.info("  %d specimens, %d events, %d-dim, loaded in %.2fs",
                 self.total_rows, len(self.event_ids),
                 self.embedding_dim, time.perf_counter() - t0)

    def __len__(self):
        return len(self.event_ids)

    def reset_progress(self):
        self._rows_processed = 0

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        indices = self.event_groups[event_id]

        beetle = self.beetle_emb[indices].float()
        colorpicker = self.colorpicker_emb[indices].float()
        scalebar = self.scalebar_emb[indices].float()
        species = self.species_idx[indices]
        domain = self.domain_id[indices]

        raw_targets = self.targets[indices[0]]
        targets = (raw_targets - self.target_means) / self.target_stds

        prev = self._rows_processed
        self._rows_processed += len(indices)
        if prev // LOG_EVERY != self._rows_processed // LOG_EVERY:
            log.info("  progress: %d / %d rows", self._rows_processed, self.total_rows)

        return {
            "beetle_emb": beetle,
            "colorpicker_emb": colorpicker,
            "scalebar_emb": scalebar,
            "species_idx": species,
            "domain_id": domain,
            "num_specimens": len(indices),
        }, targets, event_id, int(domain[0])


def collate_precomputed(batch):
    max_n = max(item[0]["num_specimens"] for item in batch)
    B = len(batch)
    D = batch[0][0]["beetle_emb"].shape[1]

    pad_beetle = torch.zeros(B, max_n, D)
    pad_color = torch.zeros(B, max_n, D)
    pad_scale = torch.zeros(B, max_n, D)
    pad_species = torch.zeros(B, max_n, dtype=torch.long)
    pad_domain = torch.zeros(B, max_n, dtype=torch.long)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    all_targets, all_eids, all_dids = [], [], []

    for i, (data, targets, eid, did) in enumerate(batch):
        n = data["num_specimens"]
        pad_beetle[i, :n] = data["beetle_emb"]
        pad_color[i, :n] = data["colorpicker_emb"]
        pad_scale[i, :n] = data["scalebar_emb"]
        pad_species[i, :n] = data["species_idx"]
        pad_domain[i, :n] = data["domain_id"]
        mask[i, :n] = True
        all_targets.append(targets)
        all_eids.append(eid)
        all_dids.append(did)

    return {
        "beetle_emb": pad_beetle,
        "colorpicker_emb": pad_color,
        "scalebar_emb": pad_scale,
        "species_idx": pad_species,
        "domain_id": pad_domain,
        "mask": mask,
        "targets": torch.stack(all_targets),
        "event_ids": all_eids,
        "domain_ids": all_dids,
    }


# ===================================================================
# Loss functions  (AGENTS.md Section 9)
# ===================================================================

def gaussian_nll(mu, sigma, y):
    """Per-sample Gaussian NLL:  0.5·log(σ²) + (y−μ)² / (2σ²)"""
    var = sigma ** 2
    nll = 0.5 * torch.log(var) + (y - mu) ** 2 / (2 * var)
    return nll  # [B, 3]


def gaussian_crps(mu, sigma, y):
    """Closed-form CRPS for Gaussian: σ·[z·(2Φ(z)−1) + 2φ(z) − 1/√π]"""
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))
    return crps  # [B, 3]


def combined_loss(mu, sigma, y, nll_weight=0.7):
    """Hybrid loss: NLL + CRPS  (Section 9.2)"""
    nll = gaussian_nll(mu, sigma, y).mean()
    crps = gaussian_crps(mu, sigma, y).mean()
    return nll_weight * nll + (1 - nll_weight) * crps, nll, crps


# ===================================================================
# Training & validation steps
# ===================================================================

def train_one_epoch(head, loader, optimizer, device, scaler=None, use_amp=False):
    head.train()
    loader.dataset.reset_progress()
    total_loss, total_nll, total_crps, n_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        beetle = batch["beetle_emb"].to(device, non_blocking=True)
        color = batch["colorpicker_emb"].to(device, non_blocking=True)
        scale = batch["scalebar_emb"].to(device, non_blocking=True)
        species = batch["species_idx"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            mu, sigma = head(beetle, color, scale, species, mask)
            loss, nll, crps = combined_loss(mu, sigma, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        total_nll += nll.item()
        total_crps += crps.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "nll": total_nll / n_batches,
        "crps": total_crps / n_batches,
    }


@torch.no_grad()
def validate(head, loader, device, target_stds, use_amp=False):
    """Compute loss + per-target CRPS in original scale."""
    head.eval()
    loader.dataset.reset_progress()
    total_loss, total_nll, total_crps, n_batches = 0.0, 0.0, 0.0, 0

    crps_sums = torch.zeros(3, device=device)
    n_events = 0

    ts = torch.tensor(target_stds, device=device)

    for batch in loader:
        beetle = batch["beetle_emb"].to(device, non_blocking=True)
        color = batch["colorpicker_emb"].to(device, non_blocking=True)
        scale = batch["scalebar_emb"].to(device, non_blocking=True)
        species = batch["species_idx"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            mu, sigma = head(beetle, color, scale, species, mask)
            loss, nll, crps = combined_loss(mu, sigma, targets)

        total_loss += loss.item()
        total_nll += nll.item()
        total_crps += crps.item()
        n_batches += 1

        mu_real = mu.float() * ts
        sigma_real = sigma.float() * ts
        y_real = targets.float() * ts
        crps_orig = gaussian_crps(mu_real, sigma_real, y_real)  # [B, 3]
        crps_sums += crps_orig.sum(dim=0)
        n_events += targets.shape[0]

    per_target_crps = (crps_sums / n_events).cpu()  # [3]
    rms_crps = float(torch.sqrt((per_target_crps ** 2).mean()))

    return {
        "loss": total_loss / n_batches,
        "nll": total_nll / n_batches,
        "crps": total_crps / n_batches,
        "crps_30d": float(per_target_crps[0]),
        "crps_1y": float(per_target_crps[1]),
        "crps_2y": float(per_target_crps[2]),
        "rms_crps": rms_crps,
    }


# ===================================================================
# Sigma calibration  (AGENTS.md Section 12)
# ===================================================================

@torch.no_grad()
def calibrate_sigma(head, loader, device, target_stds, use_amp=False):
    """Grid-search alpha in [0.5, 2.0] per target to minimize CRPS."""
    head.eval()
    loader.dataset.reset_progress()

    ts = torch.tensor(target_stds, device=device)
    all_mu, all_sigma, all_y = [], [], []

    for batch in loader:
        beetle = batch["beetle_emb"].to(device, non_blocking=True)
        color = batch["colorpicker_emb"].to(device, non_blocking=True)
        scale = batch["scalebar_emb"].to(device, non_blocking=True)
        species = batch["species_idx"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            mu, sigma = head(beetle, color, scale, species, mask)
        all_mu.append(mu.float() * ts)
        all_sigma.append(sigma.float() * ts)
        all_y.append(targets.float() * ts)

    mu_cat = torch.cat(all_mu)       # [N_events, 3]
    sigma_cat = torch.cat(all_sigma)
    y_cat = torch.cat(all_y)

    alphas = torch.linspace(0.5, 2.0, 61)
    best_alpha = [1.0, 1.0, 1.0]

    for t in range(3):
        best_crps = float("inf")
        for a in alphas:
            s_cal = sigma_cat[:, t] * float(a)
            c = gaussian_crps(mu_cat[:, t], s_cal, y_cat[:, t]).mean().item()
            if c < best_crps:
                best_crps = c
                best_alpha[t] = round(float(a), 3)
        log.info("  %s  α=%.3f  CRPS=%.4f", TARGET_NAMES[t], best_alpha[t], best_crps)

    return best_alpha


# ===================================================================
# Checkpoint helpers
# ===================================================================

def save_checkpoint(head, config, out_dir="weights"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log.info("Checkpoint saved → %s/", out_dir)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_pt", default="train_embeddings.pt")
    parser.add_argument("--val_pt", default="validation_embeddings.pt")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    device = require_gpu()
    use_amp = True
    torch.backends.cudnn.benchmark = True
    log.info("CUDA optimizations: cudnn.benchmark=True, AMP=True")

    # --- Data ---
    train_ds = PrecomputedEventDataset(args.train_pt)
    val_ds = PrecomputedEventDataset(args.val_pt)

    target_means = train_ds.target_means.tolist()
    target_stds = train_ds.target_stds.tolist()
    species_to_idx = train_ds.species_to_idx
    num_species = train_ds.num_species
    backbone_dim = train_ds.embedding_dim
    backbone_name = train_ds.backbone_name

    log.info("Species: %d  |  Backbone: %s (%d-dim)", num_species, backbone_name, backbone_dim)
    log.info("Target means: %s", target_means)
    log.info("Target stds:  %s", target_stds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_precomputed, num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_precomputed, num_workers=0,
        pin_memory=True,
    )
    log.info("Train: %d events (%d batches)  |  Val: %d events (%d batches)",
             len(train_ds), len(train_loader), len(val_ds), len(val_loader))

    # --- Model ---
    head = BeetleHead(
        backbone_dim=backbone_dim,
        num_species=num_species,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log.info("BeetleHead: %d trainable parameters", n_params)

    if not args.no_compile:
        try:
            head = torch.compile(head)
            log.info("torch.compile() applied to BeetleHead")
        except Exception:
            log.info("torch.compile() not available, skipping")

    scaler = GradScaler("cuda", enabled=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # --- Training loop ---
    best_rms_crps = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(head, train_loader, optimizer, device,
                                        scaler=scaler, use_amp=use_amp)
        val_metrics = validate(head, val_loader, device, target_stds,
                               use_amp=use_amp)
        scheduler.step(val_metrics["rms_crps"])

        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d  %.1fs  lr=%.1e  |  "
            "train loss=%.4f nll=%.4f crps=%.4f  |  "
            "val rms_crps=%.4f  (30d=%.4f  1y=%.4f  2y=%.4f)",
            epoch, args.epochs, elapsed, lr_now,
            train_metrics["loss"], train_metrics["nll"], train_metrics["crps"],
            val_metrics["rms_crps"],
            val_metrics["crps_30d"], val_metrics["crps_1y"], val_metrics["crps_2y"],
        )

        if val_metrics["rms_crps"] < best_rms_crps:
            best_rms_crps = val_metrics["rms_crps"]
            epochs_no_improve = 0
            # Save best head (calibration comes after training)
            save_checkpoint(head, {
                "species_to_idx": species_to_idx,
                "num_species": num_species,
                "target_means": target_means,
                "target_stds": target_stds,
                "sigma_calibration": [1.0, 1.0, 1.0],
                "backbone_dim": backbone_dim,
                "backbone": backbone_name,
                "best_rms_crps": best_rms_crps,
            })
            log.info("  ↑ new best rms_crps = %.4f", best_rms_crps)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log.info("Early stopping — no improvement for %d epochs", args.patience)
                break

    # --- Reload best checkpoint for calibration ---
    log.info("===== Sigma calibration =====")
    best_state = torch.load(os.path.join("weights", "model.pt"), map_location=device)
    head.load_state_dict(best_state)
    head.to(device)

    sigma_cal = calibrate_sigma(head, val_loader, device, target_stds,
                                use_amp=use_amp)
    log.info("Calibration alphas: %s", sigma_cal)

    # --- Re-save with calibration ---
    save_checkpoint(head, {
        "species_to_idx": species_to_idx,
        "num_species": num_species,
        "target_means": target_means,
        "target_stds": target_stds,
        "sigma_calibration": sigma_cal,
        "backbone_dim": backbone_dim,
        "backbone": backbone_name,
        "best_rms_crps": best_rms_crps,
    })

    # --- Final validation with calibration ---
    log.info("===== Final validation (with calibration) =====")
    # Quick re-compute CRPS with calibrated sigmas
    head.eval()
    val_ds.reset_progress()
    ts = torch.tensor(target_stds, device=device)
    sc = torch.tensor(sigma_cal, device=device)
    crps_sums = torch.zeros(3, device=device)
    n_events = 0

    with torch.no_grad():
        for batch in val_loader:
            beetle = batch["beetle_emb"].to(device, non_blocking=True)
            color = batch["colorpicker_emb"].to(device, non_blocking=True)
            scale = batch["scalebar_emb"].to(device, non_blocking=True)
            species = batch["species_idx"].to(device, non_blocking=True)
            mask_b = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                mu, sigma = head(beetle, color, scale, species, mask_b)
            mu_real = mu.float() * ts
            sigma_real = sigma.float() * ts * sc
            y_real = targets.float() * ts
            crps_orig = gaussian_crps(mu_real, sigma_real, y_real)
            crps_sums += crps_orig.sum(dim=0)
            n_events += targets.shape[0]

    per_target = (crps_sums / n_events).cpu()
    rms_final = float(torch.sqrt((per_target ** 2).mean()))
    log.info("Calibrated CRPS — 30d=%.4f  1y=%.4f  2y=%.4f  |  RMS=%.4f",
             float(per_target[0]), float(per_target[1]), float(per_target[2]), rms_final)
    log.info("Training complete. Submission-ready weights in weights/")
