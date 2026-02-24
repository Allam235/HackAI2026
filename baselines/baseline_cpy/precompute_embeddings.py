"""
One-time script: extract frozen BioCLIP2 embeddings for every specimen.

Saves one .pt file per split (train_embeddings.pt, validation_embeddings.pt).
After this, training becomes a fast tabular problem.

GPU-optimized for NVIDIA RTX 2500 (AMP, cudnn.benchmark).

Usage:
    python precompute_embeddings.py              # default batch 64
    python precompute_embeddings.py --batch 16   # smaller batch for low VRAM
"""

import argparse
import logging
import time

import numpy as np
import torch
from open_clip import create_model_and_transforms
from datasets import load_dataset

from model import ensure_rgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_COLS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
METADATA_COLS = ["eventID", "domainID", "scientificName"] + TARGET_COLS
EMBEDDING_DIM = 768  # BioCLIP 2 (ViT-L/14)
LOG_EVERY = 1000


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


# ------------------------------------------------------------------
# Backbone
# ------------------------------------------------------------------

def build_backbone(device):
    bioclip, _, preprocess = create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2",
        output_dict=True,
        require_pretrained=True,
    )
    bioclip.to(device).eval()
    log.info("Backbone: BioCLIP 2 (ViT-L/14)  →  %d-dim  (device=%s)",
             EMBEDDING_DIM, device)
    return bioclip, preprocess


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------

def extract_metadata(hf_split):
    return {col: hf_split[col] for col in METADATA_COLS}


def build_species_mapping(names):
    return {name: idx + 1 for idx, name in enumerate(sorted(set(names)))}


def compute_target_stats(meta):
    means, stds = [], []
    for col in TARGET_COLS:
        v = np.array(meta[col], dtype=np.float64)
        means.append(float(v.mean()))
        stds.append(float(v.std()))
    return means, stds


# ------------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------------

def precompute_split(hf_split, backbone, preprocess, species_to_idx,
                     device, batch_size, split_name, use_amp=False):
    total = len(hf_split)
    meta = extract_metadata(hf_split)

    all_beetle, all_color, all_scale = [], [], []
    all_species, all_domain, all_event, all_targets = [], [], [], []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        rows = hf_split[start:end]
        B = end - start

        beetle_t = torch.stack([preprocess(ensure_rgb(img)) for img in rows["file_path"]])
        color_t = torch.stack([preprocess(ensure_rgb(img)) for img in rows["colorpicker_full_path"]])
        scale_t = torch.stack([preprocess(ensure_rgb(img)) for img in rows["scalebar_full_path"]])

        all_imgs = torch.cat([beetle_t, color_t, scale_t], dim=0).to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            all_emb = backbone(all_imgs)["image_features"].cpu().half()
        b_emb, c_emb, s_emb = all_emb.split(B)

        all_beetle.append(b_emb)
        all_color.append(c_emb)
        all_scale.append(s_emb)

        all_species.extend(meta["scientificName"][start:end])
        all_domain.extend(meta["domainID"][start:end])
        all_event.extend(meta["eventID"][start:end])

        targets = torch.tensor(
            list(zip(
                meta["SPEI_30d"][start:end],
                meta["SPEI_1y"][start:end],
                meta["SPEI_2y"][start:end],
            )),
            dtype=torch.float32,
        )
        all_targets.append(targets)

        if end % LOG_EVERY < batch_size or end == total:
            log.info("[%s]  %d / %d specimens", split_name, end, total)

    species_idx = torch.tensor(
        [species_to_idx.get(s, 0) for s in all_species], dtype=torch.long,
    )

    return {
        "beetle_emb": torch.cat(all_beetle),
        "colorpicker_emb": torch.cat(all_color),
        "scalebar_emb": torch.cat(all_scale),
        "species_idx": species_idx,
        "domain_id": torch.tensor(all_domain, dtype=torch.long),
        "event_id": torch.tensor(all_event, dtype=torch.long),
        "targets": torch.cat(all_targets),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = require_gpu()
    use_amp = True
    torch.backends.cudnn.benchmark = True
    log.info("CUDA optimizations: cudnn.benchmark=True, AMP=True")

    # --- Dataset ---
    t0 = time.perf_counter()
    log.info("Loading HuggingFace dataset …")
    ds = load_dataset("imageomics/sentinel-beetles")
    log.info("Loaded in %.1fs  (train=%d  val=%d)",
             time.perf_counter() - t0, len(ds["train"]), len(ds["validation"]))

    # --- Metadata (one pass) ---
    train_meta = extract_metadata(ds["train"])
    species_to_idx = build_species_mapping(train_meta["scientificName"])
    target_means, target_stds = compute_target_stats(train_meta)
    num_species = len(species_to_idx) + 1
    log.info("Species: %d known + 1 unknown = %d", len(species_to_idx), num_species)
    log.info("Target means: %s", target_means)
    log.info("Target stds:  %s", target_stds)

    # --- Backbone ---
    backbone, preprocess = build_backbone(device)

    # --- Precompute each split ---
    for split_name in ["train", "validation"]:
        t_split = time.perf_counter()
        log.info("===== Precomputing [%s] =====", split_name)

        data = precompute_split(
            ds[split_name], backbone, preprocess,
            species_to_idx, device, args.batch, split_name,
            use_amp=use_amp,
        )

        data["species_to_idx"] = species_to_idx
        data["num_species"] = num_species
        data["target_means"] = target_means
        data["target_stds"] = target_stds
        data["backbone"] = "bioclip2"
        data["embedding_dim"] = EMBEDDING_DIM

        out_path = f"{split_name}_embeddings.pt"
        torch.save(data, out_path)
        mb = sum(v.nbytes for v in data.values() if isinstance(v, torch.Tensor)) / 1e6
        log.info("[%s] saved → %s  (%.1f MB tensors, %.1fs)",
                 split_name, out_path, mb, time.perf_counter() - t_split)

    log.info("Done. Total time: %.1fs", time.perf_counter() - t0)
