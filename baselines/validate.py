"""
Validate the submission model against the validation set.
Loads the Model class exactly as CodaBench would, then runs predict()
on each event and computes per-target CRPS and RMS-CRPS.

Usage:
    python validate.py
"""

import logging
import math
import time
from collections import defaultdict

from datasets import load_dataset
from model import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET_NAMES = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


def gaussian_crps(mu, sigma, y):
    """Closed-form CRPS for a single Gaussian prediction."""
    z = (y - mu) / sigma
    phi_z = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / math.sqrt(math.pi))


if __name__ == "__main__":
    t0 = time.perf_counter()

    log.info("Loading model …")
    m = Model()
    m.load()

    log.info("Loading validation split …")
    ds = load_dataset("imageomics/sentinel-beetles", split="validation")
    log.info("  %d specimens", len(ds))

    # Group specimens by eventID (mimics CodaBench input)
    events = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        events[row["eventID"]].append({
            "relative_img": row["file_path"],
            "colorpicker_img": row["colorpicker_full_path"],
            "scalebar_img": row["scalebar_full_path"],
            "scientificName": row["scientificName"],
            "domainID": row["domainID"],
        })

    log.info("  %d events", len(events))

    # Get ground truth targets per event (same for all specimens in event)
    event_targets = {}
    for i in range(len(ds)):
        row = ds[i]
        eid = row["eventID"]
        if eid not in event_targets:
            event_targets[eid] = {
                "SPEI_30d": row["SPEI_30d"],
                "SPEI_1y": row["SPEI_1y"],
                "SPEI_2y": row["SPEI_2y"],
            }

    crps_sums = {t: 0.0 for t in TARGET_NAMES}
    n_events = 0

    log.info("Running predictions …")
    for idx, (eid, specimens) in enumerate(events.items()):
        pred = m.predict(specimens)
        gt = event_targets[eid]

        for t in TARGET_NAMES:
            c = gaussian_crps(pred[t]["mu"], pred[t]["sigma"], gt[t])
            crps_sums[t] += c

        n_events += 1
        if (idx + 1) % 100 == 0 or (idx + 1) == len(events):
            log.info("  %d / %d events", idx + 1, len(events))

    log.info("=" * 50)
    per_target = {}
    for t in TARGET_NAMES:
        per_target[t] = crps_sums[t] / n_events
        log.info("  %s  CRPS = %.4f", t, per_target[t])

    rms = math.sqrt(sum(v ** 2 for v in per_target.values()) / len(per_target))
    log.info("  RMS-CRPS = %.4f", rms)
    log.info("=" * 50)
    log.info("Done in %.1fs", time.perf_counter() - t0)
