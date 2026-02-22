import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from datasets import load_dataset

TARGETS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

def month_from_collectdate(s: str) -> int:
    # Works for ISO-like "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
    return int(str(s)[5:7])

def main():
    ds = load_dataset("imageomics/sentinel-beetles")["train"]

    # eventID -> first row index (event-level label)
    seen = set()
    event_rows = []
    for i in range(len(ds)):
        eid = int(ds[i]["eventID"])
        if eid in seen:
            continue
        seen.add(eid)
        event_rows.append(ds[i])

    # stats helpers
    def add(stats, key, y):
        stats[key].append(y)

    by_domain = defaultdict(list)
    by_month = defaultdict(list)
    by_domain_month = defaultdict(list)

    for r in event_rows:
        dom = int(r["domainID"])
        mon = month_from_collectdate(r["collectDate"])
        y = [float(r[t]) for t in TARGETS]
        by_domain[dom].append(y)
        by_month[mon].append(y)
        by_domain_month[(dom, mon)].append(y)

    def finalize(group):
        out = {}
        for k, ys in group.items():
            arr = np.array(ys, dtype=np.float32)
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0) + 1e-6
            out[str(k)] = {"mu": mu.tolist(), "std": sd.tolist(), "n": int(arr.shape[0])}
        return out

    prior = {
        "targets": TARGETS,
        "by_domain": finalize(by_domain),
        "by_month": finalize(by_month),
        "by_domain_month": finalize(by_domain_month),
        # fallback global
        "global": {
            "mu": np.array([r[t] for t in TARGETS for r in event_rows], dtype=np.float32).reshape(-1,3).mean(axis=0).tolist()
        }
    }

    Path("baselines").mkdir(exist_ok=True)
    out_path = Path("baselines") / "prior.json"
    out_path.write_text(json.dumps(prior, indent=2))
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()