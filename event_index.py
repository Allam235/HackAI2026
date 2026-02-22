from datasets import load_dataset
import pandas as pd

def build_event_index(split="train"):
    ds = load_dataset("imageomics/sentinel-beetles")[split]
    df = ds.to_pandas()

    # Keep only what you need for training
    keep = [
        "eventID", "domainID", "scientificName", "collectDate",
        "file_path", "SPEI_30d", "SPEI_1y", "SPEI_2y"
    ]
    df = df[keep]

    # Group rows by event
    grouped = df.groupby("eventID", sort=False)
    event_ids = list(grouped.groups.keys())
    return df, grouped, event_ids

if __name__ == "__main__":
    df, grouped, event_ids = build_event_index("train")
    print("rows:", df.shape[0], "events:", len(event_ids))
    first = event_ids[0]
    print("first event size:", len(grouped.get_group(first)))
    print("targets:", grouped.get_group(first)[["SPEI_30d","SPEI_1y","SPEI_2y"]].iloc[0].tolist())