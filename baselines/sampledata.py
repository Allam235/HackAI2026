from datasets import load_dataset
ds = load_dataset("imageomics/sentinel-beetles")

print(ds)
print(ds["train"].features)
print(ds["train"][0])