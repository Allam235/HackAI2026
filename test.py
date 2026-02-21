import torch

print(torch.__version__)
print(torch.cuda.is_available())  # will be False on Mac
print(torch.backends.mps.is_available())  # True on Apple Silicon