import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import torch
    print("Torch imported successfully with CUDA disabled!")
except Exception as e:
    print("Failed to import torch:", e)
