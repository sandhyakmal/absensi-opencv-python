import numpy as np

d = np.load("data/embeddings.npz", allow_pickle=True)

print("IDs:", d.files)
for k in d.files:
    arr = d[k]
    print(f"- {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
