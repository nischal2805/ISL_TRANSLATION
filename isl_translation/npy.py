import numpy as np
from pathlib import Path

sample = list(Path("/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/train").glob("*.npy"))[0]
data = np.load(sample)
print(f"Shape: {data.shape}")  # Should be (T, 414), not (T, 138)cd