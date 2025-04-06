import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'
times = np.linspace(-0.2, 0.6, 82)

res_dir = RESULTS_DIR / 'causality' / 'sensors'

xy = [np.load(res_dir / f'{sub}_te_xy.npy') for sub in subjects]
yx = [np.load(res_dir / f'{sub}_te_yx.npy') for sub in subjects]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i, sub in enumerate(subjects):
    ax.plot(np.squeeze(xy[i]), label=f"{sub}: X → Y")
    ax.plot(np.squeeze(yx[i]), label=f"{sub}: Y → X")
ax.set_title(f"Local TE per subject")
# ax.legend()
plt.show()