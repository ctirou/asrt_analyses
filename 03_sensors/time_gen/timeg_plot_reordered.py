# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from matplotlib import colors

subjects = SUBJS15
jobs = -1
overwrite = False

data_type = "scores_blocks_reordered"

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type

# load patterns and randoms time-generalization on all epochs
data_blocks = []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    files = sorted(os.listdir(res_path), key=lambda x: int(x.split('-')[1].split('.')[0]))
    data = []
    for file in files:
        data.append(np.load(res_path / file))
    data_blocks.append(np.array(data).mean(0)) # average across blocks
data_blocks = np.array(data_blocks)

times = np.linspace(-3, 3, data_blocks.shape[-1])
chance = .25
threshold = .05
threshold = .01

res_path = ensured(res_dir / "pval")

if not op.exists(res_path / "all_data-pval.npy") or overwrite:
    print('Computing pval...')
    pval = gat_stats(data_blocks - chance, jobs) # caution: the first 3 blocks are practice, need to exclude
    np.save(res_path / "all_data-pval.npy", pval)

cmap1 = "RdBu_r"
contour_color = "#708090"
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, ax = plt.subplots(1, 1, sharex=False, layout='constrained', figsize=(10, 4))
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []

pos = ax.imshow(data_blocks.mean(0), 
        norm=norm,
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5)
ax.set_ylabel("Training time (s)", fontsize=13)
ax.set_xticks(np.arange(times[0], times[-1] + 0.5, 0.5))
ax.set_yticks(np.arange(times[0], times[-1] + 0.5, 0.5))
ax.set_title("Time generalization in reordered random trials", fontsize=16)
ax.axvline(0, color="k")
ax.axhline(0, color="k")

xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
pval = np.load(res_path / "all_data-pval.npy")
sig = pval < threshold
ax.contour(xx, yy, sig, colors=contour_color, levels=[0],
                linestyles='-', linewidths=1, alpha=1)

ax.set_xlabel("Testing time (s)", fontsize=13)

cbar = fig.colorbar(pos, ax=ax, orientation='vertical', ticks=[0.18, 0.32])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)

fig.savefig(figure_dir / "random_reordered.pdf", transparent=True)

plt.close()
