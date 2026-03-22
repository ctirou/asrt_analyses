# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as spear, ttest_1samp
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors

subjects = SUBJS15
jobs = -1
overwrite = False

data_type = "scores_blocks"

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-4, 4, 813)
win = np.where((times >= 0) & (times <= 1))[0]

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type

# load patterns and randoms time-generalization on all epochs
pats_blocks, rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    pattern, random = [], []
    for block in range(1, 24):
        pfname = res_path / f'pat-{block}.npy' if block not in [1, 2, 3] else res_path / f'pat-0-{block}.npy'
        rfname = res_path / f'rand-{block}.npy' if block not in [1, 2, 3] else res_path / f'rand-0-{block}.npy'
        pattern.append(np.diag(np.load(pfname)))
        random.append(np.diag(np.load(rfname)))
    if subject == 'sub05':
        pat_bsl = np.diag(np.load(res_path / "pat-4.npy"))
        rand_bsl = np.diag(np.load(res_path / "rand-4.npy"))
        for i in range(3):
            pattern[i] = pat_bsl.copy()
            random[i] = rand_bsl.copy()
    pats_blocks.append(np.array(pattern))
    rands_blocks.append(np.array(random))
pats_blocks = np.array(pats_blocks)[..., win] * 100
rands_blocks = np.array(rands_blocks)[..., win] * 100
contrast = pats_blocks - rands_blocks

pats_mean = pats_blocks[:, 3:].mean(1)   # (n_subjects, n_times)
rands_mean = rands_blocks[:, 3:].mean(1)
contrast_mean = contrast[:, 3:].mean(1)

n = len(subjects)
chance = 25
threshold = .5
cpat = "#FAD510"
crdm = "#FF718B"

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axes = plt.subplot_mosaic([['top', 'right'], ['bot', 'right']],
                               figsize=(10, 5), layout='tight')
ax1a, ax1b, ax2 = axes['top'], axes['bot'], axes['right']
ax1b.sharex(ax1a)

for ax in (ax1a, ax1b, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# top left: pattern and random
ax1a.axvspan(0, 0.2, color="lightgray", alpha=0.2)
ax1a.axhline(chance, color="grey", linestyle="-", alpha=0.5)
ax1a.plot(times[win], pats_mean.mean(0), label="Pattern", color=cpat)
ax1a.fill_between(times[win], pats_mean.mean(0) - pats_mean.std(0) / np.sqrt(n),
                  pats_mean.mean(0) + pats_mean.std(0) / np.sqrt(n), alpha=0.2, color=cpat)
ax1a.plot(times[win], rands_mean.mean(0), label="Random", color=crdm)
ax1a.fill_between(times[win], rands_mean.mean(0) - rands_mean.std(0) / np.sqrt(n),
                  rands_mean.mean(0) + rands_mean.std(0) / np.sqrt(n), alpha=0.2, color=crdm)
ax1a.set_ylabel("Accuracy (%)")
ax1a.set_title("Decoding")
ax1a.legend(fontsize=9, frameon=False)

# bottom left: contrast
max1 = np.argmax(pats_mean.mean(0)[times[win] <= 0.2])
min2 = np.argmin(contrast_mean.mean(0)[times[win] > 0.2])
hw = 3  # half-window size for min2
idx_02 = np.sum(times[win] <= 0.2) - 1 # last index at 0.2s

ax1b.axvspan(0, 0.2, color="lightgray", alpha=0.2)
ax1b.axhline(0, color="grey", linestyle="-", alpha=0.5)
ax1b.plot(times[win], contrast_mean.mean(0), color="g")
ax1b.fill_between(times[win], contrast_mean.mean(0) - contrast_mean.std(0) / np.sqrt(n),
                  contrast_mean.mean(0) + contrast_mean.std(0) / np.sqrt(n), alpha=0.2, color="g")
ax1b.axvspan(times[win][max1], times[win][idx_02], facecolor="b", alpha=0.2)
ax1b.axvspan(times[win][max(0, min2 + np.sum(times[win] <= 0.2) - hw)],
             times[win][min2 + np.sum(times[win] <= 0.2) + hw], facecolor="r", alpha=0.2)
ax1b.set_xlabel("Time (s)")
ax1b.set_ylabel("Diff. in accuracy (a.u.)")
ax1b.set_title("Contrast")

# right: block level
max1_idx = max1
min2_idx = min2 + np.sum(times[win] <= 0.2)
sharp = contrast[..., max1_idx:idx_02 + 1].mean(-1)
damp = contrast[..., max(0, min2_idx - hw):min2_idx + hw + 1].mean(-1) * (-1)
blocks = np.arange(1, 24)
sigma = 1.5  # smoothing kernel width (in blocks)
ax2.axvspan(1, 3, color="lightgray", alpha=0.2)
ax2.plot(blocks, sharp.mean(0), color="b", alpha=0.3)
ax2.plot(blocks, gaussian_filter1d(sharp.mean(0), sigma), color="b", label="Sharpening")
ax2.plot(blocks, damp.mean(0), color="r", alpha=0.3)
ax2.plot(blocks, gaussian_filter1d(damp.mean(0), sigma), color="r", label="Dampening")
ax2.set_xlabel("Block")
ax2.set_ylabel("Diff. in accuracy (a.u.)")
ax2.set_xticks(blocks[::2])
ax2.set_xticks(blocks, minor=True)
ax2.grid(which='major', alpha=0.4, linestyle=':')
ax2.grid(which='minor', alpha=0.15, linestyle=':')
ax2.legend(fontsize=9, frameon=False, loc='lower right')
ax2.set_title("Block level effects")

plt.savefig(figure_dir / "timeg_opt_block_supp.pdf", dpi=300, transparent=True)
plt.close()