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
data_reord = []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    files = sorted([f for f in os.listdir(res_path) if f.startswith('scores-')], key=lambda x: int(x.split('-')[-1].split('.')[0]))
    data = []
    for file in files:
        data.append(np.load(res_path / file))
    data_reord.append(np.array(data).mean(0)) # average across blocks
data_reord = np.array(data_reord)

# load patterns and randoms time-generalization on all epochs
data_orig = []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / "scores_blocks" / subject
    pattern = []
    for block in range(4, 24):
        pfname = res_path / f'pat-{block}.npy'
        rfname = res_path / f'rand-{block}.npy'
        pattern.append(np.load(pfname))
    data_orig.append(np.array(pattern))
data_orig = np.array(data_orig).mean(1)

times_orig = np.linspace(-4, 4, data_orig.shape[-1])
times = np.linspace(-3, 3, data_reord.shape[-1])
# match time points
win = np.where((times_orig >= times[0]) & (times_orig <= times[-1]))[0]
win = np.arange(win[0]-2, win[-1]+1)
data_orig = data_orig[:, win][:, :, win]

assert data_orig.shape == data_reord.shape, "Data shapes do not match after time windowing"

contrast = data_orig - data_reord

chance = .25

res_path = ensured(res_dir / "pval")

if not op.exists(res_path / "data_reord-pval.npy") or overwrite:
    print('Computing pval...')
    pval = gat_stats(data_reord - chance, jobs) # caution: the first 3 blocks are practice, need to exclude
    np.save(res_path / "data_reord-pval.npy", pval)
if not op.exists(res_path / "data_orig-pval.npy") or overwrite:
    print('Computing pval...')
    pval = gat_stats(data_orig - chance, jobs) # caution: the first 3 blocks are practice, need to exclude
    np.save(res_path / "data_orig-pval.npy", pval)
if not op.exists(res_path / "contrast-orig-reord-pval.npy") or overwrite:
    print('Computing pval...')
    pval = gat_stats(contrast, jobs) # caution: the first 3 blocks are practice, need to exclude
    np.save(res_path / "contrast-orig-reord-pval.npy", pval)

cmap1 = "RdBu_r"
cmap2 = "coolwarm"
contour_color = "#708090"
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})
threshold = .01

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 10))
plt.subplots_adjust(right=0.82, hspace=0.15)
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []

# plot original and reordered patterns
for ax, data, title in zip(axs.flat[:2], [data_orig.mean(0), data_reord.mean(0)], ["Pattern trials (Fig. 2 in main text)", "Re-ordered random trials"]):
    images.append(ax.imshow(data,
                    norm=norm,
                    interpolation="lanczos",
                    origin="lower",
                    cmap=cmap1,
                    extent=times[[0, -1, 0, -1]],
                    aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_yticks(np.arange(times[0] + 0.5, times[-1], 0.5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval_fname = "data_orig-pval.npy" if "Pattern" in title else "data_reord-pval.npy"
    pval = np.load(res_path / pval_fname)
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=contour_color, levels=[0],
                    linestyles='-', linewidths=1, alpha=1)

# plot contrast
vmin, vmax = -0.04, 0.04
im = axs[-1].imshow(contrast.mean(0),
                vmin=vmin,
                vmax=vmax,
                interpolation="lanczos",
                origin="lower",
                cmap=cmap2,
                extent=times[[0, -1, 0, -1]],
                aspect=0.5)
axs[-1].set_ylabel("Training time (s)", fontsize=13)
axs[-1].set_xticks(np.arange(times[0] + 0.5, times[-1], 0.5))
axs[-1].set_yticks(np.arange(times[0] + 0.5, times[-1], 0.5))
axs[-1].set_title("Contrast (pattern - re-ordered random)", fontsize=16)
axs[-1].axvline(0, color="k")
axs[-1].axhline(0, color="k")
axs[-1].set_xlabel("Testing time (s)", fontsize=13)
xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
pval = np.load(res_path / "contrast-orig-reord-pval.npy")
sig = pval < threshold
axs[-1].contour(xx, yy, sig, colors=contour_color, levels=[0],
                    linestyles='-', linewidths=1, alpha=1)

# Add colorbars with exactly matching widths
fig.canvas.draw()
pos0 = axs[0].get_position()
pos1 = axs[1].get_position()
pos2 = axs[2].get_position()
cbar_width = 0.025
# cbar_gap = 0.015
cbar_gap = 0.03

cax1 = fig.add_axes([pos0.x1 + cbar_gap, pos1.y0, cbar_width, pos0.y1 - pos1.y0])
cbar1 = fig.colorbar(images[0], cax=cax1, ticks=[0.18, 0.32])
cbar1.set_label("\nAccuracy", rotation=270, fontsize=13)

cax2 = fig.add_axes([pos2.x1 + cbar_gap, pos2.y0, cbar_width, pos2.y1 - pos2.y0])
cbar2 = fig.colorbar(im, cax=cax2, ticks=[vmin, vmax])
cbar2.set_label("Difference in\naccuracy", rotation=270, fontsize=13)

fig.savefig(figure_dir / "orig_and_reord.pdf", transparent=True)
plt.close()

# diag = np.diag(contrast.mean(0))
# pval_diag = np.diag(pval)
# times[pval_diag < threshold]

# plt.plot(times, diag, color='blue')
# sig = pval_diag < threshold
# for t in [-3, -1.5, 0, 1.5, 3]:
#     plt.axvspan(t, t+0.2, color="grey", alpha=0.1)
# plt.axhline(0, color="k")
# plt.fill_between(times, diag, where=sig, color="blue", alpha=0.2)