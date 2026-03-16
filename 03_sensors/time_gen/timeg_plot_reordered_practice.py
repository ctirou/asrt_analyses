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

data_type = "scores_blocks"

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type

# load patterns and randoms time-generalization on all epochs
last_blocks_reordered = [18, 19, 20]
last_blocks = [21, 22, 23]
last_data_blocks = []
last_data_reordereed = []

for subject in tqdm(subjects):
    
    # load data for last three blocks
    res_path_last = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    files = sorted([f for f in os.listdir(res_path_last) if f.startswith('pat-')], key=lambda x: int(x.split('-')[-1].split('.')[0]))
    data_last = []
    for file in files:
        if int(file.split('-')[-1].split('.')[0]) in last_blocks:
            print(f"Loading data for {subject} block {file.split('-')[-1].split('.')[0]}")
            data_last.append(np.load(res_path_last / file))
    last_data_blocks.append(np.array(data_last).mean(0)) # average across blocks

    # load data for last three blocks reordered
    res_pat_reord = RESULTS_DIR / 'TIMEG' / 'sensors' / (data_type + "_reordered") / subject
    files = sorted([f for f in os.listdir(res_pat_reord) if f.startswith('scores-0')], key=lambda x: int(x.split('-')[-1].split('.')[0]))
    data_last_reord = []
    for file in files:
        if int(file.split('-')[-1].split('.')[0]) in last_blocks_reordered:
            print(f"Loading data for {subject} block {file.split('-')[-1].split('.')[0]} (reordered)")
            data_last_reord.append(np.load(res_pat_reord / file))
    last_data_reordereed.append(np.array(data_last_reord).mean(0)) # average across blocks

last_data_blocks = np.array(last_data_blocks)
last_data_reordereed = np.array(last_data_reordereed)

times = np.linspace(-3, 3, last_data_reordereed.shape[-1])

# match time points
times_orig = np.linspace(-4, 4, last_data_blocks.shape[-1])
win = np.where((times_orig >= times[0]) & (times_orig <= times[-1]))[0]
# add 2 extra samples to win to match the original time window of last_data_blocks, before and after
win = np.arange(win[0]-2, win[-1]+1)
last_data_blocks = last_data_blocks[:, win][:, :, win]

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

fig, axs = plt.subplots(2, 1, sharex=False, layout='constrained', figsize=(7, 6))
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []
for ax, data, title in zip(axs.flat, [last_data_blocks, last_data_reordereed], ["Last three blocks (original)", "Last three blocks (reordered)"]):
    images.append(ax.imshow(data.mean(0), 
                            norm=norm,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[[0, -1, 0, -1]],
                            aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(times[0]+ 0.5, times[-1], 0.5))
    ax.set_yticks(np.arange(times[0]+ 0.5, times[-1], 0.5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

# xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
# pval = np.load(res_path / "all_data-pval.npy")
# sig = pval < threshold
# ax2.contour(xx, yy, sig, colors=contour_color, levels=[0],
#                 linestyles='-', linewidths=1, alpha=1)

axs[-1].set_xlabel("Testing time (s)", fontsize=13)

cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)

# fig.savefig(figure_dir / "random_reordered.pdf", transparent=True)
# plt.close()
