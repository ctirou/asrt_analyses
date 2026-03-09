# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as spear, ttest_1samp
from tqdm.auto import tqdm
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors

subjects = SUBJS15
jobs = -1
overwrite = True

data_type = "scores_blocks"

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-4, 4, 813)

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type

# load patterns and randoms time-generalization on all epochs
pats_blocks, rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    pattern, random = [], []
    block = 4
    for f in [1, 2]:
        pfname = res_path / f'pat-{block}-fold{f}.npy'
        rfname = res_path / f'rand-{block}-fold{f}.npy'
        pattern.append(np.load(pfname))
        random.append(np.load(rfname))
    pats_blocks.append(np.array(pattern))
    rands_blocks.append(np.array(random))
pats_blocks = np.array(pats_blocks)
rands_blocks = np.array(rands_blocks)
contrasts = pats_blocks - rands_blocks

learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)

chance = .25
threshold = .05

tmin, tmax = -3, 3
idx = np.where((times >= tmin) & (times <= tmax))[0]
res_path = ensured(res_dir / "pval_supp")

# for fold in [1, 2]:
#     if not op.exists(res_path / "all_pattern-pval.npy") or overwrite:
#         print('Computing pval for all patterns')
#         pval = gat_stats(pats_blocks[:, fold-1, idx][..., idx] - chance, jobs) # caution: the first 3 blocks are practice, need to exclude
#         np.save(res_path / f"all_pattern-fold{fold}-pval.npy", pval)
#     if not op.exists(res_path / "all_random-pval.npy") or overwrite:
#         print('Computing pval for all randoms')
#         pval = gat_stats(rands_blocks[:, fold-1, idx][..., idx] - chance, jobs)
#         np.save(res_path / f"all_random-fold{fold}-pval.npy", pval)
#     if not op.exists(res_path / "all_contrast-pval.npy") or overwrite:
#         print('Computing pval for all contrasts')
#         contrasts = pats_blocks - rands_blocks
#         pval = gat_stats(contrasts[:, fold-1, idx][..., idx], jobs)
#         np.save(res_path / f"all_contrast-fold{fold}-pval.npy", pval)

cmap1 = "RdBu_r"
cmap2 = "coolwarm"
contour_color = "black"
contour_color = "#00BFA6"
contour_color = "#708090"
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

for data_name, data_type in [("pattern", pats_blocks), ("random", rands_blocks), ("contrast", contrasts)]:
    fig, axs = plt.subplots(2, 1, sharex=False, layout='constrained', figsize=(7, 6))
    vmin, vmax = (0.18, 0.32) if data_name != "contrast" else (-0.05, 0.05)
    norm = colors.Normalize(vmin=0.18, vmax=0.32)
    images = []
    for ax, fold, title in zip(axs.flat, [1, 2], ["Fold 1", "Fold 2"]):
        data = data_type[:, fold-1, idx][..., idx]
        if data_name != "contrast":
            images.append(ax.imshow(data.mean(0), 
                                    norm=norm,
                                    interpolation="lanczos",
                                    origin="lower",
                                    cmap=cmap1,
                                    extent=times[idx][[0, -1, 0, -1]],
                                    aspect=0.5))
        else:
            images.append(ax.imshow(data.mean(0), 
                                    vmin=vmin,
                                    vmax=vmax,
                                    interpolation="lanczos",
                                    origin="lower",
                                    cmap=cmap2,
                                    extent=times[idx][[0, -1, 0, -1]],
                                    aspect=0.5))
        ax.set_ylabel("Training time (s)", fontsize=13)
        ax.set_xticks(np.arange(tmin, tmax, .5))
        ax.set_yticks(np.arange(tmin, tmax, .5))
        ax.set_title(f"{title}", fontsize=12)
        ax.axvline(0, color="k")
        ax.axhline(0, color="k")
        # xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
        # pval = np.load(res_path / f"all_{data_name}-fold{fold}-pval.npy")
        # sig = pval < threshold
        # ax.contour(xx, yy, sig, colors=contour_color, levels=[0],
        #                linestyles='-', linewidths=1, alpha=1)
        # if title == "random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
    cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
    cbar.set_label("\nAccuracy", rotation=270, fontsize=13)
    fig.suptitle(f"Time generalization in {data_name} trials")
    fig.savefig(figure_dir / f"{data_name}_block4.pdf", transparent=True)
    # plt.close()
