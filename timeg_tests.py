import os
from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from matplotlib import colors

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15

jobs = -1
overwrite = False

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-1.5, 4, 559)
times = np.linspace(-4, 4, 813)

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / 'scores_40s'

# load patterns and randoms time-generalization on all epochs
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
all_pats_bins, all_rands_bins = [], []
for subject in tqdm(subjects):    
    pattern, random = [], []
    pattern_blocks, random_blocks = [], []
    pattern_bins, random_bins = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            p, r = [], []
            for fold in range(1, 3):
                p.append(np.load(res_path / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                r.append(np.load(res_path / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pattern_bins.append(np.load(res_path / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                random_bins.append(np.load(res_path / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
            pats.append(np.array(p))
            rands.append(np.array(r))
            pattern_blocks.append(np.array(pats).mean(0))
            random_blocks.append(np.array(rands).mean(0))

        if epoch_num != 0:
            pattern.append(np.mean(pats, 0))
            random.append(np.mean(rands, 0))

    pattern = np.array(pattern).mean(1)
    random = np.array(random).mean(1)
    pattern_blocks = np.array(pattern_blocks).mean(1)
    random_blocks = np.array(random_blocks).mean(1)
    pattern_bins = np.array(pattern_bins)
    random_bins = np.array(random_bins)
    
    all_pats.append(pattern)
    all_rands.append(random)
    all_pats_blocks.append(pattern_blocks)
    all_rands_blocks.append(random_blocks)
    all_pats_bins.append(pattern_bins)
    all_rands_bins.append(random_bins)

all_pats = np.array(all_pats)
all_rands = np.array(all_rands)
all_pats_blocks = np.array(all_pats_blocks)
all_rands_blocks = np.array(all_rands_blocks)
all_pats_bins = np.array(all_pats_bins)
all_rands_bins = np.array(all_rands_bins)

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
chance = .25
threshold = .05

# ensure_dir(res_path / "pval")
# if not op.exists(res_path / "pval" / "all_pattern-pval.npy") or overwrite:
#     print('Computing pval for all patterns')
#     pval = gat_stats(all_pats - chance, jobs)
#     np.save(res_path / "pval" / "all_pattern-pval.npy", pval)
# if not op.exists(res_path / "pval" / "all_random-pval.npy") or overwrite:
#     print('Computing pval for all randoms')
#     pval = gat_stats(all_rands - chance, jobs)
#     np.save(res_path / "pval" / "all_random-pval.npy", pval)
# if not op.exists(res_path / "pval" / "all_contrast-pval.npy") or overwrite:
#     print('Computing pval for all contrasts')
#     contrasts = all_pats - all_rands
#     pval = gat_stats(contrasts, jobs)
#     np.save(res_path / "pval" / "all_contrast-pval.npy", pval)

# save learn df x time gen correlation and pvals
ensure_dir(res_path / "corr")
if not op.exists(res_path / "corr" / "rhos_learn.npy") or overwrite:
    contrasts = all_pats - all_rands
    contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
    all_rhos = []
    for sub in range(len(subjects)):
        rhos = np.empty((times.shape[0], times.shape[0]))
        vector = learn_index_df.iloc[sub][1:]  # vector to correlate with
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    np.save(res_path / "corr" / "rhos_learn.npy", all_rhos)
    # pval = gat_stats(all_rhos, -1)
    # np.save(res_path / "corr" / "pval_learn-pval.npy", pval)

mean_rsa = np.load("/Users/coum/MEGAsync/figures/RSA/sensors/mean_rsa.npy")
if not op.exists(res_path / "corr" / "rhos_rsa.npy") or overwrite:
    contrasts = all_pats - all_rands
    contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
    all_rhos = []
    for sub in range(len(subjects)):
        rhos= np.empty((times.shape[0], times.shape[0]))
        vector = mean_rsa[sub]
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    np.save(res_path / "corr" / "rhos_rsa.npy", all_rhos)
    pval = gat_stats(all_rhos, -1)
    np.save(res_path / "corr" / "pval_rsa-pval.npy", pval)

cmap1 = "RdBu_r"
cmap2 = "magma_r"
cmap2 = "coolwarm"
cmap3 = "viridis"
cmap4 = "cividis"

idx = np.where((times >= -1.5) & (times <= 3))[0]
idx = np.where((times >= -4) & (times <= 4))[0]

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(7, 6))
vmin, vmax = 0.2, 0.3
vmin, vmax = 0.18, 0.32
norm = colors.Normalize(vmin=vmin, vmax=vmax)
images = []
# for ax, data, title in zip(axs.flat, [all_patterns, all_randoms], ["pattern", "random"]):
for ax, data, title in zip(axs.flat, [all_pats, all_rands], ["pattern", "random"]):
    # images.append(ax.imshow(data[:, idx][:, :, idx].mean(0), 
    images.append(ax.imshow(data[:, :, idx][:, :, :, idx].mean((0, 1)), 
                            norm=norm,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(times[idx][0], times[idx][-1], 1))
    ax.set_yticks(np.arange(times[idx][0], times[idx][-1], 1))
    ax.set_title(f"Time generalization in {title} trials", fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    # pval = np.load(res_dir / "pval" / f"all_{title.lower()}-pval.npy")
    # sig = pval < threshold
    # ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
    #                     linestyles='--', linewidths=1, alpha=.5)
    if title == "random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)

# Contrast and learning correlation
contrasts = all_pats - all_rands
# pval_cont = np.load(res_path / "pval" / "all_contrast-pval.npy")
rhos = np.load(res_path / "corr" / "rhos_learn.npy")
# pval_rhos = np.load(res_path / "corr" / "pval_learn-pval.npy")
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout='constrained')
idx = np.where((times >= -1.5) & (times <= 3))[0]
idx = np.where((times >= -4) & (times <= 4))[0]
# norm = colors.Normalize(vmin=-0.1, vmax=0.1)
images = []
# for ax, data, title, pval, vmin, vmax in zip(axs.flat, [contrasts, rhos], \
    # ["Contrast (Pattern - Random)", "Contrast and learning correlation"], [pval_cont, pval_rhos], [-0.05, -0.2], [0.05, 0.2]):
for ax, data, title, vmin, vmax in zip(axs.flat, [contrasts.mean(1), rhos], \
    ["Contrast (Pattern - Random)", "Contrast and learning correlation"], [-0.05, -0.2], [0.05, 0.2]):
    cmap = 'coolwarm' if ax == axs.flat[0] else "BrBG"
        
    im = ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            # norm=norm,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5)
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    # xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    # sig = pval < threshold
    # ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
    #                     linestyles='--', linewidths=1, alpha=.5)
    if ax == axs.flat[-1]:
        ax.set_xlabel("Testing time (s)", fontsize=13)
        label = "Spearman's\nrho"
    else:
        label = "Difference in\naccuracy"
    # Draw an empty rectangle centered on -0.25
    rectcolor = 'black' if ax == axs.flat[0] else 'red'
    rect = plt.Rectangle([-0.75, -0.75], 0.72, 0.68, fill=False, edgecolor=rectcolor, linestyle='-', lw=2)
    ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
    cbar.set_label(label, rotation=270, fontsize=13)
    
# mean per block
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
diag_block = []
contrast_blocks = all_pats_blocks - all_rands_blocks
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = np.diag(contrast_blocks[sub, block])
        tg.append(data[idx_timeg].mean())
    diag_block.append(np.array(tg))
diag_block = np.array(diag_block)

blocks = [i for i in range(23)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.axvspan(0, 2, color='grey', alpha=0.1)
# ax.axvline(3, color='grey', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.set_xticks(blocks)
ax.set_xticklabels([str(i) for i in range(1, 24)])
for i in range(diag_block.shape[0]):
    ax.plot(blocks, diag_block[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, diag_block.mean(0), lw=3, color='#00A08A', label='Mean')
ax.legend(frameon=False)
ax.set_title('Predictive effect per block', fontstyle='italic')

# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_block = []
contrast_blocks = all_pats_blocks - all_rands_blocks
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = contrast_blocks[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    box_block.append(np.array(tg))
box_block = np.array(box_block)

blocks = [i for i in range(23)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.axvspan(0, 2, color='grey', alpha=0.1)
# ax.axvline(3, color='grey', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.set_xticks(blocks)
ax.set_xticklabels([str(i) for i in range(1, 24)])
for i in range(box_block.shape[0]):
    ax.plot(blocks, box_block[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, box_block.mean(0), lw=3, color='#00A08A', label='Mean')
ax.legend(frameon=False)
ax.set_title('Predictive effect per block', fontstyle='italic')

# mean diag per bins
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
diag_bins = []
contrast_bins = all_pats_bins - all_rands_bins
for sub in range(len(subjects)):
    tg = []
    for block in range(46):
        data = np.diag(contrast_bins[sub, block])
        tg.append(data[idx_timeg].mean())
    diag_bins.append(np.array(tg))
diag_bins = np.array(diag_bins)

# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_bins = []
contrast_blocks = all_pats_blocks - all_rands_blocks
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = contrast[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    box_bins.append(np.array(tg))
box_bins = np.array(box_bins)

blocks = [i for i in range(46)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(0, 5, color='grey', alpha=0.1)
ax.set_xticks(blocks)
ax.set_xticklabels([str(i) for i in range(1, 47)])
for i in range(diag_bins.shape[0]):
    ax.plot(blocks, diag_bins[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, diag_bins.mean(0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Predictive effect per bin', fontstyle='italic')