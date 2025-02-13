import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs
from scipy.stats import zscore, pearsonr, ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from matplotlib import colors

# analysis = "pat_bsl_filtered_3300_3160"
data_path = TIMEG_DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
jobs = -1

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-1.5, 4, 559)

# figure_dir = data_path / 'figures' / 'sensors' / lock
figure_dir = FIGURES_DIR / "time_gen" / "sensors" / lock
ensure_dir(figure_dir)

res_dir = data_path / 'results' / 'sensors' / lock

# load patterns and randoms time-generalization on all epochs
all_patterns, all_randoms = [], []
patterns, randoms = [], []
for subject in tqdm(subjects):
    pattern = np.load(op.join(res_dir, "pattern", f"{subject}-all-scores.npy"))
    all_patterns.append(pattern)
    random = np.load(op.join(res_dir, "random", f"{subject}-all-scores.npy"))
    all_randoms.append(random)
    
    pat, rand = [], []
    for i in range(5):
        pat.append(np.load(res_dir / 'pattern' / f"{subject}-{i}-scores.npy"))
        rand.append(np.load(res_dir / 'random' / f"{subject}-{i}-scores.npy"))
    
    patterns.append(np.array(pat))
    randoms.append(np.array(rand))

all_patterns, all_randoms = np.array(all_patterns), np.array(all_randoms)
patterns, randoms = np.array(patterns), np.array(randoms)

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
chance = .25
threshold = .05

ensure_dir(res_dir / "pval")
if not op.exists(res_dir / "pval" / "all_pattern-pval.npy"):
    pval = gat_stats(all_patterns - chance, jobs)
    np.save(res_dir / "pval" / "all_pattern-pval.npy", pval)
if not op.exists(res_dir / "random" / "pval" / "all_random-pval.npy"):
    pval = gat_stats(all_randoms - chance, jobs)
    np.save(res_dir / "pval" / "all_random-pval.npy", pval)
if not op.exists(res_dir / "pval" / "all_contrast-pval.npy"):
    contrasts = all_patterns - all_randoms
    pval = gat_stats(contrasts, jobs)
    np.save(res_dir / "pval" / "all_contrast-pval.npy", pval)

# ensure_dir(res_dir / "corr")
# if not op.exists(res_dir / "corr" / "rhos_blocks.npy"):
#     contrasts = patterns - randoms
#     contrasts = zscore(contrasts, axis=-1)
#     all_rhos = []
#     for sub in range(len(subjects)):
#         rhos = np.empty((times.shape[0], times.shape[0]))
#         vector = [0, 1, 2, 3, 4]  # vector to correlate with
#         contrast = contrasts[sub]
#         results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
#         for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
#             rhos[t, g] = results[idx]
#         all_rhos.append(rhos)
#     all_rhos = np.array(all_rhos)
#     np.save(res_dir / "corr" / "rhos_blocks.npy", all_rhos)
#     pval = gat_stats(all_rhos, -1)
#     np.save(res_dir / "corr" / "pval_blocks-pval.npy", pval)

# save learn df x time gen correlation and pvals
if not op.exists(res_dir / "corr" / "rhos_learn.npy"):
    contrasts = patterns - randoms
    contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
    all_rhos = []
    for sub in range(len(subjects)):
        rhos = np.empty((times.shape[0], times.shape[0]))
        vector = learn_index_df.iloc[sub]  # vector to correlate with
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    np.save(res_dir / "corr" / "rhos_learn.npy", all_rhos)
    pval = gat_stats(all_rhos, -1)
    np.save(res_dir / "corr" / "pval_learn-pval.npy", pval)
    
cmap1 = "RdBu_r"
cmap2 = "magma_r"
cmap3 = "viridis"
cmap4 = "cividis"

idx = np.where(times <= 3)[0]

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(7, 6))
norm = colors.Normalize(vmin=0.15, vmax=0.35)
images = []
for ax, data, title in zip(axs.flat, [all_patterns, all_randoms], ["Pattern", "Random"]):
    images.append(ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            norm=norm,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    pval = np.load(res_dir / "pval" / f"all_{title.lower()}-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if title == "Random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0.15, 0.35])
cbar.set_label("$Accuracy$", rotation=270, fontsize=13)
fig.savefig(figure_dir / "pattern_random.pdf", transparent=True)
plt.close()

### plot contrast ###
contrasts = all_patterns - all_randoms
pval_cont = np.load(res_dir / "pval" / "all_contrast-pval.npy")
rhos = np.load(res_dir / "corr" / "rhos_learn.npy")
pval_rhos = np.load(res_dir / "corr" / "pval_learn-pval.npy")
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

# diag = np.array([np.diag(coco) for coco in rhos])
# plt.plot(times, diag.mean(0))
# plt.axhline(0)
# pval = decod_stats(diag, -1)
# sig = pval < threshold
# plt.fill_between(times, diag.mean(0), 0, where=sig, alpha=0.5)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout='constrained')
# norm = colors.Normalize(vmin=-0.1, vmax=0.1)
images = []
for ax, data, title, pval, vmin, vmax in zip(axs.flat, [contrasts, rhos], ["Contrast", "Correlation between contrast and learning"], [pval_cont, pval_rhos], [-0.1, -0.5], [0.1, 0.5]):
    im = ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            # norm=norm,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5)
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    sig = pval < threshold
    ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if ax == axs.flat[-1]:
        ax.set_xlabel("Testing time (s)", fontsize=13)
        label = "Spearman's\nrho"
    else:
        label = "Difference in\naccuracy"

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
    cbar.set_label(label, fontstyle='italic', rotation=270, fontsize=13)
fig.savefig(figure_dir / "contrast_corr.pdf", transparent=True)
plt.close()


### plot learning index x timeg correlation ###
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
contrasts = patterns - randoms
timeg = []
for sub in range(len(subjects)):
    tg = []
    for i in range(5):
        tg.append(contrasts[sub, i, idx_timeg][:, idx_timeg].mean())
    timeg.append(np.array(tg))
timeg = np.array(timeg)
slopes, intercepts = [], []

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(timeg[sub], learn_index_df.iloc[sub], 1)
    ax.scatter(timeg[sub], learn_index_df.iloc[sub], alpha=0.3, label=f"Subject {sub+1}")
    ax.plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
timeg_range = np.linspace(timeg.min(), timeg.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax.plot(timeg_range, mean_slope * timeg_range + mean_intercept, color='black', lw=4, label='Mean fit')
fig.suptitle('Correlation between mean predictive activity and learning', fontsize=16, y=0.95)
ax.set_xlabel('Average pre-stimulus time generalization', fontsize=13)
ax.set_ylabel('Learning index', fontsize=13)
# ax.legend(frameon=False, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
timeg_flat = timeg.flatten()
learn_index_flat = learn_index_df.to_numpy().flatten()
r, pval = spear(timeg_flat, learn_index_flat)
# Add text with r and pval to the plot
textstr = f'Spearman $r$ = {r:.2f}\n$p$ = {pval:.2e}'
props = dict(boxstyle='round', facecolor='#7294D4', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
    verticalalignment='top', horizontalalignment='left', bbox=props)
# Adjust the legend to be outside the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', frameon=False, bbox_to_anchor=(1, 0.5), ncol=1)
# fig.suptitle('Correlation between mean predictive activity and learning', fontsize=14)

fig.savefig(figure_dir / "learning_index_timeg_corr.pdf", transparent=True)
plt.close()