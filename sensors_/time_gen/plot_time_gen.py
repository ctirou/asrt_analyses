import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs
from scipy.stats import spearmanr, ttest_1samp
from tqdm.auto import tqdm

analysis = "time_generalization_1024"
data_path = PRED_PATH / "no_filter"
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
jobs = -1
analysis = "csp"

# get times
epoch_fname = PRED_PATH / lock / 'sub01-0-epo.fif'
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch

figure_dir = data_path / 'figures' / 'sensors' / lock
ensure_dir(figure_dir)

res_dir = data_path / 'results' / 'sensors' / lock

# load patterns and randoms time-generalization on all epochs
patterns, randoms = [], []
for subject in tqdm(subjects):
    pattern = np.load(op.join(res_dir, f"{subject}-epochall-pattern-scores.npy"))
    patterns.append(pattern)
    random = np.load(op.join(res_dir, f"{subject}-epochall-random-scores.npy"))
    randoms.append(random)
patterns = np.array(patterns)
randoms = np.array(randoms)

# plot pattern
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    patterns.mean(0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=.20,
    vmax=.30)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("pattern", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(figure_dir, "mean_pattern.pdf"))

# plot random
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    randoms.mean(0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=.20,
    vmax=.30)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("random", style="italic")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(figure_dir, "mean_random.pdf"))

# plot contrast with significance
contrasts = patterns - randoms

# pval = gat_stats(contrasts, jobs)
# sig = np.array(pval < 0.05)

fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    contrasts.mean(0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=-0.1,
    vmax=0.1)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("contrast = pattern - random", style='italic')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("difference in accuracy")
# xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
# ax.contour(xx, yy, sig, colors='Gray', levels=[0],
#                     linestyles='solid', linewidths=1)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
fig.savefig(op.join(figure_dir, "mean_contrast.pdf"))

# look at the correlations
all_patterns, all_randoms = [], []
for subject in subjects:
    patterns, randoms = [], []
    for epoch_num in [1, 2, 3, 4]:
        pattern = np.load(op.join(res_dir, f"{subject}-epoch{epoch_num}-pattern-scores.npy"))
        patterns.append(pattern)
        random = np.load(op.join(res_dir, f"{subject}-epoch{epoch_num}-random-scores.npy"))
        randoms.append(random)
    patterns = np.array(patterns)
    randoms = np.array(randoms)
    all_patterns.append(patterns)
    all_randoms.append(randoms)
all_patterns = np.array(all_patterns)
all_randoms = np.array(all_randoms)

for trial_type, time_gen in zip(['pattern', 'random'], [all_patterns, all_randoms]):
    # Initialize the output array
    rhos = np.zeros((11, 813, 813))
    # Compute Spearman correlation for each subject
    for subject in tqdm(range(11)):
        for i in tqdm(range(813)):
            for j in range(813):
                values = time_gen[subject, :, i, j]
                rho, _ = spearmanr(time_gen[subject, :, i, j], range(len(values)))
                rhos[subject, i, j] = rho
    # Plot the mean correlation
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    im = ax.imshow(
        rhos.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.3,
        vmax=0.3)
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman correlation')
    # add significance
    pval = gat_stats(rhos, jobs)
    sig = np.array(pval < 0.05)
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    fig.savefig(op.join(figure_dir, f"mean_rho_{trial_type}_1024.pdf"))
    
chance = .25
# plot pattern
color1 = "#1982C4"
color2 = "#00BFB3"
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 15), layout='tight', sharex=True)
coco = np.array([np.diag(cock) for cock in patterns])
pval = decod_stats(coco - chance, -1)
sig = pval < 0.05
pval_unc = ttest_1samp(coco, popmean=0, axis=0)[1]
sig_unc = pval_unc < 0.05
# List to store x-coordinates where sig_unc is true
x_points = [x for x, sig in zip(times, sig_unc) if sig]
ax1.set_ylim(0.225, 0.450)
ax1.plot(times, np.diag(patterns.mean(0)), color=color1)
ax1.set_title(f"$pattern$", fontsize=15)
ax1.fill_between(times, chance, np.diag(patterns.mean(0)), color=color2, alpha=.7, where=sig, label = 'corr')
ax1.axhline(chance, alpha=.7, color='black')
ax1.axvline(-3, ls="dashed", alpha=.7, color='black')
ax1.axvline(-1.5, ls="dashed", alpha=.7, color='black')
ax1.axvline(0, ls="dashed", alpha=.7, color='black')
ax1.axvline(1.5, ls="dashed", alpha=.7, color='black')
ax1.axvline(3, ls="dashed", alpha=.7, color='black')
ax1.scatter(x_points, [0] * len(x_points), color='#DD614A', label='uncorr')
ax1.legend()
# plot random
coco = np.array([np.diag(cock) for cock in randoms])
pval = decod_stats(coco - chance, -1)
sig = pval < 0.05
pval_unc = ttest_1samp(coco, popmean=0, axis=0)[1]
sig_unc = pval_unc < 0.05
# List to store x-coordinates where sig_unc is true
x_points = [x for x, sig in zip(times, sig_unc) if sig]
ax2.set_ylim(0.225, 0.450)
ax2.plot(times, np.diag(randoms.mean(0)), color=color1, label='contrasts')
ax2.set_title(f"$random$", fontsize=15)
ax2.fill_between(times, chance, np.diag(randoms.mean(0)), color=color2, alpha=.7, where=sig)
ax2.axhline(chance, alpha=.7, color='black')
ax2.axvline(-3, ls="dashed", alpha=.7, color='black')
ax2.axvline(-1.5, ls="dashed", alpha=.7, color='black')
ax2.axvline(0, ls="dashed", alpha=.7, color='black')
ax2.axvline(1.5, ls="dashed", alpha=.7, color='black')
ax2.axvline(3, ls="dashed", alpha=.7, color='black')
ax2.scatter(x_points, [0] * len(x_points), color='#DD614A')
# plot contrast with significance
contrasts = patterns - randoms
coco = np.array([np.diag(cock) for cock in contrasts])
pval = decod_stats(coco, -1)
sig = pval < 0.05
pval_unc = ttest_1samp(coco, popmean=0, axis=0)[1]
sig_unc = pval_unc < 0.05
# List to store x-coordinates where sig_unc is true
x_points = [x for x, sig in zip(times, sig_unc) if sig]
ax3.plot(times, np.diag(contrasts.mean(0)), color=color1, label='contrasts')
ax3.set_title(f"$contrast$", fontsize=15)
ax3.fill_between(times, 0, np.diag(contrasts.mean(0)), color=color2, alpha=.7, where=sig)
ax3.axhline(0, alpha=.7, color='black')
ax3.axvline(-3, ls="dashed", alpha=.7, color='black')
ax3.axvline(-1.5, ls="dashed", alpha=.7, color='black')
ax3.axvline(0, ls="dashed", alpha=.7, color='black')
ax3.axvline(1.5, ls="dashed", alpha=.7, color='black')
ax3.axvline(3, ls="dashed", alpha=.7, color='black')
ax3.scatter(x_points, [0] * len(x_points), color='#DD614A')
fig.savefig(figure_dir / "diags.pdf")
plt.close()
