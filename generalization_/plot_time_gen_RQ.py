import os
from base import ensure_dir, gat_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs
from scipy.stats import spearmanr
from tqdm.auto import tqdm
analysis = "time_generalization"
data_path = os.getenv("PRED_PATH")
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
summary = False
epoch_fname = '/Users/romainquentin/Desktop/data/pred_asrt/stim/sub01-0-epo.fif'
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch
results_dir = op.join(data_path, 'results')
figure_dir = op.join(data_path, 'figure_results')

# load patterns and randoms time-generalization on all epochs
patterns, randoms = [], []
for subject in tqdm(subjects):
    pattern = np.load(op.join(data_path, 'results', f"{subject}-epochall-pattern-scores.npy"))
    patterns.append(pattern)
    random = np.load(op.join(data_path, 'results', f"{subject}-epochall-random-scores.npy"))
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
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(figure_dir, "mean_pattern.png"))
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
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(figure_dir, "mean_random.png"))
# plot contrast with significance
contrasts = patterns - randoms
pval = gat_stats(contrasts)
sig = np.array(pval < 0.05)
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
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                    linestyles='solid', linewidths=1)
fig.savefig(op.join(figure_dir, "mean_contrast.png"))


# look at the correlations
all_patterns, all_randoms = [], []
for subject in subjects:
    patterns, randoms = [], []
    for epoch_num in [1, 2, 3, 4]:
        pattern = np.load(op.join(data_path, 'results', f"{subject}-epoch{epoch_num}-pattern-scores.npy"))
        patterns.append(pattern)
        random = np.load(op.join(data_path, 'results', f"{subject}-epoch{epoch_num}-random-scores.npy"))
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
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman correlation')
    # add significance
    pval = gat_stats(rhos)
    sig = np.array(pval < 0.05)
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    fig.savefig(op.join(figure_dir, f"mean_rho_{trial_type}.png"))

