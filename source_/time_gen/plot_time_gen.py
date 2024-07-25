import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs, read_labels_from_annot
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import numba

analysis = "time_generalization"
data_path = PRED_PATH
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
jobs = -1
verbose = True

# get times
epoch_fname = data_path / lock / 'sub01-0-epo.fif'
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch

res_path = RESULTS_DIR
res_dir = res_path / analysis / 'source' / lock 

# figures output directory
figure_dir = HOME / 'figures' / analysis / 'source' / lock
ensure_dir(figure_dir)

@numba.jit
def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    d = rank_x - rank_y
    d_squared = np.sum(d * d)
    rho = 1 - (6 * d_squared) / (n * (n * n - 1))
    return rho

# get labels
# labels = SURFACE_LABELS
labels_annot = read_labels_from_annot(subject='sub01', parc='aparc', hemi='both', subjects_dir=FREESURFER_DIR, verbose=False)
label_names = [label.name for label in labels_annot]

for ilabel, label in enumerate(label_names):
    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)
    ensure_dir(figure_dir / label)
    # load patterns and randoms time-generalization on all epochs
    patterns, randoms = [], []
    for subject in subjects[:-3]:
        pattern = np.load(res_dir / label / 'pattern' / f"{subject}-all-scores.npy")
        patterns.append(pattern)
        random = np.load(res_dir / label / 'random' / f"{subject}-all-scores.npy")
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
        vmin=0.23,
        vmax=.27)
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(figure_dir / label / "mean_pattern.pdf")

    # plot random
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    im = ax.imshow(
        randoms.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-.30,
        vmax=.30)
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(figure_dir / label / "mean_random.pdf")

    # plot contrast with significance
    contrasts = patterns - randoms
    
    coco = np.array([np.diag(cock) for cock in contrasts])
    fig, ax = plt.subplots(1, 1, figsize=(35, 5), layout='tight')
    pval = decod_stats(coco, -1)
    sig = pval < 0.05
    ax.plot(times, np.diag(contrasts.mean(0)), label='contrasts')
    ax.fill_between(times, 0, np.diag(contrasts.mean(0)), color="grey", alpha=.3, where=sig)
    fig.savefig(figure_dir / label / "contrast_diag.pdf")

    # pval = gat_stats(contrasts, jobs)
    # pval = np.load(res_path / "pval" / label / "contrast-pval.npy")
    # sig = np.array(pval < 0.05)

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    im = ax.imshow(
        contrasts.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    # xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    # ax.contour(xx, yy, sig, colors='Gray', levels=[0],
    #                     linestyles='solid', linewidths=1)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    fig.savefig(figure_dir / label / "mean_contrast.png")

    # # look at the correlations
    # all_patterns, all_randoms = [], []
    # for subject in subjects:
    #     patterns, randoms = [], []
    #     for epoch_num in [1, 2, 3, 4]:
    #         pattern = np.load(res_dir / label / 'pattern' / f"{subject}-{epoch_num}-scores.npy")
    #         patterns.append(pattern)
    #         random = np.load(res_dir / label / 'random' / f"{subject}-{epoch_num}-scores.npy")
    #         randoms.append(random)
    #     patterns = np.array(patterns)
    #     randoms = np.array(randoms)
    #     all_patterns.append(patterns)
    #     all_randoms.append(randoms)
    # all_patterns = np.array(all_patterns)
    # all_randoms = np.array(all_randoms)

    # for trial_type, time_gen in zip(['pattern', 'random'], [all_patterns, all_randoms]):
    #     # Initialize the output array
    #     rhos = np.zeros((11, 813, 813))
    #     # Compute Spearman correlation for each subject
    #     for subject in tqdm(range(11)):
    #         for i in tqdm(range(813)):
    #             for j in range(813):
    #                 values = time_gen[subject, :, i, j]
    #                 rho = spearman_rank_correlation(time_gen[subject, :, i, j], range(len(values)))
    #                 rhos[subject, i, j] = rho
    #     # Plot the mean correlation
    #     fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    #     im = ax.imshow(
    #         rhos.mean(0),
    #         interpolation="lanczos",
    #         origin="lower",
    #         cmap="RdBu_r",
    #         extent=times[[0, -1, 0, -1]],
    #         aspect=0.5,
    #         vmin=-0.3,
    #         vmax=0.3)
    #     ax.set_xlabel("Testing Time (s)")
    #     ax.set_ylabel("Training Time (s)")
    #     ax.set_title("Temporal generalization")
    #     cbar = plt.colorbar(im, ax=ax)
    #     cbar.set_label('Spearman correlation')
    #     # # add significance
    #     # pval = gat_stats(rhos, jobs)
    #     # sig = np.array(pval < 0.05)
    #     # xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    #     # ax.contour(xx, yy, sig, colors='Gray', levels=[0],
    #     #                     linestyles='solid', linewidths=1)
    #     ax.axvline(0, color="k")
    #     ax.axhline(0, color="k")
    #     fig.savefig(op.join(figure_dir, lock, f"mean_rho_{trial_type}.png"))

