import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs
from scipy.stats import spearmanr, ttest_1samp
from tqdm.auto import tqdm
import numba

analysis = "train_rdm_test_pat"
data_path = PRED_PATH
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
jobs = -1

# get times
epoch_fname = data_path / lock / 'sub01-0-epo.fif'
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch

@numba.jit
def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    d = rank_x - rank_y
    d_squared = np.sum(d * d)
    rho = 1 - (6 * d_squared) / (n * (n * n - 1))
    return rho

for lock in ['stim', 'button']:

    figure_dir = data_path / 'figures' / 'sensors' / analysis / lock
    ensure_dir(figure_dir)

    res_dir = data_path / 'results' / 'sensors' / analysis / lock

    # load patterns and randoms time-generalization on all epochs
    all_scores, sessions = [], []
    for subject in tqdm(subjects):
        score = np.load(op.join(res_dir, 'scores', f"{subject}-all.npy"))
        all_scores.append(score)
        session = []
        for epoch_num in [1, 2, 3, 4]:
            score = np.load(op.join(res_dir, 'scores', f"{subject}-{epoch_num}.npy"))
            session.append(score)
        sessions.append(session)
    all_scores = np.array(all_scores)
    sessions = np.array(sessions)

    # plot average all sessions
    if not op.exists(fig.savefig(op.join(figure_dir, "mean_all.pdf"))):
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        im = ax.imshow(
            all_scores.mean(0),
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.5,
            vmin=.20,
            vmax=.30)
        ax.set_xlabel("Testing Time (s)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title("all", style='italic')
        ax.axvline(0, color="k")
        ax.axhline(0, color="k")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")
        pval = gat_stats(all_scores - 0.25, jobs)
        sig = np.array(pval < 0.05)
        xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
        ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                            linestyles='solid', linewidths=1)
        fig.savefig(op.join(figure_dir, "mean_all.pdf"))
    # plot per session
    for session in range(4):
        if not op.exists(op.join(figure_dir, f"mean_{session+1}.pdf")):
            fig, ax = plt.subplots(1, 1, figsize=(16, 7))
            im = ax.imshow(
                sessions[:, session, :, :].mean(0),
                interpolation="lanczos",
                origin="lower",
                cmap="RdBu_r",
                extent=times[[0, -1, 0, -1]],
                aspect=0.5,
                vmin=.20,
                vmax=.30)
            ax.set_xlabel("Testing Time (s)")
            ax.set_ylabel("Training Time (s)")
            ax.set_title(f"Session {session+1}", style='italic')
            ax.axvline(0, color="k")
            ax.axhline(0, color="k")
            pval = gat_stats(all_scores - 0.25, jobs)
            sig = np.array(pval < 0.05)
            xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
            ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                                linestyles='solid', linewidths=1)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("accuracy")
            fig.savefig(op.join(figure_dir, f"mean_{session+1}.pdf"))
    # spearman corr
    # Initialize the output array
    rhos = np.zeros((11, 813, 813))
    # Compute Spearman correlation for each subject
    for subject in tqdm(range(11)):
        for i in tqdm(range(813)):
            for j in range(813):
                values = sessions[subject, :, i, j]
                rho, _ = spearmanr(sessions[subject, :, i, j], range(len(values)))
                # rho = spearman_rank_correlation(sessions[subject, :, i, j], range(len(values)))
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
    fig.savefig(op.join(figure_dir, f"mean_rho.pdf"))
        