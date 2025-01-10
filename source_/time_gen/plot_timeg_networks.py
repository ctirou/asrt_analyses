import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs, read_labels_from_annot
from scipy.stats import ttest_1samp, spearmanr
from tqdm.auto import tqdm
import numba
import pandas as pd

data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
verbose = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
lock = 'stim'
overwrite = False
# network and custom label_names
n_parcels = 200
n_networks = 7
networks = schaefer_7[:-2] if n_networks == 7 else schaefer_17[:-2]
res_dir = data_path / 'results' / 'source' / lock / f'networks_{n_parcels}_{n_networks}'
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock / f"networks_{n_parcels}_{n_networks}"
ensure_dir(figures_dir)

names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2]

# Load data
times = np.linspace(-1.5, 1.5, 305)
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    all_patterns, all_randoms = [], []
    patterns, randoms = [], []
    
    for subject in subjects:
        pattern, random = [], []
        for i in [0, 1, 2, 3, 4]:
            pattern.append(np.load(res_dir / network / 'pattern' / f"{subject}-{i}-scores.npy"))
            random.append(np.load(res_dir / network / 'random' / f"{subject}-{i}-scores.npy"))
        
        all_pattern = np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy")
        all_patterns.append(all_pattern)
        all_random = np.load(res_dir / network / 'random' / f"{subject}-all-scores.npy")
        all_randoms.append(all_random)
        
        patterns.append(np.array(pattern))
        randoms.append(np.array(random))
    
    all_patterns = np.array(all_patterns)
    all_randoms = np.array(all_randoms)
    all_contrast = all_patterns - all_randoms
    
patterns = np.array(patterns)
randoms = np.array(randoms)
contrast = patterns - randoms

# plot average for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    # pval = gat_stats(all_contrast, -1)
    # ensure_dir(res_dir / network / "pval")
    # np.save(res_dir / network / "pval" / "all_contrast-pval.npy", pval)
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = np.array(pval < 0.05)
    im = axes[i].imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
# plot average per network
for name, network in zip(names_corrected, networks):
    ensure_dir(figures_dir / network)
    # Load data
    all_patterns, all_randoms = [], []
    for subject in subjects:
        all_pattern = np.load(res_dir / network / 'all_pattern' / f"{subject}-all-scores.npy")
        all_patterns.append(all_pattern)
        all_random = np.load(res_dir / network / 'all_random' / f"{subject}-all-scores.npy")
        all_randoms.append(all_random)
    all_patterns = np.array(all_patterns)
    all_randoms = np.array(all_randoms)
    all_contrast = all_patterns - all_randoms
    # pval = gat_stats(all_contrast, -1)
    # ensure_dir(res_dir / network / "pval")
    # np.save(res_dir / network / "pval" / "all_contrast-pval.npy", pval)
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = np.array(pval < 0.05)
    # Plot all_pattern
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
    im = ax.imshow(
        all_patterns.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0,
        vmax=.50)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"${name}$", fontsize=10)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    ax.set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(figures_dir / network / "all_pattern.pdf")
    plt.close()
    # Plot all_random
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
    im = ax.imshow(
        all_randoms.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0,
        vmax=.50)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"${name}$", fontsize=10)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    ax.set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(figures_dir / network / "all_random.pdf")
    plt.close()
    # Plot all_contrast
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
    im = ax.imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"${name}$", fontsize=10)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    ax.set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    fig.savefig(figures_dir / network / "all_contrast.pdf")
    plt.close()

for name, network in zip(names_corrected, networks):
    for sess in [0, 1, 2, 3, 4]:
        fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
        im = ax.imshow(
            patterns[:, sess].mean(0),
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.5,
            vmin=0,
            vmax=.50)
        ax.set_ylabel("Training Time (s)")
        ax.set_title(f"${name}$ - Session {sess}",)
        ax.axvline(0, color="k", alpha=.5)
        ax.axhline(0, color="k", alpha=.5)
        ax.set_xlabel("Testing Time (s)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")
        fig.savefig(figures_dir / network / f"pattern_{sess}.pdf")
        plt.close()

for name, network in zip(names_corrected, networks):
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
    im = ax.imshow(
        patterns[:, 1:].mean((0, 1)),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0,
        vmax=.50)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"${name}$",)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    ax.set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(figures_dir / network / f"pattern_{sess}_no_prac.pdf")
    plt.close()
