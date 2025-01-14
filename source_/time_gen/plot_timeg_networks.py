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

lock = 'stim'
# network and custom label_names
n_parcels = 200
n_networks = 7
networks = schaefer_7[:-2] if n_networks == 7 else schaefer_17[:-2]
res_dir = data_path / 'results' / 'source' / lock / f'networks_{n_parcels}_{n_networks}'
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock / f"networks_{n_parcels}_{n_networks}"
ensure_dir(figures_dir)

names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2]
times = np.linspace(-1.5, 1.5, 305)

# Load data
all_diags = {}
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
for network in networks:
    print(f"Processing {network}...")
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
        all_diags[network] = []
    all_pat, all_rand = [], []
    patpat, randrand = [], []
    diags =[]
    for subject in subjects:
        pat, rand = [], []
        for i in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-{i}-scores.npy"))
            rand.append(np.load(res_dir / network / 'random' / f"{subject}-{i}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
        all_pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy"))
        all_rand.append(np.load(res_dir / network / 'random' / f"{subject}-all-scores.npy"))
        diags.append(np.diag(np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy")))
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    all_diags[network] = np.array(diags)
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)
    
# plot average for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    all_contrast = all_patterns[network] - all_randoms[network]
    # pval = gat_stats(all_contrast, -1)
    # ensure_dir(res_dir / network / "pval")
    # np.save(res_dir / network / "pval" / "all_contrast-pval.npy", pval)
    # pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    # sig = np.array(pval < 0.05)
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
    # xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    # axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
    #                     linestyles='solid', linewidths=1)
fig.savefig(figures_dir / "all_contrast.pdf", transparent=True)

# plot pattern for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    im = axes[i].imshow(
        all_patterns[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0,
        vmax=0.5)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
fig.savefig(figures_dir / "all_pattern.pdf", transparent=True)

# plot pattern for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    im = axes[i].imshow(
        patterns[network].mean((0, 1)),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")

# plot pattern for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    diags = np.array([np.diag(patterns[network][sub, :, :].mean(1)) for sub in range(len(subjects))])
    axes[i].axhline(.25, color="k", alpha=.5)
    axes[i].axvspan(0, 0.2, color="grey", alpha=.1)
    axes[i].plot(times, np.diag(patterns[network].mean((0, 1))))
    axes[i].set_title(f"${name}$", fontsize=10)
    # pval = decod_stats(diags - .25, -1)
    # sig = pval < .05
    # axes[i].fill_between(times, .25, np.diag(patterns[network].mean((0, 1))), where=sig, color="red", alpha=.4)

# plot diag for all networks
fig, axes = plt.subplots(5, 1, figsize=(6, 11), sharex=True, layout='tight')
for i, (network, name) in enumerate(zip(networks, names_corrected)):
    axes[i].plot(times, all_diags[network].mean(0))
    # pval = decod_stats(all_diags[network] - .25, -1)
    # sig = pval < .05
    axes[i].axhline(.25, color="k", alpha=.5)
    axes[i].axvspan(0, 0.2, color="grey", alpha=.1)
    # axes[i].fill_between(times, .25, all_diags[network].mean(0), where=sig, color="C7")
    axes[i].set_title(f"${name}$", fontsize=10)
fig.savefig(figures_dir / "all_diag.pdf", transparent=True)

# plot average per network
for name, network in zip(names_corrected, networks):
    ensure_dir(figures_dir / network)
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), layout='tight')
    im = ax.imshow(
        all_patterns[network].mean(0),
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
        all_randoms[network].mean(0),
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
    all_contrast = all_patterns[network] - all_randoms[network]
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
