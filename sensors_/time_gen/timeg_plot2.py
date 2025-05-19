import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import *
from config import *
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm

lock = 'stim'
analysis = 'pat_high_rdm_high'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS15, EPOCHS

timesg = np.linspace(-1.5, 1.5, 307)

networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']

figures_dir = FIGURES_DIR / "test_plots"
ensure_dir(figures_dir)

threshold = 0.05
chance = 0.25
cmap = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#ECE133', '#56B4E9', '#76B041']

# --- TEMPORAL GENERALIZATION 40-TRIAL BINS ---
res_path = RESULTS_DIR / 'TIMEG' / 'source'

# data
all_patterns, all_randoms = {}, {}
for network in networks:
    if network not in all_patterns:
        all_patterns[network], all_randoms[network] = [], []
    for subject in subjects:    
        pattern, random = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            pats, rands = [], []
            for block in blocks:
                p, r = [], []
                for fold in range(1, 3):
                    p.append(np.load(res_path / network / 'scores_40s' / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    r.append(np.load(res_path / network / 'scores_40s' / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pats.append(np.array(p))
                rands.append(np.array(r))
            pattern.append(np.mean(pats, 0))
            random.append(np.mean(rands, 0))
        pattern = np.array(pattern)
        random = np.array(random)
        if epoch_num != 0:
            all_patterns[network].append(pattern.mean((0, 1)))
            all_randoms[network].append(random.mean((0, 1)))
    all_patterns[network] = np.array(all_patterns[network])
    all_randoms[network] = np.array(all_randoms[network])
    
cmap1 = "RdBu_r"
c1 = "#20B2AA"
c1 = "#00BFA6"
c1 = "#708090"

# Pattern
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    # im = axes[i].imshow(
    im = ax.imshow(
        all_patterns[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_path / network / "pval-40s" / "all_pattern-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='--', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Pattern - 40-trial bins", fontsize=12)
# fig.savefig(figures_dir / "timeg-pattern.pdf", transparent=True)
# plt.close(fig)

# Random
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    im = ax.imshow(
        all_randoms[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_path / network / "pval-40s" / "all_random-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='--', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Random - 40-trial bins", fontsize=12)
# fig.savefig(figures_dir / "timeg-random.pdf", transparent=True)
# plt.close(fig)

# Contrast
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = ax.imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_path / network / "pval-40s" / "all_contrast-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0], linestyles='-', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast - 40-trial bins", fontsize=12)
# fig.savefig(figures_dir / "timeg-contrast.pdf", transparent=True)
# plt.close(fig)

# --- TEMPORAL GENERALIZATION STRATIFIED KFOLD ---
res_dir = RESULTS_DIR / 'TIMEG' / 'source' 
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
all_diags = {}
for network in tqdm(networks):
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'scores_skf' / subject / f"pat-{j}.npy"))
            rand.append(np.load(res_dir / network / 'scores_skf' / subject / f"rand-{j}.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_dir / network / 'scores_skf' / subject / "pat-all.npy"))
        all_rand.append(np.load(res_dir / network / 'scores_skf' / subject / "rand-all.npy"))
        
        diag = np.array(all_pat) - np.array(all_rand)
        all_diag.append(np.diag(diag[i]))

    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    all_diags[network] = np.array(all_diag)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

# Pattern
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    # im = axes[i].imshow(
    im = ax.imshow(
        all_patterns[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval-skf" / "all_pattern-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='--', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Pattern - stratified kfold", fontsize=12)
# fig.savefig(figures_dir / "timeg-pattern.pdf", transparent=True)
# plt.close(fig)

# Random
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    im = ax.imshow(
        all_randoms[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval-skf" / "all_random-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='--', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Random - stratified kfold", fontsize=12)
# fig.savefig(figures_dir / "timeg-random.pdf", transparent=True)
# plt.close(fig)

# Contrast
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = ax.imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval-skf" / "all_contrast-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='--', linewidths=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast - stratified kfold", fontsize=12)
# fig.savefig(figures_dir / "timeg-contrast.pdf", transparent=True)
# plt.close(fig)

