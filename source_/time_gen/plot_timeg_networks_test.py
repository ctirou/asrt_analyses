import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, spearmanr, zscore
import numba
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import seaborn as sns

data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR

lock = 'stim'
# network and custom label_names
n_parcels = 200
n_networks = 7
networks = schaefer_7[:-2] if n_networks == 7 else schaefer_17[:-2]
# networks = networks + ['Hippocampus', 'Thalamus']
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock
ensure_dir(figures_dir)
overwrite = False

names = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2]
# names = names + ['Hippocampus', 'Thalamus']
times = np.linspace(-1.5, 1.5, 305)
chance = .25
threshold = .05

# analysis = "tg_0206_emp"
# analysis = "tg_rs_emp"
analysis = "tg_rdm_emp"
# analysis = "tg_rdm_emp_reduced"
res_dir = data_path / analysis / lock

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
all_diags = {}
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
for network in networks:
    print(f"Processing {network}...")
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-{j}-scores.npy"))
            rand.append(np.load(res_dir / network / 'random' / f"{subject}-{j}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy"))
        all_rand.append(np.load(res_dir / network / 'random' / f"{subject}-all-scores.npy"))
        
        diag = np.array(all_pat) - np.array(all_rand)
        all_diag.append(np.diag(diag[i]))
        
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    all_diags[network] = np.array(all_diag)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

cmap = "viridis"
cmap1 = "RdBu_r"
cmap2 = 'PRGn_r'
cmap3 = 'magma'

### plot pattern for all networks ###
fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True, sharey=True, layout='tight')
fig.suptitle("Pattern", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    im = axes[i].imshow(
        all_patterns[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
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
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_pattern-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
fig.savefig(figures_dir / f"{analysis}_pattern.pdf", transparent=True)

### plot random for all networks ###
fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Random", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    im = axes[i].imshow(
        all_randoms[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
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
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_random-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
fig.savefig(figures_dir / f"{analysis}_random.pdf", transparent=True)

### plot contrast for all networks ###
fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Contrast = Pattern – Random", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = axes[i].imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.01,
        vmax=0.01)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
fig.savefig(figures_dir / f"{analysis}_contrast.pdf", transparent=True)

### plot diagonal for all networks ###
fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Contrast diagonal", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    axes[i].plot(times, all_diags[network].mean(0))
    pval = decod_stats(all_diags[network], -1)
    sig = pval < threshold
    axes[i].fill_between(times, 0, all_diags[network].mean(0), where=sig, color='C7', alpha=.5)
    axes[i].axhline(0, color='grey', alpha=.5)
    axes[i].set_ylabel("difference")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvspan(0, 0.2, color='gray', alpha=0.1)
fig.savefig(figures_dir / f"{analysis}_diagonal.pdf", transparent=True)

### plot blocks x time gen correlation for all networks ###
fig, axes = plt.subplots(7, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("blocks x time gen correlation", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    rhos = np.load(res_dir / network / "corr" / "rhos_blocks.npy")
    pval = np.load(res_dir / network / "corr" / "pval_blocks-pval.npy")
    sig = pval < threshold
    im = axes[i].imshow(
        rhos.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-.2,
        vmax=.2)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"{name}", style='italic')
    axes[i].axvline(0, color="k")
    axes[i].axhline(0, color="k")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
# fig.savefig(figures_dir / "blocks_corr.pdf", transparent=True)

### plot learn df x time gen correlation for all networks ###
fig, axes = plt.subplots(7, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("learn df x time gen correlation", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    rhos = np.load(res_dir / network / "corr" / "rhos_learn.npy")
    pval = np.load(res_dir / network / "corr" / "pval_learn-pval.npy")
    sig = pval < threshold
    im = axes[i].imshow(
        rhos.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-.2,
        vmax=.2)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"{name}", style='italic')
    axes[i].axvline(0, color="k")
    axes[i].axhline(0, color="k")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
fig.savefig(figures_dir / "learn_corr.pdf", transparent=True)

# ensure_dir(figures_dir / "per_session")
# ### plot session by session ###
# for network, name in zip(networks, names):
#     contrasts = patterns[network] - randoms[network]
#     fig, axes = plt.subplots(1, 5, sharey=True, figsize=(25, 3), layout='tight')
#     fig.suptitle(f"{name}", fontsize=14)
#     for i in range(5):
#         im = axes[i].imshow(
#             contrasts[:, i].mean(0),
#             interpolation="lanczos",
#             origin="lower",
#             cmap="RdBu_r",
#             extent=times[[0, -1, 0, -1]],
#             aspect=0.5,
#             vmin=-0.01,
#             vmax=0.01)
#         axes[i].set_xlabel("Testing Time (s)")
#         axes[i].set_title(f"Session {i}", fontsize=10)
#         axes[i].axvline(0, color="k", alpha=.5)
#         axes[i].axhline(0, color="k", alpha=.5)
#         if i == 0:
#             axes[i].set_ylabel("Training Time (s)")
#         # if i == 4:
#         #     cbar = plt.colorbar(im, ax=axes[i])
#         #     cbar.set_label("accuracy")
#     fig.savefig(figures_dir / "per_session" / f"{network}.pdf", transparent=True)
#     fig.close()

### mean effect ###
mean_net = []
mean_diag = []
mean_net_sess = []
mean_diag_sess = []
filter = np.where((times < 0) & (times > -0.5))[0]
filt = np.where((times >= .2) & (times <= 0.7))[0]
for net in networks:
    sess1, sess2 = [], []
    contrast = all_patterns[net] - all_randoms[net]
    mean_net.append(contrast[:, filter, filter].mean())
    mean_diag.append(all_diags[net][:, filt].mean())
    for sub in range(len(subjects)):
        sess1.append(contrast[sub, filter, filter].mean())
        sess2.append(all_diags[net][sub, filt].mean())
    mean_net_sess.append(np.array(sess1))
    mean_diag_sess.append(np.array(sess2))
mean_net_sess = np.array(mean_net_sess)
mean_diag_sess = np.array(mean_diag_sess)

from scipy.stats import ttest_1samp
pvals = []
tvals = []
for i in range(mean_net_sess.shape[0]):
    t, p = ttest_1samp(mean_net_sess[i], 0)
    pvals.append(p)
    tvals.append(t)
pvals = np.array(pvals)
sig = pvals < 0.05

fig, ax = plt.subplots(1, 1, figsize=(7, 8), layout='tight')
cmap = sns.color_palette("colorblind")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Avenir'})
ax.bar(range(len(networks)), mean_net, label=names, color=cmap)
ax.set_xticks(range(len(networks)))
ax.set_xticklabels(names, rotation=45, ha='right')
# ax.set_xlabel('Networks')
ax.set_ylabel('Mean effect')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
# ax.legend(frameon=False)
ax.set_title('Mean prediction effect per network and region', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, s in enumerate(sig):
    if s:
        ax.text(i, mean_net[i], '*', ha='center', va='bottom', fontsize=14, color='black')
ax.text(3, 0.004, f'{analysis}', ha='left', va='top', fontsize=14, color='black')
fig.savefig(figures_dir / f"{analysis}_mean_effect.pdf", transparent=True)

# fig, ax = plt.subplots(1, 1, figsize=(7, 8), layout='tight')
# cmap = sns.color_palette("colorblind")
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Avenir'})
# # ax.bar(range(len(networks)), mean_net, label=names, color=cmap)
# ax.bar(range(len(networks)), mean_diag, label=names, color=cmap)
# ax.set_xticks(range(len(networks)))
# ax.set_xticklabels(names, rotation=45, ha='right')
# # ax.set_xlabel('Networks')
# ax.set_ylabel('Mean effect')
# ax.axhline(0, color='black', linestyle='-', linewidth=1)
# # ax.legend(frameon=False)
# ax.set_title('Mean prediction effect per network and region', fontsize=16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

