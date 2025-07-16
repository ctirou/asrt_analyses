import os
from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, spearmanr
import pandas as pd
import matplotlib.colors as mcolors

subjects, subjects_dir = SUBJS, FREESURFER_DIR

data_type1 = "scores_skf_vect"
data_type2 = "scores_skf_vect_new"
networks = NETWORKS + ["Cerebellum-Cortex"]
res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source')

figures_dir = FIGURES_DIR / "temp"

names = NETWORK_NAMES + ["Cerebellum"]
times = np.linspace(-1.5, 1.5, 307)
chance = .25
threshold = .05

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
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
            pat.append(np.load(res_path / network / data_type2 / subject /  f"pat-{j}.npy"))
            rand.append(np.load(res_path / network / data_type2 / subject /  f"rand-{j}.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_path / network / data_type1 / subject /  "pat-all.npy"))
        all_rand.append(np.load(res_path / network / data_type1 / subject /  "rand-all.npy"))
        
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

cmap = "viridis"
cmap = mcolors.LinearSegmentedColormap.from_list("Zissou1", colors["Zissou1"])
cmap = "RdBu_r"

### plot learn df x time gen correlation for all networks ###
fig, axes = plt.subplots(2, 5, figsize=(12, 6), sharex=True, layout='tight')
fig.suptitle("learn df x time gen correlation", fontsize=14)
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, names)):
    rhos = np.load(res_path / network / data_type2 / "corr" / "rhos_learn.npy")
    pval = np.load(res_path / network / data_type2 / "corr" / "pval_learn-pval.npy")
    sig = pval < threshold
    im = ax.imshow(
        rhos.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-.2,
        vmax=.2)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"{name}", style='italic')
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax.contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    if i == 10:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")

fig.savefig(figures_dir / "learn_corr.pdf", transparent=True)

# rhos diags
# fig, axes = plt.subplots(2, 5, figsize=(12, 6),sharex=True, layout='tight')
# for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, names)):
ensured(figures_dir / "timeg_corr_diags")
win = np.where((times >= -0.3) & (times < 0))[0]
for i, (network, name) in enumerate(zip(networks, names)):
    fig, ax = plt.subplots(figsize=(10, 2), layout='tight')
    rhos = np.load(res_path / network / data_type2 / "corr" / "rhos_learn.npy")
    r = np.array([np.diag(rho) for rho in rhos])
    ax.plot(times, r.mean(0), label=name)
    pval = decod_stats(r, -1)
    sig = pval < threshold
    ax.fill_between(times, r.mean(0), 0, where=sig, alpha=0.5, color='red')
    ax.set_title(f"{name}", style='italic')
    ax.axhline(0, color="k", alpha=.5)
    ax.axvline(0, color="k", alpha=.5)
    r_mean = r[:, win].mean(1)
    sig_uncorr = ttest_1samp(r_mean, 0)[1] < threshold
    if sig_uncorr:
        ax.fill_between(times[win], -0.4, -0.39, alpha=0.5, color='red')
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Spearman's rho")
    ax.set_ylim(-0.5, 0.5)
    fig.savefig(figures_dir / "timeg_corr_diags" / f"{network}.pdf", transparent=True)
    plt.close()
    
### plot session by session ###
ensure_dir(figures_dir / "per_session")
for network, name in zip(networks, names):
    contrasts = patterns[network] - randoms[network]
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=(25, 3), layout='tight')
    fig.suptitle(f"{name}", fontsize=14)
    for i in range(5):
        im = axes[i].imshow(
            contrasts[:, i].mean(0),
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.5,
            vmin=-0.01,
            vmax=0.01)
        axes[i].set_xlabel("Testing Time (s)")
        axes[i].set_title(f"Session {i}", fontsize=10)
        axes[i].axvline(0, color="k", alpha=.5)
        axes[i].axhline(0, color="k", alpha=.5)
        if i == 0:
            axes[i].set_ylabel("Training Time (s)")
        # if i == 4:
        #     cbar = plt.colorbar(im, ax=axes[i])
        #     cbar.set_label("accuracy")
    fig.savefig(figures_dir / "per_session" / f"{network}.pdf", transparent=True)
    plt.close()