import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import *
from config import *
from mne import read_epochs
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from matplotlib.ticker import FuncFormatter
from pathlib import Path

lock = 'stim'
analysis = 'pat_high_rdm_high'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS

times = np.linspace(-.2, .6, 82)
timesg = np.linspace(-1.5, 1.5, 307)

n_parcels = 200
n_networks = 17
networks = NETWORKS[:-2]

ori = "power"
figures_dir = FIGURES_DIR / "RSA" / "source" / lock / ori
ensure_dir(figures_dir)

# Load RSA data
all_highs, all_lows = {}, {}
diff_sess = {}
for network in networks:
    print(f"Processing {network}...")
    if not network in diff_sess:        
        all_highs[network] = []
        all_lows[network] = []
        diff_sess[network] = []
    for subject in subjects:        
        # RSA stuff
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        # home = Path("/Users/coum/MEGAsync/RSA")
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / lock / f'{ori}_rdm' / subject
        high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
        all_highs[network].append(high)    
        all_lows[network].append(low)
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    for i in range(5):
        rev_low = all_lows[network][:, :, i, :].mean(1) - all_lows[network][:, :, 0, :].mean(axis=1)
        rev_high = all_highs[network][:, :, i, :].mean(1) - all_highs[network][:, :, 0, :].mean(axis=1)
        diff_sess[network].append(rev_low - rev_high)
    diff_sess[network] = np.array(diff_sess[network]).swapaxes(0, 1)

threshold = 0.05
chance = 0.25
cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9']

### Plot similarity index ###
fig, axes = plt.subplots(2, 4, figsize=(12, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, NETWORK_NAMES)):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axvspan(0.28, 0.51, facecolor='green', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    high = all_highs[label][:, :, 1:, :].mean((1, 2)) - all_highs[label][:, :, 0, :].mean(1)
    low = all_lows[label][:, :, 1:, :].mean((1, 2)) - all_lows[label][:, :, 0, :].mean(axis=1)
    diff = low - high
    p_values = decod_stats(diff, jobs)
    sig = p_values < threshold
    # Main plot
    ax.plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], diff.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    ax.fill_between(times, diff.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    # ax.set_ylabel('Sim. index', fontsize=11)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    # axd[j].set_xticklabels([])
    # ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title(name)
    ax.set_ylim(-.5, 2)
    # ax.axhline(0.51, color='grey', alpha=.5)
# plt.show()
fig.suptitle(f"Similarity index – ori=${ori}$")
fig.savefig(figures_dir / f"similarity.pdf", transparent=True)
plt.close(fig)

### Plot similarity index x learning index corr ###
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices2.csv', sep="\t", index_col=0)
fig, axes = plt.subplots(2, 4, figsize=(12, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, NETWORK_NAMES)):
    ax.axvspan(0.28, 0.51, facecolor='green', edgecolor=None, alpha=.1)
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[label][sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    # axd[j].plot(times, all_rhos.mean(0), color=cmap[i])
    p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(all_rhos, -1)
    sig = p_values < 0.05
    # Main plot
    ax.plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    ax.fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    # ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=cmap[i], alpha=0.2)
    # ax.fill_between(times, 0, all_rhos.mean(0) - sem, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")    
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=cmap[i], alpha=0.2)
    # ax.fill_between(times, all_rhos.mean(0) - sem, 0, where=sig_unc, alpha=.3, label='uncorrected', facecolor="#7294D4")
    # ax.fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=.4, facecolor="#F2AD00", label='corrected')
    # ax.set_ylabel("Rho", fontsize=11)
    ax.set_title(name)
    # ax.set_ylim(-.5, .5)
    # ax.set_yticks([-.5, 0, .5])
    # ax.legend()
# plt.show()
fig.suptitle(f"Similarity index x learning index correlation – ori=${ori}$")
fig.savefig(figures_dir / "corr_learning.pdf", transparent=True)
plt.close(fig)

### Plot decoding performance ###
time_filter = np.where((timesg >= -0.2) & (timesg <= 0.6))[0]
pattern, random = {}, {}
for network in networks:
    if not network in pattern:
        pattern[network] = []
        random[network] = []
    for subject in subjects[:-2]:
        # pat = np.load(RESULTS_DIR / 'RSA' / 'source' / network / lock / 'max-power' / 'pattern' / f"{subject}-scores.npy")
        pat = np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'pattern' / f"{subject}-all-scores.npy")
        pat_diag = np.diag(pat)[time_filter]
        pattern[network].append(pat_diag)
        # rand = np.load(RESULTS_DIR / 'RSA' / 'source' / network / lock / 'max-power' / 'random' / f"{subject}-scores.npy")
        rand = np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'random' / f"{subject}-all-scores.npy")
        rand_diag = np.diag(rand)[time_filter]
        random[network].append(rand_diag)
    pattern[network] = np.array(pattern[network])
    random[network] = np.array(random[network])

fig, axes = plt.subplots(2, 4, figsize=(12, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, NETWORK_NAMES)):
    data = pattern[label]
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(.25, color='grey', alpha=.5)
    # Get significant clusters
    p_values = decod_stats(data - chance, -1)
    sig = p_values < threshold
    # Main plot
    ax.plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(data, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])    
    ax.fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])    
    ax.axhline(chance, color='grey', alpha=.5)
    ax.set_ylabel('Acc. (%)', fontsize=11)
    ax.set_ylim(0.2, 0.5)
    ax.set_title(name)
fig.suptitle(f"Pattern trials decoding – ori=${ori}$")
fig.savefig(figures_dir / "pat-decoding.pdf", transparent=True)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(12, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, NETWORK_NAMES)):
    data = random[label]
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(.25, color='grey', alpha=.5)
    # Get significant clusters
    p_values = decod_stats(data - chance, -1)
    sig = p_values < threshold
    # Main plot
    ax.plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(data, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])    
    ax.fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])    
    ax.axhline(chance, color='grey', alpha=.5)
    ax.set_ylabel('Acc. (%)', fontsize=11)
    ax.set_ylim(0.2, 0.5)
    ax.set_title(name)
fig.suptitle(f"Random trials decoding – ori=${ori}$")
fig.savefig(figures_dir / "rand-decoding.pdf", transparent=True)
plt.close(fig)

# Temporal generalization
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
for network in tqdm(networks):
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects[:-2]):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'pattern' / f"{subject}-{j}-scores.npy"))
            rand.append(np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'random' / f"{subject}-{j}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'pattern' / f"{subject}-all-scores.npy"))
        all_rand.append(np.load(TIMEG_DATA_DIR / 'results' / 'source' / 'max-power' / network / 'random' / f"{subject}-all-scores.npy"))
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

# Pattern
cmap1 = "RdBu_r"
fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True, layout='constrained')
for ax, network, name in zip(axes.flatten(), networks, NETWORK_NAMES):
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
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        ax.set_xlabel("Testing Time (s)")
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("accuracy")
# fig.savefig(figures_dir / f"pattern.pdf", transparent=True)

# Random
fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True, layout='constrained')
for ax, network, name in zip(axes.flatten(), networks, NETWORK_NAMES):
    im = ax.imshow(
        all_randoms[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    ax[i].set_ylabel("Training Time (s)")
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    ax[i].axvline(0, color="k", alpha=.5)
    ax[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        ax[i].set_xlabel("Testing Time (s)")
    # cbar = plt.colorbar(im, ax=axes[i])
    # cbar.set_label("accuracy")
# fig.suptitle("Random", fontsize=14)
# fig.savefig(figures_dir / f"random.pdf", transparent=True)

# Contrast
fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True, layout='constrained')
for ax, network, name in zip(axes.flatten(), networks, NETWORK_NAMES):
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
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        ax.set_xlabel("Testing Time (s)")
    # cbar = plt.colorbar(im, ax=axes[i])
    # cbar.set_label("accuracy")
# fig.savefig(figures_dir / f"contrast.pdf", transparent=True)
