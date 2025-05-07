import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp

subjects = SUBJS
subjects = ALL_SUBJS
lock = 'stim'
times = np.linspace(-0.2, 0.6, 82)

# --- notes --- 
# 1. find out why changes between get_all_high_low and get_all_high_low_blocks
# 3. plot the mean value per block/2

# win = np.where(times >= 0.2)[0]
win = np.where((times >= 0.28) & (times <= 0.51))[0]
# win = np.load(FIGURES_DIR / "RSA" / "sensors" / "sig_rsa.npy")
c1, c2 = "#5BBCD6", "#00A08A"

# --- RSA no shuffle ---
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "kf2_no_shfl" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    pattern_blocks, random_blocks = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            p, r = [], []
            for fold in [1, 2]:
                p.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                r.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
            pats.append(np.array(p))
            rands.append(np.array(r))
        pattern.append(np.nanmean(pats, 0))
        random.append(np.nanmean(rands, 0))
        pattern_blocks.append(np.vstack(pats))
        random_blocks.append(np.vstack(rands))
    pattern = np.array(pattern).mean(1)
    random = np.array(random).mean(1)
    pat, rand = get_all_high_low(pattern, random, sequence, False)
    all_pats.append(pat.mean(0))
    all_rands.append(rand.mean(0))
    pattern_blocks = np.vstack(pattern_blocks)
    random_blocks = np.vstack(random_blocks)
    pat_blocks, rand_blocks = get_all_high_low(pattern_blocks, random_blocks, sequence)
    all_pats_blocks.append(pat_blocks)
    all_rands_blocks.append(rand_blocks)
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)
all_pats_blocks = np.array(all_pats_blocks)
all_rands_blocks = np.array(all_rands_blocks)

fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=False, sharey=True, layout='tight')
for ax in axes.flatten():
#     ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)

# w/o practice bsl
pat = np.nanmean(all_pats[:, 1:, :], 1)
rand = np.nanmean(all_rands[:, 1:, :], 1)
diff_rp = rand - pat

axes[0, 0].plot(times, diff_rp.mean(0), color=c1)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
axes[0, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2)
axes[0, 0].set_title("no shuffle - w/o practice bsl", fontstyle='italic')
axes[1, 0].plot(times, pat.mean(0), label='pattern')
axes[1, 0].plot(times, rand.mean(0), label='random')
axes[1, 0].set_title("random vs pattern", fontstyle='italic')
axes[1, 0].legend(frameon=False)

if sig.any():
    pats_blocks = np.nanmean(all_pats_blocks[:, :, :, sig], (1, -1))
    rands_blocks = np.nanmean(all_rands_blocks[:, :, :, sig], (1, -1))
    diff_rp_blocks = rands_blocks - pats_blocks

    for i in [6, 16, 26, 36]:
        axes[2, 0].axvline(i, color='grey', linestyle='--', alpha=0.5)
    axes[2, 0].plot(np.arange(1, diff_rp_blocks.shape[1] + 1), diff_rp_blocks.mean(0), color=c2)
    axes[2, 0].set_title('40s trial bins', fontstyle='italic')
    axes[2, 0].set_xticks([i for i in range(1, 47, 5)])
    # axes[2, 0].set_xticks([6, 16, 26, 36])
    # axes[2, 0].set_xticklabels(['S1', 'S2', 'S3', 'S4'])
    for txt, xpos in zip(['S1', 'S2', 'S3', 'S4'], [6, 16, 26, 36]):
        axes[2, 0].text(xpos, 1.25, txt, ha='center', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

# w/ practice bsl
pat = np.nanmean(all_pats[:, 1:, :], 1) - all_pats[:, 0, :]
rand = np.nanmean(all_rands[:, 1:, :], 1) - all_rands[:, 0, :]
diff_rp = rand - pat

axes[0, 1].plot(times, diff_rp.mean(0), color=c1)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
pval_uncorr = ttest_1samp(diff_rp, 0, axis=0)[1]
sig_uncorr = pval_uncorr < 0.05

axes[0, 1].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2, label='corrected')
# axes[0, 1].fill_between(times, diff_rp.mean(0), 0, where=sig_uncorr, color='grey', alpha=0.2, label='uncorrected')
axes[0, 1].set_title("no shuffle - w/ practice bsl", fontstyle='italic')
axes[1, 1].plot(times, pat.mean(0), label='pattern')
axes[1, 1].plot(times, rand.mean(0), label='random')
axes[1, 1].set_title("random vs pattern", fontstyle='italic')
axes[1, 1].legend(frameon=False)

if sig.any():
    pats_blocks = np.nanmean(all_pats_blocks[:, :, :, sig], (1, -1))
    rands_blocks = np.nanmean(all_rands_blocks[:, :, :, sig], (1, -1))
    diff_rp_blocks = rands_blocks - pats_blocks
    for i in [6, 16, 26, 36]:
        axes[2, 1].axvline(i, color='grey', linestyle='--', alpha=0.5)
    axes[2, 1].plot(np.arange(1, diff_rp_blocks.shape[1] + 1), diff_rp_blocks.mean(0), color=c2)
    axes[2, 1].set_title('40s trial bins', fontstyle='italic')
    axes[2, 1].set_xticks([i for i in range(1, 47, 5)])
    # axes[2, 0].set_xticks([6, 16, 26, 36])
    # axes[2, 0].set_xticklabels(['S1', 'S2', 'S3', 'S4'])
    for txt, xpos in zip(['S1', 'S2', 'S3', 'S4'], [6, 16, 26, 36]):
        axes[2, 1].text(xpos, 1.25, txt, ha='center', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

# --- RSA shuffled ---
all_highs, all_lows = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm_fixed" / subject
    ensure_dir(res_path)
    # RSA stuff
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pats, rands = [], []
    for epoch_num in range(5):
        pats.append(np.load(res_path / f"pat-{epoch_num}.npy"))
        rands.append(np.load(res_path / f"rand-{epoch_num}.npy"))
    pats = np.array(pats)
    rands = np.array(rands)
    high, low = get_all_high_low(pats, rands, sequence, False)
    all_highs.append(high)
    all_lows.append(low)
all_highs = np.array(all_highs).mean(1)
all_lows = np.array(all_lows).mean(1)

# w/o practice bsl
pat = all_highs[:, 1:, :].mean(1)
rand = all_lows[:, 1:, :].mean(1)
diff_rp = rand - pat

fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=False, sharey=True, layout='tight')
for ax in axes.flatten():
    # ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
axes[0, 0].plot(times, diff_rp.mean(0), color=c1)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
pval_uncorr = ttest_1samp(diff_rp, 0, axis=0)[1]
sig_uncorr = pval_uncorr < 0.05
axes[0, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2, label='corrected')
# axes[0, 0].fill_between(times, diff_rp.mean(0), 0, where=sig_uncorr, color='grey', alpha=0.2, label='uncorrected')
axes[0, 0].set_title("shuffled - w/o practice bsl", fontstyle='italic')
axes[1, 0].plot(times, pat.mean(0), label='pattern')
axes[1, 0].plot(times, rand.mean(0), label='random')
axes[1, 0].set_title("random vs pattern", fontstyle='italic')
axes[1, 0].legend(frameon=False)

# w/ practice bsl
pat = all_highs[:, 1:, :].mean(1) - all_highs[:, 0, :]
rand = all_lows[:, 1:, :].mean(1) - all_lows[:, 0, :]
diff_rp = rand - pat

axes[0, 1].plot(times, diff_rp.mean(0), color=c1)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
pval_uncorr = ttest_1samp(diff_rp, 0, axis=0)[1]
sig_uncorr = pval_uncorr < 0.05
axes[0, 1].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2, label='corrected')
# axes[0, 1].fill_between(times, diff_rp.mean(0), 0, where=sig_uncorr, color='grey', alpha=0.2, label='uncorrected')
axes[0, 1].set_title("shuffled - w/ practice bsl", fontstyle='italic')
axes[1, 1].plot(times, pat.mean(0), label='pattern')
axes[1, 1].plot(times, rand.mean(0), label='random')
axes[1, 1].set_title("random vs pattern", fontstyle='italic')
axes[1, 1].legend(frameon=False)

# --- Source ---
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
threshold = 0.05
chance = 0.25
cmap = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#ECE133', '#56B4E9', '#76B041']

# --------- No shuffle ---------
all_highs, all_lows = {}, {}
for network in tqdm(networks):
    if not network in all_highs:        
        all_highs[network] = []
        all_lows[network] = []
    for subject in subjects: 
        # RSA stuff
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / lock / "kf2_no_shfl" / subject
        sequence = get_sequence(behav_dir)
        pattern, random = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            pats, rands = [], []
            for block in blocks:
                p, r = [], []
                for fold in [1, 2]:
                    p.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    r.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pats.append(np.array(p))
                rands.append(np.array(r))
            pattern.append(np.nanmean(pats, 0))
            random.append(np.nanmean(rands, 0))
        pattern = np.vstack(pattern)
        random = np.vstack(random)
        pat, rand = get_all_high_low(pattern, random, sequence, False)
        all_highs[network].append(pat.mean(0))
        all_lows[network].append(rand.mean(0))
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])

# --- w/o practice bsl ---
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, network_names)):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    diff = np.nanmean(all_lows[label][:, 6:, :], 1) - np.nanmean(all_highs[label][:, 6:, :], 1)
    pval = decod_stats(diff, -1)
    sig = pval < threshold
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
    ax.set_title(name, fontstyle='italic')
fig.suptitle("No shufffle – w/o practice bsl")

# --- w/ practice bsl ---
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, network_names)):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    low = np.nanmean(all_lows[label][:, 6:, :], 1) - np.nanmean(all_lows[label][:, :6, :], 1)
    high = np.nanmean(all_highs[label][:, 6:, :], 1) - np.nanmean(all_highs[label][:, :6, :], 1)
    diff = low - high
    pval = decod_stats(diff, -1)
    sig = pval < threshold
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
    ax.set_title(name, fontstyle='italic')
fig.suptitle("No shuffle – w/ practice bsl")

# --------- Shuffled ---------
# Load RSA data
all_highs, all_lows = {}, {}
for network in tqdm(networks):
    if not network in all_highs:
        all_highs[network] = []
        all_lows[network] = []
    for subject in subjects:        
        # RSA stuff
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / lock / 'power_rdm_fixed' / subject
        pats, rands = [], []
        for epoch_num in range(5):
            pats.append(np.load(res_path / f"pat-{epoch_num}.npy"))
            rands.append(np.load(res_path / f"rand-{epoch_num}.npy"))
        pats = np.array(pats)
        rands = np.array(rands)
        high, low = get_all_high_low(pats, rands, sequence, False)
        all_highs[network].append(high.mean(0))
        all_lows[network].append(low.mean(0))
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    
# --- w/o practice bsl ---
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, network_names)):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    diff = np.nanmean(all_lows[label][:, 1:, :], 1) - np.nanmean(all_highs[label][:, 1:, :], 1)
    pval = decod_stats(diff, -1)
    sig = pval < threshold
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
    ax.set_title(name, fontstyle='italic')
fig.suptitle("Shuffled – w/o practice bsl")

# --- w/ practice bsl ---
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharex=True, sharey=True, layout='tight')
for i, (ax, label, name) in enumerate(zip(axes.flat, networks, network_names)):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    ax.axhline(0, color='grey', alpha=.5)
    low = np.nanmean(all_lows[label][:, 1:, :], 1) - all_lows[label][:, 0, :]
    high = np.nanmean(all_highs[label][:, 1:, :], 1) - all_highs[label][:, 0, :]
    diff = low - high
    pval = decod_stats(diff, -1)
    sig = pval < threshold
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
    ax.set_title(name, fontstyle='italic')
fig.suptitle("Shuffled – w/ practice bsl")
