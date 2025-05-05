import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = ALL_SUBJS
lock = 'stim'

# --- notes --- 
# 1. find out why changes between get_all_high_low and get_all_high_low_blocks
# 3. plot the mean value per block/2

times = np.linspace(-0.2, 0.6, 82)
# win = np.where(times >= 0.2)[0]
win = np.where((times >= 0.28) & (times <= 0.51))[0]
# win = np.load(FIGURES_DIR / "RSA" / "sensors" / "sig_rsa.npy")
c1, c2 = "#D67236", "#7294D4"

# --- RSA no shuffle ---
all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_no_shfl" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 7)] if epoch_num == 0 else [i for i in range(1, 11)]
        pats, rands = [], []
        for block in blocks:
            pats.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
            rands.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
        pattern.append(np.nanmean(pats, 0))
        random.append(np.nanmean(rands, 0))
    pattern = np.array(pattern)
    random = np.array(random)
    pat_blocks, rand_blocks = get_all_high_low_blocks(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)    

pat = np.nanmean(all_pats[:, 1:, :], 1)
rand = np.nanmean(all_rands[:, 1:, :], 1)
diff_rp = rand - pat

fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey=True, layout='tight')
for ax in axes.flatten():
    ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
axes[0, 0].plot(times, diff_rp.mean(0), color=c1, linewidth=2)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
axes[0, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2)
axes[0, 0].set_title("no shuffle - w/o practice bsl", fontstyle='italic')
axes[0, 1].plot(times, pat.mean(0), label='pattern')
axes[0, 1].plot(times, rand.mean(0), label='random')
axes[0, 1].set_title("random vs pattern", fontstyle='italic')
axes[0, 1].legend(frameon=False)

pat = np.nanmean(all_pats[:, 1:, :], 1) - all_pats[:, 0, :]
rand = np.nanmean(all_rands[:, 1:, :], 1) - all_rands[:, 0, :]
diff_rp = rand - pat

axes[1, 0].plot(times, diff_rp.mean(0), color=c2, linewidth=2)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
axes[1, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c2, alpha=0.2)
axes[1, 0].set_title("no shuffle - w/ practice bsl", fontstyle='italic')
axes[1, 1].plot(times, pat.mean(0), label='pattern')
axes[1, 1].plot(times, rand.mean(0), label='random')
axes[1, 1].set_title("random vs pattern", fontstyle='italic')
axes[1, 1].legend(frameon=False)

# --- RSA shuffled ---
all_highs, all_lows = [], []
analysis = "pat_high_rdm_high"
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm_fixed" / subject
    ensure_dir(res_path)
    # RSA stuff
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
    # high, low = get_all_high_low_old(res_path, sequence, analysis, cv=True)    
    all_highs.append(high)
    all_lows.append(low)
all_highs = np.array(all_highs).mean(1)
all_lows = np.array(all_lows).mean(1)

pat = all_highs[:, 1:, :].mean(1)
rand = all_lows[:, 1:, :].mean(1)
diff_rp = rand - pat

fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey=True, layout='tight')
for ax in axes.flatten():
    ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
axes[0, 0].plot(times, diff_rp.mean(0), color=c1, linewidth=2)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
axes[0, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c1, alpha=0.2)
axes[0, 0].set_title("shuffled - w/o practice bsl", fontstyle='italic')
axes[0, 1].plot(times, pat.mean(0), label='pattern')
axes[0, 1].plot(times, rand.mean(0), label='random')
axes[0, 1].set_title("random vs pattern", fontstyle='italic')
axes[0, 1].legend(frameon=False)

pat = np.nanmean(all_pats[:, 1:, :], 1) - all_pats[:, 0, :]
rand = np.nanmean(all_rands[:, 1:, :], 1) - all_rands[:, 0, :]
diff_rp = rand - pat

axes[1, 0].plot(times, diff_rp.mean(0), color=c2, linewidth=2)
pval = decod_stats(diff_rp, -1)
sig = pval < 0.05
axes[1, 0].fill_between(times, diff_rp.mean(0), 0, where=sig, color=c2, alpha=0.2)
axes[1, 0].set_title("shuffled - w/ practice bsl", fontstyle='italic')
axes[1, 1].plot(times, pat.mean(0), label='pattern')
axes[1, 1].plot(times, rand.mean(0), label='random')
axes[1, 1].set_title("random vs pattern", fontstyle='italic')
axes[1, 1].legend(frameon=False)
