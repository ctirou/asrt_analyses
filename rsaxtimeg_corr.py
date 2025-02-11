import os.path as op
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, zscore, spearmanr as spear
from tqdm.auto import tqdm
from base import *
from config import *

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

data = 'cv'
analysis = 'pat_high_rdm_high'
lock = 'stim'

figures_dir = FIGURES_DIR

# get times
times = np.load(data_path / "times.npy")
timesg = np.linspace(-1.5, 1.5, 305)

all_highs, all_lows = [], []
patterns, randoms = [], []

for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"{data}_rdm" / subject
    ensure_dir(res_path)
        
    # RSA stuff
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
    sequence = get_sequence(behav_dir)
    high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
    all_highs.append(high)    
    all_lows.append(low)
    # Time generalization stuff    
    pat, rand = [], []
    timeg_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
    for i in range(5):
        pat.append(np.load(timeg_path / "pattern" / f"{subject}-{i}-scores.npy"))
        rand.append(np.load(timeg_path / "random" / f"{subject}-{i}-scores.npy"))
    patterns.append(np.array(pat))
    randoms.append(np.array(rand))
patterns = np.array(patterns)
randoms = np.array(randoms)
all_highs = np.array(all_highs)
all_lows = np.array(all_lows)
high = all_highs[:, :, 1:, :].mean((1, 2)) - all_highs[:, :, 0, :].mean(axis=1)
low = all_lows[:, :, 1:, :].mean((1, 2)) - all_lows[:, :, 0, :].mean(axis=1)
diff = low - high
diff_sess = list()   
for i in range(5):
    rev_low = all_lows[:, :, i, :].mean(1) - all_lows[:, :, 0, :].mean(axis=1)
    rev_high = all_highs[:, :, i, :].mean(1) - all_highs[:, :, 0, :].mean(axis=1)
    diff_sess.append(rev_low - rev_high)
diff_sess = np.array(diff_sess).swapaxes(0, 1)

# correlation between rsa and time generalization
idx_rsa = np.where((times >= .3) & (times <= .5))[0]
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]  # Correct filtering condition

rsa = diff_sess.copy()[:, :, idx_rsa].mean(2)

contrasts = patterns - randoms
timeg = []
for sub in range(len(subjects)):
    tg = []
    for i in range(5):
        tg.append(contrasts[sub, i, idx_timeg][:, idx_timeg].mean())
    timeg.append(np.array(tg))
timeg = np.array(timeg)
slopes, intercepts = [], []

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
    ax.scatter(timeg[sub], rsa[sub], alpha=0.3)
    ax.plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
    slopes.append(slope)
    intercepts.append(intercept)
ax.plot(timeg[sub], np.mean(slopes) * timeg[sub] + np.mean(intercepts), color='black', lw=4, label='Mean fit')
ax.set_xlabel('Time generalization', fontsize=12)
ax.set_ylabel('Similarity index', fontsize=12)
ax.legend(frameon=False, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# fig.tight_layout()
fig.suptitle('Correlation between RSA and time generalization - sensor space', fontsize=14)
fig.savefig(figures_dir / "RSA" / f"timeg_corr-{lock}-sensor.pdf", transparent=True)

### Source space ###

analysis = "tg_rdm_emp"
res_dir = TIMEG_DATA_DIR / analysis / lock

networks = schaefer_7[:-2] + ['Hippocampus', 'Thalamus']

src_pat, src_rand = {}, {}
for network in networks:
    print(f"Processing {network}...")
    if not network in src_pat:
        src_pat[network], src_rand[network] = [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-{j}-scores.npy"))
            rand.append(np.load(res_dir / network / 'random' / f"{subject}-{j}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    src_pat[network] = np.array(patpat)
    src_rand[network] = np.array(randrand)

rsa = diff_sess.copy()[:, :, idx_rsa].mean(2)

fig, axes = plt.subplots(1, 7, sharey=True, figsize=(20, 5), layout='tight')
for i, network in enumerate(networks):
    contrasts = src_pat[network] - src_rand[network]
    timeg = []
    for sub in range(len(subjects)):
        tg = []
        for j in range(5):
            tg.append(contrasts[sub, j, idx_timeg][:, idx_timeg].mean())
        timeg.append(np.array(tg))
    timeg = np.array(timeg)
    
    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
        axes[i].scatter(timeg[sub], rsa[sub], alpha=0.3)
        axes[i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)
    
    axes[i].plot(timeg[sub], np.mean(slopes) * timeg[sub] + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    axes[i].set_xlabel('Time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].set_title(network)
    if i == 0:
        axes[i].set_ylabel('Similarity index', fontsize=12)
fig.savefig(figures_dir / "RSA" / f"timeg_corr-{lock}-source.pdf", transparent=True)
plt.close()

all_highs, all_lows = {}, {}
diff_sess = {}
for network in networks:
    print(f"Processing {network}...")
    if not network in diff_sess:
        all_highs[network] = []
        all_lows[network] = []
        diff_sess[network] = []
    for subject in subjects:        
        # # RSA stuff
        behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject))
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / "RSA" / 'source' / network / lock / 'rdm' / subject
        high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
        all_highs[network].append(high)    
        all_lows[network].append(low)
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    # plot diff session by session
    for i in range(5):
        rev_low = all_lows[network][:, :, i, :].mean(1) - all_lows[network][:, :, 0, :].mean(axis=1)
        rev_high = all_highs[network][:, :, i, :].mean(1) - all_highs[network][:, :, 0, :].mean(axis=1)
        diff_sess[network].append(rev_low - rev_high)
    diff_sess[network] = np.array(diff_sess[network]).swapaxes(0, 1)

fig, axes = plt.subplots(1, 7, sharey=True, figsize=(20, 5), layout='tight')
for i, network in enumerate(networks):
    rsa = diff_sess[network].copy()[:, :, idx_rsa].mean(2)
    contrasts = src_pat[network] - src_rand[network]
    timeg = []
    for sub in range(len(subjects)):
        tg = []
        for j in range(5):
            tg.append(contrasts[sub, j, idx_timeg][:, idx_timeg].mean())
        timeg.append(np.array(tg))
    timeg = np.array(timeg)
    
    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
        axes[i].scatter(timeg[sub], rsa[sub], alpha=0.3)
        axes[i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)
    
    axes[i].plot(timeg[sub], np.mean(slopes) * timeg[sub] + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    axes[i].set_xlabel('Time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].set_title(network)
    if i == 0:
        axes[i].set_ylabel('Similarity index', fontsize=12)
fig.savefig(figures_dir / "RSA" / f"timeg_corr-{lock}-source_2.pdf", transparent=True)
plt.close()