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
subjects, epochs_list = SUBJS15, EPOCHS
metric = 'mahalanobis'

data = 'cv'
analysis = 'pat_high_rdm_high'
lock = 'stim'

figures_dir = FIGURES_DIR / 'time_gen' / "source"

# get times
times = np.linspace(-0.2, 0.6, 82)
timesg = np.linspace(-1.5, 1.5, 305)

# correlation between rsa and time generalization
idx_rsa = np.where((times >= .3) & (times <= .5))[0]
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]

res_dir = RESULTS_DIR / 'TIMEG' / 'source'
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']

patterns, randoms = {}, {}
for network in tqdm(networks):
    if not network in patterns:
        patterns[network], randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            p = np.load(res_dir / network / 'scores_skf' / subject / f"pat-{j}.npy")
            r = np.load(res_dir / network / 'scores_skf' / subject / f"rand-{j}.npy")
            pat.append(p)
            rand.append(r)
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

# Load RSA data
all_highs, all_lows = {}, {}
diff_sess = {}
for network in tqdm(networks):
    if not network in all_highs:
        all_highs[network], all_lows[network] = [], []
        diff_sess[network] = []
    for subject in subjects:        
        # RSA stuff
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / 'rdm_skf' / subject
        pats, rands = [], []
        for epoch_num in range(5):
            pats.append(np.load(res_path / f"pat-{epoch_num}.npy"))
            rands.append(np.load(res_path / f"rand-{epoch_num}.npy"))
        pats = np.array(pats)
        rands = np.array(rands)
        high, low = get_all_high_low(pats, rands, sequence, False)
        high, low = np.array(high).mean(0), np.array(low).mean(0)
        if subject == 'sub05':
            pat_b1 = np.load(res_path / "pat-b1.npy")
            high[0] = pat_b1.copy()
            rand_b1 = np.load(res_path / "rand-b1.npy")
            low[0] = rand_b1.copy()
        all_highs[network].append(high)
        all_lows[network].append(low)
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    # plot diff session by session
    for i in range(5):
        low_sess = all_lows[network][:, i, :] - all_lows[network][:, 0, :]
        high_sess = all_highs[network][:, i, :] - all_highs[network][:, 0, :]
        diff_sess[network].append(low_sess - high_sess)
    diff_sess[network] = np.array(diff_sess[network]).swapaxes(0, 1)
    
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axes = plt.subplots(2, 10, sharey=False, sharex=True, figsize=(20, 5), layout='tight')
for i, network in enumerate(networks):
    contrasts = patterns[network] - randoms[network]
    timeg = []
    for sub in range(len(subjects)):
        tg = []
        for j in range(5):
            tg.append(contrasts[sub, j, idx_timeg][:, idx_timeg].mean())
        timeg.append(np.array(tg))
    timeg = np.array(timeg)
    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], learn_index_df.iloc[sub], 1)
        axes[0, i].scatter(timeg[sub], learn_index_df.iloc[sub], alpha=0.3)
        axes[0, i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)

    timeg_range = np.linspace(timeg.min(), timeg.max(), 100)

    axes[0, i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    # axes[i].set_xlabel('Mean time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[0, i].spines['top'].set_visible(False)
    axes[0, i].spines['right'].set_visible(False)
    axes[0, i].set_title(network_names[i])
    if i == 0:
        axes[0, i].set_ylabel('Learning index', fontsize=12)
    learn_index_flat = learn_index_df.to_numpy().flatten()

    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], learn_index_df.iloc[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    axes[0, i].set_title(f"{network_names[i]}")
    axes[0, i].text(0.05, 0.05, f"$p=${pval:.2f}", transform=axes[0, i].transAxes, fontsize=12, verticalalignment='bottom')

    rsa = diff_sess[network].copy()[:, :, idx_rsa].mean(2)

    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
        axes[1, i].scatter(timeg[sub], rsa[sub], alpha=0.3)
        axes[1, i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)
    
    axes[1, i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    axes[1, i].set_xlabel('Mean pre-stim. contrast', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[1, i].spines['top'].set_visible(False)
    axes[1, i].spines['right'].set_visible(False)
    if i == 0:
        axes[1, i].set_ylabel('Similarity index', fontsize=12)
    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], rsa[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    
    axes[1, i].text(0.05, 0.05, f"$p=${pval:.2f}", transform=axes[1, i].transAxes, fontsize=12, verticalalignment='bottom')

    for ax in axes[0, :]:
        ax.get_shared_y_axes().join(ax, axes[0, 0])
    for ax in axes[1, :]:
        ax.get_shared_y_axes().join(ax, axes[1, 0])
        
    if i != 0:
        axes[0, i].set_yticklabels([])
        axes[1, i].set_yticklabels([])

fig.suptitle("Correlation between mean predictive activity and learning (top) and similarity index (bottom)", fontweight='bold', fontsize=16)
fig.savefig(figures_dir / f"combined_corr-all.pdf", transparent=True)
plt.close()

networks = ['SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Cerebellum-Cortex']
network_names = ['Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Control', 'Default', 'Cerebellum']
fig, axes = plt.subplots(2, 6, sharey=False, sharex=True, figsize=(12, 7), layout='tight')
for i, network in enumerate(networks):
    contrasts = patterns[network] - randoms[network]
    timeg = []
    for sub in range(len(subjects)):
        tg = []
        for j in range(5):
            tg.append(contrasts[sub, j, idx_timeg][:, idx_timeg].mean())
        timeg.append(np.array(tg))
    timeg = np.array(timeg)
    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], learn_index_df.iloc[sub], 1)
        axes[0, i].scatter(timeg[sub], learn_index_df.iloc[sub], alpha=0.3)
        axes[0, i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)

    timeg_range = np.linspace(timeg.min(), timeg.max(), 100)

    axes[0, i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    # axes[i].set_xlabel('Mean time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[0, i].spines['top'].set_visible(False)
    axes[0, i].spines['right'].set_visible(False)
    axes[0, i].set_title(network_names[i])
    if i == 0:
        axes[0, i].set_ylabel('Learning index', fontsize=12)
    learn_index_flat = learn_index_df.to_numpy().flatten()
    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], learn_index_df.iloc[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    axes[0, i].set_title(f"{network_names[i]}")
    axes[0, i].text(0.05, 0.95, f"$p=${pval:.2f}", transform=axes[0, i].transAxes, fontsize=12, verticalalignment='top')

    rsa = diff_sess[network].copy()[:, :, idx_rsa].mean(2)
    slopes, intercepts = [], []
    for sub, subject in enumerate(subjects):
        slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
        axes[1, i].scatter(timeg[sub], rsa[sub], alpha=0.3)
        axes[1, i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)
    
    axes[1, i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    axes[1, i].set_xlabel('Mean pre-stim. contrast', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[1, i].spines['top'].set_visible(False)
    axes[1, i].spines['right'].set_visible(False)
    if i == 0:
        axes[1, i].set_ylabel('Similarity index', fontsize=12)
    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], rsa[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    
    axes[1, i].text(0.05, 0.95, f"$p=${pval:.2f}", transform=axes[1, i].transAxes, fontsize=12, verticalalignment='top')

    for ax in axes[0, :]:
        ax.get_shared_y_axes().join(ax, axes[0, 0])
    for ax in axes[1, :]:
        ax.get_shared_y_axes().join(ax, axes[1, 0])
        
    if i != 0:
        axes[0, i].set_yticklabels([])
        axes[1, i].set_yticklabels([])

fig.suptitle("Correlation between mean predictive activity and learning (top)\nand mean representational change effect (bottom)", fontweight='bold', fontsize=16)

fig.savefig(figures_dir / f"combined_corr_best.pdf", transparent=True)
plt.close()