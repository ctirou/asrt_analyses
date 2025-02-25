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

figures_dir = FIGURES_DIR / 'time_gen' / "source" / lock

# get times
times = np.load(data_path / "times.npy")
timesg = np.linspace(-1.5, 1.5, 305)

# correlation between rsa and time generalization
idx_rsa = np.where((times >= .3) & (times <= .5))[0]
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]

analysis = "tg_rdm_emp"
res_dir = TIMEG_DATA_DIR / analysis / lock

networks = NETWORKS
network_names = NETWORK_NAMES

src_pat, src_rand = {}, {}
for network in tqdm(networks):
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

all_highs, all_lows = {}, {}
diff_sess = {}
for network in tqdm(networks):
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
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)

fig, axes = plt.subplots(1, 9, sharey=True, figsize=(20, 5), layout='tight')
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
    timeg_range = np.linspace(timeg.min(), timeg.max(), 100)
    
    axes[i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    axes[i].set_xlabel('Mean Time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].set_title(network_names[i])
    if i == 0:
        axes[i].set_ylabel('Similarity index', fontsize=12)
    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], rsa[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    axes[i].set_title(f"{network_names[i]}")
    
    axes[i].text(0.05, 0.95, f"$p=${pval:.2f}", transform=axes[i].transAxes, fontsize=12, verticalalignment='top')
    
fig.suptitle("Correlation between mean predictive activity and mean representational similarity", fontweight='bold',  fontsize=16)

fig.savefig(figures_dir / f"rsa_corr.pdf", transparent=True)
plt.close()


fig, axes = plt.subplots(1, 9, sharey=True, figsize=(20, 5), layout='tight')
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
        slope, intercept = np.polyfit(timeg[sub], learn_index_df.iloc[sub], 1)
        axes[i].scatter(timeg[sub], learn_index_df.iloc[sub], alpha=0.3)
        axes[i].plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=f"Subject {sub+1}")
        slopes.append(slope)
        intercepts.append(intercept)

    timeg_range = np.linspace(timeg.min(), timeg.max(), 100)

    axes[i].plot(timeg_range, np.mean(slopes) * timeg_range + np.mean(intercepts), color='black', lw=4, label='Mean fit')
    # axes[i].set_xlabel('Mean time generalization', fontsize=12)
    # axes[i].legend(frameon=False, ncol=2)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].set_title(network_names[i])
    if i == 0:
        axes[i].set_ylabel('Learning index', fontsize=12)
    learn_index_flat = learn_index_df.to_numpy().flatten()

    rhos = []
    for sub in range(len(subjects)):
        r, p = spear(timeg[sub], rsa[sub])
        rhos.append(r)        
    pval = ttest_1samp(rhos, 0)[1]
    axes[i].set_title(f"{network_names[i]}")
    
    axes[i].text(0.05, 0.95, f"$p=${pval:.2f}", transform=axes[i].transAxes, fontsize=12, verticalalignment='top')

fig.suptitle("Correlation between mean predictive activity and learning", fontweight='bold', fontsize=16)

fig.savefig(figures_dir / f"learn_corr.pdf", transparent=True)
plt.close()

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axes = plt.subplots(2, 9, sharey=False, sharex=True, figsize=(20, 5), layout='tight')
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

# fig.text(0.5, 0.02, 'Average pre-stimulus time generalization contrast', ha='center', fontsize=12)

fig.savefig(figures_dir / f"combined_corr.pdf", transparent=True)
plt.close()

networks = ['SomMot', 'DorsAttn', 'SalVentAttn', 'Default']
network_names = ['Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Default']
fig, axes = plt.subplots(2, 4, sharey=False, sharex=True, figsize=(10, 7), layout='tight')
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

fig.suptitle("Correlation between mean predictive activity and learning (top)\nand similarity index (bottom)", fontweight='bold', fontsize=16)

# fig.text(0.5, 0.02, 'Average pre-stimulus time generalization contrast', ha='center', fontsize=12)

fig.savefig(figures_dir / f"combined_corr_best.pdf", transparent=True)
plt.close()