import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import *
from config import *
from mne import read_epochs
from scipy.stats import spearmanr, ttest_1samp
from tqdm.auto import tqdm
from matplotlib.ticker import FuncFormatter

lock = 'stim'
# analysis = 'usual'
analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
# analysis = 'rdm_high_rdm_low'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
figures_dir = FIGURES_DIR / "RSA" / "source" / lock / analysis / "networks"
ensure_dir(figures_dir)
metric = 'mahalanobis'
trial_type = 'all'
# get times
times = np.load(data_path / "times.npy")

def format_func(value, tick_number):
    return f'{value:.1f}'

# labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)
n_parcels = 200
n_networks = 7
networks = schaefer_7 if n_networks == 7 else schaefer_17
names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()

def process_label(networks):
    """Process both surface and volume labels."""
    decoding, rsa, rsa_high, rsa_low, corr = {}, {}, {}, {}, {}
    for network in networks:
        print(f"Processing {network}...")
        all_high, all_low = [], []
        if not network in decoding:
            decoding[network] = []

        for subject in subjects:
            # Read the behav file to get the sequence 
            behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
            sequence = get_sequence(behav_dir)
            
            res_path = RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'rdm' / subject
            sub_high, sub_low = get_all_high_low(res_path, sequence, analysis)
            
            all_high.append(sub_high)
            all_low.append(sub_low)
            
            score = np.load(RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'scores' / subject / f"{trial_type}-scores.npy")
            decoding[network].append(score)
            
        all_high, all_low = np.array(all_high).mean(1), np.array(all_low).mean(1)
        diff_low_high = np.squeeze(all_low - all_high)
        rsa[network] = diff_low_high
        rsa_high[network] = all_high
        rsa_low[network] = all_low

        all_rhos = [
            [spearmanr([0, 1, 2, 3, 4], diff_low_high[sub, :, itime])[0] for itime in range(len(times))]
            for sub in range(len(subjects))
        ]
        corr[network] = np.array(all_rhos)
        decoding[network] = np.array(decoding[network])
        
    return decoding, rsa, rsa_high, rsa_low, corr

decoding, rsa, rsa_high, rsa_low, corr = process_label(networks)

label_names = schaefer_7
# define parameters    
chance = 25
# ncols = 4
ncols = 4 if n_networks == 7 else 6
# nrows = 10 if lock == 'stim' else 9
nrows = 2 if n_networks == 7 else 3
far_left = [0] + [i for i in range(0, len(label_names), ncols*2)]
color1, color2 = ("#1982C4", "#74B3CE") if lock == 'stim' else ("#D76A03", "#EC9F05")
color3 = "C7"

color3 = "#008080"
# color3 = "#FFA500"

# plot decoding
chance = 25
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
for i, (ax, label, name) in enumerate(zip(axs.flat, label_names, names_corrected)):
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    ax.plot(times, score.mean(0))
    # ax.fill_between(times, score.mean(0) - sem, score.mean(0) + sem, color="C7", alpha=.7)
    ax.axhline(chance, color='black', linestyle='dashed')
    ax.set_title(f"${name}$", fontsize=8)     
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    p_values = decod_stats(score, jobs)
    sig = p_values < 0.05
    ax.fill_between(times, chance, score.mean(0), where=sig, color=color3, alpha=.8, label='significant')
plt.show()

# plot in out and diff
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(15, 5))
fig.suptitle(f"high and low {lock} {analysis}", style='italic')
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    ax.plot(times, rsa_high[label].mean((0, 1)), label='high', color=color2, alpha=1)
    ax.plot(times, rsa_low[label].mean((0, 1)), label='low', color=color1, alpha=1)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black')
    if i == 0:
        ax.legend()
    ax.set_title(f"${label}$", fontsize=8)
plt.savefig(op.join(figures_dir, 'low_high.pdf'), transparent=True)
plt.close()

# plot rsa
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
fig.suptitle(f'average low - high {lock} {analysis}', style='italic')
for i, (ax, label, name) in enumerate(zip(axs.flat, label_names, names_corrected)):
    # plot reverse difference high vs. low sequence averaging all sessions
    rev_high = rsa_high[label][:, 1:, :].mean((1)) - rsa_high[label][:, 0, :]
    rev_low = rsa_low[label][:, 1:, :].mean((1)) - rsa_low[label][:, 0, :]
    rev_diff = rev_low - rev_high
    
    practice = rsa[label][:, 0, :].mean(0).astype(np.float64)
    learning = rsa[label][:, 1:5, :].mean((0, 1)).astype(np.float64)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black', alpha=.7)
    # ax.plot(times, practice, label='practice')
    # ax.plot(times, learning, label='learning')
    # ax.plot(times, rev_diff.mean(0), label='(low_post - low_pre) - (high_post - high_pre)', color='C1', alpha=0.6)

    # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
    #     tick.label.set_fontsize(8)
    # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
    #     tick.label.set_fontsize(8)
    # diff = rsa[label][:, 1:, :].mean((1)) - rsa[label][:, 0, :]
    # p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
    # sig_unc = p_values_unc < 0.05
    # ax.fill_between(times, 0, learning, where=sig_unc, color=color1, alpha=0.2)
    # p_values = decod_stats(diff, jobs)
    # sig = p_values < 0.05
    # ax.fill_between(times, 0, learning, where=sig, color=color2, alpha=0.3)
    # ax.fill_between(times, 0, rev_diff.mean(0), where=sig_unc, color=color1, alpha=0.2, label='uncorrected')
    # ax.fill_between(times, 0, rev_diff.mean(0), where=sig, color=color2, alpha=0.3, label='corrected')
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.1)
    else:
        ax.axvline(0, color='black')
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    sem = np.std(rev_diff, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, rev_diff.mean(0) - sem, rev_diff.mean(0) + sem, color=color1, alpha=.7)
    ax.set_title(f"${name}$", fontsize=8)     
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    p_values_unc = ttest_1samp(rev_diff,  axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(rev_diff, -1)
    sig = p_values < 0.05
    ax.fill_between(times, rev_diff.mean(0) - sem, rev_diff.mean(0) + sem, where=sig, color='black', alpha=.8, label='significant')
    ax.axhline(0, color='black', linestyle='dashed')
plt.savefig(op.join(figures_dir, 'low_high_ave.pdf'))
plt.close()

# plot the difference in vs. out sequence for each epoch
for k in range(1, 5):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
    fig.suptitle(f'low - high session {k} {lock} {analysis}', style='italic')
    for i, (ax, label, name) in enumerate(zip(axs.flat, label_names, names_corrected)):
        # ax.plot(times, rsa[label][:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
        # ax.plot(times, rsa[label][:, k, :].mean(0) - rsa[label][:, 0, :].mean(0), label='low - high - practice', color='C1', alpha=0.6)
        # diff = rsa[label][:, k, :] - rsa[label][:, 0, :]
        rev_high = rsa_high[label][:, k, :] - rsa_high[label][:, 0, :]
        rev_low = rsa_low[label][:, k, :] - rsa_low[label][:, 0, :]
        rev_diff = rev_low - rev_high
        sem = np.std(rev_diff, axis=0) / np.sqrt(len(subjects))
        ax.fill_between(times, rev_diff.mean(0) - sem, rev_diff.mean(0) + sem, color=color1, alpha=.7)
        p_values_unc = ttest_1samp(rev_diff, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(rev_diff, -1)
        sig = p_values < 0.05
        ax.fill_between(times, rev_diff.mean(0) - sem, rev_diff.mean(0) + sem, where=sig, alpha=0.8, color='black', label='significant')
        ax.axhline(0, color='black', linestyle='dashed')
        if lock == 'stim':
            ax.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            ax.axvline(0, color='black')
        if i ==0:    
            ax.legend()
        ax.set_title(f"${name}$", fontsize=8)     
        # plt.gca().set_ylim(-0.04, 0.12)
    plt.savefig(op.join(figures_dir, 'low_high_%s.pdf' % (str(k))))
    plt.close()

# plot correlations
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
fig.suptitle(f"{metric} correlations {lock} {analysis}", style='italic')
for i, (ax, label, name) in enumerate(zip(axs.flat, label_names, names_corrected)):
    rev_diff_corr = list()   
    for i in range(5):
        rev_low = rsa_low[label][:, i, :] - rsa_low[label][:, 0, :]
        rev_high = rsa_high[label][:, i, :] - rsa_high[label][:, 0, :]
        rev_diff_corr.append(rev_low - rev_high)
    rev_diff_corr = np.array(rev_diff_corr).swapaxes(0, 1)    
    
    rhos = [[spearmanr([0, 1, 2, 3, 4], diff[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
    rhos = np.array(rhos)
    sem = np.std(rhos, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, rhos.mean(0) - sem, rhos.mean(0) + sem, color=color1, alpha=.7, label='rhos')
    p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(rhos, -1)
    sig = p_values < 0.05
    # ax.fill_between(times, rhos.mean(0), 0, where=sig_unc, color=color1, alpha=.2, label='uncorrected')
    ax.fill_between(times, rhos.mean(0) - sem, rhos.mean(0) + sem, color='black', where=sig, label='significant', alpha=.8)
    ax.axhline(0, color="black", linestyle="dashed")
    ax.set_title(f"${name}$", fontsize=8)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black')
    ax.axhline(0, color='black', linestyle='dashed')
    # ax.legend()
plt.savefig(op.join(figures_dir, 'low_high_corr.pdf'), transparent=True)
plt.close()

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
# plot across subjects
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
fig.suptitle(f'{metric} across subjects corr {lock} {analysis}', style='italic')
for i, (ax, label, name) in enumerate(zip(axs.flat, label_names, names_corrected)):
    rev_low = rsa_low[label][:, -1, :] - rsa_low[label][:, 0, :]
    rev_high = rsa_high[label][:, -1, :] - rsa_high[label][:, 0, :]
    rev_diff_corr = rev_low - rev_high  
    all_pvalues, all_rhos = [], []
    for t in range(len(times)):
        rho, pval = spearmanr(learn_index_df["4"], rev_diff[:, t])
        all_rhos.append(rho)
        all_pvalues.append(pval)
    sem = np.std(all_rhos) / np.sqrt(len(subjects))
    ax.fill_between(times, all_rhos - sem, all_rhos + sem, color=color1, alpha=.7, label='rhos')
    # ax.plot(times, all_rhos, label='rho')
    sig = (np.asarray(all_pvalues) < 0.05)
    ax.fill_between(times, all_rhos - sem, all_rhos + sem, where=sig, color='black', alpha=.8, label='significant')  # Solid line at the bottom when sig is true
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black')
    ax.set_title(f"${name}$", fontsize=8)
    ax.axhline(0, color='black', linestyle='dashed')
plt.savefig(op.join(figures_dir, 'low_high_across_sub_corr.pdf'), transparent=True)
plt.close()

# plot within subjects
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 5))
fig.suptitle(f'{metric} within subjects corr {lock} {analysis}', style='italic')
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    rev_diff_corr = list()   
    for i in range(5):
        rev_low = rsa_low[label][:, i, :] - rsa_low[label][:, 0, :]
        rev_high = rsa_high[label][:, i, :] - rsa_high[label][:, 0, :]
        rev_diff_corr.append(rev_low - rev_high)
    rev_diff_corr = np.array(rev_diff_corr).swapaxes(0, 1)    
    all_rhos = []
    for sub in tqdm(range(len(subjects))):
        rhos = []
        for t in range(len(times)):
            rhos.append(spearmanr(learn_index_df.iloc[sub, :], rev_diff_corr[sub, :, t])[0])
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=color1, alpha=.7, label='rhos')
    p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(all_rhos, -1)
    sig = p_values < 0.05
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, color='black', alpha=.8, label='significant')
    ax.axhline(0, color="black", linestyle="dashed")
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black')
    ax.set_title(f"${name}$", fontsize=8)
    ax.axhline(0, color='black', linestyle='dashed')
plt.savefig(op.join(figures_dir, 'low_high_within_sub_corr.pdf'), transparent=True)
plt.close()