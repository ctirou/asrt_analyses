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
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from cycler import cycler

lock = 'stim'
# analysis = 'usual'
analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
# analysis = 'rdm_high_rdm_low'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'
trial_type = 'all'
# get times
times = np.load(data_path / "times.npy")

# labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)
n_parcels = 200
n_networks = 7
networks = schaefer_7[:-2] if n_networks == 7 else schaefer_17
names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2] + \
    ['Left Hippocampus', 'Right Hippocampus', 'Left Thalamus', 'Right Thalamus']
label_names = schaefer_7[:-2] if n_networks == 7 else schaefer_17

figures_dir = FIGURES_DIR / "RSA" / "source" / lock / analysis / f"networks_{n_parcels}_{n_networks}"
ensure_dir(figures_dir)

decoding = {}
all_highs, all_lows = {}, {}
for network in networks:
    print(f"Processing {network}...")
    if not network in decoding:
        decoding[network] = []
        all_highs[network] = []
        all_lows[network] = []
    for subject in subjects:        
        score = np.load(RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'scores' / subject / f"{trial_type}-scores.npy")
        decoding[network].append(score)
        # RSA stuff
        behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject))
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'rdm' / subject
        high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
        all_highs[network].append(high)    
        all_lows[network].append(low)
    decoding[network] = np.array(decoding[network])
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
# for label in ['Hippocampus-lh', 'Hippocampus-rh', 'Thalamus-Proper-lh', 'Thalamus-Proper-rh']:
#     print(f"Processing {label}...")
#     if not label in decoding:
#         decoding[label] = []
#         for subject in subjects:
#             score = np.load(RESULTS_DIR / "decoding" / "source" / lock / trial_type / label / f"{subject}-scores.npy")
#             decoding[label].append(score)
#     decoding[label] = np.array(decoding[label])

label_names = label_names + ['Hippocampus-lh', 'Hippocampus-rh', 'Thalamus-Proper-lh', 'Thalamus-Proper-rh']

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
chance = 25
threshold = .05
design = [['A', [['B1'], ['B2']], [['C1'], ['C2']]], 
          ['D', [['E1'], ['E2']], [['F1'], ['F2']]],
          ['G', [['H1'], ['H2']], [['I1'], ['I2']]],
          ['J', [['K1'], ['K2']], [['L1'], ['L2']]], 
          ['M', [['N1'], ['N2']], [['O1'], ['O2']]],
          ['P', [['Q1'], ['Q2']], [['R1'], ['R2']]], 
          ['S', [['T1'], ['T2']], [['U1'], ['U2']]]]
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
cmap = plt.cm.get_cmap('tab20', len(label_names))

fig, axd = plt.subplot_mosaic(design, sharex=True, figsize=(20, 12), layout='tight')
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
### Plot decoding ###
for i, (label, name, j) in enumerate(zip(label_names[:-4], names_corrected[:-4], ['A', 'D', 'G', 'J', 'M', 'P'])):
    if lock == 'stim':
        axd[j].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    else:
        axd[j].axvline(0, color='black', label='Button press')
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    axd[j].fill_between(times, score.mean(0) + sem, score.mean(0) - sem, alpha=.7, color='C7')
    p_values = decod_stats(score - chance, jobs)
    sig = p_values < threshold
    axd[j].fill_between(times, score.mean(0) + sem, score.mean(0) - sem, where=sig, color=cmap(i))
    axd[j].fill_between(times, score.mean(0) - sem, chance, where=sig, color=cmap(i), alpha=.5)
    axd[j].axhline(chance, color='grey', alpha=.5)
    axd[j].set_ylabel('Accuracy (%)', fontsize=12)
    axd[j].set_title(f'{name} network decoding', style='italic')
### Plot similarity index ###
for i, (label, name, j) in enumerate(zip(label_names[:-4], names_corrected[:-4], ['B1', 'E1', 'H1', 'K1', 'N1', 'Q1'])):
    axd[j].axhline(0, color='grey', alpha=.5)
    high = all_highs[label][:, :, 1:, :].mean((1, 2)) - all_highs[label][:, :, 0, :].mean(1)
    low = all_lows[label][:, :, 1:, :].mean((1, 2)) - all_lows[label][:, :, 0, :].mean(axis=1)
    diff = low - high
    axd[j].plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10, color=cmap(i))
    sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
    sig = p_values < 0.05
    axd[j].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5, color=cmap(i))
    axd[j].fill_between(times, 0, diff.mean(0) - sem, where=sig, alpha=0.3, label='Significance - corrected', facecolor="#F2AD00")
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    axd[j].fill_between(times, 0, diff.mean(0) - sem, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
    # axd[j].legend(frameon=False)
    axd[j].set_ylabel('Sim.', fontsize=11)
    axd[j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axd[j].set_xticklabels([])
    # axd[j].set_title(f'Similarity index', style='italic', fontsize=11)
### Plot cvMD ###
for i, (label, name, j) in enumerate(zip(label_names[:-4], names_corrected[:-4], ['B2', 'E2', 'H2', 'K2', 'N2', 'Q2'])):
    high = all_highs[label][:, :, 1:, :].mean((1, 2)) - all_highs[label][:, :, 0, :].mean(1)
    low = all_lows[label][:, :, 1:, :].mean((1, 2)) - all_lows[label][:, :, 0, :].mean(axis=1)
    diff = low - high
    sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
    sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
    axd[j].plot(times, high.mean(0), label='Pattern', color=cmap(i), alpha=1)
    axd[j].plot(times, low.mean(0), label='Random', color=cmap(i), alpha=1)
    axd[j].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2, color=cmap(i))
    axd[j].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.2, color=cmap(i))
    sig = p_values < 0.05
    axd[j].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig, alpha=0.3, label='Significance - corrected', facecolor="#F2AD00")
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    axd[j].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
    # axd[j].legend(frameon=False)
    axd[j].set_ylabel('cvMD', fontsize=11)
    # axd[j].set_xticklabels([])
    # axd[j].set_title(f'cvMD', style='italic', fontsize=11)
