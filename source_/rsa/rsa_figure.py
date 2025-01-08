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

def format_func(value, tick_number):
    return f'{value:.1f}'

# labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)
n_parcels = 200
n_networks = 7
networks = schaefer_7 if n_networks == 7 else schaefer_17
names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2] + \
    ['Left Hippocampus', 'Right Hippocampus', 'Left Thalamus', 'Right Thalamus']
label_names = schaefer_7 if n_networks == 7 else schaefer_17

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
        # # RSA stuff
        # behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
        # sequence = get_sequence(behav_dir)
        # res_path = RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'rdm' / subject
        # high, low = get_all_high_low(res_path, sequence, analysis, cv=False)    
        # all_highs[network].append(high)    
        # all_lows[network].append(low)
    decoding[network] = np.array(decoding[network])
    # all_highs[network] = np.array(all_highs[network])
    # all_lows[network] = np.array(all_lows[network])
# for label in ['Hippocampus-lh', 'Hippocampus-rh', 'Thalamus-Proper-lh', 'Thalamus-Proper-rh']:
#     print(f"Processing {label}...")
#     if not label in decoding:
#         decoding[label] = []
#         for subject in subjects:
#             score = np.load(RESULTS_DIR / "decoding" / "source" / lock / trial_type / label / f"{subject}-scores.npy")
#             decoding[label].append(score)
#     decoding[label] = np.array(decoding[label])

label_names = label_names[:-2] + ['Hippocampus-lh', 'Hippocampus-rh', 'Thalamus-Proper-lh', 'Thalamus-Proper-rh']

innerB = [['B1'], ['B2']]
innerD = [['D1'], ['D2']]
innerC = [['C1', 'C2']]
outer = [['A', innerB],
         [innerC, innerD]]
fig, axd = plt.subplot_mosaic(outer, sharex=False, figsize=(16, 10))
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
cmap = plt.cm.get_cmap('tab20', len(label_names))
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax not in [axd['A'], axd['C1'], axd['C2']]:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
        else:
            ax.axvline(0, color='black')
### A ###            
# Decoding of pattern trials
if lock == 'stim':
    axd['A'].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1, label='Stimulus onset')
else:
    axd['A'].axvline(0, color='black', label='Button press')
axd['A'].axhline(25, color='grey', alpha=.5, label="Chance level")
for i, (label, name, step) in enumerate(zip(label_names, names_corrected, np.arange(23, 17.5, -0.5))):
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    axd['A'].plot(times, score.mean(0), label=name, color=cmap(i))
    axd['A'].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, alpha=.3, color=cmap(i))
    p_values = decod_stats(score - 25, jobs)
    sig = p_values < 0.05
    axd['A'].fill_between(times, step, step + 0.2, where=sig, alpha=.8)
axd['A'].set_ylabel('Accuracy', fontsize=12)
axd['A'].legend(frameon=False, loc='upper left', ncol=2)
axd['A'].set_xlabel('Time (s)', fontsize=12)
axd['A'].set_title(f'Decoding of pattern trials', style='italic')

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
chance = 25
threshold = .05
design = [['A', [['B1'], ['B2']], [['C1'], ['C2']], 'D1', 'D2'], 
          ['E', [['F1'], ['F2']], [['G1'], ['G2']], 'H1', 'H2'],
          ['I', [['J1'], ['J2']], [['K1'], ['K2']], 'L1', 'L2'],
          ['M', [['N1'], ['N2']], [['O1'], ['O2']], 'P1', 'P2'], 
          ['Q', [['R1'], ['R2']], [['S1'], ['S2']], 'T1', 'T2'],
          ['U', [['V1'], ['V2']], [['W1'], ['W2']], 'X1', 'X2']]

fig, axd = plt.subplot_mosaic(design, sharex=True, figsize=(20, 12), layout='tight')
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
### Plot decoding ###
for i, (label, name, j) in enumerate(zip(label_names[:-4], names_corrected[:-4], ['A', 'E', 'I', 'M', 'Q', 'U'])):
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
# for i, (label, name, j) in enumerate(zip(label_names[:-4], names_corrected[:-4], ['B1', 'F1', 'J1', 'N1', 'R1', 'V1'])):