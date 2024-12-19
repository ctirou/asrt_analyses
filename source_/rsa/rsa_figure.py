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
names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()
label_names = schaefer_7 if n_networks == 7 else schaefer_17

figures_dir = FIGURES_DIR / "RSA" / "source" / lock / analysis / f"networks_{n_parcels}_{n_networks}"
ensure_dir(figures_dir)

decoding = {}
for network in networks:
    print(f"Processing {network}...")
    if not network in decoding:
        decoding[network] = []

    for subject in subjects:        
        score = np.load(RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'scores' / subject / f"{trial_type}-scores.npy")
        decoding[network].append(score)
        
    decoding[network] = np.array(decoding[network])

innerB = [['B1'], ['B2']]
innerD = [['D1'], ['D2']]
innerC = [['C1', 'C2']]
outer = [['A', innerB],
          [innerC, innerD]]


fig, axd = plt.subplot_mosaic(outer, sharex=False, figsize=(16, 10))
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
cmap = plt.cm.get_cmap('Dark2', len(label_names))
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax not in [axd['A'], axd['C']]:
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
axd['A'].axhline(25, color='black', linestyle='dashed')
for i, (label, name, step) in enumerate(zip(label_names[:-2], names_corrected[:-2], [23, 22.5, 22, 21.5, 21])):
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    axd['A'].plot(times, score.mean(0), label=name, color=cmap(i))
    axd['A'].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, alpha=.3, color=cmap(i))
    offset = i * 0.1  # Adjust this value to control the separation between the filled areas
    p_values = decod_stats(score - 25, jobs)
    sig = p_values < 0.05
    axd['A'].fill_between(times, step, step + 0.2, where=sig, alpha=.8, color=cmap(i))
axd['A'].set_ylabel('Accuracy')
axd['A'].legend(frameon=False)
axd['A'].set_xlabel('Time (s)')
axd['A'].set_title(f'Decoding of pattern trials', style='italic')