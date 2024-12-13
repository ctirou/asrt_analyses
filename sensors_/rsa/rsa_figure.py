import os.path as op
from cycler import cycler
import matplotlib
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from base import *
from config import *

lock = 'button'
data = 'cv'
analysis = 'pat_high_rdm_high'

overwrite = False

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

# get times
times = np.load(data_path / "times.npy")
timesg = np.load(data_path / 'times_gen.npy')

lock = 'stim'

figures_dir = FIGURES_DIR / "RSA" / "sensors" / lock
ensure_dir(figures_dir)

all_highs, all_lows = [], []

for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"{data}_rdm" / subject
    ensure_dir(res_path)
        
    # Read the behav file to get the sequence 
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
    sequence = get_sequence(behav_dir)

    high, low = get_all_high_low(res_path, sequence, analysis, cv=True)
        
    all_highs.append(high)    
    all_lows.append(low)
    
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

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)

inner = [['A1'],
         ['A2']]
outer = [[inner, 'B'],
          ['C', 'D']]

fig, axs = plt.subplot_mosaic(outer, sharex=True, figsize=(16, 10))
for ax in axs.values():
    ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black')

axs['A1'].plot(times, diff.mean(0), alpha=1, label='Low - High')
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
axs['A1'].fill_between(times, 0, diff.mean(0), where=sig, alpha=0.3, label='Significant')
axs['A1'].axhline(0, color='black', linestyle='dashed')
axs['A1'].legend()
axs['A1'].set_ylabel('Similarity index')
axs['A1'].set_title(f'Similarity index', style='italic')

axs['A2'].plot(times, high.mean(0), label='High', alpha=1)
axs['A2'].plot(times, low.mean(0), label='Low', alpha=1)
axs['A2'].fill_between(times, high.mean(0), low.mean(0), where=sig, alpha=0.3, label='Significant')
axs['A2'].legend()
axs['A2'].set_ylabel('cvMahalanobis')
axs['A2'].set_title(f'cvMahalanobis of High and Low', style='italic')
plt.setp(axs['A2'].get_xticklabels(), visible=False)

rhos = np.array([[spear([0, 1, 2, 3, 4], diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))])
axs['B'].plot(times, rhos.mean(0))
p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(rhos, -1)
sig = p_values < .05
axs['B'].fill_between(times, 0, rhos.mean(0), where=sig_unc, alpha=.2, label='Uncorrected')
axs['B'].fill_between(times, 0, rhos.mean(0), where=sig, alpha=.3, label='Corrected')
axs['B'].axhline(0, color='black', linestyle='dashed')
axs['B'].legend()
axs['B'].set_ylabel("Spearman's rho")
axs['B'].set_title(f'Within subject block correlation', style='italic')

all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
axs['C'].plot(times, all_rhos.mean(0))
p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
axs['C'].fill_between(times, all_rhos.mean(0), 0, where=sig_unc, alpha=.2, label='Uncorrected')
axs['C'].fill_between(times, all_rhos.mean(0), 0, where=sig, alpha=.4, label='Corrected')
axs['C'].axhline(0, color="black", linestyle="dashed")
axs['C'].legend()
axs['C'].set_ylabel("Spearman's rho")
axs['C'].set_title(f'Within subject learning index correlation', style='italic')
