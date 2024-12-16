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

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

data = 'cv'
analysis = 'pat_high_rdm_high'
lock = 'button'

# get times
times = np.load(data_path / "times.npy")

figures_dir = FIGURES_DIR / "RSA" / "sensors"
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

innerB = [['B1'],
         ['B2']]
innerD = [['D1'],
          ['D2']]

outer = [['A', innerB],
          ['C', innerD]]

fig, axs = plt.subplot_mosaic(outer, sharex=False, figsize=(16, 10))
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
for ax in axs.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax != axs['C']:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.2)
        else:
            ax.axvline(0, color='black')
### B1 ####        
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
for start, end in zip(times[:-1], times[1:]):
    if sig[np.where(times == start)[0][0]]:
        axs['B1'].axvspan(start, end, facecolor="#F2AD00", alpha=0.3, edgecolor=None)
sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
axs['B1'].plot(times, diff.mean(0), alpha=1, label='Low - High')
axs['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2)
axs['B1'].axhline(0, color='black', linestyle='dashed')
axs['B1'].legend()
axs['B1'].set_ylabel('Similarity index')
axs['B1'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axs['B1'].set_title(f'Similarity index', style='italic')
plt.setp(axs['B1'].get_xticklabels(), visible=False)

### B2 ###
sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
for start, end in zip(times[:-1], times[1:]):
    if sig[np.where(times == start)[0][0]]:
        axs['B2'].axvspan(start, end, facecolor="#F2AD00", alpha=0.3, edgecolor=None)
axs['B2'].plot(times, high.mean(0), label='High', alpha=1)
axs['B2'].plot(times, low.mean(0), label='Low', alpha=1)
axs['B2'].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2)
axs['B2'].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.2)
# axs['B2'].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig, alpha=0.3, label='Significant')
axs['B2'].legend()
axs['B2'].set_ylabel('cvMahalanobis')
axs['B2'].set_title(f'cvMahalanobis of High and Low', style='italic')
plt.setp(axs['B2'].get_xticklabels(), visible=False)

### D1 ###
rhos = np.array([[spear([0, 1, 2, 3, 4], diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))])
sem = np.std(rhos, axis=0) / np.sqrt(len(subjects))
axs['D1'].plot(times, rhos.mean(0))
p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(rhos, -1)
sig = p_values < .05
axs['D1'].fill_between(times, rhos.mean(0) - sem, rhos.mean(0) + sem, alpha=0.2)
# axs['D1'].fill_between(times, 0, rhos.mean(0), where=sig_unc, alpha=.2, label='Uncorrected')
# axs['D1'].fill_between(times, 0, rhos.mean(0), where=sig, alpha=.3, label='Corrected')
axs['D1'].axhline(0, color='black', linestyle='dashed')
axs['D1'].set_ylabel("Spearman's rho")
axs['D1'].set_title(f'Within subject block correlation', style='italic')
plt.setp(axs['D1'].get_xticklabels(), visible=False)

### D2 ###
all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
axs['D2'].plot(times, all_rhos.mean(0))
p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
axs['D2'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2)
# axs['D2'].fill_between(times, all_rhos.mean(0), 0, where=sig_unc, alpha=.2, label='Uncorrected')
# axs['D2'].fill_between(times, all_rhos.mean(0), 0, where=sig, alpha=.4, label='Corrected')
axs['D2'].axhline(0, color="black", linestyle="dashed")
axs['D2'].set_ylabel("Spearman's rho")
axs['D2'].set_xlabel('Time (s)')
axs['D2'].set_title(f'Within subject learning index correlation', style='italic')

### C ###
idx_times = np.where((times >= 0.3) & (times <= 0.5))[0]
mdiff = diff_sess[:, :, idx_times].mean(2)
rhos, pvals = [], []
for sess in range(5):
    rho, pval = spear(learn_index_df.iloc[:, sess], mdiff[:, sess])
    rhos.append(rho)
    pvals.append(pval)
rhos = np.array(rhos)
pvals = np.array(pvals)
sig = pvals < .05
axs['C'].plot(range(5), rhos, marker='o', linestyle='-', alpha=0.6)
axs['C'].axhline(0, color='black', linestyle='dashed')
for i, (rho, pval) in enumerate(zip(rhos, pvals)):
    if sig[i]:
        axs['C'].text(i, rho, '*', fontsize=12, ha='center', va='bottom', color='red')
axs['C'].set_xticks(range(5))
axs['C'].set_xticklabels([f'Session {i}' for i in range(5)])
axs['C'].set_ylabel("Spearman's rho")
axs['C'].set_title(f'Correlation with learning index', style='italic')

plt.savefig(figures_dir /  f"{lock}.pdf", transparent=True)
plt.close()
# handles, labels = [], []
# for ax in axs.values():
#     for handle, label in zip(*ax.get_legend_handles_labels()):
#         if label not in labels:
#             handles.append(handle)
#             labels.append(label)
# fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.subplots_adjust(right=0.85)