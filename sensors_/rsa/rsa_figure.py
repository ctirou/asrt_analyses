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
lock = 'stim'

# get times
times = np.load(data_path / "times.npy")

figures_dir = FIGURES_DIR / "RSA" / "sensors"
ensure_dir(figures_dir)

all_highs, all_lows = [], []
decoding = []

for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"{data}_rdm" / subject
    ensure_dir(res_path)
        
    # Read the behav file to get the sequence 
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
    sequence = get_sequence(behav_dir)
    high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
    all_highs.append(high)    
    all_lows.append(low)
    
    res_path = RESULTS_DIR / 'decoding' / 'sensors' / lock / 'pattern'
    decoding.append(np.load(res_path / f"{subject}-all-scores.npy"))
    
all_highs = np.array(all_highs)
all_lows = np.array(all_lows)
decoding = np.array(decoding)

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

def sig_span(ax, sig, times, lab=True):
    for i, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if sig[np.where(times == start)[0][0]]:
            if lab:
                label = 'Significant' if not any(sig[:i]) else None
            else:
                label = None
            ax.axvspan(start, end, facecolor="#F2AD00", alpha=0.3, label=label, edgecolor=None)

innerB = [['B1'],
         ['B2']]
innerD = [['D1'],
          ['D2']]

outer = [['A', innerB],
          ['C', innerD]]

fig, axd = plt.subplot_mosaic(outer, sharex=False, figsize=(16, 10))
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax not in [axd['A'], axd['C']]:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
        else:
            ax.axvline(0, color='black')

### A ###            
if lock == 'stim':
    axd['A'].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1, label='Stimulus onset')
else:
    axd['A'].axvline(0, color='black', label='Button press')
p_values = decod_stats(decoding - 0.25, -1)
sig = p_values < 0.05
# sig_span(axd['A'], sig, times)
sem = np.std(decoding, axis=0) / np.sqrt(len(subjects))
axd['A'].plot(times, decoding.mean(0), alpha=1)
axd['A'].fill_between(times, decoding.mean(0) - sem, decoding.mean(0) + sem, alpha=0.2)
axd['A'].axhline(0.25, color='black', linestyle='dashed')
axd['A'].set_ylabel('Accuracy')
axd['A'].fill_between(times, .23, .233, where=sig, alpha=0.3, color="#F2AD00", capstyle='round', label='Significant')
axd['A'].legend(frameon=False)
axd['A'].set_title(f'Decoding of Pattern trials', style='italic')
            
### B1 ####        
axd['B1'].plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10)
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
sig_span(axd['B1'], sig, times)
sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
axd['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5)
axd['B1'].axhline(0, color='black', linestyle='dashed')
axd['B1'].legend(frameon=False)
axd['B1'].set_ylabel('Similarity index')
axd['B1'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axd['B1'].set_xticklabels([])
axd['B1'].set_title(f'Similarity index', style='italic')

### B2 ###
sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
sig_span(axd['B2'], sig, times, lab=False)
axd['B2'].plot(times, high.mean(0), label='Pattern', alpha=1)
axd['B2'].plot(times, low.mean(0), label='Random', alpha=1)
axd['B2'].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2)
axd['B2'].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.2)
# axd['B2'].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig, alpha=0.3, label='Significant')
# axd['B2'].legend(["_", 'Pattern', 'Random'])
axd['B2'].legend(frameon=False)
axd['B2'].set_ylabel('cvMahalanobis')
# axd['B2'].set_xticklabels([])
axd['B2'].set_title(f'cvMahalanobis of Random and Pattern trials', style='italic')

### D1 ###
rhos = np.array([[spear([0, 1, 2, 3, 4], diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))])
sem = np.std(rhos, axis=0) / np.sqrt(len(subjects))
axd['D1'].plot(times, rhos.mean(0))
p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(rhos, -1)
sig = p_values < .05
axd['D1'].fill_between(times, rhos.mean(0) - sem, rhos.mean(0) + sem, alpha=0.2)
# axd['D1'].fill_between(times, 0, rhos.mean(0), where=sig_unc, alpha=.2, label='Uncorrected')
# axd['D1'].fill_between(times, 0, rhos.mean(0), where=sig, alpha=.3, label='Corrected')
axd['D1'].axhline(0, color='black', linestyle='dashed')
axd['D1'].set_ylabel("Spearman's rho")
axd['D1'].set_xticklabels([])
axd['D1'].set_title(f'Within subject block correlation', style='italic')

### D2 ###
all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
axd['D2'].plot(times, all_rhos.mean(0))
p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
axd['D2'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2)
# axd['D2'].fill_between(times, all_rhos.mean(0), 0, where=sig_unc, alpha=.2, label='Uncorrected')
# axd['D2'].fill_between(times, all_rhos.mean(0), 0, where=sig, alpha=.4, label='Corrected')
axd['D2'].axhline(0, color="black", linestyle="dashed")
axd['D2'].set_ylabel("Spearman's rho")
axd['D2'].set_xlabel('Time (s)')
axd['D2'].set_title(f'Within subject learning index correlation', style='italic')

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
axd['C'].plot(range(5), rhos, marker='o', linestyle='-', alpha=0.6)
axd['C'].axhline(0, color='black', linestyle='dashed')
for i, (rho, pval) in enumerate(zip(rhos, pvals)):
    if sig[i]:
        axd['C'].text(i, rho, '*', fontsize=12, ha='center', va='bottom', color='red')
axd['C'].set_xticks(range(5))
axd['C'].set_xticklabels([f'Session {i}' for i in range(5)])
axd['C'].set_ylabel("Spearman's rho")
axd['C'].set_title(f'Correlation with learning index', style='italic')
# handles, labels = axd['B1'].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', title='Conditions')

plt.savefig(figures_dir /  f"{lock}.pdf", transparent=True)
plt.close()
# handles, labels = [], []
# for ax in axd.values():
#     for handle, label in zip(*ax.get_legend_handles_labels()):
#         if label not in labels:
#             handles.append(handle)
#             labels.append(label)
# fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.subplots_adjust(right=0.85)