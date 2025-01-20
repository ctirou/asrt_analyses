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
# timesg = np.load(data_path / "times_gen.npy")

figures_dir = FIGURES_DIR / "RSA" / "sensors"
ensure_dir(figures_dir)

all_highs, all_lows = [], []
decoding, pca_decod, no_pca_decod, max_power  = [], [], [], []
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
    # Decoding stuff
    res_path = RESULTS_DIR / 'decoding' / 'sensors' / lock / 'pattern'
    decoding.append(np.load(res_path / f"{subject}-all-scores.npy"))
    res_path = RESULTS_DIR / 'decoding' / 'source' / lock / 'pattern' / 'all'
    pca_decod.append(np.load(res_path / f"{subject}-scores.npy"))
    res_path = RESULTS_DIR / 'decoding' / 'source' / lock / 'pattern' / 'all-nopca'
    no_pca_decod.append(np.load(res_path / f"{subject}-scores.npy"))
    # res_path = RESULTS_DIR / 'decoding' / 'source' / lock / 'pattern' / 'max-power'
    # max_power.append(np.load(res_path / f"{subject}-scores.npy"))
    # Time generalization stuff    
    pat, rand = [], []
    timeg_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
    for i in range(5):
        pat.append(np.load(timeg_path / f"{subject}-epoch{i}-pattern-scores.npy"))
        rand.append(np.load(timeg_path / f"{subject}-epoch{i}-random-scores.npy"))
    patterns.append(np.array(pat))
    randoms.append(np.array(rand))
    
decoding = np.array(decoding)
pca_decod = np.array(pca_decod)
no_pca_decod = np.array(no_pca_decod)
max_power = np.array(max_power)

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

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)

innerB = [['B1'], ['B2']]
innerD = [['D1'], ['D2']]
innerC = [['C1', 'C2']]

outer = [['A', innerB],
         [innerC, innerD]]

fig, axd = plt.subplot_mosaic(outer, sharex=False, figsize=(16, 11), layout='tight')
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Avenir'})
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax not in [axd['A'], axd['C1'], axd['C2']]:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
        else:
            ax.axvline(0, color='black')
            
### A ### Decoding
if lock == 'stim':
    axd['A'].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1, label='Stimulus onset')
else:
    axd['A'].axvline(0, color='black', label='Button press')
axd['A'].axhline(0.25, color='grey', label='Chance level', alpha=0.5)
p_values = decod_stats(decoding - 0.25, -1)
sig = p_values < 0.05
sem = np.std(decoding, axis=0) / np.sqrt(len(subjects))
axd['A'].plot(times, decoding.mean(0), alpha=1)
axd['A'].fill_between(times, decoding.mean(0) - sem, decoding.mean(0) + sem, alpha=0.2)
axd['A'].fill_between(times, decoding.mean(0) - sem, .25, where=sig, alpha=0.3, facecolor="#F2AD00", capstyle='round', label='Significance - corrected')
axd['A'].set_ylabel('Accuracy', fontsize=11)
axd['A'].legend(loc='upper left', frameon=False)
axd['A'].set_xlabel('Time (s)', fontsize=11)
axd['A'].set_title(f'Sensor space decoding', style='italic', fontsize=13)

### B1 ### Similarity index
axd['B1'].axhline(0, color='grey', alpha=0.5)
axd['B1'].plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10)
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
axd['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5)
axd["B1"].fill_between(times, 0, diff.mean(0) - sem, where=sig, alpha=0.3, label='Significance - corrected', facecolor="#F2AD00")
axd['B1'].legend(frameon=False)
axd['B1'].set_ylabel('Similarity index', fontsize=11)
axd['B1'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axd['B1'].set_xticklabels([])
axd['B1'].set_title(f'Similarity index', style='italic', fontsize=13)

### B2 ### cvMD
sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
axd['B2'].plot(times, high.mean(0), label='Pattern', color="#FFA07A", alpha=1)
axd['B2'].plot(times, low.mean(0), label='Random', color="#5BBCD6", alpha=1)
axd['B2'].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2, color="#FFA07A")
axd['B2'].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.2, color="#5BBCD6")
axd['B2'].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig, alpha=0.3, facecolor="#F2AD00", label='Significance - corrected')
axd['B2'].legend(frameon=False)
axd['B2'].set_ylabel('cvMD', fontsize=11)
# axd['B2'].set_xticklabels([])
axd['B2'].set_title(f'cvMD within random and pattern elements', style='italic', fontsize=13)

### D1 ### Within subject block correlation
axd['D1'].axhline(0, color='grey', alpha=0.5)
rhos = np.array([[spear([0, 1, 2, 3, 4], diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))])
sem = np.std(rhos, axis=0) / np.sqrt(len(subjects))
axd['D1'].plot(times, rhos.mean(0))
p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(rhos, -1)
sig = p_values < .05
axd['D1'].fill_between(times, rhos.mean(0) - sem, rhos.mean(0) + sem, alpha=0.2)
axd['D1'].fill_between(times, 0, rhos.mean(0) - sem, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
axd['D1'].fill_between(times, 0, rhos.mean(0) - sem, where=sig, alpha=.3, facecolor="#F2AD00", label='Significance - corrected')
axd['D1'].set_ylabel("Spearman's rho", fontsize=11)
axd['D1'].set_xticklabels([])
axd['D1'].legend(frameon=False, loc="lower right")
axd['D1'].set_title(f'Within subject block correlation', style='italic', fontsize=13)

### D2 ### Within subject learning index correlation
axd['D2'].axhline(0, color="grey", alpha=0.5)
all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
axd['D2'].plot(times, all_rhos.mean(0))
p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
axd['D2'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2)
axd['D2'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
axd['D2'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=.4, facecolor="#F2AD00", label='Significance - corrected')
axd['D2'].set_ylabel("Spearman's rho", fontsize=11)
axd['D2'].set_xlabel('Time (s)', fontsize=11)
axd['D2'].legend(frameon=False, loc="lower right")
axd['D2'].set_title(f'Within subject learning index correlation', style='italic', fontsize=13)

### C1 ### Blocks fit
cmap = plt.cm.get_cmap('tab20', len(subjects))
idx_rsa = np.where((times >= 0.3) & (times <= 0.5))[0]
mdiff = diff_sess[:, :, idx_rsa].mean(2)
sess = np.array([[0, 1, 2, 3, 4] for _ in range(len(subjects))])
slopes, intercepts = [], []
for sub, subject in enumerate(subjects):
    # Linear fit
    slope, intercept = np.polyfit(sess[sub][1:], mdiff[sub][1:], 1)
    axd['C1'].plot(
        sess[sub], 
        slope * sess[sub] + intercept, 
        alpha=0.6, 
        # label=f'{subject} Fit', 
        color=cmap(sub))
    # Scatter points for raw data
    axd['C1'].scatter(
        sess[sub], 
        mdiff[sub], 
        alpha=0.6, 
        # label=f'{subject} Data', 
        color=cmap(sub), 
        marker='o')
    slopes.append(slope)
    intercepts.append(intercept)
# Mean fit line
mean_slope, mean_intercept = np.mean(slopes), np.mean(intercepts)
axd['C1'].plot(
    sess[sub], 
    mean_slope * sess[sub] + mean_intercept, 
    color='black', 
    lw=4, 
    label='Mean fit')
# Labels, legend, and grid for C1
axd['C1'].set_xlabel('Session', fontsize=11)
# axd['C1'].set_xticks([0, 1, 2, 3, 4])
axd['C1'].set_xticks([0, 1, 2, 3, 4])
# axd['C1'].legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
# axd['C1'].grid(True, linestyle='--', alpha=0.7)
# axd['C1'].set_yticklabels([])'
axd['C1'].set_ylabel('Similarity Index', fontsize=11)
axd['C1'].legend(frameon=False)
axd['C1'].set_title(f'Similarity index and blocks fit', style='italic', fontsize=13)

### C2 ### Learning index fit
slopes, intercepts = [], []
for sub, subject in enumerate(subjects):
    # Linear fit
    slope, intercept = np.polyfit(learn_index_df.iloc[sub][1:], mdiff[sub][1:], 1)
    axd['C2'].plot(
        learn_index_df.iloc[sub], 
        slope * learn_index_df.iloc[sub] + intercept, 
        alpha=0.6, 
        # label=f'{subject} Fit', 
        color=cmap(sub))
    # Scatter points for raw data
    axd['C2'].scatter(
        learn_index_df.iloc[sub], 
        mdiff[sub], 
        alpha=0.6, 
        # label=f'{subject} Data', 
        color=cmap(sub), 
        marker='o')
    slopes.append(slope)
    intercepts.append(intercept)
# Mean fit line
mean_slope, mean_intercept = np.mean(slopes), np.mean(intercepts)
axd['C2'].plot(
    learn_index_df.iloc[sub], 
    mean_slope * learn_index_df.iloc[sub] + mean_intercept, 
    color='black', 
    lw=4, 
    label='Mean fit')
# Labels, legend, and grid for C2
# axd['C2'].legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
# axd['C2'].grid(True, linestyle='--', alpha=0.7)
axd['C2'].legend(frameon=False)
axd['C2'].sharey(axd['C1'])
axd['C2'].set_title(f'Similarity index and learning index fit', style='italic', fontsize=13)

plt.savefig(figures_dir /  f"{lock}.pdf", transparent=True)
plt.close()

# correlation between rsa and time generalization
idx_rsa = np.where((times >= .3) & (times <= .5))[0]
idx_timeg = np.where((timesg >= -.5 ) & (timesg <= 0))[0]
rsa = diff_sess.copy()[:, :, idx_rsa].mean(2)
contrasts = patterns - randoms
timeg = np.array([[np.diag(coco[i, :, :]) for i in range(5)] for coco in contrasts])[:, :, idx_timeg].mean(2)
slopes, intercepts = [], []
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.set_prop_cycle(cycler('color', colors['Darjeeling1'] + colors['Darjeeling2'] + colors['Moonrise3']))
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
    ax.plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6, label=subject)
    slopes.append(slope)
    intercepts.append(intercept)
ax.plot(timeg[sub], np.mean(slopes) * timeg[sub] + np.mean(intercepts), color='black', lw=4, label='Mean')
ax.set_xlabel('Time generalization')
ax.set_ylabel('Similarity index')
ax.legend(frameon=False)

# ### C1 ###
# # correlation between rsa and learning index
# idx_rsa = np.where((times >= 0.3) & (times <= 0.5))[0]
# mdiff = diff_sess[:, :, idx_rsa].mean(2)
# slopes, intercepts = [], []
# for sub, subject in enumerate(subjects):
#     # Linear fit
#     slope, intercept = np.polyfit(learn_index_df.iloc[sub], mdiff[sub], 1)
#     axd['C1'].plot(learn_index_df.iloc[sub], slope * learn_index_df.iloc[sub] + intercept, alpha=0.6)
#     slopes.append(slope)
#     intercepts.append(intercept)
# axd['C1'].plot(learn_index_df.iloc[sub], np.mean(slopes) * learn_index_df.iloc[sub] + np.mean(intercepts), color='black', lw=4, label='Mean fit')
# axd['C1'].set_xlabel('Learning index')
# axd['C1'].set_ylabel('Similarity index')
# axd['C1'].legend(frameon=False)

# ### C2 ###
# # correlation between rsa and blocks
# sess = np.array([[0, 1, 2, 3, 4] for _ in range(len(subjects))])
# slopes, intercepts = [], []
# for sub, subject in enumerate(subjects):
#     # Linear fit
#     slope, intercept = np.polyfit(sess[sub], mdiff[sub], 1)
#     axd['C2'].plot(sess[sub], slope * sess[sub] + intercept, alpha=0.6)
#     slopes.append(slope)
#     intercepts.append(intercept)
# axd['C2'].plot(sess[sub], np.mean(slopes) * sess[sub] + np.mean(intercepts), color='black', lw=4, label='Mean fit')
# axd['C2'].set_xlabel('Blocks')
# axd['C2'].set_xticks([0, 1, 2, 3, 4])
# axd['C2'].sharey(axd['C1'])

# for i, (rho, pval) in enumerate(zip(rhos, pvals)):
#     if sig[i]:
#         axd['C'].text(i, rho, '*', fontsize=12, ha='center', va='bottom', color='red')

# handles, labels = [], []
# for ax in axd.values():
#     for handle, label in zip(*ax.get_legend_handles_labels()):
#         if label not in labels:
#             handles.append(handle)
#             labels.append(label)
# fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.subplots_adjust(right=0.85)