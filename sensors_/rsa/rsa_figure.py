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
    # # Time generalization stuff    
    # pat, rand = [], []
    # timeg_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
    # for i in range(5):
    #     pat.append(np.load(timeg_path / f"{subject}-epoch{i}-pattern-scores.npy"))
    #     rand.append(np.load(timeg_path / f"{subject}-epoch{i}-random-scores.npy"))
    # patterns.append(np.array(pat))
    # randoms.append(np.array(rand))
    
decoding = np.array(decoding) * 100
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
chance = 25
threshold = 0.05

innerB = [['B1'], ['B2']]
innerD = [['D1'], ['D2']]
innerC = [['C1', 'C2']]
outer = [['A', innerB],
         ['C', 'D']]

cmap1 = colors['Darjeeling1']

c1 = "#0173B2"
c2 = "#029E73"
c3 = "#CA9161"
c4 = "#CC78BC"
c5 = "#ECE133"
c6 = "#D55E00"

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axd = plt.subplot_mosaic(outer, 
                              sharex=False, 
                              figsize=(15, 10), 
                              layout='tight',
                              gridspec_kw={
                                  'height_ratios': [1, .7],
                                #   'width_ratios': [.3, .3, 1  , .5, .5]
                                  })
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax not in [axd['A'], axd['C']]:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=0, alpha=.1)
        else:
            ax.axvline(0, color='black')
### A ### Decoding
if lock == 'stim':
        axd['A'].axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=0, alpha=.1, label='Stimulus onset')
else:
    axd['A'].axvline(0, color='black', label='Button press')
axd['A'].axhline(chance, color='grey', label='Chance level', alpha=0.5)
p_values = decod_stats(decoding - chance, -1)
sig = p_values < 0.05
sem = np.std(decoding, axis=0) / np.sqrt(len(subjects))
# axd['A'].plot(times, decoding.mean(0), alpha=1)
# Main plot
axd['A'].plot(times, decoding.mean(0), alpha=1, zorder=10, color='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['A'].plot(times[start:end], decoding.mean(0)[start:end], alpha=1, zorder=15, color=c1)
axd['A'].fill_between(times, decoding.mean(0) - sem, decoding.mean(0) + sem, alpha=0.2, zorder=10, facecolor='C7')    
# Highlight significant regions
axd['A'].fill_between(times, decoding.mean(0) - sem, decoding.mean(0) + sem, where=sig, alpha=0.2, zorder=15, facecolor=c1, label='Significance')    
axd['A'].fill_between(times, decoding.mean(0) - sem, chance, where=sig, alpha=0.1, zorder=15, facecolor=c1)
# axd['A'].fill_between(times, decoding.mean(0) - sem, decoding.mean(0) + sem, alpha=0.2)
# axd['A'].fill_between(times, decoding.mean(0) - sem, .25, where=sig, alpha=0.3, facecolor="#F2AD00", capstyle='round', label='Significance - corrected')
axd['A'].set_ylabel('Accuracy (%)', fontsize=11)
axd['A'].legend(loc='upper left', frameon=False)
axd['A'].set_xlabel('Time (s)', fontsize=11)
axd['A'].set_title(f'Time course decoding', fontsize=13)

### B1 ### Similarity index
axd['B1'].axhline(0, color='grey', alpha=0.5)
axd['B1'].plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10)
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
# Main plot
axd['B1'].plot(times, diff.mean(0), alpha=1, zorder=10, color='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['B1'].plot(times[start:end], diff.mean(0)[start:end], alpha=1, zorder=10, color=c2)
axd['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')    
# Highlight significant regions
axd['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, where=sig, alpha=0.3, zorder=5, facecolor=c2, label='Significance')
axd['B1'].fill_between(times, diff.mean(0) - sem, 0, where=sig, alpha=0.2, zorder=5, facecolor=c2)
# axd['B1'].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5)
# axd["B1"].fill_between(times, 0, diff.mean(0) - sem, where=sig, alpha=0.3, label='Significance - corrected', facecolor="#F2AD00")
axd['B1'].legend(frameon=False)
axd['B1'].set_ylabel('Similarity index', fontsize=11)
axd['B1'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axd['B1'].set_xticklabels([])
axd['B1'].set_title(f'Average similarity index time course', fontsize=13)

### B2 ### cvMD
sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
# axd['B2'].plot(times, high.mean(0), label='Pattern', color="c3", alpha=1)
# axd['B2'].plot(times, low.mean(0), label='Random', color="c4", alpha=1)
# axd['B2'].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2, color="c3")
# axd['B2'].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.2, color="c4")
# axd['B2'].fill_between(times, high.mean(0) + sem_high, low.mean(0) - sem_low, where=sig, alpha=0.3, facecolor="#F2AD00", label='Significance - corrected')
# High
# Main plot
axd['B2'].plot(times, high.mean(0), alpha=1, zorder=10, color='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['B2'].plot(times[start:end], high.mean(0)[start:end], alpha=1, zorder=10, color=c3)
axd['B2'].fill_between(times, high.mean(0) - sem, high.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')    
# Highlight significant regions
axd['B2'].fill_between(times, high.mean(0) - sem, high.mean(0) + sem, where=sig, alpha=0.3, zorder=5, facecolor=c3, label='Pattern significance')    
# Low
# Main plot
axd['B2'].plot(times, low.mean(0), alpha=1, zorder=10, color='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['B2'].plot(times[start:end], low.mean(0)[start:end], alpha=1, zorder=10, color=c4)
axd['B2'].fill_between(times, low.mean(0) - sem, low.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
# Highlight significant regions
axd['B2'].fill_between(times, low.mean(0) - sem, low.mean(0) + sem, where=sig, alpha=0.3, zorder=5, facecolor=c4, label='Random significance')    
axd['B2'].legend(frameon=False, loc='lower left')
axd['B2'].set_ylabel('cvMD', fontsize=11)
# axd['B2'].set_xticklabels([])
axd['B2'].set_title(f'Cross-validated Mahalanobis distance within random and pattern elements', fontsize=13)
axd['B2'].set_xlabel('Time (s)', fontsize=11)


### D2 ### Within subject learning index correlation
diff_sess = zscore(diff_sess, axis=1)
axd['D'].axhline(0, color="grey", alpha=0.5)
all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
# axd['D'].plot(times, all_rhos.mean(0))
p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
# Main plot
axd['D'].plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['D'].plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=c6)
axd['D'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')    
# Highlight significant regions
axd['D'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.2, zorder=5, facecolor=c6, label='Significance')
axd['D'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=0.1, zorder=5, facecolor=c6)
# axd['D'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2)
# axd['D'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
# axd['D'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=.4, facecolor="#F2AD00", label='Significance - corrected')
axd['D'].set_ylabel("Spearman's rho", fontsize=11)
axd['D'].set_xlabel('Time (s)', fontsize=11)
axd['D'].legend(frameon=False, loc="lower right")
axd['D'].set_title(f'Correlation between similarity index and learning time course', fontsize=13)

cmap = plt.cm.get_cmap('tab20', len(subjects))
idx_rsa = np.where((times >= 0.3) & (times <= 0.5))[0]
mdiff = diff_sess[:, :, idx_rsa].mean(2)

### C2 ### Learning index fit
slopes, intercepts = [], []
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(mdiff[sub], learn_index_df.iloc[sub], 1)
    axd['C'].scatter(mdiff[sub], learn_index_df.iloc[sub], alpha=0.3)
    axd['C'].plot(mdiff[sub], slope * mdiff[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
rangee = np.linspace(mdiff.min(), mdiff.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
axd['C'].plot(rangee, mean_slope * rangee + mean_intercept, color='black', lw=4, label='Mean fit\nacross participants')

axd['C'].set_xlabel('Mean similarity index', fontsize=11)
axd['C'].set_ylabel('Learning index', fontsize=11)
axd['C'].set_title(f'Fit between mean representational similarity effect and learning', fontsize=13)

rhos = []
for sub in range(len(subjects)):
    r, p = spear(mdiff[sub], learn_index_df.iloc[sub])
    rhos.append(r)
pval = ttest_1samp(rhos, 0)[1]
axd['C'].legend(frameon=False, title=f"$p=${pval:.3f}", loc='upper left')

plt.savefig(figures_dir /  f"{lock}-rsa.pdf", transparent=True)
plt.close()
