import os.path as op
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, zscore, spearmanr as spear
from tqdm.auto import tqdm
from base import *
from config import *

subjects = SUBJS15
data_path = DATA_DIR
metric = 'mahalanobis'

analysis = 'pat_high_rdm_high'
lock = 'stim'

# get times
times = np.linspace(-0.2, 0.6, 82)

figures_dir = FIGURES_DIR / "RSA" / "sensors"
ensure_dir(figures_dir)

timesg = np.linspace(-4, 4, 813)
filt = np.where((timesg > -0.21) & (timesg <= 0.6))[0]

all_highs, all_lows = [], []
patterns, randoms = [], []
# Decoding stuff
all_decoding = {}
all_decoding['pattern'] = []
all_decoding['random'] = []
for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / "rdm_skf" / subject
    ensure_dir(res_path)
        
    # RSA stuff
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pats, rands = [], []
    for epoch_num in range(5):
        pats.append(np.load(res_path / f"pat-{epoch_num}.npy"))
        rands.append(np.load(res_path / f"rand-{epoch_num}.npy"))
    pats = np.array(pats)
    rands = np.array(rands)
    
    high, low = get_all_high_low(pats, rands, sequence, False)
        
    high = np.array(high).mean(0)
    low = np.array(low).mean(0)

    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-b1.npy")
        rand_bsl = np.load(res_path / "rand-b1.npy")
        high[0] = pat_bsl
        low[0] = rand_bsl
    
    all_highs.append(high)
    all_lows.append(low)
    
    # Decoding stuff
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / 'scores_skf' / subject
    pat = np.diag(np.load(res_path / "pat-all.npy"))[filt]
    rand = np.diag(np.load(res_path / "rand-all.npy"))[filt]
    
    all_decoding['pattern'].append(pat)
    all_decoding['random'].append(rand)

for trial_type in ['pattern', 'random']:
    all_decoding[trial_type] = np.array(all_decoding[trial_type]) * 100

patterns = np.array(patterns)
randoms = np.array(randoms)

all_highs = np.array(all_highs)
all_lows = np.array(all_lows)

high = all_highs[:, 1:, :].mean(1) - all_highs[:, 0, :]
low = all_lows[:, 1:, :].mean(1) - all_lows[:, 0, :]
# high = all_highs[:, 1:, :].mean(1)
# low = all_lows[:, 1:, :].mean(1)
diff_lh = low - high

diff_sess = list()   
for i in range(5):
    # rev_low = all_lows[:, i, :]
    # rev_high = all_highs[:, i, :]
    rev_low = all_lows[:, i, :] - all_lows[:, 0, :]
    rev_high = all_highs[:, i, :] - all_highs[:, 0, :]
    diff_sess.append(rev_low - rev_high)
diff_sess = np.array(diff_sess).swapaxes(0, 1)

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
chance = 25
threshold = 0.05

outer = [['A', 'B'],
         ['C', 'D']]

cmap1 = colors['Darjeeling1']

c1 = "#0173B2"
c2 = "#3B9AB2"

c2 = "#56B4E9"

c3 = "#029E73"
c4 = "#CC78BC"
c5 = "#ECE133"
c6 = "#029E73"

cpat = "#FAD510"
crdm = "#FF718B"

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axd = plt.subplot_mosaic(outer, 
                              figsize=(15, 7), 
                              layout='tight',
                            #   gridspec_kw={'height_ratios': [0.5]}
                              )
for ax in axd.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_prop_cycle(cycler('color', colors['Darjeeling1']))
    if ax != axd['D']:
        if lock == 'stim':
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=-1, alpha=.1)
        else:
            ax.axvline(0, color='black')

### B1 ### cvMD
sem_high = np.std(high, axis=0) / np.sqrt(len(subjects))
sem_low = np.std(low, axis=0) / np.sqrt(len(subjects))
axd['A'].axhline(0, color='grey', alpha=0.5)
# High
axd['A'].plot(times, high.mean(0), alpha=1, zorder=10, color=cpat, label='Pattern')
# Plot significant regions separately
# for start, end in contiguous_regions(sig):
#     axd['A'].plot(times[start:end], high.mean(0)[start:end], alpha=1, zorder=10, color=c3)
axd['A'].fill_between(times, high.mean(0) - sem_high, high.mean(0) + sem_high, alpha=0.2, zorder=5, facecolor=cpat)    
# Highlight significant regions
# axd['A'].fill_between(times, high.mean(0) - sem, high.mean(0) + sem, where=sig, alpha=0.3, zorder=5, facecolor=c3)    
# Low
axd['A'].plot(times, low.mean(0), alpha=1, zorder=10, color=crdm, label='Random')
# Plot significant regions separately
# for start, end in contiguous_regions(sig):
#     axd['A'].plot(times[start:end], low.mean(0)[start:end], alpha=1, zorder=10, color=c4)
axd['A'].fill_between(times, low.mean(0) - sem_low, low.mean(0) + sem_low, alpha=0.1, zorder=5, facecolor=crdm)
# Highlight significant regions
# axd['A'].fill_between(times, low.mean(0) - sem, low.mean(0) + sem, where=sig, alpha=0.3, zorder=5, facecolor=c4)    
axd['A'].legend(frameon=False, loc='upper left')
axd['A'].set_ylabel('cvMD', fontsize=11)
axd['A'].set_ylim(-0.7, 0.7)
axd['A'].text(0.1, 0.55, '$Stimulus$', fontsize=11, ha='center')
axd['A'].set_title(f'Mahalanobis distance within pairs', fontsize=13)

### B2 ### Similarity index
axd['C'].axhline(0, color='grey', alpha=0.5)
p_values = decod_stats(diff_lh, -1)
sig = p_values < 0.05
axd['C'].plot(times, diff_lh.mean(0), alpha=1, zorder=10, color='C7')
sem = np.std(diff_lh, axis=0) / np.sqrt(len(subjects))
axd['C'].fill_between(times, diff_lh.mean(0) - sem, diff_lh.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
# Plot significant regions separately
for start, end in contiguous_regions(sig):
    axd['C'].plot(times[start:end], diff_lh.mean(0)[start:end], alpha=1, zorder=10, color=c2)
idx_rsa = np.where(sig)[0] # to compute mean later
# np.save(figures_dir / 'sig_rsa.npy', idx_rsa)
# Plot the entire line in the default color
# Fill the entire area with a semi-transparent color
# Overlay significant regions with the specified color
axd['C'].fill_between(times, diff_lh.mean(0) - sem, diff_lh.mean(0) + sem, where=sig, alpha=0.4, zorder=5, facecolor=c2)
axd['C'].fill_between(times, diff_lh.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=c2)
# axd['C'].legend(frameon=False, loc="upper left")
axd['C'].text(np.mean(times[sig]), 0.1, '*', fontsize=25, ha='center', va='center', color=c2, weight='bold')
axd['C'].set_ylabel('Similarity index', fontsize=11)
axd['C'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axd['C'].set_xlabel('Time (s)', fontsize=11)
axd['C'].set_title('Similarity index time course', fontsize=13)

### D ### Correlation with learning index
diff_sess = zscore(diff_sess, axis=1)
axd['B'].axhline(0, color="grey", alpha=0.5)
all_rhos = np.array([[spear(learn_index_df.iloc[sub], diff_sess[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
# Plot the entire line in the default color
axd['B'].plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
# Fill the entire area with a semi-transparent color
axd['B'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
# Overlay significant regions with the specified color
for start, end in contiguous_regions(sig):
    axd['B'].plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=c6)
# Highlight significant regions
axd['B'].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.4, zorder=10, facecolor=c6)
axd['B'].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=c6)
axd['B'].set_ylabel("Spearman's rho", fontsize=11)
axd['B'].set_xlabel('Time (s)', fontsize=11)
axd['B'].text(np.mean(times[sig]), 0.1, '*', fontsize=25, ha='center', va='center', color=c6, weight='bold')
# axd['B'].legend(frameon=False, loc="lower right")
axd['B'].set_title('Similarity index and learning correlation time course', fontsize=13)
cmap = plt.cm.get_cmap('tab20', len(subjects))

### C2 ### Learning index fit
idx_rsa = np.where((times >= 0.3) & (times <= 0.6))[0]
mdiff = diff_sess[:, :, idx_rsa].mean(2)
np.save(figures_dir / "mean_rsa.npy", mdiff)
slopes, intercepts = [], []
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(mdiff[sub], learn_index_df.iloc[sub], 1)
    axd['D'].scatter(mdiff[sub], learn_index_df.iloc[sub], alpha=0.3)
    axd['D'].plot(mdiff[sub], slope * mdiff[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
rangee = np.linspace(mdiff.min(), mdiff.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
axd['D'].plot(rangee, mean_slope * rangee + mean_intercept, color='black', lw=4, label='Mean fit')
axd['D'].set_xlabel('Mean similarity index', fontsize=11)
axd['D'].set_ylabel('Learning index (ms)', fontsize=11)
axd['D'].set_title(f'Similarity index and learning fit', fontsize=13)
rhos = []
for sub in range(len(subjects)):
    r, p = spear(mdiff[sub], learn_index_df.iloc[sub])
    rhos.append(r)
pval = ttest_1samp(rhos, 0)[1]
print(f"Spearman's rho: {np.mean(rhos):.2f}, p-value: {pval:.3f}")
ptext = f"p = {pval:.2f}" if pval > 0.001 else "p < 0.001"
axd['D'].legend(frameon=False, title=ptext, loc='upper left')

plt.savefig(figures_dir /  "rsa-final_v3.pdf", transparent=True)
plt.close()

# fig, ax = plt.subplots(1, 1, figsize=(12, 4), layout='tight')
# ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=-1, alpha=.1)
# ax.axhline(0, color='grey', alpha=0.5)
# practice = all_lows[:, 0, :] - all_highs[:, 0, :]
# ax.plot(times, practice.mean(0), label='prac', color='black')
# for j in range(1, 5):
#     ax.plot(times, diff_sess[:, j, :].mean(0), label=j)
# ax.legend(ncol=2, frameon=False)
# ax.set_title('Sensor space RSA', fontstyle='italic')