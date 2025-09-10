import os.path as op
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from base import *
from config import *

subjects = SUBJS15
data_path = DATA_DIR

# get times
times = np.linspace(-0.2, 0.6, 82)

figures_dir = FIGURES_DIR / "RSA" / "sensors"
ensure_dir(figures_dir)

data_type = "rdm_blocks_new"
# data_type = "rdm_blocks"
bsl_practice = False

all_patterns, all_randoms = [], []
# all_highs, all_lows = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / data_type / subject
    ensure_dir(res_path)
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    # pattern_sessions, random_sessions = [], []
    pattern_blocks, random_blocks = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            pattern_blocks.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
            random_blocks.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
        # pat_files = list(res_path.glob(f"pat-{epoch_num}-*.npy"))
        # rand_files = list(res_path.glob(f"rand-{epoch_num}-*.npy"))
        # pat_files.sort()
        # rand_files.sort()
        # for pat_file, rand_file in zip(pat_files, rand_files):
        #     pats.append(np.load(pat_file))
        #     rands.append(np.load(rand_file))
        # pattern_sessions.append(np.nanmean(pats, axis=0))
        # random_sessions.append(np.nanmean(rands, axis=0))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-1-1.npy")
        rand_bsl = np.load(res_path / "rand-1-1.npy")
        for i in range(3):
            pattern_blocks[i] = pat_bsl.copy()
            random_blocks[i] = rand_bsl.copy()
        # pattern_sessions[0] = pat_bsl
        # random_sessions[0] = rand_bsl
    # pattern_sessions = np.array(pattern_sessions)
    # random_sessions = np.array(random_sessions)
    # high, low = get_all_high_low(pattern_sessions, random_sessions, sequence, False)
    # high = np.array(high).mean(0)
    # low = np.array(low).mean(0)
    # all_highs.append(high)
    # all_lows.append(low)
    pattern_blocks = np.array(pattern_blocks)
    random_blocks = np.array(random_blocks)
    high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False)
    high = np.array(high).mean(0)
    low = np.array(low).mean(0)
    all_patterns.append(high)
    all_randoms.append(low)

# time resolved
all_patterns = np.array(all_patterns)
all_randoms = np.array(all_randoms)
high = np.nanmean(all_patterns[:, 3:, :], 1) - np.nanmean(all_patterns[:, :3, :], 1) if bsl_practice \
    else np.nanmean(all_patterns[:, 3:, :], 1)
low = np.nanmean(all_randoms[:, 3:, :], 1) - np.nanmean(all_randoms[:, :3, :], 1) if bsl_practice \
    else np.nanmean(all_randoms[:, 3:, :], 1)
diff_lh = low - high

# block resolved
bsl_high = np.nanmean(all_patterns[:, :3, :], 1)
bsl_low = np.nanmean(all_randoms[:, :3, :], 1)
high_b = all_patterns - bsl_high[:, np.newaxis, :] if bsl_practice \
    else all_patterns
low_b = all_randoms - bsl_low[:, np.newaxis, :] if bsl_practice \
    else all_randoms
diff_b = low_b - high_b

# # session resolved
# all_highs = np.array(all_highs)
# all_lows = np.array(all_lows)
# diff_sess = list()   
# for i in range(5):
#     rev_low = all_lows[:, i, :] - all_lows[:, 0, :] if bsl_practice \
#         else all_lows[:, i, :]
#     rev_high = all_highs[:, i, :] - all_highs[:, 0, :] if bsl_practice \
#         else all_highs[:, i, :]
#     diff_sess.append(rev_low - rev_high)
# diff_sess = np.array(diff_sess).swapaxes(0, 1)

# learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
# Baseline correct learn_index_blocks by subtracting the mean across blocks for each subject
learn_index_blocks = learn_index_blocks.sub(learn_index_blocks.mean(axis=1), axis=0)

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
        ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=-1, alpha=.1)

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
# axd['A'].set_ylim(-0.7, 0.7)
axd['A'].text(0.1, 1.5, '$Stimulus$', fontsize=11, ha='center', va='top')
axd['A'].set_xlabel('Time (s)', fontsize=11)
axd['A'].set_title(f'Mahalanobis distance within pairs', fontsize=13)

### B2 ### Similarity index

gam_sig = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors.csv")
gam_sig = gam_sig[gam_sig['metric'] == 'RS']
arr = np.zeros(len(times), dtype=bool)
arr[gam_sig['start'][0]:gam_sig['end'][0] + 1] = True

win = np.where((times >= 0.3) & (times <= 0.55))[0]
axd['C'].axhline(0, color='grey', alpha=0.5)
pval = decod_stats(diff_lh, -1)
sig = pval < 0.05
sig = arr.copy()
sig_rsa = sig.copy()
pval_unc = ttest_1samp(diff_lh, 0)[1]
sig_unc = pval_unc < 0.05
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
axd['C'].text(np.mean(times[sig]), -0.1, '*', fontsize=25, ha='center', va='center', color=c2, weight='bold')
# mdiff = diff_lh[:, win].mean(1)
# mdiff_sig = ttest_1samp(mdiff, 0)[1] < 0.05
# if mdiff_sig:
#     axd['C'].fill_between(times[win], -0.20, -0.18, alpha=0.7, zorder=5, facecolor=c2)
#     axd['C'].text(np.mean(times[win]), -0.28, '*', fontsize=25, ha='center', va='center', color=c2, weight='bold')
# axd['C'].legend(frameon=False, loc="upper left")
axd['C'].set_ylabel('Similarity index', fontsize=11)
axd['C'].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axd['C'].set_xlabel('Time (s)', fontsize=11)
axd['C'].set_title('Similarity index time course', fontsize=13)

### D ### Correlation with learning index

gam_sig = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors.csv")
gam_sig = gam_sig[gam_sig['metric'] == 'RS CORR']
arr = np.zeros(len(times), dtype=bool)
arr[gam_sig['start'][1]:gam_sig['end'][1] + 1] = True

# Center diff_b for each subject by subtracting their mean across all blocks and times
diff_c = diff_b - np.nanmean(diff_b, axis=1, keepdims=True)
axd['B'].axhline(0, color="grey", alpha=0.5)
all_rhos = np.array([[spear(learn_index_blocks.iloc[sub], diff_c[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
sem = np.nanstd(all_rhos, axis=0) / np.sqrt(len(subjects))
p_values = decod_stats(all_rhos, -1)
sig = p_values < 0.05
sig = arr.copy()
# Plot the entire line in the default color
axd['B'].plot(times, np.nanmean(all_rhos, axis=0), alpha=1, zorder=10, color='C7')
# Fill the entire area with a semi-transparent color
axd['B'].fill_between(times, np.nanmean(all_rhos, axis=0) - sem, np.nanmean(all_rhos, axis=0) + sem, alpha=0.2, zorder=5, facecolor='C7')
# Overlay significant regions with the specified color
for start, end in contiguous_regions(sig):
    axd['B'].plot(times[start:end], np.nanmean(all_rhos, axis=0)[start:end], alpha=1, zorder=10, color=c6)
    cluster_center = np.mean(times[start:end])
    axd['B'].text(cluster_center, -0.07, '*', fontsize=25, ha='center', va='center', color=c6, weight='bold')
# Highlight significant regions
axd['B'].fill_between(times, np.nanmean(all_rhos, axis=0) - sem, np.nanmean(all_rhos, axis=0) + sem, where=sig, alpha=0.4, zorder=10, facecolor=c6)
axd['B'].fill_between(times, np.nanmean(all_rhos, axis=0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=c6)
axd['B'].set_ylabel("Spearman's rho", fontsize=11)
axd['B'].set_xlabel('Time (s)', fontsize=11)
# axd['B'].legend(frameon=False, loc="lower right")
axd['B'].set_title('Similarity index and learning correlation time course', fontsize=13)
# win = np.where((times >= 0.3) & (times <= 0.55))[0]
# m_rho = np.nanmean(all_rhos[:, win], axis=1)
# m_rho_sig = ttest_1samp(m_rho, 0)[1] < 0.05
# if m_rho_sig:
#     axd['B'].fill_between(times[win], -0.1, -0.09, alpha=0.7, zorder=5, facecolor=c6)
#     axd['B'].text(np.mean(times[win]), -0.15, '*', fontsize=25, ha='center', va='center', color=c6, weight='bold')

### C2 ### Learning index fit
cmap = plt.cm.get_cmap('tab20', len(subjects))
idx_rsa = np.where((times >= 0.3) & (times <= 0.55))[0]
idx_rsa = sig_rsa.copy()
idx_rsa = sig.copy()
mdiff = diff_b[:, :, idx_rsa].mean(-1)
mdiff = mdiff - np.nanmean(mdiff, axis=1, keepdims=True)  # center by subject
slopes, intercepts = [], []
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(mdiff[sub], learn_index_blocks.iloc[sub], 1)
    axd['D'].scatter(mdiff[sub], learn_index_blocks.iloc[sub], alpha=0.3)
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
    r, p = spear(mdiff[sub], learn_index_blocks.iloc[sub])
    rhos.append(r)
pval = ttest_1samp(rhos, 0)[1]
print(f"Spearman's rho: {np.mean(rhos):.2f}, p-value: {pval:.3f}")
ptext = f"p = {pval:.2f}" if pval > 0.001 else "p < 0.001"
axd['D'].legend(frameon=False, title=ptext, loc='lower right')
# if pval < 0.05:
#     axd['D'].text(-1.13, 10, '*', fontsize=25, ha='center', va='center', color='black', weight='bold')

fname = 'gam_rsa_new' if data_type.endswith('new') else 'gam_rsa'
fname += '_bsl.pdf' if bsl_practice else '_no_bsl.pdf'
plt.savefig(figures_dir /  fname, transparent=True)
plt.close()

# ---------------------- No practice correlation figure ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), layout='tight')
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Spearman's rho time course
gam_sig = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors_no_prac.csv")
gam_sig = gam_sig[gam_sig['metric'] == 'RS CORR']
arr2 = np.zeros(len(times), dtype=bool)
arr2[gam_sig['start'][1]:gam_sig['end'][1] + 1] = True
diff_c = diff_b - np.nanmean(diff_b, axis=1, keepdims=True)
ax1.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=-1, alpha=.1)
ax1.axhline(0, color="grey", alpha=0.5)
all_rhos = np.array([[spear(learn_index_blocks.iloc[sub, 3:], diff_c[sub, 3:, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
sem = np.nanstd(all_rhos, axis=0) / np.sqrt(len(subjects))
# p_values = decod_stats(all_rhos, -1)
# sig = p_values < 0.05
sig = arr2.copy()
ax1.plot(times, np.nanmean(all_rhos, axis=0), alpha=1, zorder=10, color='C7')
ax1.fill_between(times, np.nanmean(all_rhos, axis=0) - sem, np.nanmean(all_rhos, axis=0) + sem, alpha=0.2, zorder=5, facecolor='C7')
for start, end in contiguous_regions(sig):
    ax1.plot(times[start:end], np.nanmean(all_rhos, axis=0)[start:end], alpha=1, zorder=10, color=c6)
    cluster_center = np.mean(times[start:end])
    ax1.text(cluster_center, -0.13, '***', fontsize=25, ha='center', va='center', color=c6, weight='bold')
ax1.fill_between(times, np.nanmean(all_rhos, axis=0) - sem, np.nanmean(all_rhos, axis=0) + sem, where=sig, alpha=0.4, zorder=10, facecolor=c6)
ax1.fill_between(times, np.nanmean(all_rhos, axis=0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=c6)
ax1.set_ylabel("Spearman's rho", fontsize=11)
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_title('Similarity index and learning correlation time course', fontsize=13)

# Learning index fit
cmap = plt.cm.get_cmap('tab20', len(subjects))
idx_rsa = arr.copy()
mdiff = diff_b[:, 3:, idx_rsa].mean(-1)
mdiff = mdiff - np.nanmean(mdiff, axis=1, keepdims=True)  # center by subject
slopes, intercepts = [], []
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(mdiff[sub], learn_index_blocks.iloc[sub, 3:], 1)
    ax2.scatter(mdiff[sub], learn_index_blocks.iloc[sub, 3:], alpha=0.3)
    ax2.plot(mdiff[sub], slope * mdiff[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
rangee = np.linspace(mdiff.min(), mdiff.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax2.plot(rangee, mean_slope * rangee + mean_intercept, color='black', lw=4, label='Mean fit')
ax2.set_xlabel('Mean similarity index', fontsize=11)
ax2.set_ylabel('Learning index (ms)', fontsize=11)
ax2.set_title(f'Similarity index and learning fit', fontsize=13)
rhos = []
for sub in range(len(subjects)):
    r, p = spear(mdiff[sub], learn_index_blocks.iloc[sub, 3:])
    rhos.append(r)
pval = ttest_1samp(rhos, 0)[1]
print(f"Spearman's rho: {np.mean(rhos):.2f}, p-value: {pval:.3f}")
ptext = f"p = {pval:.2f}" if pval > 0.001 else "p < 0.001"
ax2.legend(frameon=False, title=ptext, loc='lower right')
if pval < 0.05:
    ax2.text(-1.5, -25, '*', fontsize=25, ha='center', va='center', color='black', weight='bold')

fname = 'rsa_no_prac_corr.pdf'
plt.savefig(figures_dir /  fname, transparent=True)
plt.close()
