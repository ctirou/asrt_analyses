# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from base import *
from config import *
from tqdm.auto import tqdm
import os.path as op

subjects = SUBJS15
path_data = HOME / 'raw_behavs'
figures_dir = FIGURES_DIR
saving = False

sessions = ['0', '1', '2', '3', '4']
blocks = np.arange(1, 24)
n = len(subjects)

online_pattern, online_random = [], []
offline_pattern, offline_random = [], []

n_bin = 6

for subject in tqdm(subjects):
    
    # Sort behav files
    path_to_behav_dir = path_data / subject
    behav_dir = os.listdir(path_to_behav_dir)
    behav_files_filter = [f for f in behav_dir if not f.startswith('.')]
    behav_files = sorted([f for f in behav_files_filter if '_eASRT_Practice' in f or '_eASRT_Epoch' in f])
    behav_sessions = [behav_files[-1]] + behav_files[:-1]
    
    behav_df = []            
    for i, behav_session in enumerate(behav_sessions):
        behav_fname = path_data / subject / behav_session
        behav = pd.read_csv(behav_fname, sep='\t')
        # behav.reset_index(inplace=True)
        if i == 0:
            behav.columns = [col for col in behav.columns if col not in ['isi_if_correct', 'isi_if_incorrect']] + ['isi_if_correct', 'isi_if_incorrect']
        behav['session'] = i
        behav_df.append(behav)
    behav_df = pd.concat(behav_df, ignore_index=True)
    
    behav_df.loc[behav_df.session != 0, 'block'] += 3
    behav_df = behav_df[behav_df.talalat == 1].reset_index(drop=True)
    blocks = behav_df.block.unique()
    
    pat, rand = [], []
    
    for block in blocks:
        
        pat_block_df = behav_df[(behav_df.trialtype == 1) & (behav_df.block == block)]
        head = pat_block_df.head(n_bin).RT.to_list()
        tail = pat_block_df.tail(n_bin).RT.to_list()
        pat.append([np.mean(head), np.mean(tail)])
        
        rand_block_df = behav_df[(behav_df.trialtype == 2) & (behav_df.block == block)]
        head = rand_block_df.head(n_bin).RT.to_list()
        tail = rand_block_df.tail(n_bin).RT.to_list()
        rand.append([np.mean(head), np.mean(tail)])
        
    pat, rand = np.array(pat), np.array(rand) # shape (n_blocks, 2)
    
    online_pat = pat[:, 1] - pat[:, 0]
    online_rand = rand[:, 1] - rand[:, 0]
    
    offline_pat = pat[1:, 0] - pat[:-1, 1]
    offline_rand = rand[1:, 0] - rand[:-1, 1]
    
    online_pattern.append(online_pat)
    online_random.append(online_rand)
    offline_pattern.append(offline_pat)
    offline_random.append(offline_rand)
    
online_pattern = np.array(online_pattern) # shape (n_subjects, n_blocks)
online_random = np.array(online_random)
offline_pattern = np.array(offline_pattern) # shape (n_subjects, n_blocks - 1)
offline_random = np.array(offline_random)    

plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})
color1 = "#FFD966"
color2 = "#FF718A"

# fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True, layout='tight')
# for axx in ax.flatten():
#     axx.spines['top'].set_visible(False)
#     axx.spines['right'].set_visible(False)

# # online
# sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)
# ax[0].plot(blocks, np.nanmean(online_random, axis=0), color=color2, label='Random', zorder=9, alpha=1)
# ax[0].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color=color2, alpha=0.1)

# sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
# ax[0].plot(blocks, np.nanmean(online_pattern, axis=0), color=color1, label='Pattern', zorder=10, alpha=1)
# ax[0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)

# ax[0].set_title('Online', fontstyle='italic')
# ax[0].set_xticks(blocks[::2])
# ax[0].set_xlabel('Block')
# ax[0].set_ylabel(f'RT Difference (a.u.) - {n_bin} bins')
# ax[0].grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)

# # offline
# sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)
# ax[1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color=color2, zorder=10, alpha=1)
# ax[1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color=color2, alpha=0.1)

# sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
# ax[1].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color=color1, zorder=9, alpha=1)
# ax[1].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)
# ax[1].set_title('Offline', fontstyle='italic')
# ax[1].set_xticks(blocks[1::2])
# ax[1].set_xlabel('Block')
# ax[1].grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)
# if saving: 
#     fig.savefig(figures_dir / 'online_offline_learning.png', dpi=300)
#     plt.close()
    
fig, ax = plt.subplots(4, 2, figsize=(7, 10), sharey=True, sharex=False, layout='tight')
for axx in ax.flatten():
    axx.grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)
    axx.grid(True, which='minor', alpha=0.1, linestyle=':', color='gray', linewidth=0.3)
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    axx.set_xticks(blocks[::2])
    axx.axvspan(1, 3, color='gray', alpha=0.1)
    
# Online pattern
ax[0, 0].set_title('Online', fontstyle='italic')
sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
ax[0, 0].plot(blocks, np.nanmean(online_pattern, axis=0), color=color1, label='Pattern', zorder=10, alpha=1)
ax[0, 0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)
ax[0, 0].set_ylabel('Pattern')
# Offline pattern
ax[0, 1].set_title('Offline', fontstyle='italic')
sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
ax[0, 1].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color=color1, zorder=9, alpha=1)
ax[0, 1].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)

# Online random
sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)
ax[1, 0].plot(blocks, np.nanmean(online_random, axis=0), color=color2, label='Random', zorder=9, alpha=1)
ax[1, 0].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color=color2, alpha=0.1)
ax[1, 0].set_ylabel('Random')
# Offline random
sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)
ax[1, 1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color=color2, zorder=10, alpha=1)
ax[1, 1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color=color2, alpha=0.1)

# Combined pattern and random
# online
sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)
ax[2, 0].plot(blocks, np.nanmean(online_random, axis=0), color=color2, label='Random', zorder=9, alpha=1)
ax[2, 0].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color=color2, alpha=0.1)
sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
ax[2, 0].plot(blocks, np.nanmean(online_pattern, axis=0), color=color1, label='Pattern', zorder=10, alpha=1)
ax[2, 0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)
ax[2, 0].set_ylabel('Combined')
# offline
sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)
ax[2, 1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color=color2, zorder=10, alpha=1)
ax[2, 1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color=color2, alpha=0.1)
sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
ax[2, 1].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color=color1, zorder=9, alpha=1)
ax[2, 1].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)

# Within trials
# pattern
sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
ax[3, 0].set_title('Pattern', fontstyle='italic')
ax[3, 0].plot(blocks, np.nanmean(online_pattern, axis=0), color='green', zorder=10, alpha=1, label='Online')
ax[3, 0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color='green', alpha=0.2)
sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
ax[3, 0].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color='blue', zorder=9, alpha=1, label='Offline')
ax[3, 0].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color='blue', alpha=0.2)
ax[3, 0].set_ylabel('Within Trials')
ax[3, 0].legend(frameon=False, fontsize=8)
ax[3, 0].set_xlabel('Block')
# random
sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)
ax[3, 1].set_title('Random', fontstyle='italic')
ax[3, 1].plot(blocks, np.nanmean(online_random, axis=0), color='green', zorder=10, alpha=1, label='Online')
ax[3, 1].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color='green', alpha=0.2)
sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)
ax[3, 1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color='blue', zorder=9, alpha=1, label='Offline')
ax[3, 1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color='blue', alpha=0.2)
ax[3, 1].set_xlabel('Block')

if saving:
    fig.savefig(figures_dir / 'online_offline_learning_detailed.png', dpi=300)
    plt.close()
    
# RS offline
data_type = "rdm_blocks" 
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / data_type / subject
    # read behav        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern_blocks, random_blocks = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            pattern_blocks.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
            random_blocks.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-1-1.npy")
        rand_bsl = np.load(res_path / "rand-1-1.npy")
        for i in range(3):
            pattern_blocks[i] = pat_bsl.copy()
            random_blocks[i] = rand_bsl.copy()
    pattern_blocks = np.array(pattern_blocks)
    random_blocks = np.array(random_blocks)
    high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False)
    all_pats.append(high.mean(0))
    all_rands.append(low.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)
bsl_pat = np.nanmean(all_pats[:, :3, :], 1) 
bsl_rand = np.nanmean(all_rands[:, :3, :], 1)
pat = all_pats - bsl_pat[:, np.newaxis, :]
rand = all_rands - bsl_rand[:, np.newaxis, :]
diff_rp = rand - pat
# extract index from GAMM segments
segments = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors.csv")
start = segments.iloc[0]['start'] if data_type.endswith("new") else segments.iloc[1]['start']
end = segments.iloc[0]['end'] if data_type.endswith("new") else segments.iloc[1]['end']
idx = np.arange(int(start), int(end)+1)
# block-wise mean for the significant time window
diff_rp_blocks = np.nanmean(diff_rp[:, :, idx], axis=(-1)) # shape (n_subjects, n_blocks)

offline_rsa = np.diff(diff_rp_blocks, axis=1) # shape (n_subjects, n_blocks - 1)
online_rsa = diff_rp_blocks.copy() # shape (n_subjects, n_blocks)

# PA offline
# --- Temporal generalization sensors --- blocks ---
data_type = 'scores_blocks'
subjects = SUBJS15
times = np.linspace(-4, 4, 813)
pats_blocks, rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    pattern, random = [], []
    for block in range(1, 24):
        pfname = res_path / f'pat-{block}.npy' if block not in [1, 2, 3] else res_path / f'pat-0-{block}.npy'
        rfname = res_path / f'rand-{block}.npy' if block not in [1, 2, 3] else res_path / f'rand-0-{block}.npy'
        pattern.append(np.load(pfname))
        random.append(np.load(rfname))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-4.npy")
        rand_bsl = np.load(res_path / "rand-4.npy")
        for i in range(3):
            pattern[i] = pat_bsl.copy()
            random[i] = rand_bsl.copy()
    pats_blocks.append(np.array(pattern))
    rands_blocks.append(np.array(random))
pats_blocks = np.array(pats_blocks)
rands_blocks = np.array(rands_blocks)
conts_blocks = pats_blocks - rands_blocks
# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_blocks = []
for sub in range(len(subjects)):
    tg = []
    dg = []
    for block in range(23):
        data = conts_blocks[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    box_blocks.append(np.array(tg))
box_blocks = np.array(box_blocks)

offline_pa = np.diff(box_blocks, axis=1)
online_pa = box_blocks.copy()

# plot all offlines

blocks = np.arange(1, 24)
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})

fig, ax = plt.subplots(3, 1, figsize=(5, 12), sharey=False, layout='tight')
for axx in ax.flatten():
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    axx.grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)
    axx.grid(True, which='minor', alpha=0.1, linestyle=':', color='gray', linewidth=0.3)
    axx.set_xticks(blocks[::2])
    axx.axhline(0, color='gray', alpha=0.5, linestyle='-', linewidth=1)
    axx.axvspan(1, 3, color='gray', alpha=0.1)
# PA
mean_pa = np.nanmean(offline_pa, axis=0)
sem = np.nanstd(offline_pa, axis=0) / np.sqrt(n)
ax[0].set_title('PA', fontstyle='italic')
ax[0].plot(blocks[1:], mean_pa, color='blue', zorder=9, alpha=0.3)
ax[0].plot(blocks[1:], gaussian_filter1d(mean_pa, sigma=2), color='blue', zorder=10, alpha=1)
ax[0].fill_between(blocks[1:], mean_pa - sem, mean_pa + sem, color='blue', alpha=0.1)
# Behavioral
offline_behav = offline_pattern - offline_random
mean_beh = np.nanmean(offline_behav, axis=0)
sem = np.nanstd(offline_behav, axis=0) / np.sqrt(n)
ax[1].set_title('Behavior', fontstyle='italic')
ax[1].plot(blocks[1:], mean_beh, color='purple', zorder=11, alpha=0.3)
ax[1].plot(blocks[1:], gaussian_filter1d(mean_beh, sigma=2), color='purple', zorder=12, alpha=1)
ax[1].fill_between(blocks[1:], mean_beh - sem, mean_beh + sem, color='purple', alpha=0.2)
# RSA
mean_rsa = np.nanmean(offline_rsa, axis=0)
sem = np.nanstd(offline_rsa, axis=0) / np.sqrt(n)
ax[2].set_title('RS', fontstyle='italic')
ax[2].plot(blocks[1:], mean_rsa, color='green', zorder=10, alpha=0.3)
ax[2].plot(blocks[1:], gaussian_filter1d(mean_rsa, sigma=2), color='green', zorder=11, alpha=1)
ax[2].fill_between(blocks[1:], mean_rsa - sem, mean_rsa + sem, color='green', alpha=0.2)
ax[2].set_xlabel('Block')

# compute correlation between PA and behavior
from scipy.stats import ttest_1samp, spearmanr as spear
online_behav = online_pattern - online_random

offline_behav = offline_behav - np.nanmean(offline_behav, axis=0)
online_behav = online_behav - np.nanmean(online_behav, axis=0)
offline_pa = offline_pa - np.nanmean(offline_pa, axis=0)
online_pa = online_pa - np.nanmean(online_pa, axis=0)
offline_rsa = offline_rsa - np.nanmean(offline_rsa, axis=0)
online_rsa = online_rsa - np.nanmean(online_rsa, axis=0)

offline_pattern = offline_pattern - np.nanmean(offline_pattern, axis=0)
online_pattern = online_pattern - np.nanmean(online_pattern, axis=0)

cmap = plt.cm.get_cmap('tab20', len(subjects))

print("\n---\n")

rhos_pa_off = np.array([[spear(offline_behav[sub], offline_pa[sub])[0]] for sub in range(len(subjects))])
rhos_pa_off, _, _ = fisher_z_and_ttest(rhos_pa_off)
pval_unc = ttest_1samp(rhos_pa_off, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Offline - PA vs Behavior: r = {rhos_pa_off.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")

rhos_rsa_off = np.array([[spear(offline_behav[sub], offline_rsa[sub])[0]] for sub in range(len(subjects))])
rhos_rsa_off, _, _ = fisher_z_and_ttest(rhos_rsa_off)
pval_unc = ttest_1samp(rhos_rsa_off, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Offline - RSA vs Behavior: r = {rhos_rsa_off.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")

rhos_pa_on = np.array([[spear(online_behav[sub, 1:], offline_pa[sub])[0]] for sub in range(len(subjects))])
rhos_pa_on, _, _ = fisher_z_and_ttest(rhos_pa_on)
pval_unc = ttest_1samp(rhos_pa_on, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Online - PA vs Behavior: r = {rhos_pa_on.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")

rhos_rsa_on = np.array([[spear(online_behav[sub, 1:], offline_rsa[sub])[0]] for sub in range(len(subjects))])
rhos_rsa_on, _, _ = fisher_z_and_ttest(rhos_rsa_on)
pval_unc = ttest_1samp(rhos_rsa_on, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Online - RSA vs Behavior: r = {rhos_rsa_on.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")



rhos_pa_pat = np.array([[spear(offline_pa[sub], offline_pattern[sub])[0]] for sub in range(len(subjects))])
rhos_pa_pat, _, _ = fisher_z_and_ttest(rhos_pa_pat)
pval_unc = ttest_1samp(rhos_pa_pat, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Offline - PA vs Pattern: r = {rhos_pa_pat.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")

rhos_rsa_pat = np.array([[spear(offline_rsa[sub], offline_pattern[sub])[0]] for sub in range(len(subjects))])
rhos_rsa_pat, _, _ = fisher_z_and_ttest(rhos_rsa_pat)
pval_unc = ttest_1samp(rhos_rsa_pat, 0)[1]
sig_unc = pval_unc < 0.05
print(f"Offline - RSA vs Pattern: r = {rhos_rsa_pat.mean():.3f}, p_unc = {pval_unc[0]:.3f}, sig_unc = {sig_unc[0]}")

# Mecha vs pattern
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharey=True, layout='tight')
for axx in ax.flatten():
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
# PA vs pattern offline
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_pa[sub]
    y = offline_pattern[sub]
    slope, intercept = np.polyfit(x, y, 1)
    ax[0].scatter(x, y, alpha=0.3)
    ax[0].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_pa.min(), offline_pa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[0].plot(rangee, mean_slope * rangee + mean_intercept, color='blue', lw=4, label='Mean fit')
ax[0].set_title('Offline - PA vs Pattern', fontstyle='italic')
ax[0].set_xlabel('PA')
ax[0].set_ylabel('Pattern')
# RSA vs pattern offline
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_rsa[sub]
    y = offline_pattern[sub]
    slope, intercept = np.polyfit(x, y, 1)
    ax[1].scatter(x, y, alpha=0.3)
    ax[1].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_rsa.min(), offline_rsa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[1].plot(rangee, mean_slope * rangee + mean_intercept, color='blue', lw=4, label='Mean fit')
ax[1].set_title('Offline - RSA vs Pattern', fontstyle='italic')
ax[1].set_xlabel('RSA')
ax[1].set_ylabel('Pattern') 


# Mecha vs behavior
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharey=True, layout='tight')
for axx in ax.flatten():
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)

# PA vs behavior offline
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_pa[sub]
    y = offline_behav[sub]
    slope, intercept = np.polyfit(x, y, 1)
    ax[0, 0].scatter(x, y, alpha=0.3)
    ax[0, 0].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_pa.min(), offline_pa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[0, 0].plot(rangee, mean_slope * rangee + mean_intercept, color='blue', lw=4, label='Mean fit')
ax[0, 0].set_title('Offline - PA vs Behavior', fontstyle='italic')
ax[0, 0].set_xlabel('PA')
ax[0, 0].set_ylabel('Behavior')

# RSA vs behavior offline
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_rsa[sub]
    y = offline_behav[sub]
    slope, intercept = np.polyfit(x, y, 1)
    ax[0, 1].scatter(x, y, alpha=0.3)
    ax[0, 1].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_rsa.min(), offline_rsa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[0, 1].plot(rangee, mean_slope * rangee + mean_intercept, color='blue', lw=4, label='Mean fit')
ax[0, 1].set_title('Offline - RSA vs Behavior', fontstyle='italic')
ax[0, 1].set_xlabel('RSA')
ax[0, 1].set_ylabel('Behavior')

# PA vs behavior online
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_pa[sub]
    y = online_behav[sub, 1:]
    slope, intercept = np.polyfit(x, y, 1)
    ax[1, 0].scatter(x, y, alpha=0.3)
    ax[1, 0].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_pa.min(), offline_pa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[1, 0].plot(rangee, mean_slope * rangee + mean_intercept, color='green', lw=4, label='Mean fit')
ax[1, 0].set_title('Online - PA vs Behavior', fontstyle='italic')
ax[1, 0].set_xlabel('PA')
ax[1, 0].set_ylabel('Behavior')

# RSA vs behavior online
slopes, intercepts = [], []
for sub in range(len(subjects)):
    x = offline_rsa[sub]
    y = online_behav[sub, 1:]
    slope, intercept = np.polyfit(x, y, 1)
    ax[1, 1].scatter(x, y, alpha=0.3)
    ax[1, 1].plot(x, slope * x + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
rangee = np.linspace(offline_rsa.min(), offline_rsa.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax[1, 1].plot(rangee, mean_slope * rangee + mean_intercept, color='green', lw=4, label='Mean fit')
ax[1, 1].set_title('Online - RSA vs Behavior', fontstyle='italic')
ax[1, 1].set_xlabel('RSA')
ax[1, 1].set_ylabel('Behavior')

if saving:
    fig.savefig(figures_dir / 'online_offline_scatter.png', dpi=300)
    plt.close()

# Correlate online measures and offline behavior within subjects
rhos_pa_on_beh = np.array([[spear(offline_behav[sub], online_pa[sub, 1:])[0]] for sub in range(len(subjects))])
rhos_rsa_on_beh = np.array([[spear(offline_behav[sub], online_rsa[sub, 1:])[0]] for sub in range(len(subjects))])
rhos_pa_on_beh, _, _ = fisher_z_and_ttest(rhos_pa_on_beh)
rhos_rsa_on_beh, _, _ = fisher_z_and_ttest(rhos_rsa_on_beh)
pval_pa = ttest_1samp(rhos_pa_on_beh, 0)[1]
pval_rsa = ttest_1samp(rhos_rsa_on_beh, 0)[1]
sig_pa = pval_pa < 0.05
sig_rsa = pval_rsa < 0.05
print("\n---\n")
print(f"Online PA vs Offline Behavior: r = {rhos_pa_on_beh.mean():.3f}, p = {pval_pa[0]:.3f}, sig = {sig_pa[0]}")
print(f"Online RSA vs Offline Behavior: r = {rhos_rsa_on_beh.mean():.3f}, p = {pval_rsa[0]:.3f}, sig = {sig_rsa[0]}")


# Flatten all data for correlation
offline_pa_flat = offline_pa.flatten()
offline_rsa_flat = offline_rsa.flatten()
offline_behav_flat = offline_behav.flatten()

online_pa_flat = online_pa.flatten()
online_rsa_flat = online_rsa.flatten()
online_behav_flat = online_behav[:, 1:].flatten()

print("\n---\n")
flat_rhos_pa_off, pval = spear(offline_behav_flat, offline_pa_flat)[0], spear(offline_behav_flat, offline_pa_flat)[1]
print(f"Offline - PA vs Behavior (flat): r = {flat_rhos_pa_off:.3f}, p = {pval:.3f}")
flat_rhos_rsa_off, pval = spear(offline_behav_flat, offline_rsa_flat)[0], spear(offline_behav_flat, offline_rsa_flat)[1]
print(f"Offline - RSA vs Behavior (flat): r = {flat_rhos_rsa_off:.3f}, p = {pval:.3f}")

flat_rhos_pa_on, pval = spear(online_behav_flat, offline_pa_flat)[0], spear(online_behav_flat, offline_pa_flat)[1]
print(f"Online - PA vs Behavior (flat): r = {flat_rhos_pa_on:.3f}, p = {pval:.3f}")
flat_rhos_rsa_on, pval = spear(online_behav_flat, offline_rsa_flat)[0], spear(online_behav_flat, offline_rsa_flat)[1]
print(f"Online - RSA vs Behavior (flat): r = {flat_rhos_rsa_on:.3f}, p = {pval:.3f}")

flat_rhos_pa_pat_off, pval = spear(offline_pattern.flatten(), offline_pa_flat)[0], spear(offline_pattern.flatten(), offline_pa_flat)[1]
print(f"Offline - Pattern vs PA (flat): r = {flat_rhos_pa_pat_off:.3f}, p = {pval:.3f}")

flat_rhos_rsa_pat_off, pval = spear(offline_pattern.flatten(), offline_rsa_flat)[0], spear(offline_pattern.flatten(), offline_rsa_flat)[1]
print(f"Offline - Pattern vs RSA (flat): r = {flat_rhos_rsa_pat_off:.3f}, p = {pval:.3f}")