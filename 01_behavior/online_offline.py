# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from tqdm.auto import tqdm

subjects = SUBJS15
path_data = HOME / 'raw_behavs'
figures_dir = FIGURES_DIR
saving = True

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

fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True, layout='tight')
for axx in ax.flatten():
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)

# online
sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)
ax[0].plot(blocks, np.nanmean(online_random, axis=0), color=color2, label='Random', zorder=9, alpha=1)
ax[0].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color=color2, alpha=0.1)

sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
ax[0].plot(blocks, np.nanmean(online_pattern, axis=0), color=color1, label='Pattern', zorder=10, alpha=1)
ax[0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)

ax[0].set_title('Online', fontstyle='italic')
ax[0].set_xticks(blocks[::2])
ax[0].set_xlabel('Block')
ax[0].set_ylabel(f'RT Difference (a.u.) - {n_bin} bins')
ax[0].grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)

# offline
sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)
ax[1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color=color2, zorder=10, alpha=1)
ax[1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color=color2, alpha=0.1)

sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
ax[1].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color=color1, zorder=9, alpha=1)
ax[1].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color=color1, alpha=0.2)
ax[1].set_title('Offline', fontstyle='italic')
ax[1].set_xticks(blocks[1::2])
ax[1].set_xlabel('Block')
ax[1].grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)
if saving: 
    fig.savefig(figures_dir / 'online_offline_learning.png', dpi=300)
    plt.close()
    
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