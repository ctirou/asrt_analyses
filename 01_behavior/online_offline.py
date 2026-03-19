# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from config import *
from tqdm.auto import tqdm
from base import get_sequence

path_data = HOME / 'raw_behavs'
figures_dir = FIGURES_DIR

subjects = [sub for sub in SUBJS15 if sub != 'sub05']

sessions = ['0', '1', '2', '3', '4']
blocks = np.arange(1, 24)
n = len(subjects)

# online_dict = {}
# offline_dict = {}

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
    
    # online_dict[subject] = {}
    # offline_dict[subject] = {}

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

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(blocks, online_pat, label='Online Pattern', color='blue')
    # ax[0].plot(blocks, online_rand, label='Online Random', color='orange')
    # ax[0].set_title(f'{subject} - Online Learning')
    # ax[0].set_xlabel('Block')
    # ax[0].set_ylabel('RT Difference (Tail - Head)')
    # ax[0].legend()
    # ax[0].grid()
    
    # ax[1].plot(blocks[1:], offline_pat, label='Offline Pattern', color='blue')
    # ax[1].plot(blocks[1:], offline_rand, label='Offline Random', color='orange')
    # ax[1].set_title(f'{subject} - Offline Learning')
    # ax[1].set_xlabel('Block')
    # ax[1].set_ylabel('RT Difference (Next Head - Previous Tail)')
    # ax[1].legend()
    # ax[1].grid()
    # plt.tight_layout()
    
online_pattern = np.array(online_pattern) # shape (n_subjects, n_blocks)
online_random = np.array(online_random)
offline_pattern = np.array(offline_pattern) # shape (n_subjects, n_blocks - 1)
offline_random = np.array(offline_random)    

plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})
color1 = "#FFD966"
color2 = "#FF718A"

fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True, layout='tight')

for axx in ax.flatten():
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)

# online
sem_pat = np.nanstd(online_pattern, axis=0) / np.sqrt(n)
sem_rand = np.nanstd(online_random, axis=0) / np.sqrt(n)

ax[0].plot(blocks, np.nanmean(online_random, axis=0), color=color2, label='Random')
ax[0].fill_between(blocks, np.nanmean(online_random, axis=0) - sem_rand, np.nanmean(online_random, axis=0) + sem_rand, color=color2, alpha=0.3)

ax[0].plot(blocks, np.nanmean(online_pattern, axis=0), color=color1, label='Pattern')
ax[0].fill_between(blocks, np.nanmean(online_pattern, axis=0) - sem_pat, np.nanmean(online_pattern, axis=0) + sem_pat, color=color1, alpha=0.3)

ax[0].set_title('Online Learning\nTail - Head')
ax[0].set_xticks(blocks[::2])
ax[0].set_xlabel('Block')
ax[0].set_ylabel('RT Difference (a.u.)')
ax[0].grid(alpha=0.3)

# ax[0].legend()

# offline
sem_pat = np.nanstd(offline_pattern, axis=0) / np.sqrt(n)
sem_rand = np.nanstd(offline_random, axis=0) / np.sqrt(n)

ax[1].plot(blocks[1:], np.nanmean(offline_random, axis=0), color=color2)
ax[1].fill_between(blocks[1:], np.nanmean(offline_random, axis=0) - sem_rand, np.nanmean(offline_random, axis=0) + sem_rand, color=color2, alpha=0.3)

ax[1].plot(blocks[1:], np.nanmean(offline_pattern, axis=0), color=color1)
ax[1].fill_between(blocks[1:], np.nanmean(offline_pattern, axis=0) - sem_pat, np.nanmean(offline_pattern, axis=0) + sem_pat, color=color1, alpha=0.3)

ax[1].set_title('Offline Learning\nNext Head - Previous Tail')
ax[1].set_xticks(blocks[1::2])
ax[1].set_xlabel('Block')
ax[1].grid(alpha=0.2, linestyle='--', color='gray', linewidth=0.5)
# ax[1].set_ylabel('RT Difference (a.u.)')
plt.show()