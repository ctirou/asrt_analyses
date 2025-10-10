# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

path_data = HOME / 'raw_behavs'
figures_dir = FIGURES_DIR

subjects = SUBJS15

pattern_RT = {f'Epoch_{i}': [] for i in range(5)}
random_high_RT = {f'Epoch_{i}': [] for i in range(5)}
random_low_RT = {f'Epoch_{i}': [] for i in range(5)}
all_RT = {f'Epoch_{i}': [] for i in range(5)}

sessions = ['0', '1', '2', '3', '4']
blocks = np.arange(23)
n = len(subjects)

subdict = {}
learn_index_dict = {}

learn_index_blocks_d = {}
pattern_blocks = {}
random_high_blocks = {}
random_low_blocks = {}
all_blocks = {}

for subject in tqdm(subjects):
    
    # Sort behav files
    path_to_behav_dir = path_data / subject
    behav_dir = os.listdir(path_to_behav_dir)
    behav_files_filter = [f for f in behav_dir if not f.startswith('.')]
    behav_files = sorted([f for f in behav_files_filter if '_eASRT_Practice' in f or '_eASRT_Epoch' in f])
    behav_sessions = [behav_files[-1]] + behav_files[:-1]
    
    subdict[subject] = {}
    learn_index_dict[subject] = {}
    
    learn_index_blocks_d[subject] = {}
    pattern_blocks[subject] = {}
    random_high_blocks[subject] = {}
    random_low_blocks[subject] = {}
    all_blocks[subject] = {}
    
    for i, behav_session in enumerate(behav_sessions):
        
        subdict[subject][i] = {"all": [],
                               "pattern": [], 
                               "random_high": [],
                               "random_low": [],
                               'one': [],
                               'two': []}
        
        behav_fname = path_data / subject / behav_session
        behav_df = pd.read_csv(behav_fname, sep='\t')
        behav_df.reset_index(inplace=True)
        if i == 0:
            behav_df.columns = [col for col in behav_df.columns if col not in ['isi_if_correct', 'isi_if_incorrect']] + [''] * len(['isi_if_correct', 'isi_if_incorrect'])

        patterns, randoms = [], []
        ones, twos = [], []
        
        # Initialize RT lists for each epoch
        
        for j, k in enumerate(behav_df['RT']):
            if behav_df['triplet'][j] in [30, 32, 34]:
                if behav_df['talalat'][j] == 1:
                    all_RT[f'Epoch_{i}'].append(behav_df['RT'][j])
                    subdict[subject][i]["all"].append((behav_df['RT'][j])) 
                    if behav_df['triplet'][j] == 30:
                        pattern_RT[f'Epoch_{i}'].append(behav_df['RT'][j])
                        subdict[subject][i]["pattern"].append((behav_df['RT'][j])) 
                    elif behav_df['triplet'][j] == 32: # sub11 has 34 for random_high instead of 32
                        random_high_RT[f'Epoch_{i}'].append(behav_df['RT'][j])
                        subdict[subject][i]["random_high"].append((behav_df['RT'][j]))
                    elif behav_df['triplet'][j] == 34:
                        random_low_RT[f'Epoch_{i}'].append(behav_df['RT'][j])
                        subdict[subject][i]["random_low"].append((behav_df['RT'][j]))        
                    
                    if behav_df['trialtype'][j] == 1:
                        subdict[subject][i]["one"].append((behav_df['RT'][j]))
                    elif behav_df['trialtype'][j] == 2:
                        subdict[subject][i]["two"].append((behav_df['RT'][j]))
                    
            else:
                continue
        
        subdict[subject][i]["all"] = np.mean(subdict[subject][i]["all"]) if subdict[subject][i]["all"] else np.nan
        subdict[subject][i]["pattern"] = np.mean(subdict[subject][i]["pattern"]) if subdict[subject][i]["pattern"] else np.nan
        subdict[subject][i]["random_high"] = np.mean(subdict[subject][i]["random_high"]) if subdict[subject][i]["random_high"] else np.nan
        subdict[subject][i]["random_low"] = np.mean(subdict[subject][i]["random_low"]) if subdict[subject][i]["random_low"] else np.nan
        
        patterns.append(np.mean(subdict[subject][i]["pattern"]))
        randoms.append(np.mean(subdict[subject][i]["random_high"]))
        
        ones.append(np.mean(subdict[subject][i]["one"]))
        twos.append(np.mean(subdict[subject][i]["two"]))
        
        # learning_index = (np.mean(randoms) - np.mean(patterns)) / np.mean(randoms)
        prac_index = np.mean(twos) - np.mean(ones) if i == 0 else 0
        learning_index = np.mean(randoms) - np.mean(patterns)
        learn_index_dict[subject][i] = learning_index if i != 0 else prac_index
        
        nblocks = np.unique(behav_df.block)
        for block in nblocks:
            
            idx = "0" + str(block) if i == 0 else str(block)

            learn_index_blocks_d[subject][idx] = 0
            pattern_blocks[subject][idx] = 0
            random_high_blocks[subject][idx] = 0
            
            pat, rand = [], []
            one, two = [], []
            all_of_them = []
            for j, _ in enumerate(behav_df.RT):
                if behav_df.block[j] == block:
                    all_of_them.append(behav_df.RT[j])
                    if behav_df.triplet[j] == 30:
                        pat.append(behav_df.RT[j])
                    elif behav_df.triplet[j] == 32:
                        rand.append(behav_df.RT[j])
                    if i == 0:
                        if behav_df.trialtype[j] == 1:
                            one.append(behav_df.RT[j])
                        elif behav_df.trialtype[j] == 2:
                            two.append(behav_df.RT[j])
            
            index = np.mean(rand) - np.mean(pat)
            prac_index = np.mean(two) - np.mean(one) if i == 0 else 0
            learn_index_blocks_d[subject][idx] = index if idx not in ['01', '02', '03'] else prac_index
            
            pattern_blocks[subject][idx] = np.mean(pat) if idx not in ['01', '02', '03'] else np.mean(one)
            random_high_blocks[subject][idx] = np.mean(rand) if idx not in ['01', '02', '03'] else np.mean(two)
            all_blocks[subject][idx] = np.mean(all_of_them)

color1 = "#FFD966"
color2 = "#FF718A"
color3 = "black" 

# ----------------------------------- USING BLOCKS -----------------------------------                
# Save block learning indices to CSV
learn_index_blocks_df = pd.DataFrame.from_dict(learn_index_blocks_d, orient='index')
if not op.exists(figures_dir / 'behav' / 'learning_indices_blocks.csv'):
    learn_index_blocks_df.to_csv(figures_dir / 'behav' / 'learning_indices_blocks.csv', sep=',')
pattern_blocks_df = pd.DataFrame.from_dict(pattern_blocks, orient='index')
if not op.exists(figures_dir / 'behav' / 'pattern_blocks.csv'):
    pattern_blocks_df.to_csv(figures_dir / 'behav' / 'pattern_blocks.csv', sep=',')
random_high_blocks_df = pd.DataFrame.from_dict(random_high_blocks, orient='index')
if not op.exists(figures_dir / 'behav' / 'random_high_blocks.csv'):
    random_high_blocks_df.to_csv(figures_dir / 'behav' / 'random_high_blocks.csv', sep=',')
all_blocks_df = pd.DataFrame.from_dict(all_blocks, orient='index')
if not op.exists(figures_dir / 'behav' / 'all_blocks.csv'):
    all_blocks_df.to_csv(figures_dir / 'behav' / 'all_blocks.csv', sep=',')
    
# Combined RT and learning index
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})
fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout="tight")
ax.autoscale()
blocks = ['01', '02', '03'] + [str(i) for i in range(1, 21)]
x = np.arange(len(blocks))  
width = 0.3
# Reaction times
for subject in subjects:
    for j, i in enumerate(blocks):
        xpos = x[j]
        # All (center, no dodge)
        ax.scatter(xpos, all_blocks_df.loc[subject][i],
                   color='grey', marker=".", alpha=0.2)
        if i not in ['01', '02', '03']:
            # Pattern (dodged left)
            ax.scatter(xpos - width, pattern_blocks_df.loc[subject][i],
                       color=color1, marker=".", alpha=0.3)
            # Random (dodged right)
            ax.scatter(xpos + width, random_high_blocks_df.loc[subject][i],
                       color=color2, marker=".", alpha=0.2)
ax.plot(x, all_blocks_df.mean(axis=0), '-o',
        color=color3, label="All", markersize=7, alpha=.7)
ax.plot(x[3:] - width, pattern_blocks_df.mean(axis=0)[3:], '-o',
        color=color1, label="Pattern", markersize=7, alpha=1)
ax.plot(x[3:] + width, random_high_blocks_df.mean(axis=0)[3:], '-o',
        color=color2, label="Random", markersize=7, alpha=1)
ax.legend(loc='lower left', frameon=False, title=f"n = {n}")
ax.set_ylabel("Reaction time (ms)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelbottom=False)
# Learning index
divider = make_axes_locatable(ax)
axlow = divider.append_axes("bottom", 1.2, pad=0.2, sharex=ax)
axlow.autoscale()
learning_indices_mean = learn_index_blocks_df.mean(axis=0)
learning_indices_stderr = learn_index_blocks_df.std(axis=0)/np.sqrt(len(subjects))
bar_width = 0.6
axlow.bar(blocks, learning_indices_mean, yerr=learning_indices_stderr, alpha=0.7, capsize=5, color="#029E73", width=bar_width)
axlow.set_ylabel("Learning index (ms)", fontsize=12)
axlow.spines['top'].set_visible(False)
axlow.spines['right'].set_visible(False)
axlow.set_xticks(x)
axlow.set_xticklabels(blocks)
axlow.set_xlabel("Block", fontsize=12)
fig.savefig(figures_dir / 'behav' / 'combined_blocks.pdf', transparent=True)
plt.close()

# ----------------------------------- USING RUNS -----------------------------------
# Save session learning indices to CSV
learn_index_df = pd.DataFrame.from_dict(learn_index_dict, orient='index')
if not op.exists(figures_dir / 'behav' / 'learning_indices_runs.csv'):
    learn_index_df.to_csv(figures_dir / 'behav' / 'learning_indices_runs.csv', sep='\t')

# Calculate means and standard errors
mean_all = [np.mean(all_RT[f'Epoch_{i}']) for i in range(5)]
stderr_all = [np.std(all_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(5)]
mean_pattern = [np.mean(pattern_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_pattern = [np.std(pattern_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_high = [np.mean(random_high_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_random_high = [np.std(random_high_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_low = [np.mean(random_low_RT[f'Epoch_{i}']) for i in range(5)]
stderr_random_low = [np.std(random_low_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(5)]

# Combined RT and learning index
fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout="tight")
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})
ax.autoscale()
for subject in subjects:
    for i in range(5):
        ax.scatter(str(i), subdict[subject][i]["all"], color=color3, marker=".", alpha=0.3)
        if i > 0:
            ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, marker=".", alpha=0.4)
            ax.scatter(str(i), subdict[subject][i]["random_high"], color=color2, marker=".", alpha=0.4)
ax.plot(sessions, mean_all, '-o', color=color3, label="All", markersize=7, alpha=.7)
ax.plot(sessions[1:], mean_pattern, '-o', color=color1, label="Pattern", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_random_high, '-o', color=color2, label="Random", markersize=7, alpha=1)
ax.legend(loc='lower left', frameon=False, title=f"n = {n}")
ax.set_ylabel("Reaction time (ms)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelbottom=False)
divider = make_axes_locatable(ax)
axlow = divider.append_axes("bottom", 1.2, pad=0.4, sharex=ax)
axlow.autoscale()
learning_indices_mean = learn_index_df.mean(axis=0)
learning_indices_stderr = learn_index_df.sem(axis=0)
bar_width = 0.5
axlow.bar(sessions, learning_indices_mean, yerr=learning_indices_stderr, alpha=0.7, capsize=5, color="#029E73", width=bar_width)
axlow.set_ylabel("Learning index (ms)", fontsize=12)
axlow.spines['top'].set_visible(False)
axlow.spines['right'].set_visible(False)
axlow.set_xticklabels(['Practice', '1', '2', '3', '4'])
axlow.set_xlabel("Run", fontsize=12)
fig.savefig(figures_dir / 'behav' / 'combined_runs.pdf', transparent=True)
plt.close()