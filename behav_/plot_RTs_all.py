import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from tqdm.auto import tqdm

path_data = DATA_DIR
figures_dir = FIGURES_DIR

subjects = SUBJS

pattern_RT = {
    'Epoch_0': list(),
    'Epoch_1': list(),
    'Epoch_2': list(),
    'Epoch_3': list(),
    'Epoch_4': list(),
}

random_high_RT = {
    'Epoch_0': list(),
    'Epoch_1': list(),
    'Epoch_2': list(),
    'Epoch_3': list(),
    'Epoch_4': list(),
}

random_low_RT = {
    'Epoch_0': list(),
    'Epoch_1': list(),
    'Epoch_2': list(),
    'Epoch_3': list(),
    'Epoch_4': list(),
}

sessions = ['0', '1', '2', '3', '4']
n = len(subjects)

subdict = {}
learn_index_dict = {}

for subject in tqdm(subjects):

    all_RT = list()
    pat_RT = list()
    rand_RT = list()
    
    subdict[subject] = {}
    learn_index_dict[subject] = {}
    
    for i in range(5):

        subdict[subject][i] = {"pattern": [], 
                               "random_high": [],
                               "random_low": []}
        
        fname_behav = op.join(path_data, 'behav', f'{subject}-{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)
        
        highs, lows = [], []
        
        for j, k in enumerate(behav_df['RTs']):
            all_RT.append(k)
            if behav_df['triplets'][j] == 30:
                pattern_RT[f'Epoch_{i}'].append(behav_df['RTs'][j]) # faire liste et moyennes puis append contenu liste dans dico
                subdict[subject][i]["pattern"].append((behav_df['RTs'][j])) 
            elif behav_df['triplets'][j] == 32:
                random_high_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                subdict[subject][i]["random_high"].append((behav_df['RTs'][j]))
            elif behav_df['triplets'][j] == 34:
                random_low_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                subdict[subject][i]["random_low"].append((behav_df['RTs'][j]))
        
        subdict[subject][i]["pattern"] = np.mean(subdict[subject][i]["pattern"]) 
        subdict[subject][i]["random_high"] = np.mean(subdict[subject][i]["random_high"])
        subdict[subject][i]["random_low"] = np.mean(subdict[subject][i]["random_low"])

        highs.append(np.mean(subdict[subject][i]["pattern"]))
        highs.append(np.mean(subdict[subject][i]["random_high"]))
        lows.append(np.mean(subdict[subject][i]["random_low"]))
        
        learning_index = (np.mean(lows) - np.mean(highs)) / np.mean(lows)
        learn_index_dict[subject][i] = learning_index if not np.isnan(learning_index) else 0

# Save learning indices to CSV
learn_index_df = pd.DataFrame.from_dict(learn_index_dict, orient='index')
if not op.exists(figures_dir / 'behav' / 'learning_indices.csv'):
    learn_index_df.to_csv(figures_dir / 'behav' / 'learning_indices.csv')

# Calculate means and standard errors
mean_pattern = [np.mean(pattern_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_pattern = [np.std(pattern_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_high = [np.mean(random_high_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_random_high = [np.std(random_high_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_low = [np.mean(random_low_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_random_low = [np.std(random_low_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]

color1 = "#00BFB3"
color2 = "#DD614A"
color3 = "#1982C4"
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.autoscale()
ax.plot(sessions[1:], mean_pattern, '-o', color=color1, label="pattern", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_random_high, '-o', color=color2, label="random high", markersize=7, alpha=.9)
ax.plot(sessions[1:], mean_random_low, '-o', color=color3, label="random low", markersize=7, alpha=.9)
# Scatter plot individual subject values
for subject in subjects:
    for i in range(1, 5):
        ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, marker=".", alpha=0.3)
        ax.scatter(str(i), subdict[subject][i]["random_high"], color=color2, marker=".", alpha=0.2)
        ax.scatter(str(i), subdict[subject][i]["random_low"], color=color3, marker=".", alpha=0.2)
# # Add asterisks above all mean random values
# for i, mean_r in enumerate(mean_random):
#     ax.annotate('*', (sessions[i], mean_r + 20), ha='center', color='black', fontsize=14)
# ax.legend(loc='lower left', frameon=False)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax.set_xlabel("Session")
ax.set_ylabel("Reaction Time (ms)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(figures_dir / 'behav' / 'mean_RT3.pdf', transparent=True)

# Plot learning index as a histogram
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.autoscale()
learning_indices_mean = learn_index_df.mean(axis=0)
learning_indices_stderr = learn_index_df.sem(axis=0)
ax.bar(sessions, learning_indices_mean, yerr=learning_indices_stderr, color=color1, alpha=0.7, capsize=5)
ax.set_xlabel("Session")
ax.set_ylabel("Learning Index")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(figures_dir / 'behav' / 'learning_index_all.pdf', transparent=True)
