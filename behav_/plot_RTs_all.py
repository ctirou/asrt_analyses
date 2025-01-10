import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        subdict[subject][i] = {"all": [],
                               "pattern": [], 
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
        
        subdict[subject][i]["all"] = np.mean(all_RT)
        subdict[subject][i]["pattern"] = np.mean(subdict[subject][i]["pattern"]) 
        subdict[subject][i]["random_high"] = np.mean(subdict[subject][i]["random_high"])
        subdict[subject][i]["random_low"] = np.mean(subdict[subject][i]["random_low"])

        highs.append(np.mean(subdict[subject][i]["pattern"]))
        # highs.append(np.mean(subdict[subject][i]["random_high"]))
        lows.append(np.mean(subdict[subject][i]["random_high"]))
        
        learning_index = (np.mean(lows) - np.mean(highs)) / np.mean(lows)
        # learning_index = np.mean(lows) - np.mean(highs)
        learn_index_dict[subject][i] = learning_index if i != 0 else 0

# Save learning indices to CSV
learn_index_df = pd.DataFrame.from_dict(learn_index_dict, orient='index')
if not op.exists(figures_dir / 'behav' / 'learning_indices.csv'):
    learn_index_df.to_csv(figures_dir / 'behav' / 'learning_indices.csv', sep='\t')

# Calculate means and standard errors
mean_all = [np.mean(subdict[subject][i]["all"]) for i in range(5)]
stderr_all = [np.std(subdict[subject][i]["all"]) / np.sqrt(n) for i in range(5)]
mean_pattern = [np.mean(pattern_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_pattern = [np.std(pattern_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_high = [np.mean(random_high_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_random_high = [np.std(random_high_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_low = [np.mean(random_low_RT[f'Epoch_{i}']) for i in range(5)]
stderr_random_low = [np.std(random_low_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(5)]

color1 = "#00BFB3"
color2 = "#DD614A"
color3 = "#1982C4"
color4 = "#FFB300"  # New contrasting color

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# Plot individual subject values
for subject in subjects:
    ax.plot(sessions, [subdict[subject][i]["all"] for i in range(5)], marker=".", alpha=0.3, label=f'{subject}')
plt.plot(sessions, mean_all, '-o', color=color4, label="mean", markersize=7, alpha=1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax.set_xlabel("Session")
ax.set_ylabel("Reaction Time (ms)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# fig.savefig(figures_dir / 'behav' / 'mean_RT_all.pdf', transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Avenir'})
ax.autoscale()
# Scatter plot individual subject values
for subject in subjects:
    for i in range(5):
        ax.scatter(str(i), subdict[subject][i]["all"], color=color4, marker=".", alpha=0.3)
        ax.scatter(str(i), subdict[subject][i]["random_low"], color=color3, marker=".", alpha=0.2)    
        if i > 0:
            ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, marker=".", alpha=0.3)
            ax.scatter(str(i), subdict[subject][i]["random_high"], color=color2, marker=".", alpha=0.2)
ax.plot(sessions, mean_all, '-o', color=color4, label="All", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_pattern, '-o', color=color1, label="Pattern", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_random_high, '-o', color=color2, label="Random high", markersize=7, alpha=.9)
ax.plot(sessions, mean_random_low, '-o', color=color3, label="Random low", markersize=7, alpha=.9)
# # Add asterisks above all mean random values
# for i, mean_r in enumerate(mean_random):
#     ax.annotate('*', (sessions[i], mean_r + 20), ha='center', color='black', fontsize=14)
# ax.legend(loc='lower left', frameon=False)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="Trial types")
# ax.set_xlabel("Session", fontsize=12)
ax.set_ylabel("Reaction Time (ms)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['Practice', '1', '2', '3', '4'])
plt.tight_layout()
fig.savefig(figures_dir / 'behav' / 'mean_RT.pdf', transparent=True)

# Plot learning index as a histogram
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.autoscale()
learning_indices_mean = learn_index_df.mean(axis=0)
learning_indices_stderr = learn_index_df.sem(axis=0)
ax.bar(sessions, learning_indices_mean, yerr=learning_indices_stderr, alpha=0.7, capsize=5)
ax.set_xlabel("Session")
ax.set_ylabel("Learning Index")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(figures_dir / 'behav' / 'learning_index_all.pdf', transparent=True)

# Combined RT and learning index
fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout="tight")
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Avenir'})
ax.autoscale()
# Scatter plot individual subject values
for subject in subjects:
    for i in range(5):
        ax.scatter(str(i), subdict[subject][i]["all"], color=color4, marker=".", alpha=0.3)
        ax.scatter(str(i), subdict[subject][i]["random_low"], color=color3, marker=".", alpha=0.2)    
        if i > 0:
            ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, marker=".", alpha=0.3)
            ax.scatter(str(i), subdict[subject][i]["random_high"], color=color2, marker=".", alpha=0.2)
ax.plot(sessions, mean_all, '-o', color=color4, label="All", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_pattern, '-o', color=color1, label="Pattern", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_random_high, '-o', color=color2, label="Random high", markersize=7, alpha=.9)
ax.plot(sessions, mean_random_low, '-o', color=color3, label="Random low", markersize=7, alpha=.9)
# # Add asterisks above all mean random values
# for i, mean_r in enumerate(mean_random):
#     ax.annotate('*', (sessions[i], mean_r + 20), ha='center', color='black', fontsize=14)
# ax.legend(loc='lower left', frameon=False)
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="Trial types")
ax.legend(loc='lower left', frameon=False, title="Trial types")
ax.set_ylabel("Reaction Time (ms)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelbottom=False)  # Hide x-axis tick labels
# create new Axes on the right and on the top of the current Axes
divider = make_axes_locatable(ax)
# below height and pad are in inches
axlow = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
axlow.autoscale()
learning_indices_mean = learn_index_df.mean(axis=0)
learning_indices_stderr = learn_index_df.sem(axis=0)
bar_width = 0.5  # Adjust the width of the bars
axlow.bar(sessions, learning_indices_mean, yerr=learning_indices_stderr, alpha=0.7, capsize=5, color="#7294D4", width=bar_width)
axlow.set_ylabel("Learning Index", fontsize=12)
axlow.spines['top'].set_visible(False)
axlow.spines['right'].set_visible(False)
axlow.set_xticklabels(['Practice', '1', '2', '3', '4'])
axlow.set_xlabel("Session", fontsize=12)
axlow.set_ylim(bottom=0)  # Set the lower limit of the y-axis to 0 to reduce the height
fig.savefig(figures_dir / 'behav' / 'combined.pdf', transparent=True)


