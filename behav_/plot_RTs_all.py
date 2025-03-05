import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

path_data = DATA_DIR
figures_dir = FIGURES_DIR

# subjects = SUBJS
subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

pattern_RT = {f'Epoch_{i}': [] for i in range(5)}
random_high_RT = {f'Epoch_{i}': [] for i in range(5)}
random_low_RT = {f'Epoch_{i}': [] for i in range(5)}
all_RT = {f'Epoch_{i}': [] for i in range(5)}

sessions = ['0', '1', '2', '3', '4']
n = len(subjects)

subdict = {}
learn_index_dict = {}

for subject in tqdm(subjects):

    
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
        
        patterns, randoms = [], []
        
        for j, k in enumerate(behav_df['RTs']):
            if behav_df['triplets'][j] in [30, 32, 34]:
                all_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                subdict[subject][i]["all"].append((behav_df['RTs'][j])) 
                if behav_df['triplets'][j] == 30:
                    pattern_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                    subdict[subject][i]["pattern"].append((behav_df['RTs'][j])) 
                elif behav_df['triplets'][j] == 32:
                    random_high_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                    subdict[subject][i]["random_high"].append((behav_df['RTs'][j]))
                elif behav_df['triplets'][j] == 34:
                    random_low_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                    subdict[subject][i]["random_low"].append((behav_df['RTs'][j]))
            else:
                continue
        
        subdict[subject][i]["all"] = np.mean(subdict[subject][i]["all"]) if subdict[subject][i]["all"] else np.nan
        subdict[subject][i]["pattern"] = np.mean(subdict[subject][i]["pattern"]) if subdict[subject][i]["pattern"] else np.nan
        subdict[subject][i]["random_high"] = np.mean(subdict[subject][i]["random_high"]) if subdict[subject][i]["random_high"] else np.nan
        subdict[subject][i]["random_low"] = np.mean(subdict[subject][i]["random_low"]) if subdict[subject][i]["random_low"] else np.nan

        patterns.append(np.mean(subdict[subject][i]["pattern"]))
        # patterns.append(np.mean(subdict[subject][i]["random_high"]))
        randoms.append(np.mean(subdict[subject][i]["random_high"]))
        
        learning_index = (np.mean(randoms) - np.mean(patterns)) / np.mean(randoms)
        # learning_index = np.mean(randoms) - np.mean(patterns)
        learn_index_dict[subject][i] = learning_index if i != 0 else 0

# Save learning indices to CSV
learn_index_df = pd.DataFrame.from_dict(learn_index_dict, orient='index')
if not op.exists(figures_dir / 'behav' / 'learning_indices2.csv'):
    learn_index_df.to_csv(figures_dir / 'behav' / 'learning_indices2.csv', sep='\t')

# Calculate means and standard errors
mean_all = [np.mean(all_RT[f'Epoch_{i}']) for i in range(5)]
stderr_all = [np.std(all_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(5)]
mean_pattern = [np.mean(pattern_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_pattern = [np.std(pattern_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_high = [np.mean(random_high_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_random_high = [np.std(random_high_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
mean_random_low = [np.mean(random_low_RT[f'Epoch_{i}']) for i in range(5)]
stderr_random_low = [np.std(random_low_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(5)]

color1 = "#5B9BD5"
color2 = "#61CBF5"
color3 = "#A6CAEC"
color4 = "black" 

# Combined RT and learning index
fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout="tight")
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Arial'})
ax.autoscale()
# Scatter plot individual subject values
for subject in subjects:
    for i in range(5):
        ax.scatter(str(i), subdict[subject][i]["all"], color=color4, marker=".", alpha=0.3)
        # ax.scatter(str(i), subdict[subject][i]["random_low"], color=color3, marker=".", alpha=0.2)
        if i > 0:
            ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, marker=".", alpha=0.3)
            ax.scatter(str(i), subdict[subject][i]["random_high"], color=color2, marker=".", alpha=0.2)
ax.plot(sessions, mean_all, '-o', color=color4, label="All", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_pattern, '-o', color=color1, label="Pattern pair", markersize=7, alpha=1)
ax.plot(sessions[1:], mean_random_high, '-o', color=color2, label="Random pair", markersize=7, alpha=.9)
# ax.plot(sessions, mean_random_low, '-o', color=color3, label="Random low", markersize=7, alpha=.9)
# # Add asterisks above all mean random values
# for i, mean_r in enumerate(mean_random_high):
#     ax.annotate('*', (sessions[i], mean_r + 20), ha='center', color='black', fontsize=14)
ax.legend(loc='lower left', frameon=False, title=f"n = {n}")
ax.set_ylabel("Reaction time (ms)", fontsize=12)
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
axlow.bar(sessions, learning_indices_mean, yerr=learning_indices_stderr, alpha=0.7, capsize=5, color="#029E73", width=bar_width)
axlow.set_ylabel("Learning index", fontsize=12)
axlow.spines['top'].set_visible(False)
axlow.spines['right'].set_visible(False)
axlow.set_xticklabels(['Practice', '1', '2', '3', '4'])
axlow.set_xlabel("Session", fontsize=12)
axlow.set_ylim(bottom=0)  # Set the lower limit of the y-axis to 0 to reduce the height
fig.savefig(figures_dir / 'behav' / 'combined_3.pdf', transparent=True)
plt.close()