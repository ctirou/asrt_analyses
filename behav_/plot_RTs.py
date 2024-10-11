import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_DIR, SUBJS, NEW_FIG_DIR
from tqdm.auto import tqdm

path_data = DATA_DIR
figures_dir = NEW_FIG_DIR

subjects = SUBJS

pattern_RT = {
    'Epoch_1': list(),
    'Epoch_2': list(),
    'Epoch_3': list(),
    'Epoch_4': list(),
}

random_RT = {
    'Epoch_1': list(),
    'Epoch_2': list(),
    'Epoch_3': list(),
    'Epoch_4': list(),
}

sessions = ['1', '2', '3', '4']
n = len(subjects)

subdict = {}

for subject in tqdm(subjects):

    all_RT = list()
    pat_RT = list()
    rand_RT = list()
    
    subdict[subject] = {}
    
    for i in range(1, 5):

        subdict[subject][i] = {"pattern": [], 
                               "random": []}
        
        fname_behav = op.join(path_data, 'behav', f'{subject}-{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)
        
        for j, k in enumerate(behav_df['RTs']):
            all_RT.append(k)
            if behav_df['triplets'][j] == 30:
                pattern_RT[f'Epoch_{i}'].append(behav_df['RTs'][j]) # faire liste et moyennes puis append contenu liste dans dico
                subdict[subject][i]["pattern"].append((behav_df['RTs'][j])) 
            elif behav_df['triplets'][j] == 32 or behav_df['triplets'][j] == 34:
                random_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])
                subdict[subject][i]["random"].append((behav_df['RTs'][j]))
        
        subdict[subject][i]["pattern"] = np.mean(subdict[subject][i]["pattern"]) 
        subdict[subject][i]["random"] = np.mean(subdict[subject][i]["random"])

# Calculate means and standard errors
mean_pattern = [np.mean(pattern_RT[f'Epoch_{i}']) for i in range(1, 5)]
mean_random = [np.mean(random_RT[f'Epoch_{i}']) for i in range(1, 5)]
stderr_pattern = [np.std(pattern_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]
stderr_random = [np.std(random_RT[f'Epoch_{i}']) / np.sqrt(n) for i in range(1, 5)]

color1 = "#70AD47"
color2 = "C7"
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.autoscale()
ax.plot(sessions, mean_pattern, '-o', color=color1, label="Paired elements", markersize=7, alpha=1)
ax.plot(sessions, mean_random, '-o', color=color2, label="Unpaired elements", markersize=7, alpha=.9)
# Scatter plot individual subject values
for subject in subjects:
    for i in range(1, 5):
        ax.scatter(str(i), subdict[subject][i]["pattern"], color=color1, alpha=0.3)
        ax.scatter(str(i), subdict[subject][i]["random"], color=color2, alpha=0.2)
# Add asterisks above all mean random values
for i, mean_r in enumerate(mean_random):
    ax.annotate('*', (sessions[i], mean_r + 15), ha='center', color='black', fontsize=14)
ax.legend(loc='lower left', frameon=False)
ax.set_xlabel("Session")
ax.set_ylabel("Reaction Time (ms)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(figures_dir / 'behav' / 'mean_RT2.pdf')