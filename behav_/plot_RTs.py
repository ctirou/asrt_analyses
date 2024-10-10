import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, RESULTS_DIR, SUBJS, NEW_FIG_DIR


path_data = DATA_DIR
# path_plots = op.join(RESULTS_DIR, 'figures', 'behav')
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

x = ['1', '2', '3', '4']
n = len(subjects)

for subject in subjects:

    all_RT = list()
    pat_RT = list()
    rand_RT = list()

    for i in range(1, 5):
        
        fname_behav = op.join(path_data, 'behav', f'{subject}-{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)
        
        for j, k in enumerate(behav_df['RTs']):
            all_RT.append(k)
            if behav_df['triplets'][j] == 30:
                pattern_RT[f'Epoch_{i}'].append(behav_df['RTs'][j]) # faire liste et moyennes puis append contenu liste dans dico
            elif behav_df['triplets'][j] == 32 or  behav_df['triplets'][j] == 34:
                random_RT[f'Epoch_{i}'].append(behav_df['RTs'][j])

mpat = list()
spat = list()
mrand = list()
srand = list()

for i in range(1, 5):
    
    spat.append(np.std(pattern_RT[f'Epoch_{i}']))
    mpat.append(np.mean(pattern_RT[f'Epoch_{i}']))
    
    srand.append(np.std(random_RT[f'Epoch_{i}']))
    mrand.append(np.mean(random_RT[f'Epoch_{i}']))

pdf = pd.DataFrame.from_dict(pattern_RT, orient='index').T
pdf.rename(columns={'Epoch_1':'1','Epoch_2':'2','Epoch_3':'3','Epoch_4':'4'}, inplace=True)

rdf = pd.DataFrame.from_dict(random_RT, orient='index').T
rdf.rename(columns={'Epoch_1':'1','Epoch_2':'2','Epoch_3':'3','Epoch_4':'4'}, inplace=True)

learn_index = (rdf - pdf)/rdf
learn_index.dropna()

x = pdf.columns

perr = np.std(pdf)/np.sqrt(n)
rerr = np.std(rdf)/np.sqrt(n)

t = np.mean(learn_index)
sdt = np.std(learn_index)/np.sqrt(n)

color1 = "#70AD47"
color2 = "#5B9BD5"
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.autoscale()
ax.plot(x, np.mean(pdf), label="Paired elements", color=color1)
ax.plot(x, np.mean(rdf), label="Unpaired elements", color=color2)
ax.errorbar(x, np.mean(pdf), yerr=perr, fmt='-o', color=color1, alpha=.7)
ax.errorbar(x, np.mean(rdf), yerr=rerr, fmt='-o', color=color2, alpha=.7)
ax.legend(loc='lower left', frameon=False, title='n = 11')
ax.set_xlabel("Session")
ax.set_ylabel("Reaction Time (ms)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add asterisks above all error bars
for i, (mean_rdf, err_rdf) in enumerate(zip(np.mean(rdf), rerr)):
    ax.annotate('*', (x[i], mean_rdf + err_rdf + 0.2), ha='center', color='black', fontsize=14)
fig.savefig(figures_dir / 'behav' / 'mean_RT2.pdf')