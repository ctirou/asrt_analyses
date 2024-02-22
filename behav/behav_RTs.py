import os
import os.path as op
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import pingouin as pg
from config import DATA_DIR, RESULTS_DIR, SUBJS

path_data = DATA_DIR
path_plots = op.join(RESULTS_DIR, 'figures')

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

for subject in subjects:

    all_RT = list()
    pat_RT = list()
    rand_RT = list()

    for i in range(1, 5):
        
        fname_behav = op.join(path_data, 'behav', f'{subject}_{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)
        
        for j, k in enumerate(behav_df['RTs']):
            all_RT.append(k)
            if behav_df['triplets'][j] == 30:
                pattern_RT[f'Epoch_{i}'].append(behav_df['RTs'][j]) #faire liste et moyennes puis append contenu liste dans dico
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

    with sns.plotting_context('notebook'):
        custom_params = {'axes.spines.right':False, 'axes.spines.top':False}
        sns.set_theme(style="ticks", rc=custom_params)
        
        learn_index = (rdf - pdf)/rdf
        learn_index.dropna()

        x = pdf.columns

        perr = np.std(pdf)/np.sqrt(14)
        rerr = np.std(rdf)/np.sqrt(14)

        t = np.mean(learn_index)
        sdt = np.std(learn_index)/np.sqrt(14)

        plt.rcParams['font.sans-serif'] = ['Avenir']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['axes.linewidth'] = 1.3
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.autoscale()
        ax.plot(x, np.mean(pdf), label="Paired elements", color='mediumseagreen')
        ax.plot(x, np.mean(rdf), label="Unpaired elements", color='C7')
        ax.errorbar(x, np.mean(pdf), yerr=perr, fmt='-o', color='mediumseagreen', alpha=.7)
        ax.errorbar(x, np.mean(rdf), yerr=rerr, fmt='-o', color='C7', alpha=.7)
        ax.legend()
        ax.set_xlabel("Session")
        ax.set_ylabel("Reaction Time (ms)")
        # ax.grid(alpha=.2, which='both')

        # ax2.plot(x, t)
        # ax2.fill_between(x, t+sdt, t-sdt, facecolor='lightblue', alpha=0.5)
        # ax2.set_xlabel("Blocks")
        # ax2.set_ylabel("Learning index")
        # # rajouter lignes a 0 learn index
        # plt.show()
        plt.close()
        fig.savefig(op.join(path_plots, 'behav_%s.png') % subject)

# print(ttest_rel(np.mean(pdf['1']), np.mean(rdf['1']), nan_policy='propagate', alternative='two-sided')[1])