import pandas as pd
import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'

times = np.linspace(-0.2, 0.6, 82)

# load RSA data
pattern, random = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "loocv_rdm_blocks" / subject        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pat, rand = get_all_high_low_blocks(res_path, sequence)
    pattern.append(pat)
    random.append(rand)
pattern = np.array(pattern).mean(1)
random = np.array(random).mean(1)
win = np.where((times >= 0.28) & (times <= 0.51))[0]
m_pattern = pattern[:, :, win].mean(-1)
m_random = random[:, :, win].mean(-1)
prac_pat = m_pattern[:, :3].mean(-1)
prac_rand = m_random[:, :3].mean(-1)
sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(23):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

# load sensor time gen data
timeg_data_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
timesg = np.linspace(-1.5, 4, 559)
pattern, random = [], []
for subject in tqdm(subjects):
    pat, rand = [], []
    for block in range(1, 24):
        pat.append(np.load(timeg_data_path / 'split_all_pattern' /  f"{subject}-{block}.npy"))
        rand.append(np.load(timeg_data_path / 'split_all_random' / f"{subject}-{block}.npy"))
    pattern.append(np.array(pat))
    random.append(np.array(rand))
pattern, random = np.array(pattern), np.array(random)
contrast = pattern - random
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
# mean diag
mean_diag = []
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = np.diag(contrast[sub, block])
        tg.append(data[idx_timeg].mean())
    mean_diag.append(np.array(tg))
mean_diag = np.array(mean_diag)
    
    
X = mean_diag.copy() # Predictive coding
Y = sim_index.copy() # Representational change

from scipy.stats import zscore
# Apply z-score normalization to each subject individually
X_norm = np.array([zscore(X[sub, :]) for sub in range(X.shape[0])])
Y_norm = np.array([zscore(Y[sub, :]) for sub in range(Y.shape[0])])

data = {
    'participant': np.repeat(np.arange(len(subjects)), 23),
    'block': np.tile(np.arange(23), len(subjects)),
    'X': X_norm.flatten(),
    'Y': Y_norm.flatten()
    }

coef_xy, pv_xy = [], []
coef_yx, pv_yx = [], []

for lag in range(1, 6):
    
    df = pd.DataFrame(data)
    # Create lagged variables per participant
    df['X_lag'] = df.groupby('participant')['X'].shift(lag)
    df['Y_lag'] = df.groupby('participant')['Y'].shift(lag)
    # Remove missing lagged values
    df = df.dropna().reset_index(drop=True)
    
    result_XtoY = mixed_model_pvalues(df, dependent='Y', predictor='X_lag', group='participant')
    result_YtoX = mixed_model_pvalues(df, dependent='X', predictor='Y_lag', group='participant')
    
    coef_xy.append(result_XtoY['coef'])
    coef_yx.append(result_YtoX['coef'])
    
    pv_xy.append(result_XtoY['LikelihoodRatio_pvalue'])
    pv_yx.append(result_YtoX['LikelihoodRatio_pvalue'])

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='tight')
ax[0].plot(coef_xy, label='X->Y', color='blue')
ax[0].plot(coef_yx, label='Y->X', color='orange')
ax[0].set_title('$Coefficient$')
ax[0].set_ylabel('Coefficient')
ax[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
ax[0].set_xticks(range(5), [f'Lag {i+1}' for i in range(5)])
ax[0].legend()

ax[1].plot(pv_xy, label='X->Y', color='blue')
ax[1].plot(pv_yx, label='Y->X', color='orange')
ax[1].set_title('$pvalues$')
ax[1].set_ylabel('Likelihood Ratio pvalue')
ax[1].axhline(0.05, color='grey', linestyle='--', alpha=1, label='alpha = 0.05')
ax[1].set_xticks(range(5), [f'Lag {i+1}' for i in range(5)])
ax[1].legend()
fig.suptitle('Causality analysis between predictive coding and representational change', fontsize=16)

# load source time gen data
subjects = SUBJS
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
timesg = np.linspace(-1.5, 1.5, 307)
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
res_dir = TIMEG_DATA_DIR / 'results' / 'source' / 'max-power'
data_diag = {}
patterns, randoms = {}, {}
for network in tqdm(networks):
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        data_diag[network] = []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in range(1, 24):
            p = np.load(res_dir / network / 'split_all_pattern' / f"{subject}-{j}.npy")
            r = np.load(res_dir / network / 'split_all_random' / f"{subject}-{j}.npy")
            pat.append(np.diag(p)[idx_timeg].mean())
            rand.append(np.diag(r)[idx_timeg].mean())
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)
    data_diag[network] = patterns[network] - randoms[network]

cmap = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#ECE133', '#56B4E9', '#76B041']

fig, axs = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (network, ax, name) in enumerate(zip(networks, axs.flatten(), network_names)):
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    ax.plot([i for i in range(1, 24)], np.poly1d(np.polyfit(range(1, 24), data_diag[network].mean(0), 1))(range(1, 24)), color='black', linestyle='--', alpha=0.5, label='linear fit')
    ax.plot([i for i in range(1, 24)], data_diag[network].mean(0), color=cmap[i], alpha=1)
    # ax.plot([i for i in range(1, 24)], mean_diag.mean(0), color='grey', alpha=0.5)
    ax.set_title(name)
    if ax == axs.flatten()[0]:
        ax.legend(frameon=False, loc='upper left')
fig.suptitle('Mean predictive coding dynamics', fontsize=16)

# load source RSA data
subjects = SUBJS
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.28) & (times <= 0.51))[0]
res_path = RESULTS_DIR / 'RSA' / 'source'

pattern, random = {}, {}
sim_index = dict()
for network in tqdm(networks):
    if network not in pattern:
        pattern[network], random[network] = [], []
        sim_index[network] = []
        # prac_pat, prac_rand = dict(), dict()
    for subject in subjects:
        res_dir = res_path / network / lock / "loocv_rdm_blocks" / subject
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        pat, rand = get_all_high_low_blocks(res_dir, sequence)
        pattern[network].append(pat)
        random[network].append(rand)
    pattern[network] = np.array(pattern[network])[:, :, :, win].mean((1, -1))
    random[network] = np.array(random[network])[:, :, :, win].mean((1, -1))
        
    for subject, _ in enumerate(subjects):
        prac_pat = pattern[network][subject, :3].mean(-1)
        prac_rand = random[network][subject, :3].mean(-1)
        sub_sim = list()
        for i in range(23):
            diff = (random[network][subject, i] - prac_rand) - (pattern[network][subject, i] - prac_pat)
            sub_sim.append(diff)
        sim_index[network].append(np.array(sub_sim))
    sim_index[network] = np.array(sim_index[network])

fig, axs = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
for i, (network, ax, name) in enumerate(zip(networks, axs.flatten(), network_names)):
    ax.axvspan(1, 3, color='grey', alpha=.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    ax.plot([i for i in range(1, 24)], np.poly1d(np.polyfit(range(1, 24), sim_index[network].mean(0), 1))(range(1, 24)), color='black', linestyle='--', alpha=0.5, label='linear fit')
    ax.plot([i for i in range(1, 24)], sim_index[network].mean(0), color=cmap[i], alpha=1)
    ax.set_title(name)
    if ax == axs.flatten()[0]:
        ax.legend(frameon=False, loc='upper left')
fig.suptitle('Mean representational change dynamics', fontsize=16)

xy_matrix = np.zeros((len(networks), len(networks)))
yx_matrix = np.zeros((len(networks), len(networks)))

xy_matrix_pv = np.zeros((len(networks), len(networks)))
yx_matrix_pv = np.zeros((len(networks), len(networks)))

X = data_diag.copy()
Y = sim_index.copy()
lag = 3

for i, net1 in enumerate(networks):
    for j, net2 in enumerate(networks):
        
        data = {
            'participant': np.repeat(np.arange(len(subjects)), 23),
            'block': np.tile(np.arange(23), len(subjects)),
            'X': X[net1].flatten(),
            'Y': Y[net2].flatten()
            }
        
        df = pd.DataFrame(data)
        # Create lagged variables per participant
        df['X_lag'] = df.groupby('participant')['X'].shift(lag)
        df['Y_lag'] = df.groupby('participant')['Y'].shift(lag)
        # Remove missing lagged values
        df = df.dropna().reset_index(drop=True)
        
        result_XtoY = mixed_model_pvalues(df, dependent='Y', predictor='X_lag', group='participant')
        result_YtoX = mixed_model_pvalues(df, dependent='X', predictor='Y_lag', group='participant')
        
        xy_matrix[i, j] = result_XtoY['coef']
        yx_matrix[i, j] = result_YtoX['coef']

        xy_matrix_pv[i, j] = result_XtoY['LikelihoodRatio_pvalue']
        yx_matrix_pv[i, j] = result_YtoX['LikelihoodRatio_pvalue']

import seaborn as sns

red_names = ['Vis', 'Mot', 'DAtt', 'VAtt', 'Limb', 'Ctrl', 'Dflt', 'Hpc', 'Thal', 'Crblm']

fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout='tight')
cbar = sns.heatmap(xy_matrix_pv, vmin=0, vmax=0.05, annot=xy_matrix, fmt=".2f", cbar_kws={'label': 'p-value'}).collections[0].colorbar
cbar.ax.set_ylabel('p-value', rotation=-90, va="bottom")
ax.set_xticklabels(red_names)
ax.set_yticklabels(red_names)
ax.xaxis.tick_top()
fig.suptitle("X -> Y : Predictive coding drives representational change")

fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout='tight')
cbar = sns.heatmap(yx_matrix_pv, vmin=0, vmax=0.05, annot=yx_matrix, fmt=".2f", cbar_kws={'label': 'p-value'}).collections[0].colorbar
cbar.ax.set_ylabel('p-value', rotation=-90, va="bottom")
ax.set_xticklabels(red_names)
ax.set_yticklabels(red_names)
ax.xaxis.tick_top()
fig.suptitle("Y -> X : Representational change drives predictive coding")
