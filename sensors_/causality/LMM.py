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

subjects = SUBJS
# load source time gen data
networks = NETWORKS + ['Cerebellum-Cortex']
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
for i, (network, ax) in enumerate(zip(networks, axs.flatten())):
    ax.plot([i for i in range(1, 24)], data_diag[network].mean(0), color=cmap[i], alpha=1)
    ax.plot([i for i in range(1, 24)], mean_diag.mean(0), color='grey', alpha=0.5)
    ax.set_title(network)
    
X = mean_diag.copy() # Predictive coding
Y = sim_index.copy() # Representational change

data = {
    'participant': np.repeat(np.arange(len(subjects)), 23),
    'block': np.tile(np.arange(23), len(subjects)),
    'X': X.flatten(),
    'Y': Y.flatten()
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