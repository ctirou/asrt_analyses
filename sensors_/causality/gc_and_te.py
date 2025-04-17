import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'

times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.28) & (times <= 0.51))[0]

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

# load time gen data
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

# # mean box
# means = []
# for sub in range(len(subjects)):
#     tg = []
#     for block in range(23):
#         data = contrast[sub, block, idx_timeg, :][:, idx_timeg]
#         tg.append(data.mean())
#     means.append(np.array(tg))
# means = np.array(means)
blocks = [i for i in range(1, 24)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
for ax in fig.axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    # ax.axvspan(0, 2, color='grey', alpha=0.1)
    ax.set_xticks(blocks)
    ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
    # ax.axvspan(3, 7, color='purple', alpha=0.1)
    # ax.axvspan(8, 12, color='purple', alpha=0.1)s
    # ax.axvspan(13, 17, color='purple', alpha=0.1)
    # ax.axvspan(18, 22, color='purple', alpha=0.1)

for i in range(sim_index.shape[0]):
    ax1.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax1.plot(blocks, sim_index.mean(0), lw=3, color='#00A08A', label='Mean')
ax1.set_ylabel('Mean RSA effect')
ax1.legend(frameon=False)
ax1.set_title('Representational change')

for i in range(mean_diag.shape[0]):
    ax2.plot(blocks, mean_diag[i], alpha=0.5, color=cmap(i))
ax2.plot(blocks, mean_diag.mean(0), lw=3, color='#FD6467', label='Mean')
ax2.set_xlabel('Block')
ax2.set_ylabel('Mean predictive effect')
ax2.legend(frameon=False)
ax2.set_title('Predictive coding')
fig.tight_layout()

X = mean_diag.copy() # Predictive coding
Y = sim_index.copy() # Representational change

from scipy.stats import zscore
# Apply z-score normalization to each subject individually
X_norm = np.array([zscore(X[sub, :]) for sub in range(X.shape[0])])
Y_norm = np.array([zscore(Y[sub, :]) for sub in range(Y.shape[0])])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
for ax in fig.axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    # ax.axvspan(0, 2, color='grey', alpha=0.1)
    ax.set_xticks(blocks)
    ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
    # ax.axvspan(3, 7, color='purple', alpha=0.1)
    # ax.axvspan(8, 12, color='purple', alpha=0.1)s
    # ax.axvspan(13, 17, color='purple', alpha=0.1)
    # ax.axvspan(18, 22, color='purple', alpha=0.1)

for i in range(Y_norm.shape[0]):
    ax1.plot(blocks, Y_norm[i], alpha=0.5, color=cmap(i))
ax1.plot(blocks, Y_norm.mean(0), lw=3, color='#00A08A', label='Mean')
ax1.set_ylabel('Mean RSA effect')
ax1.legend(frameon=False)
ax1.set_title('Representational change')

for i in range(X_norm.shape[0]):
    ax2.plot(blocks, X_norm[i], alpha=0.5, color=cmap(i))
ax2.plot(blocks, X_norm.mean(0), lw=3, color='#FD6467', label='Mean')
ax2.set_xlabel('Block')
ax2.set_ylabel('Mean predictive effect')
ax2.legend(frameon=False)
ax2.set_title('Predictive coding')
fig.tight_layout()


fig, ax = plt.subplots(1, 1)
ax.set_xticks([i for i in range(1, 24)])
ax.axvspan(1, 3, color='grey', alpha=0.1)
ax.plot([i for i in range(1, 24)], X.mean(0), label='x: predictive coding')
ax.plot([i for i in range(1, 24)], Y.mean(0), label='y: representational change')
ax.set_xlabel('Block')
ax.set_xticklabels([str(i) for i in range(1, 24)])
ax.legend()
ax.set_title('Z-score normalization')

### Granger causality
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import pandas as pd

stationary_flags, X_stationary, diffs_applied = ensure_stationarity(X_norm, max_diff=4)
_, Y_stationary, _ = ensure_stationarity(Y_norm, max_diff=4)

x, y = X_stationary.copy(), Y_stationary.copy()
try:
    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
except AssertionError as e:
    small = x.shape[1] if x.shape[1] < y.shape[1] else y.shape[1]
    x = x[:, :small]
    y = y[:, :small]
assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"

# X -> Y
max_lags = 6
pvalues = []
for sub in range(len(subjects)):
    data = pd.DataFrame({'X': X_norm[sub], 'Y': Y_norm[sub]})
    results = grangercausalitytests(data, max_lags, verbose=False)
    pval = [round(results[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lags)]
    pvalues.append(pval)    
pvalues = np.array(pvalues)
sig = pvalues < 0.05
print(np.unique(sig, return_counts=True))

### Cross-correlation
from scipy.signal import correlate
from scipy.stats import ttest_1samp
# Generate lag indices
lags = np.arange(-len(X_norm.mean(0)) + 1, len(X_norm.mean(0)))

corr_list = []
for sub in range(len(subjects)):
    corr = correlate(X[sub], Y[sub], mode='full')
    corr_list.append(corr)
corr_list = np.array(corr_list)

pval = decod_stats(corr_list, -1)
pval_unc = ttest_1samp(corr_list, axis=0, popmean=0)[1]

sig = pval < 0.05
sig_unc = pval_unc < 0.05

whre = np.where(pval != 1)[0]

fig, ax = plt.subplots(1, 1)
ax.plot(lags, corr_list.mean(0), label='Cross-correlation')
ax.axvline(0, color='grey', linestyle='--', label="Zero Lag")
ax.axhline(0, color='grey', linestyle='-')
ax.set_xlabel("Lag")
ax.set_ylabel("Cross-Correlation")
ax.set_title("Cross-Correlation Between X and Y")
ax.axvline(lags[whre], color='black', label='sig')
ax.legend()

### Transfer Entropy - using PyInform
from pyinform.transferentropy import transfer_entropy
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# X: Predictive coding, Y: Representational change
X, Y = mean_diag, sim_index
# Apply z-score normalization to each subject individually
X_norm = np.array([zscore(X[sub, :]) for sub in range(X.shape[0])])
Y_norm = np.array([zscore(Y[sub, :]) for sub in range(Y.shape[0])])
# Apply Min-Max scaling to each subject individually
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
X_discrete = X_scaled.astype(int)
Y_discrete = Y_scaled.astype(int)

optimal_lags, global_lag = find_optimal_lags(X, Y)
optimal_lags, global_lag = find_optimal_lags(X_norm, Y_norm)
print("Optimal lags per subject:", optimal_lags)
print("Global (median) lag:", global_lag)

res_dir = RESULTS_DIR / 'causality' / 'sensors' / "basic"
ensure_dir(res_dir)
# Iterate over subjects
for isub, sub in enumerate(subjects):
    print(f"Computing TE for {sub} ({isub})...")
    try:
        lag = global_lag
        for l in range(1, global_lag + 1):
            # Compute Transfer Entropy
            xy = transfer_entropy(X[isub], Y[isub], k=l, local=True)
            yx = transfer_entropy(Y[isub], X[isub], k=l, local=True)
            # Save local TE results as .npy files
            np.save(res_dir / f"{sub}_te_xy-{l}.npy", xy)
            np.save(res_dir / f"{sub}_te_yx-{l}.npy", yx)
    except Exception as e:
        print(f"Error computing TE for {sub}: {e}")
print("\nAll computations complete!")

res_dir = RESULTS_DIR / 'causality' / 'sensors' / "normalized"
ensure_dir(res_dir)
# Iterate over subjects
for isub, sub in enumerate(subjects):
    print(f"Computing TE for {sub} ({isub})...")
    try:
        lag = global_lag
        for l in range(1, global_lag + 1):
            # Compute Transfer Entropy
            xy = transfer_entropy(X_norm[isub], Y_norm[isub], k=l, local=True)
            yx = transfer_entropy(Y_norm[isub], X_norm[isub], k=l, local=True)
            # Save local TE results as .npy files
            np.save(res_dir / f"{sub}_te_xy-{l}.npy", xy)
            np.save(res_dir / f"{sub}_te_yx-{l}.npy", yx)
    except Exception as e:
        print(f"Error computing TE for {sub}: {e}")
print("\nAll computations complete!")

res_dir = RESULTS_DIR / 'causality' / 'sensors' / "discretized"
ensure_dir(res_dir)
# Iterate over subjects
for isub, sub in enumerate(subjects):
    print(f"Computing TE for {sub} ({isub})...")
    try:
        lag = global_lag
        for l in range(1, global_lag + 1):
            # Compute Transfer Entropy
            xy = transfer_entropy(X_discrete[isub], Y_discrete[isub], k=l, local=True)
            yx = transfer_entropy(Y_discrete[isub], X_discrete[isub], k=l, local=True)
            # Save local TE results as .npy files
            np.save(res_dir / f"{sub}_te_xy-{l}.npy", xy)
            np.save(res_dir / f"{sub}_te_yx-{l}.npy", yx)
    except Exception as e:
        print(f"Error computing TE for {sub}: {e}")
print("\nAll computations complete!")
