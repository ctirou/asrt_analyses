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

# Calculate peak for all participants
# Then mean
blocks = np.arange(23)
peaks = list()
for subject in range(len(subjects)):
    peaks.append(blocks[np.argmax(sim_index[subject])])
peaks = np.array(peaks)
peak_rsa = int(round(peaks.mean()))
# print('Peak:', m_peak)

# load time gen data
timeg_data_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
timesg = np.linspace(-1.5, 4, 559)

pattern, random = [], []
for subject in tqdm(subjects):
    pat, rand = [], []
    for epoch_num in range(5):
        blocks = range(1, 4) if epoch_num == 0 else range(1, 6)
        for block in blocks:
            pat.append(np.load(timeg_data_path / 'split_pattern' /  f"{subject}-{epoch_num}-{block}.npy"))
            rand.append(np.load(timeg_data_path / 'split_random' / f"{subject}-{epoch_num}-{block}.npy"))
    pattern.append(np.array(pat))
    random.append(np.array(rand))
pattern, random = np.array(pattern), np.array(random)
contrast = pattern - random

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

# pval = gat_stats(contrast.mean(1), -1)
# sig = pval < 0.05
# plt.imshow(contrast.mean((0, 1)),
#            interpolation="lanczos",
#                             origin="lower",
#                             cmap='RdBu_r',
#                             extent=timesg[[0, -1, 0, -1]],
#                             aspect=0.5)
# plt.axhline(0, color='black', lw=1)
# plt.axvline(0, color='black', lw=1)
# xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
# plt.contour(xx, yy, sig, colors='black', levels=[0],
#                     linestyles='--', linewidths=1, alpha=.5)
# plt.show()

# # mean box
# means = []
# for sub in range(len(subjects)):
#     tg = []
#     for block in range(23):
#         data = contrast[sub, block, idx_timeg, :][:, idx_timeg]
#         tg.append(data.mean())
#     means.append(np.array(tg))
# means = np.array(means)

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

blocks = np.arange(23)
peaks = list()
for subject in range(len(subjects)):
    peaks.append(blocks[np.argmax(mean_diag[subject])])
peaks = np.array(peaks)
peak_tg = int(round(peaks.mean()))

# cmap = plt.cm.get_cmap('tab20', len(subjects))
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
# for ax in fig.axes:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
#     # ax.axvspan(0, 2, color='grey', alpha=0.1)
#     ax.set_xticks(blocks)
#     ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
#     # ax.axvspan(3, 7, color='purple', alpha=0.1)
#     # ax.axvspan(8, 12, color='purple', alpha=0.1)s
#     # ax.axvspan(13, 17, color='purple', alpha=0.1)
#     # ax.axvspan(18, 22, color='purple', alpha=0.1)

# for i in range(sim_index.shape[0]):
#     ax1.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
# ax1.plot(blocks, sim_index.mean(0), lw=3, color='#00A08A', label='Mean')
# ax1.set_ylabel('Mean RSA effect')
# ax1.legend(frameon=False)
# ax1.set_title('Representational change')

# for i in range(mean_diag.shape[0]):
#     ax2.plot(blocks, mean_diag[i], alpha=0.5, color=cmap(i))
# ax2.plot(blocks, mean_diag.mean(0), lw=3, color='#FD6467', label='Mean')
# ax2.set_xlabel('Block')
# ax2.set_ylabel('Mean predictive effect')
# ax2.legend(frameon=False)
# ax2.set_title('Predictive coding')
# # fig.tight_layout()

X = mean_diag.copy() # Predictive coding
Y = sim_index.copy() # Representational change

from scipy.stats import zscore
# Apply z-score normalization to each subject individually
X_norm = np.array([zscore(X[sub, :]) for sub in range(X.shape[0])])
Y_norm = np.array([zscore(Y[sub, :]) for sub in range(Y.shape[0])])

# fig, ax = plt.subplots(1, 1)
# ax.set_xticks([i for i in range(1, 24)])
# ax.axvspan(1, 3, color='grey', alpha=0.1)
# ax.plot([i for i in range(1, 24)], X_norm.mean(0), label='x: predictive coding')
# ax.plot([i for i in range(1, 24)], Y_norm.mean(0), label='y: representational change')
# ax.set_xlabel('Block')
# ax.set_xticklabels([str(i) for i in range(1, 24)])
# ax.legend()
# ax.set_title('Z-score normalization')

# Example usage
stationary_flags, X_stationary, diffs_applied = ensure_stationarity(X_norm, max_diff=4)
_, Y_stationary, _ = ensure_stationarity(Y_norm, max_diff=4)

# # Example usage
# stationary_flags, X_stationary, diffs_applied = ensure_stationarity(mean_diag)
# _, Y_stationary, _ = ensure_stationarity(sim_index)
# # Print summary
# print(f"Subjects that remained non-stationary: {np.sum(~np.array(stationary_flags))} / {len(stationary_flags)}")
# print(f"Max differencing applied: {max(diffs_applied)}")

x, y = X_stationary.copy(), Y_stationary.copy()
try:
    assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"
except AssertionError as e:
    small = x.shape[1] if x.shape[1] < y.shape[1] else y.shape[1]
    x = x[:, :small]
    y = y[:, :small]
assert x.shape == y.shape, f"Shapes do not match: {x.shape} != {y.shape}"

# bees = [i for i in range(x.shape[1])]
# fig, ax = plt.subplots(1, 1)
# ax.set_xticks(bees)
# ax.axvspan(0, 2, color='grey', alpha=0.1)
# ax.plot(bees, x.mean(0), label='x: predictive coding')
# ax.plot(bees, y.mean(0), label='y: representational change')
# ax.set_xticklabels(bees)
# ax.set_xlabel('Block')
# ax.legend()
# ax.set_title('Stationary data')

# from statsmodels.tsa.stattools import grangercausalitytests
# from statsmodels.tsa.api import VAR
# import pandas as pd

# # X -> Y
# AIC, BIC, HQIC = [], [], []
# for sub in range(len(subjects)):
#     print(f"\n##### Subject {sub} #####")
#     data = pd.DataFrame({'X': x[sub], 'Y': y[sub]})
#     # gc_res = grangercausalitytests(data, 5, verbose=True)
    
#     model = VAR(data)
#     lag_order = model.select_order(5) # Check this    
#     AIC.append(lag_order.aic)
#     BIC.append(lag_order.bic)
#     HQIC.append(lag_order.hqic)

# print("\nMean AIC:", np.mean(AIC))
# print("Mean BIC:", np.mean(BIC))
# print("Mean HQIC:", np.mean(HQIC))

# # Y -> X
# AIC, BIC, HQIC = [], [], []
# for sub in range(len(subjects)):
#     print(f"\n##### Subject {sub} #####")
#     data = pd.DataFrame({'X': y[sub], 'Y': x[sub]})
#     # gc_res = grangercausalitytests(data, [6], verbose=True)
#     model = VAR(data)
#     lag_order = model.select_order(5)
#     AIC.append(lag_order.aic)
#     BIC.append(lag_order.bic)
#     HQIC.append(lag_order.hqic)

# print("\nMean AIC:", np.mean(AIC))
# print("Mean BIC:", np.mean(BIC))
# print("Mean HQIC:", np.mean(HQIC))

### Cross-correlation
from scipy.signal import correlate
correlations = []
for sub in range(len(subjects)):
    corr = correlate(X_norm[sub], Y_norm[sub], mode='full')
    correlations.append(corr)
correlations = np.array(correlations)
# Generate lag indices
lags = np.arange(-len(X_norm.mean(0)) + 1, len(X_norm.mean(0)))

# fig, ax = plt.subplots(1, 1)
# ax.axvline(0, color='grey', linestyle='--', label="Zero Lag")
# ax.axhline(0, color='grey', linestyle='-')
# ax.plot(lags, correlations.mean(0), label='Cross-correlation')
# # ax.fill_between(0, correlations.mean(0), where=sig, alpha=0.2, zorder=5, facecolor='C7')
# correlation = correlate(X_norm.mean(0), Y_norm.mean(0), mode='full')
# ax.set_xlabel("Lag")
# ax.set_ylabel("Cross-Correlation")
# ax.set_title("Cross-Correlation Between X and Y")

# Find min and max lags
min_lag_idx = np.argmin(correlations.mean(0))
min_lag = lags[min_lag_idx]
print(f"Min lag: {min_lag}")
max_lag_idx = np.argmax(correlations.mean(0))
max_lag = lags[max_lag_idx]
print(f"Max lag: {max_lag}")

### Transfer Entropy
from pyinform.transferentropy import transfer_entropy

# X: Predictive coding, Y: Representational change
X, Y = mean_diag, sim_index

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
X_discrete = X_scaled.astype(int)
Y_discrete = Y_scaled.astype(int)

# fig, ax = plt.subplots(1, 1)
# ax.plot(X_discrete.mean(0), label='X: Predictive coding')
# ax.plot(Y_discrete.mean(0), label='Y: Representational change')
# ax.set_xlabel('Block')
# ax.set_ylabel('Scaled values')
# ax.set_title('Scaled data')
# ax.legend()

k = 3 # Number of nearest neighbors
# Calculate transfer entropy
te_XY = transfer_entropy(X_discrete.mean(0), Y_discrete.mean(0), k=k, local=False)
te_YX = transfer_entropy(Y_discrete.mean(0), X_discrete.mean(0), k=k, local=False)
# Print results
print(f"Transfer Entropy (X → Y): {te_XY}")
print(f"Transfer Entropy (Y → X): {te_YX}")

nn_xy = {}
nn_yx = {}
diff = {}
for lag in range(1, 7):
    if not lag in nn_xy:
        nn_xy[lag] = []
        nn_yx[lag] = []
    for sub in range(len(subjects)):
        X = X_discrete[sub]
        Y = Y_discrete[sub]
        te_XY = transfer_entropy(X, Y, k=lag, local=True)
        te_YX = transfer_entropy(Y, X, k=lag, local=True)
        nn_xy[lag].append(te_XY)
        nn_yx[lag].append(te_YX)
    nn_xy[lag] = np.squeeze(nn_xy[lag])
    nn_yx[lag] = np.squeeze(nn_yx[lag])
    diff[lag] = nn_xy[lag] - nn_yx[lag]
    
sig = {}
for lag in range(1, 7):
    pv = decod_stats(diff[lag], -1)
    sig[lag] = pv < 0.05

# fig, ax = plt.subplots(1, 1)
# for lag in range(1, 7):
#     ax.plot(nn_xy[lag].mean(0), label=f'lag={lag}')
# ax.set_xlabel("Lag")
# ax.set_ylabel("Transfer Entropy")
# ax.set_title("Transfer Entropy X → Y")
# ax.legend()

# fig, ax = plt.subplots(1, 1)
# for lag in range(1, 7):
#     ax.plot(nn_yx[lag].mean(0), label=f'lag={lag}')
# ax.set_xlabel("Lag")
# ax.set_ylabel("Transfer Entropy")
# ax.set_title("Transfer Entropy Y → X")
# ax.legend()

from sklearn.feature_selection import mutual_info_regression

X = X_discrete.copy()
Y = Y_discrete.copy() # Representational change

def optimal_lag_mi(X_sub, Y_sub, max_lag=10):
    """
    Find optimal lag for Transfer Entropy using Mutual Information for one subject.
    
    Args:
        X_sub (array): Time series for X (shape: n_time_points).
        Y_sub (array): Time series for Y (shape: n_time_points).
        max_lag (int): Maximum lag to test.

    Returns:
        best_lag (int): Optimal lag with highest Mutual Information.
    """
    mi_scores = []
    for lag in range(1, max_lag + 1):
        mi = mutual_info_regression(X_sub[:-lag].reshape(-1, 1), Y_sub[lag:].reshape(-1, 1), random_state=42)
        # mi = mutual_info_regression(X_sub[:-lag].reshape(-1, 1), Y_sub[lag:].reshape(-1, 1))
        mi_scores.append(mi[0])  # Store MI score for this lag

    best_lag = np.argmax(mi_scores) + 1  # Best lag is the one with highest MI
    return best_lag, mi_scores

# ---- Run for all subjects ----
def find_optimal_lags(X, Y, max_lag=10):
    """
    Compute optimal lag for TE per subject.
    
    Args:
        X (array): Shape (n_subjects, n_time_points).
        Y (array): Shape (n_subjects, n_time_points).
        max_lag (int): Maximum lag to test.

    Returns:
        optimal_lags (list): Optimal lag per subject.
        global_lag (int): Median of all subject lags.
    """
    optimal_lags = []
    for sub in range(X.shape[0]):  # Iterate over subjects
        best_lag, _ = optimal_lag_mi(X[sub], Y[sub], max_lag)
        optimal_lags.append(best_lag)
    
    global_lag = int(np.median(optimal_lags))  # Take median for a global choice
    return optimal_lags, global_lag

optimal_lags, global_lag = find_optimal_lags(X, Y)
print("Optimal lags per subject:", optimal_lags)
print("Global (median) lag:", global_lag)

subs = subjects
tes_xy, tes_yx = [], []
for isub, sub in enumerate(subs):
    # lag = optimal_lags[isub]
    lag = min(optimal_lags[isub], len(X[isub]) - 1)  # Ensure lag is valid
    xy = transfer_entropy(X[isub], Y[isub], k=lag, local=False)
    print("TE X -> Y:", xy, "for", sub, "with lag =", lag)
    yx = transfer_entropy(Y[isub], X[isub], k=lag, local=False)
    print("TE Y -> X:", yx, "for", sub, "with lag =", lag)
    tes_xy.append(xy)
    tes_yx.append(yx)
tes_xy, tes_yx = np.array(tes_xy), np.array(tes_yx)

# Try with IDTxl
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network

tes_xy, tes_yx = [], []
for isub, sub in enumerate(subs):
    lag = min(optimal_lags[isub], X.shape[1] - 1)  # Ensure lag does not exceed time points
    
    # Format data for IDTxl: Shape (n_timepoints, n_variables)
    data_array = np.vstack([X[isub], Y[isub]])  # Shape (n_timepoints, 2)
    data = Data(data_array, dim_order='ps')  # 'p' = points, 's' = sources
    
    network = BivariateTE()
    settings = {'source_target': [[0, 1]], # X → Y
                'cmi_estimator': 'JidtKraskovCMI', 
                'max_lag_sources': int(lag), # Optimal lag per subject
                'min_lag_sources': 0, # Lag - 1  
                    } 
    
    # TE X -> Y
    results = network.analyse_network(settings=settings, data=data)
    # results.print_edge_list(weights="max_te_lag", fdr=False)
    # plot_network(results=results, weights="max_te_lag", fdr=False)
    # plt.show()