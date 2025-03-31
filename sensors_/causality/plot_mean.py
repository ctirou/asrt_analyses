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

blocks = np.arange(23)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(3, 7, color='purple', alpha=0.1)
ax.axvspan(8, 12, color='purple', alpha=0.1)
ax.axvspan(13, 17, color='purple', alpha=0.1)
ax.axvspan(18, 22, color='purple', alpha=0.1)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5)
# ax.set_xticks(range(1, 24))
ax.set_xticks(blocks)
ax.plot(blocks, sim_index.mean(0), lw=3, color='red', label='RSA')
ax.set_xlabel('Block')
ax.set_ylabel('Mean RSA effect')
ax.axvspan(peak_rsa-0.05, peak_rsa+0.05, color='red', alpha=0.5, label='peak')
ax.axvspan(0, 2, color='grey', alpha=0.1, label='practice')
ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
ax.legend()
plt.show()

import matplotlib.pyplot as plt
plt.imshow(pattern.mean((0, 1)),
           interpolation="lanczos",
                            origin="lower",
                            cmap='RdBu_r',
                            extent=timesg[[0, -1, 0, -1]],
                            aspect=0.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.show()

plt.imshow(random.mean((0, 1)),
           interpolation="lanczos",
                            origin="lower",
                            cmap='RdBu_r',
                            extent=timesg[[0, -1, 0, -1]],
                            aspect=0.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.show()

plt.imshow(contrast.mean((0, 1)),
           interpolation="lanczos",
                            origin="lower",
                            cmap='RdBu_r',
                            extent=timesg[[0, -1, 0, -1]],
                            aspect=0.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.show()

# pval = gat_stats(contrast.mean(1), -1)
# sig = pval < 0.05
plt.imshow(contrast.mean((0, 1)),
           interpolation="lanczos",
                            origin="lower",
                            cmap='RdBu_r',
                            extent=timesg[[0, -1, 0, -1]],
                            aspect=0.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
# xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
# plt.contour(xx, yy, sig, colors='black', levels=[0],
#                     linestyles='--', linewidths=1, alpha=.5)
plt.show()

# mean box
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
means = []
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = contrast[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    means.append(np.array(tg))
means = np.array(means)

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
    peaks.append(blocks[np.argmax(means[subject])])
peaks = np.array(peaks)
peak_tg = int(round(peaks.mean()))

from scipy.stats import zscore
X_scaled = zscore(mean_diag, axis=1)
y_scaled = zscore(sim_index, axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

ax1.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax1.axvspan(0, 2, color='grey', alpha=0.1)
ax1.axvspan(3, 7, color='purple', alpha=0.1)
ax1.axvspan(8, 12, color='purple', alpha=0.1)
ax1.axvspan(13, 17, color='purple', alpha=0.1)
ax1.axvspan(18, 22, color='purple', alpha=0.1)
for i in range(sim_index.shape[0]):
    ax1.plot(blocks, sim_index[i], alpha=0.5)
# ax.set_xticks(range(1, 24))
ax1.set_xticks(blocks)
ax1.plot(blocks, sim_index.mean(0), lw=3, color='red', label='Mean')
ax1.set_ylabel('Mean RSA effect')
ax1.axvspan(peak_rsa-0.05, peak_rsa+0.05, color='red', alpha=0.5, label='Mean peak')
ax1.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
ax1.legend()
ax1.set_title('Representational change')

ax2.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax2.axvspan(0, 2, color='grey', alpha=0.1)
ax2.axvspan(3, 7, color='purple', alpha=0.1)
ax2.axvspan(8, 12, color='purple', alpha=0.1)
ax2.axvspan(13, 17, color='purple', alpha=0.1)
ax2.axvspan(18, 22, color='purple', alpha=0.1)
for i in range(means.shape[0]):
    ax2.plot(blocks, means[i], alpha=0.5)
# ax.set_xticks(range(1, 24))
ax2.set_xticks(blocks)
ax2.plot(blocks, means.mean(0), lw=3, color='blue', label='Mean')
# ax.plot(blocks, sim_index.mean(0), lw=3, label='Representational change')
ax2.set_xlabel('Block')
ax2.set_ylabel('Mean predictive effect')
ax2.axvspan(peak_tg-0.05, peak_tg+0.05, color='blue', alpha=0.5, label='Mean peak')
ax2.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
ax2.legend()
ax2.set_title('Predictive coding')

data = pd.DataFrame({'X': mean_diag.mean(0), 'Y': sim_index.mean(0)})

def check_stationarity(series):
    """
    Check if the series is stationary using the Augmented Dickey-Fuller test.
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    new_series = np.zeros((series.shape[0], series.shape[1] - 1))
    sig = []
    for sub in range(series.shape[0]):
        result = adfuller(series[sub])
        if result[1] < 0.05:
            sig.append(True)
            new_series[sub, :] = series[sub, 1:]
        else:
            sig.append(False)
            new_series[sub, :] = np.diff(series[sub, :])
    return sig, new_series

sig_tg, new_X = check_stationarity(mean_diag)
sig_rsa, new_Y = check_stationarity(sim_index)

sig_x, _ = check_stationarity(new_X)
sig_y, _ = check_stationarity(new_Y)

x = np.array([np.diff(mean_diag[sub])[1:] for sub in range(len(subjects))])
y = np.array([np.diff(sim_index[sub])[1:] for sub in range(len(subjects))])

plt.plot([i for i in range(22)], x.mean(0), label='x mean')
plt.plot([i for i in range(22)], y.mean(0), label='y mean')
plt.legend()
plt.show()

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import pandas as pd

# X : predictive coding
# Y : representational change

# X -> Y
AIC, BIC, HQIC = [], [], []
for sub in range(len(subjects)):
    print(f"\n##### Subject {sub} #####")
    data = pd.DataFrame({'X': mean_diag[sub], 'Y': sim_index[sub]})
    gc_res = grangercausalitytests(data, 6, verbose=True)
    model = VAR(data)
    lag_order = model.select_order(6)
    AIC.append(lag_order.aic)
    BIC.append(lag_order.bic)
    HQIC.append(lag_order.hqic)
print("\nMean AIC:", np.mean(AIC))
print("Mean BIC:", np.mean(BIC))
print("Mean HQIC:", np.mean(HQIC))

print("\nMedian AIC:", np.median(AIC))
print("Median BIC:", np.median(BIC))
print("Median HQIC:", np.median(HQIC))

# Y -> X
AIC, BIC, HQIC = [], [], []
for sub in range(len(subjects)):
    print(f"\n##### Subject {sub} #####")
    data = pd.DataFrame({'X': sim_index[sub], 'Y': mean_diag[sub]})
    gc_res = grangercausalitytests(data, [4], verbose=True)
    model = VAR(data)
    lag_order = model.select_order(6)
    AIC.append(lag_order.aic)
    BIC.append(lag_order.bic)
    HQIC.append(lag_order.hqic)
print("Mean AIC:", np.mean(AIC))
print("Mean BIC:", np.mean(BIC))
print("Mean HQIC:", np.mean(HQIC))