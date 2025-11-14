# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr as spear, ttest_1samp
from tqdm.auto import tqdm
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors

subjects = SUBJS15
jobs = -1
overwrite = False

data_type = "scores_blocks"

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-4, 4, 813)

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type

# load patterns and randoms time-generalization on all epochs
pats_blocks, rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / data_type / subject
    pattern, random = [], []
    for block in range(1, 24):
        pfname = res_path / f'pat-{block}.npy' if block not in [1, 2, 3] else res_path / f'pat-0-{block}.npy'
        rfname = res_path / f'rand-{block}.npy' if block not in [1, 2, 3] else res_path / f'rand-0-{block}.npy'
        pattern.append(np.load(pfname))
        random.append(np.load(rfname))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-4.npy")
        rand_bsl = np.load(res_path / "rand-4.npy")
        for i in range(3):
            pattern[i] = pat_bsl.copy()
            random[i] = rand_bsl.copy()
    pats_blocks.append(np.array(pattern))
    rands_blocks.append(np.array(random))
pats_blocks = np.array(pats_blocks)
rands_blocks = np.array(rands_blocks)

learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)    

chance = .25
threshold = .05
threshold = .01

idx = np.where((times >= -1.5) & (times <= 3))[0]
res_path = ensured(res_dir / "pval")
if not op.exists(res_path/ "all_pattern-pval.npy") or overwrite:
    print('Computing pval for all patterns')
    pval = gat_stats(pats_blocks[:, 3:, idx][:, 3:, :, idx].mean(1) - chance, jobs) # caution: the first 3 blocks are practice, need to exclude
    np.save(res_path/ "all_pattern-pval.npy", pval)
if not op.exists(res_path/ "all_random-pval.npy") or overwrite:
    print('Computing pval for all randoms')
    pval = gat_stats(rands_blocks[:, 3:, idx][:, 3:, :, idx].mean(1) - chance, jobs)
    np.save(res_path/ "all_random-pval.npy", pval)
if not op.exists(res_path/ "all_contrast-pval.npy") or overwrite:
    print('Computing pval for all contrasts')
    contrasts = pats_blocks - rands_blocks
    pval = gat_stats(contrasts[:, 3:, idx][:, 3:, :, idx].mean(1), jobs)
    np.save(res_path/ "all_contrast-pval.npy", pval)

filt = np.where((times >= -1.5) & (times <= 3))[0]
ensure_dir(res_dir / "corr")
if not op.exists(res_dir / "corr" / "rhos_learn.npy") or overwrite:
    contrasts = pats_blocks - rands_blocks
    contrasts = contrasts[:, :, filt][:, :, :, filt]
    all_rhos = []
    for sub in range(len(subjects)):
        print(f"Computing Spearman correlation for subject {sub+1}/{len(subjects)}")
        rhos = np.empty((times[filt].shape[0], times[filt].shape[0]))
        vector = learn_index_blocks.iloc[sub]  # vector to correlate with
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times[filt])) for g in range(len(times[filt])))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times[filt])) for g in range(len(times[filt]))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    all_rhos = fisher_z_transform_3d(all_rhos)
    np.save(res_dir / "corr" / "rhos_learn.npy", all_rhos)
    pval = gat_stats(all_rhos, -1)
    np.save(res_dir / "corr" / "pval_learn-pval.npy", pval)

filt = np.where((times >= -.51) & (times <= .3))[0]
new_times = times[filt]
sig_times = np.linspace(-1.5, 3, 457)
sig_filt = np.where((sig_times >= -.51) & (sig_times <= .3))[0]
chance = 25

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), layout='tight')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, zorder=-1, alpha=.1)

ax1.axhline(chance, color='grey', alpha=0.5)
ax2.axhline(0, color='grey', alpha=0.5)

# Plot pattern and random diagonals with significance
for data, title in zip([pats_blocks, rands_blocks], ["pattern", "random"]):
    color =  "#FAD510" if title == "pattern" else "#FF718B"
    # zorder = 10 if title == "random" else 5
    zorder = 10
    alpha_fill = 0.2 if title == 'pattern' else 0.1
    # alpha_fill = 0.2
    diag_data = np.array([np.diag(data[sub, 3:].mean(0)) for sub in range(len(subjects))])[:, filt] * 100
    mean_diag = diag_data.mean(0)
    sem_diag = diag_data.std(0) / np.sqrt(len(subjects))
    ax1.plot(new_times, mean_diag, color=color, alpha=1, zorder=zorder, label=title.capitalize())
    pval = np.diag(np.load(res_path/ f"all_{title.lower()}-pval.npy"))[sig_filt]
    sig = pval < threshold
    ax1.fill_between(new_times, mean_diag - sem_diag, mean_diag + sem_diag, facecolor=color, alpha=alpha_fill, zorder=5)
    # for start, end in contiguous_regions(sig):
    #     ax1.plot(new_times[start:end], mean_diag[start:end], alpha=1, zorder=zorder, color=color, label=title.capitalize())
    # ax1.fill_between(new_times, mean_diag - sem_diag, mean_diag + sem_diag, where=sig, alpha=0.4, zorder=5, facecolor=color)
    # ax1.fill_between(new_times, mean_diag - sem_diag, chance, where=sig, alpha=0.3, zorder=5, facecolor=color)
    # break
ax1.set_ylim(None, 45)
ax1.text(0.1, 44.5, '$Stimulus$', fontsize=11, ha='center', va='top')
ax1.legend(frameon=False, loc='upper left')
ax1.set_ylabel('Decoding accuracy (%)', fontsize=11)
ax1.set_title("Pattern and Random decoding", fontsize=13)

# Plot contrast diagonal with significance
contrasts = pats_blocks - rands_blocks
diag_contrasts = np.array([np.diag(contrasts[sub, 3:].mean(0)) for sub in range(len(subjects))])[:, filt]
mean_diag_contrasts = diag_contrasts.mean(0)
sem_diag_contrasts = diag_contrasts.std(0) / np.sqrt(len(subjects))
pval_contrasts = np.diag(np.load(res_path/ "all_contrast-pval.npy"))[sig_filt]
sig_contrasts = pval_contrasts < threshold
ax2.plot(new_times, mean_diag_contrasts, label="Contrast (Pattern - Random)", color='C7', alpha=1, zorder=10)
ax2.fill_between(new_times, mean_diag_contrasts - sem_diag_contrasts, mean_diag_contrasts + sem_diag_contrasts, facecolor='C7', alpha=0.2, zorder=5)
for start, end in contiguous_regions(sig_contrasts):
    ax2.plot(new_times[start:end], mean_diag_contrasts[start:end], alpha=1, zorder=10, color="#56B4E9")
ax2.fill_between(new_times, mean_diag_contrasts - sem_diag_contrasts, mean_diag_contrasts + sem_diag_contrasts, where=sig_contrasts, alpha=0.4, zorder=5, facecolor="#56B4E9")
ax2.text(-0.25, 0.04, '***', fontsize=25, ha='center', va='center', color="#56B4E9", weight='bold')

ax2.set_title("Contrast (Pattern - Random)", fontsize=13)
ax2.set_ylabel('Difference in accuracy (%)', fontsize=11)
ax2.set_xlabel('Time (s)', fontsize=11)

fname = 'timeg_diag-sensors.pdf'
plt.savefig(figure_dir / fname, transparent=True)