import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = SUBJS15
lock = 'stim'

times = np.linspace(-0.2, 0.6, 82)
# win = np.where(times >= 0.2)[0]
win = np.where((times >= 0.3) & (times <= 0.5))[0]
# win = np.load(FIGURES_DIR / "RSA" / "sensors" / "sig_rsa.npy")

# --------- RSA per block ---------
# --- within session ---
all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock 
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    for epoch_num in range(5):
        # blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(1 + 5 * (epoch_num - 1), 5 * (epoch_num) + 1)]
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(1, 6)]
        pats, rands = [], []
        for block in blocks:
            pats.append(np.load(res_path / "split_pattern" / f"{subject}-{epoch_num}-{block}.npy"))
            rands.append(np.load(res_path / "split_random" / f"{subject}-{epoch_num}-{block}.npy"))
        pattern.append(np.array(pats))
        random.append(np.array(rands))
    pattern = np.vstack(pattern)
    random = np.vstack(random)
    pat_blocks, rand_blocks = get_all_high_low(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)    

m_pattern = np.nanmean(all_pats[:, :, win], axis=2)
m_random = np.nanmean(all_rands[:, :, win], axis=2)
prac_pat = np.nanmean(all_pats[:, :3, win], (1, 2))
prac_rand = np.nanmean(all_rands[:, :3, win], (1, 2))

sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(23):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

blocks = [i for i in range(1, all_pats.shape[1] + 1)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(1, 3, color='grey', alpha=0.1)
ax.set_xticks(blocks)
ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, sim_index.mean(0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Representational change effect per block - within session', fontstyle='italic')

# --- across sessions ---
all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock 
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    blocks = [i for i in range(1, 21)]
    for block in blocks:
        pattern.append(np.load(res_path / "split_all_pattern" / f"{subject}-{block}.npy"))
        random.append(np.load(res_path / "split_all_random" / f"{subject}-{block}.npy"))
    pattern = np.array(pattern)
    random = np.array(random)
    pat_blocks, rand_blocks = get_all_high_low(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)    

m_pattern = np.nanmean(all_pats[:, :, win], axis=2)
m_random = np.nanmean(all_rands[:, :, win], axis=2)
prac_pat = np.nanmean(all_pats[:, :3, win], (1, 2))
prac_rand = np.nanmean(all_rands[:, :3, win], (1, 2))

sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(20):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

blocks = [i for i in range(1, all_pats.shape[1] + 1)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(1, 3, color='grey', alpha=0.1)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, sim_index.mean(0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Representational change effect per block - across sessions', fontstyle='italic')

# --------- RSA per trial bins ---------
# --- within session ---
all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_pat = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_pattern" / subject
    res_rand = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_random" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 13)] if epoch_num == 0 else [i for i in range(1, 21)]
        pats, rands = [], []
        for block in blocks:
            pats.append(np.load(res_pat / f"{subject}-{epoch_num}-{block}.npy"))
            rands.append(np.load(res_rand / f"{subject}-{epoch_num}-{block}.npy"))
        pattern.append(np.array(pats))
        random.append(np.array(rands))
    pattern = np.vstack(pattern)
    random = np.vstack(random)

    pat_blocks, rand_blocks = get_all_high_low(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)

m_pattern = np.nanmean(all_pats[:, :, win], axis=2)
m_random = np.nanmean(all_rands[:, :, win], axis=2)
prac_pat = np.nanmean(all_pats[:, :12, win], (1, 2))
prac_rand = np.nanmean(all_rands[:, :12, win], (1, 2))

sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(all_pats.shape[1]):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

blocks = [i for i in range(1, all_pats.shape[1] + 1)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(1, 12, color='grey', alpha=0.1)
ax.set_xticks(blocks)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, np.nanmean(sim_index, 0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Representational change effect per trial bin in session')

# --- across session ---
all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_pat = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_all_pattern" / subject
    res_rand = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_all_random" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    for block in np.arange(1, 81):
        pattern.append(np.load(res_pat / f"{subject}-{block}.npy"))
        random.append(np.load(res_rand / f"{subject}-{block}.npy"))
    pattern = np.array(pattern)
    random = np.array(random)
    pat_blocks, rand_blocks = get_all_high_low(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)

prac_pat = prac_pat.copy()
prac_rand = prac_rand.copy()
m_pattern = np.nanmean(all_pats[:, :, win], axis=2)
m_random = np.nanmean(all_rands[:, :, win], axis=2)

sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(all_pats.shape[1]):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

blocks = [i for i in range(1, all_pats.shape[1] + 1)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(1, 12, color='grey', alpha=0.1)
ax.set_xticks(blocks)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, np.nanmean(sim_index, 0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Representational change effect per trial bin in all sessions')

# --------- Temporal generalization ---------
# --- within session ---

timeg_data_path = TIMEG_DATA_DIR / 'results' / 'sensors'
timesg = np.linspace(-1.5, 4, 559)
pattern, random = [], []
for subject in tqdm(subjects):
    pat, rand = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(1, 6)]
        for block in blocks:
            pat.append(np.load(timeg_data_path / 'split_pattern' /  f"{subject}-{epoch_num}-{block}.npy"))
            rand.append(np.load(timeg_data_path / 'split_random' / f"{subject}-{epoch_num}-{block}.npy"))
    pattern.append(np.array(pat))
    random.append(np.array(rand))
pattern, random = np.array(pattern), np.array(random)
contrast = pattern - random

# mean diag
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
mean_diag = []
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = np.diag(contrast[sub, block])
        tg.append(data[idx_timeg].mean())
    mean_diag.append(np.array(tg))
mean_diag = np.array(mean_diag)

blocks = [i for i in range(23)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(0, 2, color='grey', alpha=0.1)
ax.set_xticks(blocks)
ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
for i in range(mean_diag.shape[0]):
    ax.plot(blocks, mean_diag[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, mean_diag.mean(0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Predictive effect per block - within session', fontstyle='italic')

# --- across session ---
timeg_data_path = TIMEG_DATA_DIR / 'results' / 'sensors'
timesg = np.linspace(-1.5, 4, 559)
pattern, random = [], []
for subject in tqdm(subjects):
    pat, rand = [], []
    blocks = [i for i in range(1, 24)]
    for block in blocks:
        pat.append(np.load(timeg_data_path / 'split_all_pattern' /  f"{subject}-{block}.npy"))
        rand.append(np.load(timeg_data_path / 'split_all_random' / f"{subject}-{block}.npy"))
    pattern.append(np.array(pat))
    random.append(np.array(rand))
pattern, random = np.array(pattern), np.array(random)
contrast = pattern - random

# mean diag
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
mean_diag = []
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = np.diag(contrast[sub, block])
        tg.append(data[idx_timeg].mean())
    mean_diag.append(np.array(tg))
mean_diag = np.array(mean_diag)

blocks = [i for i in range(23)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(0, 2, color='grey', alpha=0.1)
ax.set_xticks(blocks)
ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
for i in range(mean_diag.shape[0]):
    ax.plot(blocks, mean_diag[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, mean_diag.mean(0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Predictive effect per block - across session', fontstyle='italic')

# --------- RSA per trial bins ---------
# --- within session ---

all_pats, all_rands = [], []
for subject in tqdm(subjects):
    res_pat = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_pattern" / subject
    res_rand = RESULTS_DIR / 'RSA' / 'sensors' / lock / "split_20s_random" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 13)] if epoch_num == 0 else [i for i in range(1, 21)]
        pats, rands = [], []
        for block in blocks:
            pats.append(np.load(res_pat / f"{subject}-{epoch_num}-{block}.npy"))
            rands.append(np.load(res_rand / f"{subject}-{epoch_num}-{block}.npy"))
        pattern.append(np.array(pats))
        random.append(np.array(rands))
    pattern = np.vstack(pattern)
    random = np.vstack(random)

    pat_blocks, rand_blocks = get_all_high_low(pattern, random, sequence)
    all_pats.append(pat_blocks.mean(0))
    all_rands.append(rand_blocks.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)

m_pattern = np.nanmean(all_pats[:, :, win], axis=2)
m_random = np.nanmean(all_rands[:, :, win], axis=2)
prac_pat = np.nanmean(all_pats[:, :12, win], (1, 2))
prac_rand = np.nanmean(all_rands[:, :12, win], (1, 2))

sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(all_pats.shape[1]):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

blocks = [i for i in range(1, all_pats.shape[1] + 1)]
cmap = plt.cm.get_cmap('tab20', len(subjects))
fig, ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True, layout='tight')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.axvspan(1, 12, color='grey', alpha=0.1)
ax.set_xticks(blocks)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5, color=cmap(i))
ax.plot(blocks, np.nanmean(sim_index, 0), lw=3, color='#00A08A', label='Mean')
ax.set_ylabel('Mean RSA effect')
ax.legend(frameon=False)
ax.set_title('Representational change effect per trial bin in session')