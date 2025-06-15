import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp

subjects = SUBJS15
times = np.linspace(-0.2, 0.6, 82)

# win = np.where((times >= 0.28) & (times <= 0.51))[0]
win = np.where((times >= 0.3) & (times <= 0.5))[0]
# win = np.where(times >= 0.2)[0]
# win = np.load(FIGURES_DIR / "RSA" / "sensors" / "sig_rsa.npy")
c1, c2 = "#5BBCD6", "#00A08A"

# --- RSA sensors --- 40 trial bins ---
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
all_pats_bins, all_rands_bins = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / "rdm_40s" / subject
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern, random = [], []
    pattern_blocks, random_blocks = [], []
    pattern_bins, random_bins = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            p, r = [], []
            for fold in [1, 2]:
                p.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                r.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pattern_bins.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                random_bins.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
            pats.append(np.array(p))
            rands.append(np.array(r))
            pattern_blocks.append(np.nanmean(pats, 0))
            random_blocks.append(np.nanmean(rands, 0))
        pattern.append(np.nanmean(pats, 0))
        random.append(np.nanmean(rands, 0))
    if subject == 'sub05':
        for i in range(2):
            pattern[0][i] = pattern[1][0].copy()
            random[0][i] = random[1][0].copy()
        for i in range(3):
            for j in range(2):
                pattern_blocks[i][j] = np.mean(pattern_blocks[3], 0).copy()
                random_blocks[i][j] = np.mean(random_blocks[3], 0).copy()
        for i in range(6):
            pattern_bins[0][i] = np.mean(pattern_bins[1][:6], 0).copy()
            random_bins[0][i] = np.mean(random_bins[1][:6], 0).copy()
    pattern = np.array(pattern).mean(1)
    random = np.array(random).mean(1)
    pat, rand = get_all_high_low(pattern, random, sequence, False)
    all_pats.append(pat.mean(0))
    all_rands.append(rand.mean(0))
    
    pattern_blocks = np.array(pattern_blocks).mean(1)
    random_blocks = np.array(random_blocks).mean(1)
    pat_blocks, rand_blocks = get_all_high_low(pattern_blocks, random_blocks, sequence)
    all_pats_blocks.append(pat_blocks.mean(0))
    all_rands_blocks.append(rand_blocks.mean(0))

    pattern_bins = np.array(pattern_bins)
    random_bins = np.array(random_bins)
    pat_bins, rand_bins = get_all_high_low(pattern_bins, random_bins, sequence)
    all_pats_bins.append(pat_bins.mean(0))
    all_rands_bins.append(rand_bins.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)
all_pats_blocks = np.array(all_pats_blocks)
all_rands_blocks = np.array(all_rands_blocks)
all_pats_bins = np.array(all_pats_bins)
all_rands_bins = np.array(all_rands_bins)
# apply practice bsl
pat = all_pats[:, 1:, :].mean(1) - all_pats[:, 0, :]
rand = all_rands[:, 1:, :].mean(1) - all_rands[:, 0, :]
diff_rp = rand - pat
idx = np.where((times >= 0.3) & (times <= 0.5))[0]
bsl_pat = np.nanmean(all_pats_bins[:, :6, idx], axis=(1, 2))
bsl_rand = np.nanmean(all_rands_bins[:, :6, idx], axis=(1, 2))
pats_bins = np.nanmean(all_pats_bins[:, :, idx], 2) - bsl_pat[:, np.newaxis]
rands_bins = np.nanmean(all_rands_bins[:, :, idx], 2) - bsl_rand[:, np.newaxis]
diff_rp_bins = rands_bins - pats_bins
# get table
rows = list()
for i, subject in enumerate(subjects):
    for block in range(diff_rp_bins.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": diff_rp_bins[i, block]
        })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "RSA" / "sensors" / "rsa_bins_sensors.csv", index=False, sep=",")

# --- RSA source --- 40 trial bins ---
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
all_pats, all_rands = {}, {}
all_pats_blocks, all_rands_blocks = {}, {}
all_pats_bins, all_rands_bins = {}, {}
for network in tqdm(networks):
    if not network in all_pats:        
        all_pats[network], all_rands[network] = [], []
        all_pats_blocks[network], all_rands_blocks[network] = [], []
        all_pats_bins[network], all_rands_bins[network] = [], []
    for subject in subjects: 
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / "rdm_40s" / subject
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        pattern, random = [], []
        pattern_blocks, random_blocks = [], []
        pattern_bins, random_bins = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            pats, rands = [], []
            for block in blocks:
                p, r = [], []
                for fold in [1, 2]:
                    p.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    r.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
                    pattern_bins.append(np.load(res_path / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    random_bins.append(np.load(res_path / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pats.append(np.array(p))
                rands.append(np.array(r))
                pattern_blocks.append(np.array(pats).mean(0))
                random_blocks.append(np.array(rands).mean(0))
            pattern.append(np.nanmean(pats, 0))
            random.append(np.nanmean(rands, 0))
        if subject == 'sub05':
            for i in range(2):
                pattern[0][i] = pattern[1][0].copy()
                random[0][i] = random[1][0].copy()
            for i in range(3):
                for j in range(2):
                    pattern_blocks[i][j] = np.mean(pattern_blocks[3], 0).copy()
                    random_blocks[i][j] = np.mean(random_blocks[3], 0).copy()
            for i in range(6):
                pattern_bins[0][i] = np.mean(pattern_bins[1][:6], 0).copy()
                random_bins[0][i] = np.mean(random_bins[1][:6], 0).copy()
        pattern = np.array(pattern).mean(1)
        random = np.array(random).mean(1)
        pat, rand = get_all_high_low(pattern, random, sequence, False)
        all_pats[network].append(pat.mean(0))
        all_rands[network].append(rand.mean(0))
        pattern_blocks = np.array(pattern_blocks).mean(1)
        random_blocks = np.array(random_blocks).mean(1)
        pat_blocks, rand_blocks = get_all_high_low(pattern_blocks, random_blocks, sequence)
        all_pats_blocks[network].append(pat_blocks.mean(0))
        all_rands_blocks[network].append(rand_blocks.mean(0))
        pattern_bins = np.array(pattern_bins)
        random_bins = np.array(random_bins)
        pat_bins, rand_bins = get_all_high_low(pattern_bins, random_bins, sequence)
        all_pats_bins[network].append(pat_bins.mean(0))
        all_rands_bins[network].append(rand_bins.mean(0))
    all_pats[network] = np.array(all_pats[network])
    all_rands[network] = np.array(all_rands[network])
    all_pats_blocks[network] = np.array(all_pats_blocks[network])
    all_rands_blocks[network] = np.array(all_rands_blocks[network])
    all_pats_bins[network] = np.array(all_pats_bins[network])
    all_rands_bins[network] = np.array(all_rands_bins[network])
# baseline with practice
diff = dict()
rows = list()
for network in networks:
    bsl_pat = np.nanmean(all_pats_bins[network][:, :6, idx], axis=(1, 2))
    bsl_rand = np.nanmean(all_rands_bins[network][:, :6, idx], axis=(1, 2))
    mean_pat = np.nanmean(all_pats_bins[network][:, :, idx], axis=(-1)) - bsl_pat[:, np.newaxis]
    mean_rand = np.nanmean(all_rands_bins[network][:, :, idx], axis=(-1)) - bsl_rand[:, np.newaxis]
    diff[network] =  mean_rand - mean_pat
    # get table
    for i, subject in enumerate(subjects):
        for block in range(diff[network].shape[1]):
            rows.append({
                "network": network_names[networks.index(network)],
                "subject": subject,
                "block": block + 1,
                "value": diff[network][i, block]
            })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "RSA" / "source" / "rsa_bins_source.csv", index=False, sep=",")

# --- Temporal generalization sensors --- 40 trial bins ---
data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
jobs = -1
times = np.linspace(-4, 4, 813)
filt = np.where((times >= -1.5) & (times <= 3))[0]
times_filt = times[filt]
figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")
res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / 'scores_40s'
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
all_pats_bins, all_rands_bins = [], []
for subject in tqdm(subjects):    
    pattern, random = [], []
    pattern_blocks, random_blocks = [], []
    pattern_bins, random_bins = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            p, r = [], []
            for fold in range(1, 3):
                p.append(np.load(res_path / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                r.append(np.load(res_path / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pattern_bins.append(np.load(res_path / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                random_bins.append(np.load(res_path / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
            pats.append(np.array(p))
            rands.append(np.array(r))
            pattern_blocks.append(np.array(pats).mean(0))
            random_blocks.append(np.array(rands).mean(0))
        if epoch_num != 0:
            pattern.append(np.mean(pats, 0))
            random.append(np.mean(rands, 0))
    pattern = np.array(pattern).mean(1)
    random = np.array(random).mean(1)
    pattern_blocks = np.array(pattern_blocks).mean(1)
    random_blocks = np.array(random_blocks).mean(1)
    pattern_bins = np.array(pattern_bins)
    random_bins = np.array(random_bins)
    all_pats.append(pattern)
    all_rands.append(random)
    all_pats_blocks.append(pattern_blocks)
    all_rands_blocks.append(random_blocks)
    all_pats_bins.append(pattern_bins)
    all_rands_bins.append(random_bins)
all_pats = np.array(all_pats).mean(1)[:, filt][:, :, filt]
all_rands = np.array(all_rands).mean(1)[:, filt][:, :, filt]
all_pats_blocks = np.array(all_pats_blocks)
all_rands_blocks = np.array(all_rands_blocks)
all_pats_bins = np.array(all_pats_bins)
all_rands_bins = np.array(all_rands_bins)
# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_bins = []
contrast_bins = all_pats_bins - all_rands_bins
for sub in range(len(subjects)):
    tg = []
    for bin in range(46):
        data = contrast_bins[sub, bin, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    box_bins.append(np.array(tg))
box_bins = np.array(box_bins)
# get table
rows = list()
for i, subject in enumerate(subjects):
    for block in range(box_bins.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": box_bins[i, block]
        })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "time_gen" / "sensors" / "timeg_bins_sensors.csv", index=False, sep=",")

# --- Temporal generalization source --- 40 trial bins ---
subjects = SUBJS15
res_dir = RESULTS_DIR / 'TIMEG' / 'source'
networks = 
timesg = np.linspace(-1.5, 1.5, 307)
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
all_pats_bins, all_rands_bins = {}, {}
all_box_bins = {}
for network in tqdm(networks):
    if not network in all_pats_bins:
        all_pats_bins[network], all_rands_bins[network] = [], []
        all_box_bins[network] = []
    for i, subject in enumerate(subjects):
        pattern_bins, random_bins = [], []
        for epoch_num in [0, 1, 2, 3, 4]:
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            for block in blocks:
                for fold in range(1, 3):
                    pattern_bins.append(np.load(res_dir / network / 'scores_40s' / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    random_bins.append(np.load(res_dir / network / 'scores_40s' / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
        if subject == 'sub05':
            for i in range(6):
                pattern_bins[0][i] = np.mean(pattern_bins[1][:6], 0).copy()
                random_bins[0][i] = np.mean(random_bins[1][:6], 0).copy()
        all_pats_bins[network].append(np.array(pattern_bins))
        all_rands_bins[network].append(np.array(random_bins))
    all_pats_bins[network] = np.array(all_pats_bins[network])
    all_rands_bins[network] = np.array(all_rands_bins[network])
    contrast = all_pats_bins[network] - all_rands_bins[network]
    box_bins = []
    for sub in range(len(subjects)):
        tg = []
        for bin in range(46):
            # data is a square matrix of shape (len(idx_tg), len(idx_tg))
            data = contrast[sub, bin, idx_timeg, :][:, idx_timeg]
            tg.append(data.mean())
        box_bins.append(np.array(tg))
    all_box_bins[network] = np.array(box_bins)

rows = list()
for i, network in enumerate(networks):
    diff = all_box_bins[network]
    # get table
    for j, subject in enumerate(subjects):
        for block in range(diff.shape[1]):
            rows.append({
                "network": network_names[i],
                "subject": subject,
                "block": block + 1,
                "value": diff[j, block]
            })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "time_gen" / "source" / "timeg_bins_source.csv", index=False, sep=",")