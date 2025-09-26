import os.path as op
import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr as spear
from scipy.ndimage import gaussian_filter1d

subjects = SUBJS15
times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.3) & (times <= 0.5))[0]
c1, c2 = "#5BBCD6", "#00A08A"

# --- RSA sensors --- blocks ---
data_type = "rdm_blocks_new" # "rdm_blocks_new" for all trials or "rdm_blocks" for correct trials only
bsl_practice = False
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / data_type / subject
    # read behav        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern_blocks, random_blocks = [], []
    for epoch_num in range(5):
        blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
        pats, rands = [], []
        for block in blocks:
            pattern_blocks.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
            random_blocks.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-1-1.npy")
        rand_bsl = np.load(res_path / "rand-1-1.npy")
        for i in range(3):
            pattern_blocks[i] = pat_bsl.copy()
            random_blocks[i] = rand_bsl.copy()
    pattern_blocks = np.array(pattern_blocks)
    random_blocks = np.array(random_blocks)
    high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False)
    all_pats.append(high.mean(0))
    all_rands.append(low.mean(0))
all_pats = np.array(all_pats)
all_rands = np.array(all_rands)
bsl_pat = np.nanmean(all_pats[:, :3, :], 1) 
bsl_rand = np.nanmean(all_rands[:, :3, :], 1)
pat = all_pats - bsl_pat[:, np.newaxis, :] if bsl_practice else all_pats
rand = all_rands - bsl_rand[:, np.newaxis, :] if bsl_practice else all_rands
diff_rp = rand - pat

# extract index from GAMM segments
segments = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors.csv")
start = segments.iloc[0]['start'] if data_type.endswith("new") else segments.iloc[1]['start']
end = segments.iloc[0]['end'] if data_type.endswith("new") else segments.iloc[1]['end']
idx = np.arange(int(start), int(end)+1)

# save table
diff_rp_blocks = np.nanmean(diff_rp[:, :, idx], axis=(-1))
rows = list()
for i, subject in enumerate(subjects):
    for block in range(diff_rp_blocks.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": diff_rp_blocks[i, block]
        })
df = pd.DataFrame(rows)
# fname = "rsa_blocks_sensors" if not data_type.endswith("new") else "rsa_blocks_sensors_new"
# fname += "_no_bsl.csv" if not bsl_practice else "_bsl.csv"
fname = 'rsa_sensors_br'
fname += '_all.csv' if data_type.endswith("new") else '.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# save table
diff_rp_tr = np.nanmean(diff_rp[:, 3:], 1)
rows = list()
for s, subject in enumerate(subjects):
    for t, time in enumerate(times):
        rows.append({
            "subject": subject,
            'time': t,
            "value": diff_rp_tr[s, t]
        })
df = pd.DataFrame(rows)
# fname = "rsa_blocks_sensors" if not data_type.endswith("new") else "rsa_blocks_sensors_new"
# fname += "_no_bsl.csv" if not bsl_practice else "_bsl.csv"
fname = 'rsa_sensors_tr_all.csv' if data_type.endswith("new") else 'rsa_sensors_tr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
learn_index_blocks = learn_index_blocks.sub(learn_index_blocks.mean(axis=1), axis=0)
all_rhos = np.array([[spear(learn_index_blocks.iloc[sub], diff_rp[sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
# save table
rows = list()
for s, subject in enumerate(subjects):
    for t, time in enumerate(times):
        rows.append({
            "subject": subject,
            'time': t,
            "value": all_rhos[s, t]
        })
df = pd.DataFrame(rows)
fname = 'rsa_sensors_tr_all_corr.csv' if data_type.endswith("new") else 'rsa_sensors_tr_corr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
learn_index_blocks = learn_index_blocks.iloc[:, 3:].sub(learn_index_blocks.iloc[:, 3:].mean(axis=1), axis=0)
all_rhos = np.array([[spear(learn_index_blocks.iloc[sub], diff_rp[sub, 3:, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
# save table
rows = list()
for s, subject in enumerate(subjects):
    for t, time in enumerate(times):
        rows.append({
            "subject": subject,
            'time': t,
            "value": all_rhos[s, t]
        })
df = pd.DataFrame(rows)
fname = 'rsa_sensors_tr_no_prac_corr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# RSA source --- blocks ---
data_type = "rdm_blocks_vect_new" # "rdm_blocks_vect_new" for all trials or "rdm_blocks_vect" for correct trials only
bsl_practice = False
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
diff_rp = {}
corr_rp = {}
learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
corr_rp_no_prac = {}
learn_index_blocks_no_prac = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
learn_index_blocks_no_prac = learn_index_blocks_no_prac.iloc[:, 3:].sub(learn_index_blocks_no_prac.iloc[:, 3:].mean(axis=1), axis=0)
for network in tqdm(networks):
    if not network in diff_rp:
        diff_rp[network] =  []
        corr_rp[network] = []
        corr_rp_no_prac[network] = []
    for isub, subject in enumerate(subjects):
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / data_type / subject
        # read behav
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        pattern_blocks, random_blocks = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            pats, rands = [], []
            for block in blocks:
                pattern_blocks.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
                random_blocks.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
        if subject == 'sub05':
            pat_bsl = np.load(res_path / "pat-1-1.npy")
            rand_bsl = np.load(res_path / "rand-1-1.npy")
            for i in range(3):
                pattern_blocks[i] = pat_bsl.copy()
                random_blocks[i] = rand_bsl.copy()
        pattern_blocks = np.array(pattern_blocks)
        random_blocks = np.array(random_blocks)
        high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False)
        bsl_pat = np.nanmean(high[:, :3, :], (0, 1))
        bsl_rand = np.nanmean(low[:, :3, :], (0, 1))
        pat = np.nanmean(high, 0) - bsl_pat[np.newaxis, :] if bsl_practice else np.nanmean(high, 0)
        rand = np.nanmean(low, 0) - bsl_rand[np.newaxis, :] if bsl_practice else np.nanmean(low, 0)
        diff = rand - pat
        diff_rp[network].append(diff)
        corr_rp[network].append([np.array([spear(learn_index_blocks.iloc[isub], diff[:, t])[0] for t in range(len(times))])])
        corr_rp_no_prac[network].append([np.array([spear(learn_index_blocks_no_prac.iloc[isub], diff[3:, t])[0] for t in range(len(times))])])
    diff_rp[network] = np.array(diff_rp[network])
    corr_rp[network] = np.array(corr_rp[network]).squeeze()
    corr_rp[network], _, _ = fisher_z_and_ttest(corr_rp[network])
    corr_rp_no_prac[network] = np.array(corr_rp_no_prac[network]).squeeze()
    corr_rp_no_prac[network], _, _ = fisher_z_and_ttest(corr_rp_no_prac[network])

# extract index from GAMM segments
segments = pd.read_csv(FIGURES_DIR / "TM" / "segments_tr_sensors.csv")
start = segments.iloc[0]['start'] if data_type.endswith("new") else segments.iloc[1]['start']
end = segments.iloc[0]['end'] if data_type.endswith("new") else segments.iloc[1]['end']
idx = np.arange(int(start), int(end)+1)

# save table
rows = list()
for i, network in enumerate(networks):
    diff = np.nanmean(diff_rp[network][:, :, idx], axis=(-1))
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
fname = 'rsa_source_br'
fname += '_all.csv' if data_type.endswith("new") else '.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# plot time resolved RSA source
xmin, xmax = 0.3, 0.55
win = np.where((times >= xmin) & (times <= xmax))[0]
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    data = diff_rp[network][:, 3:, :]
    ax.plot(times, np.nanmean(data, axis=(0, 1)), color=c1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sig = decod_stats(np.nanmean(data, axis=(1)), -1) < 0.05
    ax.fill_between(times, np.nanmean(data, axis=(0, 1)), 0, where=sig, color='red', alpha=0.3) 
    ax.set_title(network_names[i], fontstyle='italic')
    smoothed = gaussian_filter1d(np.nanmean(data, axis=(0, 1)), sigma=1.5)
    mdiff = np.nanmean(data[:, :, win], axis=(1, -1))
    mdiff_sig = ttest_1samp(mdiff, 0)[1] < 0.05
    if mdiff_sig:
        ax.axvspan(times[win][0], times[win][-1], facecolor='orange', edgecolor=None, alpha=0.2, zorder=5)
title = 'RS source - time resolved (correct trials only)' if not data_type.endswith("new") else 'RS source - time resolved (all trials)'
title += ' - no practice baseline' if not bsl_practice else ' - practice baseline'
fig.suptitle(title, fontsize=14)
fname = "rsa_time_resolved_source" if not data_type.endswith("new") else "rsa_time_resolved_source_new"
fname += "_no_bsl.pdf" if not bsl_practice else "_bsl.pdf"
fig.savefig(FIGURES_DIR / "RSA" / fname, transparent=True)
plt.close(fig)

# save table
rows = list()
for i, network in enumerate(networks):
    diff = np.nanmean(diff_rp[network][:, 3:], axis=(1))
    # get table
    for j, subject in enumerate(subjects):
        for t, time in enumerate(times):
            rows.append({
                "network": network_names[i],
                "subject": subject,
                "time": t,
                "value": diff[j, t]
            })
df = pd.DataFrame(rows)
# fname = "rsa_blocks_source" if not data_type.endswith("new") else "rsa_blocks_source_new"
# fname += "_no_bsl.csv" if not bsl_practice else "_bsl.csv"
fname = 'rsa_source_tr_all.csv' if data_type.endswith("new") else 'rsa_source_tr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# save table correlations
rows = list()
for i, network in enumerate(networks):
    # get table
    for j, subject in enumerate(subjects):
        for t, time in enumerate(times):
            rows.append({
                "network": network_names[i],
                "subject": subject,
                "time": t,
                "value": corr_rp[network][j, t]
            })
df = pd.DataFrame(rows)
fname = 'rsa_source_tr_all_corr.csv' if data_type.endswith("new") else 'rsa_source_tr_corr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# save table correlations no practice
rows = list()
for i, network in enumerate(networks):
    # get table
    for j, subject in enumerate(subjects):
        for t, time in enumerate(times):
            rows.append({
                "network": network_names[i],
                "subject": subject,
                "time": t,
                "value": corr_rp_no_prac[network][j, t]
            })
df = pd.DataFrame(rows)
fname = 'rsa_source_tr_all_corr_no_corr.csv' if data_type.endswith("new") else 'rsa_source_tr_corr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

seg_df = pd.read_csv("/Users/coum/MEGAsync/figures/TM/em_segments_rs_tr_source_no_prac.csv")
seg_df = seg_df[seg_df['metric'] == 'RS CORR']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(82, dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_rs_tr_source_no_prac.csv")
sig_df = sig_df[sig_df['metric'] == 'RS CORR']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i+10] == 'ns':
            del sig_dict[net]

# plot it
times = np.linspace(-0.2, 0.6, 82)
cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]
fig, axes = plt.subplots(5, 2, figsize=(7, 9), sharey=True, sharex=True, layout="tight")
for i, ax in enumerate(axes.flatten()):
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    all_rhos = corr_rp[networks[i]]
    sem = np.std(all_rhos, axis=0) / np.sqrt(all_rhos.shape[0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='black', alpha=1)
    network = networks[i]
    # Main plot
    ax.plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
        # Plot significant regions separately
    sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(all_rhos.shape[1], dtype=bool)
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    ax.set_title(network_names[i], fontsize=13, fontstyle='italic')
    
    if ax in axes[:, 0]:
        ax.set_ylabel("Spearman's rho", fontsize=11)
    
    # Only set xlabel for axes in the bottom row
    if i >= (axes.shape[0] - 1) * axes.shape[1]:
        ax.set_xlabel("Time (s)", fontsize=11)
    
plt.savefig("/Users/coum/MEGAsync/figures/RSA/source/rsa_source_no_prac_corr.pdf", transparent=True)
plt.close(fig)


"""""
 - - ---- --- -- - -- --- - -- - - -- -- - - - - - - --- TEMPORAL GENERALIZATION -  - - - -- - - - - -  - - - - - - - - -- - - - - ---- --- - - - - - 
"""""
    
# --- Temporal generalization sensors --- blocks ---
data_type = 'scores_lobotomized'
subjects = SUBJS15
times = np.linspace(-4, 4, 813)
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
conts_blocks = pats_blocks - rands_blocks

# Plot conts_blocks with imshow
fig, ax = plt.subplots(layout='constrained')
im = ax.imshow(
    # conts_blocks.mean((0, 1)),
    pats_blocks.mean((0, 1)),
    interpolation="lanczos",
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    extent=[times[0], times[-1], times[0], times[-1]],
    # vmin=-0.05,
    # vmax=0.05
    vmin=0.2,
    vmax=0.3
)
ax.set_title(subject)
ax.set_xlabel("Train time (s)")
ax.set_ylabel("Test time (s)")
fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02)
fig.suptitle("Temporal Generalization Contrast (conts_blocks)")
plt.show()

# export time resolved diagonals
cont_tr = []
pat_tr, rand_tr = [], []
idx = np.where((times >= -1.5) & (times <= 3))[0]
for s in range(len(subjects)):
    scont, spat, srand = [], [], []
    for b in range(23):
        scont.append(np.diag(conts_blocks[s, b, idx]))
        spat.append(np.diag(pats_blocks[s, b, idx]))
        srand.append(np.diag(rands_blocks[s, b, idx]))
    cont_tr.append(np.array(scont))
    pat_tr.append(np.array(spat))
    rand_tr.append(np.array(srand))
    # cont_tr.append(np.array([np.diag(conts_blocks[s, b, idx]) for b in range(23)]))
    # pat_tr.append(np.array([np.diag(pats_blocks[s, b, idx]) for b in range(23)]))
    # rand_tr.append(np.array([np.diag(rands_blocks[s, b, idx]) for b in range(23)]))
cont_tr = np.array(cont_tr)
pat_tr, rand_tr = np.array(pat_tr), np.array(rand_tr)

fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
ax.axvspan(0, 0.2, color='grey', alpha=0.1)
ax.plot(times[idx], pat_tr.mean((0, 1)), label='Pattern')
ax.plot(times[idx], rand_tr.mean((0, 1)), label='Random')
ax.axhline(0.25, color='grey', linestyle='-', alpha=0.5)
# ax.plot(times[idx], cont_tr.mean((0, 1)), label='Contrast')
# ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
ax.legend()

# save table for mean box
for data, data_fname in zip([cont_tr, pat_tr, rand_tr], ['contrast', 'pattern', 'random']):
    rows = list()
    for i, subject in enumerate(subjects):
        for block in range(data.shape[1]):
            for t in range(data.shape[2]):
                rows.append({
                    "subject": subject,
                    "block": block + 1,
                    "time": t,
                    "value": data[i, block, t]
                })
    df = pd.DataFrame(rows)
    df_fname = f"timeg_sensors-tr_{data_fname}.csv"
    df.to_csv(FIGURES_DIR / "TM" / "data" / df_fname, index=False, sep=",")

# time resolved correlations with learning index
cont_corr_tr = []
for s in range(len(subjects)):
    cont_corr_tr.append(np.array([spear(learn_index_blocks.iloc[s], cont_tr[s, :, t])[0] for t in range(cont_tr.shape[-1])]))
cont_corr_tr = np.array(cont_corr_tr)
rows = list()
for i, subject in enumerate(subjects):
    for t in range(cont_corr_tr.shape[-1]):
        rows.append({
            "subject": subject,
            "time": t,
            "value": cont_corr_tr[i, t]
            })
df = pd.DataFrame(rows)
df_fname = "timeg_sensors-tr_cont-corr.csv"
df.to_csv(FIGURES_DIR / "TM" / "data" / df_fname, index=False, sep=",")

# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_blocks = []
diag_blocks = []
for sub in range(len(subjects)):
    tg = []
    dg = []
    for block in range(23):
        data = conts_blocks[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
        dg.append(np.diag(conts_blocks[sub, block])[idx_timeg].mean())
    box_blocks.append(np.array(tg))
    diag_blocks.append(np.array(dg))
box_blocks = np.array(box_blocks)
diag_blocks = np.array(diag_blocks)

# plot
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
blocks = np.arange(1, 24)
ax.axvspan(1, 3, color='orange', alpha=0.1,  )
# Highlight each group of 5 blocks after practice
for start in range(4, 24, 5):
    ax.axvspan(start, start + 5, color='green', alpha=0.1)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
sem = np.std(box_blocks, axis=0) / np.sqrt(box_blocks.shape[0])
mean = box_blocks.mean(0)
ax.plot(blocks, mean, color=c1)
ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
# Smooth the mean curve for visualization
smoothed = gaussian_filter1d(box_blocks.mean(0), sigma=1.5)
ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
ax.set_xticks(np.arange(1, 24, 4))
ax.set_xlabel('Block')
# ax.grid(True, linestyle='-', alpha=0.3)
ax.legend()
ax.set_title('PA sensors - blocks - mean box')
# fig.savefig(FIGURES_DIR / "time_gen" / "sensors" / "timeg_blocks_sensors.pdf", transparent=True)
# plt.close(fig)

# save table for mean box
rows = list()
for i, subject in enumerate(subjects):
    for block in range(box_blocks.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": box_blocks[i, block]
        })
df = pd.DataFrame(rows)
# df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_sensors.csv", index=False, sep=",")

# plot
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
blocks = np.arange(1, 24)
ax.axvspan(1, 3, color='orange', alpha=0.1,  )
# Highlight each group of 5 blocks after practice
for start in range(4, 24, 5):
    ax.axvspan(start, start + 5, color='green', alpha=0.1)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
sem = np.std(diag_blocks, axis=0) / np.sqrt(diag_blocks.shape[0])
mean = diag_blocks.mean(0)
ax.plot(blocks, mean, color=c1)
ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
# Smooth the mean curve for visualization
smoothed = gaussian_filter1d(diag_blocks.mean(0), sigma=1.5)
ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
ax.set_xticks(np.arange(1, 24, 4))
ax.set_xlabel('Block')
# ax.grid(True, linestyle='-', alpha=0.3)
ax.legend()
ax.set_title('PA sensors - blocks - mean diagonal')
# fig.savefig(FIGURES_DIR / "time_gen" / "sensors" / "timeg_blocks_sensors.pdf", transparent=True)
# plt.close(fig)

# save table for diagonals
rows = list()
for i, subject in enumerate(subjects):
    for block in range(diag_blocks.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": diag_blocks[i, block]
        })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_sensors-diag.csv", index=False, sep=",")

# Temporal generalization source --- blocks ---
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
timesg = np.linspace(-1.5, 1.5, 307)
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
cont_blocks = {}
pat_blocks = {}
rand_blocks = {}
contrast_net = dict()
pattern_net, random_net = dict(), dict()
diag_net = dict()
# data_type  = "scores_blocks_vect_0200_new"
# data_type  = "scores_lobo_vector_new"
data_type = 'scores_lobotomized'
for network in tqdm(networks):
    pats_blocks, rands_blocks = [], []
    if not network in pat_blocks:
        cont_blocks[network] = []
        pat_blocks[network] = []
        rand_blocks[network] = []
        diag_net[network] = []
    for subject in subjects:
        res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / subject
        pattern, random = [], []
        for block in range(1, 24):
            if network in networks[:-3]:
                pfname = res_path / f'pat-{block}.npy' if block not in [1, 2, 3] else res_path / f'pat-0-{block}.npy'
                rfname = res_path / f'rand-{block}.npy' if block not in [1, 2, 3] else res_path / f'rand-0-{block}.npy'
            else:
                pfname = res_path / f'pat-4-{block}.npy' if block not in [1, 2, 3] else res_path / f'pat-0-{block}.npy'
                rfname = res_path / f'rand-4-{block}.npy' if block not in [1, 2, 3] else res_path / f'rand-0-{block}.npy'
            pattern.append(np.load(pfname))
            random.append(np.load(rfname))
        if subject == 'sub05':
            pat_bsl = np.load(res_path / "pat-4.npy") if network in networks[:-3] else np.load(res_path / "pat-4-4.npy")
            rand_bsl = np.load(res_path / "rand-4.npy") if network in networks[:-3] else np.load(res_path / "rand-4-4.npy")
            for i in range(3):
                pattern[i] = pat_bsl.copy()
                random[i] = rand_bsl.copy()
        pats_blocks.append(np.array(pattern))
        rands_blocks.append(np.array(random))
    pats_blocks, rands_blocks = np.array(pats_blocks), np.array(rands_blocks)
    contrast = pats_blocks - rands_blocks
    contrast_net[network] = contrast
    pattern_net[network] = pats_blocks
    random_net[network] = rands_blocks
    # mean box
    box_blocks_c = []
    box_blocks_p = []
    box_blocks_r = []
    diag_blocks = []
    for sub in range(len(subjects)):
        tg_p, tg_r = [], []
        tg_c = []
        dg = []
        for block in range(23):
            # contrast
            data_c = contrast[sub, block, idx_timeg, :][:, idx_timeg]
            tg_c.append(data_c.mean())
            # pattern
            data_p = pats_blocks[sub, block, idx_timeg, :][:, idx_timeg]
            tg_p.append(data_p.mean())
            # random
            data_r = rands_blocks[sub, block, idx_timeg, :][:, idx_timeg]
            tg_r.append(data_r.mean())
            # diagonal
            dg.append(np.diag(contrast[sub, block])[idx_timeg].mean())
        box_blocks_c.append(np.array(tg_c))
        box_blocks_p.append(np.array(tg_p))
        box_blocks_r.append(np.array(tg_r))
        diag_blocks.append(np.array(dg))
    cont_blocks[network] = np.array(box_blocks_c)
    pat_blocks[network] = np.array(box_blocks_p)
    rand_blocks[network] = np.array(box_blocks_r)
    diag_net[network] = np.array(diag_blocks)

# export time resolved diagonals
idxt = np.where((timesg >= -1) & (timesg <= 1.5))[0]
cont_tr = {}
pat_tr, rand_tr = {}, {}
for network in networks:
    cont_tr[network] = []
    pat_tr[network] = []
    rand_tr[network] = []
    for s in range(len(subjects)):
        cont_tr[network].append(np.diag(contrast_net[network].mean(1)[s]))
        pat_tr[network].append(np.diag(pattern_net[network].mean(1)[s]))
        rand_tr[network].append(np.diag(random_net[network].mean(1)[s]))
    cont_tr[network] = np.array(cont_tr[network])
    pat_tr[network] = np.array(pat_tr[network])
    rand_tr[network] = np.array(rand_tr[network])

# get significant time points from GAMM csv --- contrast
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_cont_source.csv")
seg_df = seg_df[seg_df['metric'] == 'PA']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(len(idxt), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_cont_source.csv")
sig_df = sig_df[sig_df['metric'] == 'PA']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
            del sig_dict[net]

fig, ax = plt.subplots(5, 2, sharey=True, sharex=False, layout='tight')
for ax, network, name in zip(ax.flatten(), networks, network_names):
    data = cont_tr[network][:, idxt]
    mytime = timesg[idxt]
    # mytime = timesg.copy()
    ax.plot(mytime, data.mean(0), color="grey", alpha=0.5)
    ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    # ax.axhline(0.25, color='grey', linestyle='-', alpha=0.5)
    ax.set_title(name, fontstyle='italic')
    sig = sig_dict[name] if name in sig_dict else np.zeros(data.shape[1], dtype=bool)
    for start, end in contiguous_regions(sig):
        ax.plot(mytime[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=c1)

seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_pat_rand_source.csv")

# save time resolved diagonals
idxt = np.where((timesg >= -1) & (timesg <= 1.5))[0]
for data, data_fname in zip([cont_tr, pat_tr, rand_tr], ['contrast', 'pattern', 'random']):
    rows = list()
    for i, network in enumerate(networks):
        # get table
        for j, subject in enumerate(subjects):
            for t, idx in enumerate(idxt):
                rows.append({
                    "network": network_names[i],
                    "subject": subject,
                    "time": t,
                    "value": data[network][j, idx] - 0.25 if data_fname in ['pattern', 'random'] else data[network][j, idx]
                })
    df = pd.DataFrame(rows)
    fname = f'pa_source_tr_{data_fname}_all.csv' if data_type.endswith("new") else f'pa_source_tr_{data_fname}.csv'
    df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# export time resolved for preactivation
tpoi = int(np.where(np.isclose(timesg, 0.37, atol=0.005))[0])
cont_tr = {}
pat_tr, rand_tr = {}, {}
for network in networks:
    cont_tr[network] = []
    pat_tr[network] = []
    rand_tr[network] = []
    for s in range(len(subjects)):
        cont_tr[network].append(contrast_net[network].mean(1)[s, tpoi])
        pat_tr[network].append(pattern_net[network].mean(1)[s, tpoi])
        rand_tr[network].append(random_net[network].mean(1)[s, tpoi])
    cont_tr[network] = np.array(cont_tr[network])
    pat_tr[network] = np.array(pat_tr[network])
    rand_tr[network] = np.array(rand_tr[network])
# save time resolved diagonals
idxt = np.where((timesg >= -1) & (timesg <= 1.5))[0]
for data, data_fname in zip([cont_tr, pat_tr, rand_tr], ['contrast', 'pattern', 'random']):
    rows = list()
    for i, network in enumerate(networks):
        # get table
        for j, subject in enumerate(subjects):
            for t, idx in enumerate(idxt):
                rows.append({
                    "network": network_names[i],
                    "subject": subject,
                    "time": t,
                    "value": data[network][j, idx] - 0.25 if data_fname in ['pattern', 'random'] else data[network][j, idx]
                })
    df = pd.DataFrame(rows)
    fname = f'preact_source_tr_{data_fname}_all.csv' if data_type.endswith("new") else f'preact_source_tr_{data_fname}.csv'
    df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")


# export time resolved diagonals
idxt = np.where((timesg >= -1) & (timesg <= 1.5))[0]
cont_tr = {}
pat_tr, rand_tr = {}, {}
for network in tqdm(networks):
    cont_tr[network] = []
    pat_tr[network] = []
    rand_tr[network] = []
    for s in range(len(subjects)):
        cont_b = []
        pat_b = []
        rand_b = []
        for b in range(23):
            cont_b.append(np.diag(contrast_net[network][s, b]))
            pat_b.append(np.diag(pattern_net[network][s, b]))
            rand_b.append(np.diag(random_net[network][s, b]))
        cont_tr[network].append(np.array(cont_b))
        pat_tr[network].append(np.array(pat_b))
        rand_tr[network].append(np.array(rand_b))
    cont_tr[network] = np.array(cont_tr[network])[:, :, idxt]
    pat_tr[network] = np.array(pat_tr[network])[:, :, idxt]
    rand_tr[network] = np.array(rand_tr[network])[:, :, idxt]

corr_network = {}
for network in networks:
    corr_network[network] = []
    for s in range(len(subjects)):
        corr_network[network].append(np.array([spear(learn_index_blocks.iloc[s], cont_tr[network][s, :, t])[0] for t in range(len(idxt))]))
    corr_network[network] = np.array(corr_network[network])
    corr_network[network], _, _ = fisher_z_and_ttest(corr_network[network])
    
# save time resolved diagonal correlations
rows = list()
for i, network in enumerate(networks):
    # get table
    for j, subject in enumerate(subjects):
        for t, _ in enumerate(idxt):
            rows.append({
                "network": network_names[i],
                "subject": subject,
                "time": t,
                "value": corr_network[network][j, t]
            })
df = pd.DataFrame(rows)
df_fname = 'timeg_source-tr_cont-corr_all.csv' if data_type.endswith("new") else 'timeg_source-tr_cont-corr.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / df_fname, index=False, sep=",")
    
ensured(FIGURES_DIR / "temp" / "timeg_pval")
# Contrast
cmap1 = "RdBu_r"
c1 = "#708090"
c1 = "#00BFA6"

fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
idx = np.where((timesg >= -0.5) & (timesg < 0))[0]
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    data = contrast_net[network][:, 3:]
    print(f"Plotting {network}...")
    im = ax.imshow(
        data.mean((0, 1)),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    pval_path = RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / 'pval'
    if not op.exists(pval_path / "all_contrast-pval.npy"):
        pval = gat_stats(data.mean(1), -1)
        np.save(pval_path / "all_contrast-pval.npy", pval)
    else:
        pval = np.load(pval_path / "all_contrast-pval.npy")
    sig = pval < 0.05
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='-', linewidths=1, alpha=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast time generalization")
# fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_time_resolved_source.pdf", transparent=True)
# plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
idx = np.where((timesg >= -0.5) & (timesg < 0))[0]
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    data = contrast_net[network][:, 3:]
    ax.plot(timesg, np.diag(data.mean((0, 1))), color=c1, linewidth=2, label='Diagonal')
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    diags = [np.diag(d) for d in data.mean(1)]
    pval = decod_stats(diags, -1)
    sig = pval < 0.05
    ax.fill_between(timesg, np.diag(data.mean((0, 1))), 0, where=sig, color='red', alpha=0.3)
    mdiags = [np.mean(d[idx]) for d in diags]
    sig_unc = ttest_1samp(mdiags, 0)[1] < 0.05
    if sig_unc:
        ax.fill_between(timesg[idx], -0.02, -0.018, facecolor='red', edgecolor=None, alpha=1, zorder=5)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast time generalization diag")
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_diag.pdf", transparent=True)
# plt.close(fig)

# plot contrast - box mean
blocks = np.arange(1, 24)
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    ax.set_xticks(np.arange(1, 24, 4))
    ax.axvspan(1, 3, color='orange', alpha=0.1)
    # Highlight each group of 5 blocks after practice
    for start in range(4, 24, 5):
        ax.axvspan(start, start + 5, color='green', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sem = cont_blocks[network].std(axis=0) / np.sqrt(cont_blocks[network].shape[0])
    mean = cont_blocks[network].mean(axis=0)
    ax.plot(blocks, mean, color=c1)
    ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
    # Smooth the mean curve for visualization
    smoothed = gaussian_filter1d(cont_blocks[network].mean(0), sigma=1.5)
    ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
    # ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_title(network_names[i], fontstyle='italic')
    if i == 0:
        ax.legend()
    # Only set xlabel for axes in the bottom row
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Block')
fig.suptitle('PA source - box mean blocks', fontsize=14)
# fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_mean_blocks-box.pdf", transparent=True)
# plt.close(fig)
# save table
rows = list()
for i, network in enumerate(networks):
    diff = cont_blocks[network]
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
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_source.csv", index=False, sep=",")

# plot contrast - diag mean
blocks = np.arange(1, 24)
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    ax.set_xticks(np.arange(1, 24, 4))
    ax.axvspan(1, 3, color='orange', alpha=0.1)
    # Highlight each group of 5 blocks after practice
    for start in range(4, 24, 5):
        ax.axvspan(start, start + 5, color='green', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sem = diag_net[network].std(axis=0) / np.sqrt(diag_net[network].shape[0])
    mean = diag_net[network].mean(axis=0)
    ax.plot(blocks, mean, color=c1)
    ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
    # Smooth the mean curve for visualization
    smoothed = gaussian_filter1d(diag_net[network].mean(0), sigma=1.5)
    ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
    # ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_title(network_names[i], fontstyle='italic')
    if i == 0:
        ax.legend()
    # Only set xlabel for axes in the bottom row
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Block')
fig.suptitle('PA source - diag mean blocks', fontsize=14)
# fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_mean_blocks-diag.pdf", transparent=True)
# plt.close(fig)
# save table
rows = list()
for i, network in enumerate(networks):
    diff = diag_net[network]
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
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_source-diag.csv", index=False, sep=",")

# plot correlation (block level)
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
idx = np.where((timesg >= -0.5) & (timesg < 0))[0]
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    rho = np.load(RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / 'corr' / 'rhos_learn.npy')
    pval = np.load(RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / 'corr' / 'pval_learn-pval.npy')
    print(f"Plotting {network}...")
    im = ax.imshow(
        rho.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=timesg[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.05,
        vmax=0.05)
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    sig = pval < 0.05
    ax.contour(xx, yy, sig, colors=c1, levels=[0],
                        linestyles='-', linewidths=1, alpha=1)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast time generalization")
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_corr.pdf", transparent=True)

fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
idx = np.where((timesg >= -0.75) & (timesg < 0))[0]
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    rho = np.load(RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / 'corr' / 'rhos_learn.npy')
    ax.plot(timesg, np.diag(rho.mean(0)), color=c1, linewidth=2, label='Diagonal')
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    diags = [np.diag(d) for d in rho]
    pval = decod_stats(diags, -1)
    sig = pval < 0.05
    ax.fill_between(timesg, np.diag(rho.mean(0)), 0, where=sig, color='red', alpha=0.3)
    mdiags = [np.mean(d[idx]) for d in diags]
    sig_unc = ttest_1samp(mdiags, 0)[1] < 0.05
    if sig_unc:
        print(f"{network} sig *******************")
        ax.fill_between(timesg[idx], -0.1, -0.13, facecolor='red', edgecolor=None, alpha=1, zorder=5)
    else:
        print(f"{network} ns")
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
    mrho = []
    for r in rho:
        mrho.append(r[idx][:, idx].mean())
    mrho = np.array(mrho)
    sig_unc = ttest_1samp(mrho, 0)[1] < 0.05
    if sig_unc:
        print(f"{network} box sig *******************")
    else:
        print(f"{network} box ns")
fig.suptitle("Contrast diag corr")
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_corr_diag.pdf", transparent=True)
# plt.close(fig)
