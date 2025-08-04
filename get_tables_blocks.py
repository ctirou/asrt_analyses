import os.path as op
import numpy as np
from base import decod_stats, get_sequence, get_all_high_low, ensured, gat_stats
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter1d

subjects = SUBJS15
times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.3) & (times <= 0.5))[0]
c1, c2 = "#5BBCD6", "#00A08A"

# --- RSA sensors --- blocks ---
data_type = "rdm_blocks"
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

# plot
idx = np.where((times >= 0.3) & (times <= 0.55))[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), layout='tight')
blocks = np.arange(1, 24)
ax1.axvspan(1, 3, color='orange', alpha=0.1)
# Highlight each group of 5 blocks after practice
for start in range(4, 24, 5):
    end = min(start + 4, 23)
    ax1.axvspan(start, end, color='green', alpha=0.1)
ax1.axhline(0, color='grey', linestyle='-', alpha=0.5)
sem = np.nanstd(diff_rp[:, :, idx], axis=(0, -1)) / np.sqrt(diff_rp.shape[0])
mean = np.nanmean(diff_rp[:, :, idx], axis=((0, -1)))
# ax.plot(blocks, np.nanmean(diff_rp[:, :, idx], axis=(0, -1)))
ax1.plot(blocks, mean, color=c1)
ax1.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
# Smooth the mean curve for visualization
smoothed = gaussian_filter1d(np.nanmean(diff_rp[:, :, idx], axis=(0, -1)), sigma=1.5)
ax1.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
ax1.set_xticks(np.arange(1, 24, 4))
# ax.grid(True, linestyle='-', alpha=0.3)
# ax.text(2, 0.4 , "Prac.", color='orange', fontsize=14, ha='center', va='center', fontstyle='italic')
ax1.set_xlabel('Block')
ax1.legend()
title = 'RS sensors (correct trials only)' if not data_type.endswith("new") else 'RS sensors (all trials)'
title += ' - no practice baseline' if not bsl_practice else ' - practice baseline'
# ax1.set_title(title, fontstyle='italic')
# fname = "rsa_blocks_sensors" if not data_type.endswith("new") else "rsa_blocks_sensors_new"
# fname += "_no_bsl.pdf" if not bsl_practice else "_bsl.pdf"
# fig.savefig(FIGURES_DIR / "RSA" / "sensors" / fname, transparent=True)
# plt.close(fig)

# plot time resolved RSA sensors
xmin, xmax = 0.3, 0.55
win = np.where((times >= xmin) & (times <= xmax))[0]
# fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
ax2.plot(times, np.nanmean(diff_rp, axis=(0, 1)), color=c1)
ax2.axhline(0, color='grey', linestyle='-', alpha=0.5)
sig = decod_stats(np.nanmean(diff_rp, axis=(1)), -1) < 0.05
ax2.fill_between(times, np.nanmean(diff_rp, axis=(0, 1)), 0, where=sig, color='red', alpha=0.3, label='Significant cluster')
mdiff = np.nanmean(diff_rp[:, :, win], axis=(1, -1))
mdiff_sig = ttest_1samp(mdiff, 0)[1] < 0.05
if mdiff_sig:
    ax2.axvspan(times[win][0], times[win][-1], facecolor='orange', edgecolor=None, alpha=0.2, zorder=5, label=f'Significant time window - [{xmin}-{xmax}] s')
    # ax.text((xmin+xmax)/2, -0.5, '*', fontsize=16, ha='center', va='center', color='red', weight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Similarity index (random - pattern)')
ax2.legend()
fig.suptitle(title, fontstyle='italic')
fname = "rsa_sensors" if not data_type.endswith("new") else "rsa_sensors_new"
fname += "_no_bsl.pdf" if not bsl_practice else "_bsl.pdf"
fig.savefig(FIGURES_DIR / "RSA" / fname, transparent=True)
plt.close(fig)

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
fname = 'rsa_sensors.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# RSA source --- blocks ---
data_type = "rdm_blocks_vect_new"
bsl_practice = False
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
diff_rp = {}
for network in tqdm(networks):
    if not network in diff_rp:
        diff_rp[network] =  []
    for subject in subjects:
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
        diff_rp[network].append(rand - pat)
    diff_rp[network] = np.array(diff_rp[network])

# plot
idx = np.where((times >= 0.3) & (times <= 0.55))[0]
# idx = np.where((times >= 0.3) & (times <= 0.6))[0]
blocks = np.arange(1, 24)
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    ax.axvspan(1, 3, color='orange', alpha=0.1)
    # Highlight each group of 5 blocks after practice
    for start in range(4, 24, 5):
        end = min(start + 4, 23)
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sem = np.nanstd(diff_rp[network][:, :, idx], axis=(0, -1)) / np.sqrt(diff_rp[network].shape[0])
    mean = np.nanmean(diff_rp[network][:, :, idx], axis=((0, -1)))
    ax.plot(blocks, mean, color=c1)
    ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
    # Smooth the mean curve for visualization
    smoothed = gaussian_filter1d(np.nanmean(diff_rp[network][:, :, idx], axis=(0, -1)), sigma=1.5)
    ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
    ax.set_xticks(np.arange(1, 24, 4))
    # ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_title(network_names[i], fontstyle='italic')
    if i == 0:
        ax.legend()
    # Only set xlabel for axes in the bottom row
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Block')
title = 'RS source - blocks (correct trials only)' if not data_type.endswith("new") else 'RS source - blocks (all trials)'
title += ' - no practice baseline' if not bsl_practice else ' - practice baseline'
fig.suptitle(title, fontsize=14)
fname = "rsa_blocks_source" if not data_type.endswith("new") else "rsa_blocks_source_new"
fname += "_no_bsl.pdf" if not bsl_practice else "_bsl.pdf"
fig.savefig(FIGURES_DIR / "RSA" / fname, transparent=True)
plt.close(fig)

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
# fname = "rsa_blocks_source" if not data_type.endswith("new") else "rsa_blocks_source_new"
# fname += "_no_bsl.csv" if not bsl_practice else "_bsl.csv"
fname = 'rsa_source.csv'
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

"""""
 - - ---- --- -- - -- --- - -- - - -- -- --- --- - - - - - TEMPORAL GENERALIZATION
"""""
    
# --- Temporal generalization sensors --- blocks ---
data_type = 'scores_lobotomized'
subjects = SUBJS15
times = np.linspace(-4, 4, 813)
filt = np.where((times >= -1.5) & (times <= 3))[0]
times_filt = times[filt]
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

# mean box
idx_timeg = np.where((times >= -0.75) & (times < 0))[0]
box_blocks = []
diag_blocks = []
conts_blocks = pats_blocks - rands_blocks
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
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_sensors.csv", index=False, sep=",")

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
idx_timeg = np.where((timesg >= -0.75) & (timesg < 0))[0]
cont_blocks = {}
pat_blocks = {}
rand_blocks = {}
contrast_net = dict()
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
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_mean_blocks-diag.pdf", transparent=True)
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
