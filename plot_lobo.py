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
all_pats, all_rands = [], []
all_pats_blocks, all_rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / "rdm_lobo_new" / subject
    # read behav        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pattern_blocks, random_blocks = [], []
    for block in range(1, 24):
        pattern_blocks.append(np.load(res_path / f"pat-{block}.npy"))
        random_blocks.append(np.load(res_path / f"rand-{block}.npy"))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-1.npy")
        rand_bsl = np.load(res_path / "rand-1.npy")
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
pat = all_pats - bsl_pat[:, np.newaxis, :]
rand = all_rands - bsl_rand[:, np.newaxis, :]
diff_rp = rand - pat

# plot
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
blocks = np.arange(1, 24)
idx = np.where((times >= 0.3) & (times <= 0.55))[0]
ax.axvspan(1, 3, color='orange', alpha=0.1)
# Highlight each group of 5 blocks after practice
for start in range(4, 24, 5):
    end = min(start + 4, 23)
    ax.axvspan(start, end, color='green', alpha=0.1)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
sem = np.nanstd(diff_rp[:, :, idx], axis=(0, -1)) / np.sqrt(diff_rp.shape[0])
mean = np.nanmean(diff_rp[:, :, idx], axis=((0, -1)))
# ax.plot(blocks, np.nanmean(diff_rp[:, :, idx], axis=(0, -1)))
ax.plot(blocks, mean, color=c1)
ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
# Smooth the mean curve for visualization
smoothed = gaussian_filter1d(np.nanmean(diff_rp[:, :, idx], axis=(0, -1)), sigma=1.5)
ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
ax.set_xticks(np.arange(1, 24, 4))
# ax.grid(True, linestyle='-', alpha=0.3)
# ax.text(2, 0.4 , "Prac.", color='orange', fontsize=14, ha='center', va='center', fontstyle='italic')
ax.set_xlabel('Block')
ax.legend()
ax.set_title('RS sensors - blocks', fontstyle='italic')
fig.savefig(FIGURES_DIR / "RSA" / "sensors" / "rsa_blocks_sensors.pdf", transparent=True)
# plt.close(fig)

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
df.to_csv(FIGURES_DIR / "RSA" / "sensors" / "rsa_blocks_sensors.csv", index=False, sep=",")
df.to_csv(FIGURES_DIR / "TM" / "data" / "rsa_blocks_sensors.csv", index=False, sep=",")

# plot time resolved RSA sensors
xmin, xmax = 0.3, 0.55
win = np.where((times >= xmin) & (times <= xmax))[0]
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
ax.axvspan(0, 0.2, color='grey', alpha=0.1)
ax.plot(times, np.nanmean(diff_rp, axis=(0, 1)), color=c1)
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
sig = decod_stats(np.nanmean(diff_rp, axis=(1)), -1) < 0.05
ax.fill_between(times, np.nanmean(diff_rp, axis=(0, 1)), 0, where=sig, color='red', alpha=0.3)
mdiff = np.nanmean(diff_rp[:, :, win], axis=(1, -1))
mdiff_sig = ttest_1samp(mdiff, 0)[1] < 0.05
# if mdiff_sig:
#     ax.axvspan(times[win][0], times[win][-1], facecolor='red', edgecolor=None, alpha=0.2, zorder=5)
#     ax.text((xmin+xmax)/2, -0.5, '*', fontsize=16, ha='center', va='center', color='red', weight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Similarity index (random - pattern)')
ax.set_title('RS sensors - time resolved', fontstyle='italic')
fig.savefig(FIGURES_DIR / "RSA" / "sensors" / "rsa_time_resolved_sensors.pdf", transparent=True)
plt.close(fig)

# RSA source --- blocks ---
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
diff_rp = {}
for network in tqdm(networks):
    if not network in diff_rp:
        diff_rp[network] =  []
    for subject in subjects:
        # res_path = RESULTS_DIR / 'RSA' / 'source' / network / "rdm_blocks_vect_0200" / subject
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / "rdm_blocks_vect_new" / subject
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
        pat = np.nanmean(high, 0) - bsl_pat[np.newaxis, :]
        rand = np.nanmean(low, 0) - bsl_rand[np.newaxis, :]
        diff_rp[network].append(rand - pat)
    diff_rp[network] = np.array(diff_rp[network])

# plot
idx = np.where((times >= 0.3) & (times <= 0.5))[0]
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
fig.suptitle('RS source - blocks', fontsize=14)
fig.savefig(FIGURES_DIR / "RSA" / "source" / "rsa_blocks_source.pdf", transparent=True)
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
df.to_csv(FIGURES_DIR / "RSA" / "source" / "rsa_blocks_source.csv", index=False, sep=",")
df.to_csv(FIGURES_DIR / "TM" / "data" / "rsa_blocks_source.csv", index=False, sep=",")

# plot time resolved RSA source
xmin, xmax = 0.3, 0.55
win = np.where((times >= xmin) & (times <= xmax))[0]
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    ax.plot(times, np.nanmean(diff_rp[network], axis=(0, 1)), color=c1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sig = decod_stats(np.nanmean(diff_rp[network], axis=(1)), -1) < 0.05
    ax.fill_between(times, np.nanmean(diff_rp[network], axis=(0, 1)), 0, where=sig, color='red', alpha=0.3) 
    ax.set_title(network_names[i], fontstyle='italic')
    smoothed = gaussian_filter1d(np.nanmean(diff_rp[network], axis=(0, 1)), sigma=1.5)
    mdiff = np.nanmean(diff_rp[network][:, :, win], axis=(1, -1))
    mdiff_sig = ttest_1samp(mdiff, 0)[1] < 0.05
    if mdiff_sig:
        ax.axvspan(times[win][0], times[win][-1], facecolor='red', edgecolor=None, alpha=0.2, zorder=5)
        ax.text((xmin+xmax)/2, -0.5, '*', fontsize=16, ha='center', va='center', color='red', weight='bold')
fig.suptitle('RS source - time resolved', fontsize=14)
fig.savefig(FIGURES_DIR / "RSA" / "source" / "rsa_time_resolved_source.pdf", transparent=True)
# plt.close(fig)
    
# --- Temporal generalization sensors --- blocks ---
subjects = SUBJS15
times = np.linspace(-4, 4, 813)
filt = np.where((times >= -1.5) & (times <= 3))[0]
times_filt = times[filt]
pats_blocks, rands_blocks = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'TIMEG' / 'sensors' / 'scores_lobo_new' / subject
    pattern, random = [], []
    for block in range(1, 24):
        pattern.append(np.load(res_path / f"pat-{block}.npy"))
        random.append(np.load(res_path / f"rand-{block}.npy"))
    if subject == 'sub05':
        pat_bsl = np.load(res_path / "pat-1.npy")
        rand_bsl = np.load(res_path / "rand-1.npy")
        for i in range(3):
            pattern[i] = pat_bsl.copy()
            random[i] = rand_bsl.copy()
    pats_blocks.append(np.array(pattern))
    rands_blocks.append(np.array(random))
pats_blocks = np.array(pats_blocks)
rands_blocks = np.array(rands_blocks)
# mean box
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
box_blocks = []
conts_blocks = pats_blocks - rands_blocks
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = conts_blocks[sub, block, idx_timeg, :][:, idx_timeg]
        tg.append(data.mean())
    box_blocks.append(np.array(tg))
box_blocks = np.array(box_blocks)

# plot
fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
blocks = np.arange(1, 24)
ax.axvspan(1, 3, color='orange', alpha=0.1,  )
# Highlight each group of 5 blocks after practice
for start in range(4, 24, 5):
    end = min(start + 4, 23)
    ax.axvspan(start, end, color='green', alpha=0.1)
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
ax.set_title('PA sensors - blocks')
fig.savefig(FIGURES_DIR / "time_gen" / "sensors" / "timeg_blocks_sensors.pdf", transparent=True)
plt.close(fig)

# save table
rows = list()
for i, subject in enumerate(subjects):
    for block in range(box_blocks.shape[1]):
        rows.append({
            "subject": subject,
            "block": block + 1,
            "value": box_blocks[i, block]
        })
df = pd.DataFrame(rows)
df.to_csv(FIGURES_DIR / "time_gen" / "sensors" / "timeg_blocks_sensors.csv", index=False, sep=",")
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_blocks_sensors.csv", index=False, sep=",")

# plot time resolved PA sensors
patt_ave, rand_ave = pats_blocks[:, 3:].mean(1), rands_blocks[:, 3:].mean(1)
data = patt_ave - rand_ave
ensured(FIGURES_DIR / "temp" / "timeg_pval")

from matplotlib import colors

cmap1 = plt.get_cmap('RdBu_r')
vmin, vmax = -0.05, 0.05
idx = np.where((times >= -1.5) & (times <=3))[0]
fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
norm = colors.Normalize(vmin=vmin, vmax=vmax)
images = []
images.append(ax.imshow(data[:, idx][:, :, idx].mean(0), 
# images.append(ax.imshow(data.mean(0), 
                        # norm=norm,
                        vmin = vmin,
                        vmax = vmax,
                        interpolation="lanczos",
                        origin="lower",
                        cmap=cmap1,
                        extent=times[idx][[0, -1, 0, -1]],
                        # extent=times[[0, -1, 0, -1]],
                        aspect=0.5))
ax.set_ylabel("Training time (s)")
ax.set_xticks(np.arange(-1, 3, .5))
ax.set_yticks(np.arange(-1, 3, .5))
ax.set_title("Time generalization - time resolved", fontsize=16)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
# xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
if not op.exists(FIGURES_DIR / "temp" / "timeg_pval" / "Sensors-pval3.npy"):
    pval = gat_stats(data[:, idx][:, :, idx], -1)
    # pval = gat_stats(data, -1)
    np.save(FIGURES_DIR / "temp" / "timeg_pval" / "Sensors-pval3.npy", pval)
else:
    pval = np.load(FIGURES_DIR / "temp" / "timeg_pval" / "Sensors-pval3.npy")
sig = pval < 0.001
ax.contour(xx, yy, sig, colors='black', levels=[0],
                    linestyles='-', linewidths=1, alpha=.5)
ax.set_xlabel("Testing time (s)")
cbar = fig.colorbar(images[0], ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
cbar.set_label("\nDifference in accuracy", rotation=270)
ax.set_title("Time generalization - time resolved", fontstyle='italic')
fig.savefig(FIGURES_DIR / "time_gen" / "sensors" / "timeg_time_resolved_sensors.pdf", transparent=True)
plt.close(fig)

# Temporal generalization source --- blocks ---
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
timesg = np.linspace(-1.5, 1.5, 307)
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
cont_blocks = {}
pat_blocks = {}
rand_blocks = {}
contrast_net = dict()
# data_type  = "scores_blocks_vect_0200_new"
data_type  = "scores_blocks_vector_new"
for network in tqdm(networks):
    pats_blocks, rands_blocks = [], []
    if not network in pat_blocks:
        cont_blocks[network] = []
        pat_blocks[network] = []
        rand_blocks[network] = []
    for subject in subjects:
        res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / subject
        pattern, random = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            for block in blocks:
                pattern.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
                random.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
        if subject == 'sub05':
            pat_bsl = np.load(res_path / "pat-1-1.npy")
            rand_bsl = np.load(res_path / "rand-1-1.npy")
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
    for sub in range(len(subjects)):
        tg_p, tg_r = [], []
        tg_c = []
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
        box_blocks_c.append(np.array(tg_c))
        box_blocks_p.append(np.array(tg_p))
        box_blocks_r.append(np.array(tg_r))
    cont_blocks[network] = np.array(box_blocks_c)
    pat_blocks[network] = np.array(box_blocks_p)
    rand_blocks[network] = np.array(box_blocks_r)
    
ensured(FIGURES_DIR / "temp" / "timeg_pval")
# Contrast
cmap1 = "RdBu_r"
c1 = "#708090"
c1 = "#00BFA6"

fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharex=True, sharey=True, layout='constrained')
idx = np.where((timesg >= -0.5) & (timesg < 0))[0]
for i, (ax, network, name) in enumerate(zip(axes.flatten(), networks, network_names)):
    data = contrast_net[network][:, 3:]
    # print(f"Plotting {network}...")
    # im = ax.imshow(
    #     data.mean((0, 1)),
    #     interpolation="lanczos",
    #     origin="lower",
    #     cmap=cmap1,
    #     extent=timesg[[0, -1, 0, -1]],
    #     aspect=0.5,
    #     vmin=-0.05,
    #     vmax=0.05)
    ax.plot(timesg, np.diag(data.mean((0, 1))), color=c1, linewidth=2, label='Diagonal')
    ax.set_title(f"{name}", fontsize=10, fontstyle="italic")
    # xx, yy = np.meshgrid(timesg, timesg, copy=False, indexing='xy')
    # if not op.exists(FIGURES_DIR / "temp" / "timeg_pval" / f"{network}-pval.npy"):
    #     pval = gat_stats(data.mean(1), -1)
    #     np.save(FIGURES_DIR / "temp" / "timeg_pval" / f"{network}-pval.npy", pval)
    # else:
    #     pval = np.load(FIGURES_DIR / "temp" / "timeg_pval" / f"{network}-pval.npy")
    diags = [np.diag(d) for d in data.mean(1)]
    pval = decod_stats(diags, -1)
    sig = pval < 0.05
    ax.fill_between(timesg, np.diag(data.mean((0, 1))), 0, where=sig, color='red', alpha=0.3)
    mdiags = [np.mean(d[idx]) for d in diags]
    sig_unc = ttest_1samp(mdiags, 0)[1] < 0.05
    if sig_unc:
        ax.fill_between(timesg[idx], -0.02, -0.018, facecolor='red', edgecolor=None, alpha=1, zorder=5)
    # ax.contour(xx, yy, sig, colors='black', levels=[0],
    #                     linestyles='--', linewidths=1, alpha=.5)
    ax.axvline(0, color="k", alpha=.5)
    ax.axhline(0, color="k", alpha=.5)
fig.suptitle("Contrast time generalization")
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_time_resolved_source.pdf", transparent=True)
plt.close(fig)

# plot contrast
blocks = np.arange(1, 24)
fig, axes = plt.subplots(2, 5, figsize=(15, 4), sharey=True, sharex=True, layout='tight')
for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
    ax.axvspan(1, 3, color='orange', alpha=0.1)
    # Highlight each group of 5 blocks after practice
    for start in range(4, 24, 5):
        end = min(start + 4, 23)
        ax.axvspan(start, end, color='green', alpha=0.1)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
    sem = cont_blocks[network].std(axis=0) / np.sqrt(cont_blocks[network].shape[0])
    mean = cont_blocks[network].mean(axis=0)
    ax.plot(blocks, mean, color=c1)
    ax.fill_between(blocks, mean - sem, mean + sem, color=c1, alpha=0.3)
    # Smooth the mean curve for visualization
    smoothed = gaussian_filter1d(cont_blocks[network].mean(0), sigma=1.5)
    ax.plot(blocks, smoothed, color='red', linestyle='--', label='Gaussian smoothed')
    ax.set_xticks(np.arange(1, 24, 4))
    # ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_title(network_names[i], fontstyle='italic')
    if i == 0:
        ax.legend()
    # Only set xlabel for axes in the bottom row
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('Block')
fig.suptitle('PA source - contrast blocks', fontsize=14)
fig.savefig(FIGURES_DIR / "time_gen" / "source" / "timeg_blocks_source.pdf", transparent=True)
plt.close(fig)

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
df.to_csv(FIGURES_DIR / "time_gen" / "source" / "timeg_blocks_source.csv", index=False, sep=",")
df.to_csv(FIGURES_DIR / "TM" / "data" / "timeg_blocks_source.csv", index=False, sep=",")

# # plot pattern
# fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True, layout='tight')
# for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
#     ax.axvspan(1, 3, color='orange', alpha=0.1)
#     # Highlight each group of 5 blocks after practice
#     for start in range(4, 24, 5):
#         end = min(start + 4, 23)
#         ax.axvspan(start, end, color='green', alpha=0.1)
#     ax.axhline(0.25, color='grey', linestyle='-', alpha=0.5)
#     ax.plot(blocks, pat_blocks[network].mean(0))
#     # Smooth the mean curve for visualization
#     smoothed = gaussian_filter1d(pat_blocks[network].mean(0), sigma=1.5)
#     ax.plot(blocks, smoothed, color='red', linestyle='--', label='smoothed')
#     ax.set_xticks(np.arange(1, 24, 4))
#     # ax.grid(True, linestyle='-', alpha=0.3)
#     ax.set_title(network_names[i], fontstyle='italic')
#     if i == 0:
#         ax.legend()
#     # Only set xlabel for axes in the bottom row
#     if ax.get_subplotspec().is_last_row():
#         ax.set_xlabel('Block')
# fig.suptitle('PA source - pattern blocks', fontsize=14)

# # plot random
# fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True, layout='tight')
# for i, (ax, network) in enumerate(zip(axes.flatten(), networks)):
#     ax.axvspan(1, 3, color='orange', alpha=0.1)
#     # Highlight each group of 5 blocks after practice
#     for start in range(4, 24, 5):
#         end = min(start + 4, 23)
#         ax.axvspan(start, end, color='green', alpha=0.1)
#     ax.axhline(0.25, color='grey', linestyle='-', alpha=0.5)
#     ax.plot(blocks, rand_blocks[network].mean(0))
#     # Smooth the mean curve for visualization
#     smoothed = gaussian_filter1d(rand_blocks[network].mean(0), sigma=1.5)
#     ax.plot(blocks, smoothed, color='red', linestyle='--', label='smoothed')
#     ax.set_xticks(np.arange(1, 24, 4))
#     # ax.grid(True, linestyle='-', alpha=0.3)
#     ax.set_title(network_names[i], fontstyle='italic')
#     if i == 0:
#         ax.legend()
#     # Only set xlabel for axes in the bottom row
#     if ax.get_subplotspec().is_last_row():
#         ax.set_xlabel('Block')
# fig.suptitle('PA source - random blocks', fontsize=14)
    
