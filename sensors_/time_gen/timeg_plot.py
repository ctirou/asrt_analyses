import os
from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore
from matplotlib import colors

# analysis = "pat_bsl_filtered_3300_3160"
subjects = SUBJS15
jobs = -1
overwrite = False

def compute_spearman(t, g, vector, contrasts):
    return spear(vector, contrasts[:, t, g])[0]

times = np.linspace(-4, 4, 813)

figure_dir = ensured(FIGURES_DIR / "time_gen" / "sensors")

res_dir = RESULTS_DIR / 'TIMEG' / 'sensors' / "scores_skf"

# load patterns and randoms time-generalization on all epochs
all_patterns, all_randoms = [], []
patterns, randoms = [], []
for subject in tqdm(subjects):
    pattern = np.load(res_dir / subject / "pat-all.npy")
    all_patterns.append(pattern)
    random = np.load(res_dir / subject / "rand-all.npy")
    all_randoms.append(random)
    
    pat, rand = [], []
    for i in range(5):
        pat.append(np.load(res_dir / subject / f"pat-{i}.npy"))
        rand.append(np.load(res_dir / subject / f"rand-{i}.npy"))
    
    patterns.append(np.array(pat))
    randoms.append(np.array(rand))

all_patterns, all_randoms = np.array(all_patterns), np.array(all_randoms)
patterns, randoms = np.array(patterns), np.array(randoms)

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
chance = .25
threshold = .05

idx = np.where((times >= -1.5) & (times <= 3))[0]
ensure_dir(res_dir / "pval-all")
if not op.exists(res_dir / "pval-all" / "all_pattern-pval.npy") or overwrite:
    print('Computing pval for all patterns')
    pval = gat_stats(all_patterns[:, idx][:, :, idx] - chance, jobs)
    np.save(res_dir / "pval-all" / "all_pattern-pval.npy", pval)
if not op.exists(res_dir / "pval-all" / "all_random-pval.npy") or overwrite:
    print('Computing pval for all randoms')
    pval = gat_stats(all_randoms[:, idx][:, :, idx] - chance, jobs)
    np.save(res_dir / "pval-all" / "all_random-pval.npy", pval)
if not op.exists(res_dir / "pval-all" / "all_contrast-pval.npy") or overwrite:
    print('Computing pval for all contrasts')
    contrasts = all_patterns - all_randoms
    pval = gat_stats(contrasts[:, idx][:, :, idx], jobs)
    np.save(res_dir / "pval-all" / "all_contrast-pval.npy", pval)

filt = np.where((times >= -1.5) & (times <= 3))[0]
# save learn df x time gen correlation and pvals
ensure_dir(res_dir / "corr-all")
if not op.exists(res_dir / "corr-all" / "rhos_learn.npy") or overwrite:
    contrasts = patterns - randoms
    contrasts = contrasts[:, :, filt][:, :, :, filt]
    contrasts = zscore(contrasts, axis=-1)
    all_rhos = []
    for sub in range(len(subjects)):
        rhos = np.empty((times.shape[0], times.shape[0]))
        vector = learn_index_df.iloc[sub]  # vector to correlate with
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times[filt])) for g in range(len(times[filt])))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times[filt])) for g in range(len(times[filt]))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    np.save(res_dir / "corr-all" / "rhos_learn.npy", all_rhos)
    all_rhos = np.load(res_dir / "corr-all" / "rhos_learn.npy")
    pval = gat_stats(all_rhos, -1)
    np.save(res_dir / "corr" / "pval_learn-pval.npy", pval)

mean_rsa = np.load("/Users/coum/MEGAsync/figures/RSA/sensors/mean_rsa.npy")
if not op.exists(res_dir / "corr" / "rhos_rsa.npy") or overwrite:
    contrasts = patterns - randoms
    contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
    all_rhos = []
    for sub in range(len(subjects)):
        rhos= np.empty((times.shape[0], times.shape[0]))
        vector = mean_rsa[sub]
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    np.save(res_dir / "corr" / "rhos_rsa.npy", all_rhos)
    pval = gat_stats(all_rhos, -1)
    np.save(res_dir / "corr" / "pval_rsa-pval.npy", pval)

cmap1 = "RdBu_r"
cmap2 = "magma_r"
cmap2 = "coolwarm"
cmap3 = "viridis"
cmap4 = "cividis"

idx = np.where((times >= -1.5) & (times <= 3))[0]

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(7, 6))
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []
for ax, data, title in zip(axs.flat, [all_patterns, all_randoms], ["pattern", "random"]):
# for ax, data, title in zip(axs.flat, [patterns, randoms], ["pattern", "random"]):
    images.append(ax.imshow(data[:, idx][:, :, idx].mean(0), 
    # images.append(ax.imshow(data[:, :, idx][:, :, :, idx].mean((0, 1)), 
                            norm=norm,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(f"Time generalization in {title} trials", fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    pval = np.load(res_dir / "pval-all" / f"all_{title.lower()}-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if title == "random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)

fig.savefig(figure_dir / "pattern_random.pdf", transparent=True)
plt.close()

### plot contrast ###
contrasts = all_patterns - all_randoms
pval_cont = np.load(res_dir / "pval-all" / "all_contrast-pval.npy")

rhos = np.load(res_dir / "corr-all" / "rhos_learn.npy")
pval_rhos = np.load(res_dir / "corr-all" / "pval_learn-pval.npy")
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

# diag = np.array([np.diag(coco) for coco in rhos])
# plt.plot(times, diag.mean(0))
# plt.axhline(0)
# pval = decod_stats(diag, -1)
# sig = pval < threshold
# plt.fill_between(times, diag.mean(0), 0, where=sig, alpha=0.5)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout='constrained')
norm = colors.Normalize(vmin=-0.1, vmax=0.1)
images = []
for ax, data, title, pval, vmin, vmax in zip(axs.flat, [contrasts, rhos], \
    ["Contrast (Pattern - Random)", "Contrast and learning correlation"], [pval_cont, pval_rhos], [-0.05, -0.2], [0.05, 0.2]):
    cmap = 'coolwarm' if ax == axs.flat[0] else "BrBG"
        
    im = ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            # norm=norm,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5)
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    sig = pval < threshold
    ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if ax == axs.flat[-1]:
        ax.set_xlabel("Testing time (s)", fontsize=13)
        label = "Spearman's\nrho"
    else:
        label = "Difference in\naccuracy"
    # Draw an empty rectangle centered on -0.25
    rectcolor = 'black' if ax == axs.flat[0] else 'red'
    rect = plt.Rectangle([-0.75, -0.75], 0.72, 0.68, fill=False, edgecolor=rectcolor, linestyle='-', lw=2)
    ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
    cbar.set_label(label, rotation=270, fontsize=13)

fig.savefig(figure_dir / "contrast_corr.pdf", transparent=True)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=True, layout='constrained')
norm = colors.Normalize(vmin=-0.1, vmax=0.1)
images = []
data, title, pval, vmin, vmax = contrasts, "Contrast (Pattern - Random)", pval_cont, -0.05, 0.05
cmap = 'coolwarm'
im = ax.imshow(data[:, idx][:, :, idx].mean(0), 
                        # norm=norm,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="lanczos",
                        origin="lower",
                        cmap=cmap,
                        extent=times[idx][[0, -1, 0, -1]],
                        aspect=0.5)
ax.set_ylabel("Training time (s)", fontsize=13)
ax.set_xticks(np.arange(-1, 3, .5))
ax.set_yticks(np.arange(-1, 3, .5))
ax.set_title(title, fontsize=16)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
sig = pval < threshold
ax.contour(xx, yy, sig, colors='black', levels=[0],
                    linestyles='--', linewidths=1, alpha=.5)
if ax == axs.flat[-1]:
    ax.set_xlabel("Testing time (s)", fontsize=13)
    label = "Spearman's\nrho"
else:
    label = "Difference in\naccuracy"
# Draw an empty rectangle centered on -0.25
# rectcolor = 'black' if ax == axs.flat[0] else 'red'
# rect = plt.Rectangle([-0.75, -0.75], 0.72, 0.68, fill=False, edgecolor=rectcolor, linestyle='-', lw=2)
# ax.add_patch(rect)
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
cbar.set_label(label, rotation=270, fontsize=13)

vmin, vmax = -0.2, 0.2
all_rhos = np.load(res_dir / "corr" / "rhos_rsa.npy")
pval = np.load(res_dir / "corr" / "pval_rsa-pval.npy")
sig = pval < threshold

fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=True, layout='constrained')
cmap = "BrBG"
im = ax.imshow(all_rhos[:, idx][:, :, idx].mean(0), 
                        # norm=norm,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="lanczos",
                        origin="lower",
                        cmap=cmap,
                        extent=times[idx][[0, -1, 0, -1]],
                        aspect=0.5)
ax.set_ylabel("Training time (s)", fontsize=13)
ax.set_xticks(np.arange(-1, 3, .5))
ax.set_yticks(np.arange(-1, 3, .5))
ax.set_title("timeg x mean rep. change corr.", fontsize=16)
ax.axvline(0, color="k")
ax.axhline(0, color="k")
xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
sig = pval < threshold
ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                    linestyles='--', linewidths=1, alpha=.5)
ax.set_xlabel("Testing time (s)", fontsize=13)
label = "Spearman's\nrho"
# Draw an empty rectangle centered on -0.25
# rectcolor = 'black' if ax == axs.flat[0] else 'red'
# rect = plt.Rectangle([-0.75, -0.75], 0.72, 0.68, fill=False, edgecolor=rectcolor, linestyle='-', lw=2)
# ax.add_patch(rect)
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
cbar.set_label(label, rotation=270, fontsize=13)

fig.savefig(figure_dir / "timegxrsa_corr.pdf", transparent=True)
plt.close()

### plot learning index x timeg correlation ###
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
contrasts = patterns - randoms
timeg = []
for sub in range(len(subjects)):
    tg = []
    for i in range(5):
        # tg.append(contrasts[sub, i, idx_timeg][:, idx_timeg].mean())
        tg.append(np.diag(contrasts[sub, i][idx_timeg]).mean())
    timeg.append(np.array(tg))
timeg = np.array(timeg)
rhos = []
for sub in range(len(subjects)):
    r, _ = spear(timeg[sub], learn_index_df.iloc[sub])
    rhos.append(r)    
pval = ttest_1samp(rhos, 0)[1]
slopes, intercepts = [], []

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout='constrained')
fig, ax1 = plt.subplots(1, 1, figsize=(7, 3), layout='constrained')
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(timeg[sub], learn_index_df.iloc[sub], 1)
    ax1.scatter(timeg[sub], learn_index_df.iloc[sub], alpha=0.3)
    ax1.plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
timeg_range = np.linspace(timeg.min(), timeg.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax1.plot(timeg_range, mean_slope * timeg_range + mean_intercept, color='black', lw=4, label='Mean fit')
ax1.set_title('Predictive activity and learning fit', fontsize=16)
# fig.suptitle('Correlation between mean predictive activity\nand learning', fontsize=16, y=0.95)
ax1.set_xlabel('Average pre-stimulus contrast', fontsize=13)
ax1.set_ylabel('Learning index', fontsize=13)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
timeg_flat = timeg.flatten()
textstr = "$p$ < 0.001" if pval < 0.001 else f'$p$ = {pval:.2e}'
# Adjust the legend to be outside the plot
ax1.legend(frameon=False, title=textstr, loc="lower right")
fig.savefig(figure_dir / "learn_corr.pdf", transparent=True)
# fig.savefig(figure_dir / "learn_corr2.pdf", transparent=True)
plt.close()

all_highs, all_lows = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm" / subject
    ensure_dir(res_path)
    # RSA stuff
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    high, low = get_all_high_low(res_path, sequence, "pat_high_rdm_high", cv=True)    
    all_highs.append(high)    
    all_lows.append(low)
all_highs = np.array(all_highs)
all_lows = np.array(all_lows)
diff_sess = list()   
for i in range(5):
    rev_high = all_highs[:, :, i, :].mean(1) - all_highs[:, :, 0, :].mean(axis=1)
    rev_low = all_lows[:, :, i, :].mean(1) - all_lows[:, :, 0, :].mean(axis=1)
    diff_sess.append(rev_low - rev_high)
diff_sess = np.array(diff_sess).swapaxes(0, 1)

# correlation between rsa and time generalization
times_rsa = np.linspace(-0.2, 0.6, 82)
idx_rsa = np.where((times_rsa >= .3) & (times_rsa <= .5))[0]
idx_rsa = np.load("/Users/coum/MEGAsync/figures/RSA/sensors/sig_rsa.npy")
idx_timeg = np.where((times >= -0.5) & (times < 0))[0]
rsa = diff_sess.copy()[:, :, idx_rsa].mean(2)
slopes, intercepts = [], []
rhos = []
# rsa = mean_rsa.copy()
for sub in range(len(subjects)):
    r, p = spear(timeg[sub], rsa[sub])
    rhos.append(r)    
pval = ttest_1samp(rhos, 0)[1]

fig, ax2 = plt.subplots(1, 1, figsize=(6, 3), layout='constrained')
# Plot for individual subjects
for sub, subject in enumerate(subjects):
    slope, intercept = np.polyfit(timeg[sub], rsa[sub], 1)
    ax2.scatter(timeg[sub], rsa[sub], alpha=0.3)
    ax2.plot(timeg[sub], slope * timeg[sub] + intercept, alpha=0.6)
    slopes.append(slope)
    intercepts.append(intercept)
# Plot the mean fit line over the full range of timeg
timeg_range = np.linspace(timeg.min(), timeg.max(), 100)
mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
ax2.plot(timeg_range, mean_slope * timeg_range + mean_intercept, color='black', lw=4, label='Mean fit')
ax2.set_xlabel('Average pre-stimulus contrast', fontsize=13)
ax2.set_ylabel('Similarity index', fontsize=13)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
textstr = "$p$ < 0.001" if pval < 0.001 else f'$p$ = {pval:.2e}'
ax2.legend(frameon=False, title=textstr, loc="lower right")
# ax2.set_title("Correlation between mean predictive activity and mean representational similarity", fontsize=16)
# fig.suptitle("Correlation between mean predictive activity\nand mean representational similarity", y=0.95, fontsize=16)
ax2.set_title("Predictive activity and representational change fit", fontsize=16)
fig.savefig(figure_dir / "rsa_corr.pdf", transparent=True)
# fig.savefig(figure_dir / "combined_corr.pdf", transparent=True)
plt.close()

outerA = [['A1'], ['A2']]
outerB = [['B1'], ['B2']]
outerC = [['C1'], ['C2']]
outerD = [['D1'], ['D2']]
design = [[outerA, outerB],
          [outerC, outerD]]

fig, axd = plt.subplot_mosaic(design, figsize=(13, 10), layout='constrained')
# pattern and random
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []
for ax, data, title in zip([axd['A1'], axd['A2']], [all_patterns, all_randoms], ["pattern", "random"]):
    images.append(ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            norm=norm,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap1,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5))
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(f"Time generalization in {title} trials", fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    pval = np.load(res_dir / "pval" / f"all_{title.lower()}-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if title == "random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
cbar = fig.colorbar(images[0], ax=[axd["A1"], axd["A2"]], orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)
### plot contrast ###
contrasts = all_patterns - all_randoms
pval_cont = np.load(res_dir / "pval" / "all_contrast-pval.npy")
rhos = np.load(res_dir / "corr" / "rhos_learn.npy")
pval_rhos = np.load(res_dir / "corr" / "pval_learn-pval.npy")
images = []
for ax, data, title, pval, vmin, vmax in zip([axd['B1'], axd['B2']], [contrasts, rhos], ["Contrast (Pattern - Random)", "Contrast and learning correlation"], [pval_cont, pval_rhos], [-0.07, -0.4], [0.07, 0.4]):
    cmap = 'coolwarm' if ax == axd['B1'] else "BrBG_r"
        
    im = ax.imshow(data[:, idx][:, :, idx].mean(0), 
                            # norm=norm,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation="lanczos",
                            origin="lower",
                            cmap=cmap,
                            extent=times[idx][[0, -1, 0, -1]],
                            aspect=0.5)
    ax.set_ylabel("Training time (s)", fontsize=13)
    ax.set_xticks(np.arange(-1, 3, .5))
    ax.set_yticks(np.arange(-1, 3, .5))
    ax.set_title(title, fontsize=16)
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
    sig = pval < threshold
    ax.contour(xx, yy, sig[idx][:, idx], colors='black', levels=[0],
                        linestyles='--', linewidths=1, alpha=.5)
    if ax == axd['B2']:
        ax.set_xlabel("Testing time (s)", fontsize=13)
        label = "Spearman's\nrho"
    else:
        label = "Difference in\naccuracy"
    # Draw an empty rectangle centered on -0.25
    rect = plt.Rectangle([-0.75, -0.75], 0.72, 0.68, fill=False, edgecolor='black', linestyle='--', lw=1.5)
    ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
    cbar.set_label(label, rotation=270, fontsize=13)