import os
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
session_on = False

data_type = "scores_lobotomized"

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

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
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

# if session_on:
#     for i in range(5):
#         # Patterns
#         if not op.exists(res_path/ f"pat-{i}-pval.npy") or overwrite:
#             print(f'Computing pval for patterns session {i}')
#             pval = gat_stats(patterns[:, i, idx][:, :, idx] - chance, jobs)
#             np.save(res_path/ f"pat-{i}-pval.npy", pval)
#         else:
#             print(f'Pattern session {i} already exists')
#         # Randoms
#         if not op.exists(res_path/ f"rand-{i}-pval.npy") or overwrite:
#             print(f'Computing pval for randoms session {i}')
#             pval = gat_stats(randoms[:, i, idx][:, :, idx] - chance, jobs)
#             np.save(res_path/ f"rand-{i}-pval.npy", pval)
#         else:
#             print(f'Random session {i} already exists')
#         # Contrast
#         if not op.exists(res_path/ f"con-{i}-pval.npy") or overwrite:
#             print(f'Computing pval for contrasts session {i}')
#             contrasts = patterns[:, i, idx][:, :, idx] - randoms[:, i, idx][:, :, idx]
#             pval = gat_stats(contrasts, jobs)
#             np.save(res_path/ f"con-{i}-pval.npy", pval)
#         else:
#             print(f'Contrast session {i} already exists')

filt = np.where((times >= -1.5) & (times <= 3))[0]
# save learn df x time gen correlation and pvals
ensure_dir(res_dir / "corr")
if not op.exists(res_dir / "corr" / "rhos_learn.npy") or overwrite:
    contrasts = pats_blocks - rands_blocks
    contrasts = contrasts[:, :, filt][:, :, :, filt]
    # contrasts = zscore(contrasts, axis=-1)
    all_rhos = []
    for sub in range(len(subjects)):
        print(f"Computing Spearman correlation for subject {sub+1}/{len(subjects)}")
        rhos = np.empty((times[filt].shape[0], times[filt].shape[0]))
        # vector = learn_index_df.iloc[sub]  # vector to correlate with
        vector = learn_index_blocks.iloc[sub]  # vector to correlate with
        contrast = contrasts[sub]
        results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times[filt])) for g in range(len(times[filt])))
        for idx, (t, g) in enumerate([(t, g) for t in range(len(times[filt])) for g in range(len(times[filt]))]):
            rhos[t, g] = results[idx]
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    # all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
    all_rhos = fisher_z_transform_3d(all_rhos)
    np.save(res_dir / "corr" / "rhos_learn.npy", all_rhos)
    # all_rhos = np.load(res_dir / "corr" / "rhos_learn.npy")
    pval = gat_stats(all_rhos, -1)
    np.save(res_dir / "corr" / "pval_learn-pval.npy", pval)

cmap1 = "RdBu_r"
cmap2 = "coolwarm"
contour_color = "black"
contour_color = "#00BFA6"
contour_color = "#708090"


idx = np.where((times >= -1.5) & (times <= 3))[0]

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(7, 6))
norm = colors.Normalize(vmin=0.18, vmax=0.32)
images = []
for ax, data, title in zip(axs.flat, [pats_blocks, rands_blocks], ["pattern", "random"]):
    # for ax, data, title in zip(axs.flat, [patterns, randoms], ["pattern", "random"]):
    images.append(ax.imshow(data[:, 3:, idx][:, 3:, :, idx].mean((0, 1)), 
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
    pval = np.load(res_path/ f"all_{title.lower()}-pval.npy")
    sig = pval < threshold
    ax.contour(xx, yy, sig, colors=contour_color, levels=[0],
                        linestyles='-', linewidths=1, alpha=1)
    if title == "random":
        ax.set_xlabel("Testing time (s)", fontsize=13)
cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
cbar.set_label("\nAccuracy", rotation=270, fontsize=13)

fig.savefig(figure_dir / "pattern_random.pdf", transparent=True)
plt.close()

### plot contrast ###
contrasts = pats_blocks - rands_blocks
win = np.where((times[filt] <= -0.5) & (times[filt] < 0))[0]
mean = np.array([cont[win, win].mean() for cont in contrasts.mean(1)])
sig_mean = ttest_1samp(mean, 0, axis=0)[1] < threshold
contrasts = contrasts[:, 3:, idx][:, 3:, :, idx].mean((0, 1))
pval_cont = np.load(res_dir / "pval" /  "all_contrast-pval.npy")

rhos = np.load(res_dir / "corr" / "rhos_learn.npy").mean(0)
pval_rhos = np.load(res_dir / "corr" / "pval_learn-pval.npy")

csig = "#0F0D0E"
fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout='constrained')
norm = colors.Normalize(vmin=-0.1, vmax=0.1)
images = []
for ax, data, title, pval, vmin, vmax in zip(axs.flat, [contrasts, rhos], \
    ["Contrast (Pattern - Random)", "Contrast and learning correlation"], [pval_cont, pval_rhos], [-0.04, -0.05], [0.04, 0.05]):
    cmap = 'coolwarm' if ax == axs.flat[0] else "BrBG"
    im = ax.imshow(data, 
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
    ax.contour(xx, yy, sig, colors=contour_color, levels=[0],
                        linestyles='-', linewidths=1, alpha=1)
    if ax == axs.flat[-1]:
        ax.set_xlabel("Testing time (s)", fontsize=13)
        label = "Spearman's\nrho"
    else:
        label = "Difference in\naccuracy"
    # Draw an empty rectangle centered on -0.25
    # rectcolor = 'black' if ax == axs.flat[0] else 'red'
    rectcolor = 'black'
    if ax == axs.flat[0]:
        rect = plt.Rectangle([-0.5, -0.5], 0.48, 0.48, fill=False, edgecolor=csig, linestyle='--', lw=2, zorder=10)
        ax.add_patch(rect)
        if sig_mean:
            ax.text(-0.6, -0.35, "*", fontsize=25, color=csig, ha='right', va='center', weight='bold')
    #     rect1 = plt.Rectangle([-0.75, 0.05], 0.72, 0.68, fill=False, edgecolor='white', linestyle='--', lw=2, zorder=10)
    #     ax.add_patch(rect1)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1, ticks=[vmin, vmax])
    cbar.set_label(label, rotation=270, fontsize=13)

fig.savefig(figure_dir / "contrast_corr.pdf", transparent=True)
plt.close()

# # ----- Plot sessions ----- #

# # Patterns
# fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, layout='tight')
# norm = colors.Normalize(vmin=0.18, vmax=0.32)
# images = []
# for i, ax in enumerate(axs.flat):
#     pval = np.load(res_path/ f"pat-{i}-pval.npy")
#     images.append(ax.imshow(patterns[:, i, idx][:, :, idx].mean(0), 
#                             norm=norm,
#                             interpolation="lanczos",
#                             origin="lower",
#                             cmap=cmap1,
#                             extent=times[idx][[0, -1, 0, -1]],
#                             aspect=0.5))
#     ax.set_title(f"Session {i+1}", fontsize=10, fontstyle="italic")
#     xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
#     sig = pval < threshold
#     ax.contour(xx, yy, sig, colors='black', levels=[0],
#                         linestyles='--', linewidths=1)
#     ax.axvline(0, color="k", alpha=.5)
#     ax.axhline(0, color="k", alpha=.5)
# cbar = fig.colorbar(images[0], ax=axs[-1], orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
# cbar.set_label("\nAccuracy", rotation=270, fontsize=13)
# fig.suptitle("Pattern", fontsize=12)

# # Randoms
# fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, layout='tight')
# images = []
# for i, ax in enumerate(axs.flat):
#     pval = np.load(res_path/ f"rand-{i}-pval.npy")
#     images.append(ax.imshow(randoms[:, i, idx][:, :, idx].mean(0), 
#                             norm=norm,
#                             interpolation="lanczos",
#                             origin="lower",
#                             cmap=cmap1,
#                             extent=times[idx][[0, -1, 0, -1]],
#                             aspect=0.5))
#     ax.set_title(f"Session {i+1}", fontsize=10, fontstyle="italic")
#     xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
#     sig = pval < threshold
#     ax.contour(xx, yy, sig, colors='black', levels=[0],
#                         linestyles='--', linewidths=1)
#     ax.axvline(0, color="k", alpha=.5)
#     ax.axhline(0, color="k", alpha=.5)
# cbar = fig.colorbar(images[0], ax=axs[-1], orientation='vertical', fraction=.1, ticks=[0.18, 0.32])
# cbar.set_label("\nAccuracy", rotation=270, fontsize=13)
# fig.suptitle("Random", fontsize=12)

# # Contrast
# fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True, layout='tight')
# norm = colors.Normalize(vmin=-0.1, vmax=0.1)
# images = []
# for i, ax in enumerate(axs.flat):
#     pval = np.load(res_path/ f"con-{i}-pval.npy")
#     contrasts = patterns - randoms
#     images.append(ax.imshow(contrasts[:, i, idx][:, :, idx].mean(0), 
#                             norm=norm,
#                             interpolation="lanczos",
#                             origin="lower",
#                             cmap=cmap2,
#                             extent=times[idx][[0, -1, 0, -1]],
#                             aspect=0.5))
#     ax.set_title(f"Session {i+1}", fontsize=10, fontstyle="italic")
#     xx, yy = np.meshgrid(times[idx], times[idx], copy=False, indexing='xy')
#     sig = pval < threshold
#     ax.contour(xx, yy, sig, colors='black', levels=[0],
#                         linestyles='--', linewidths=1)
#     ax.axvline(0, color="k", alpha=.5)
#     ax.axhline(0, color="k", alpha=.5)
# cbar = fig.colorbar(images[-1], ax=axs[-1], orientation='vertical', fraction=.1, ticks=[-0.1, 0.1])
# cbar.set_label("\nDifference in accuracy", rotation=270, fontsize=13)
# fig.suptitle("Contrast (Pattern - Random)", fontsize=12)