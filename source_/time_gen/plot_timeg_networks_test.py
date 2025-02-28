import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, spearmanr, zscore, linregress
import numba
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm.auto import tqdm
from mne.viz import Brain

data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR

lock = 'stim'
# network and custom label_names
n_parcels = 200
n_networks = 7
networks = NETWORKS
network_names = NETWORK_NAMES
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock
ensure_dir(figures_dir)
overwrite = False

names = NETWORK_NAMES
times = np.linspace(-1.5, 1.5, 305)
chance = .25
threshold = .05

# analysis = "tg_0206_emp"
# analysis = "tg_rs_emp"
analysis = "tg_rdm_emp"
# analysis = "tg_rdm_emp_reduced"
res_dir = data_path / analysis / lock

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
all_diags = {}
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
for network in tqdm(networks):
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-{j}-scores.npy"))
            rand.append(np.load(res_dir / network / 'random' / f"{subject}-{j}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy"))
        all_rand.append(np.load(res_dir / network / 'random' / f"{subject}-all-scores.npy"))
        
        diag = np.array(all_pat) - np.array(all_rand)
        all_diag.append(np.diag(diag[i]))
        
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    all_diags[network] = np.array(all_diag)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)

cmap = "viridis"
cmap1 = "RdBu_r"
cmap2 = "coolwarm"
cmap3 = 'magma'

### plot pattern for all networks ###
fig, axes = plt.subplots(9, 1, figsize=(6, 12), sharex=True, sharey=True, layout='constrained')
# fig.suptitle("Pattern", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    im = axes[i].imshow(
        all_patterns[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.24,
        vmax=0.26)
    axes[i].set_ylabel("Training Time (s)")
    # axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_pattern-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
# fig.savefig(figures_dir / f"pattern.pdf", transparent=True)

### plot random for all networks ###
fig, axes = plt.subplots(7, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Random", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    im = axes[i].imshow(
        all_randoms[network].mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=0.2,
        vmax=0.3)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_random-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
fig.savefig(figures_dir / f"random.pdf", transparent=True)

### plot contrast for all networks ###
fig, axes = plt.subplots(9, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Contrast = Pattern – Random", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = axes[i].imshow(
        all_contrast.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-0.01,
        vmax=0.01)
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvline(0, color="k", alpha=.5)
    axes[i].axhline(0, color="k", alpha=.5)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = pval < threshold
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
fig.savefig(figures_dir / f"contrast.pdf", transparent=True)

### plot diagonal for all networks ###
fig, axes = plt.subplots(9, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("Contrast diagonal", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    axes[i].plot(times, all_diags[network].mean(0))
    pval = decod_stats(all_diags[network], -1)
    sig = pval < threshold
    axes[i].fill_between(times, 0, all_diags[network].mean(0), where=sig, color='C7', alpha=.5)
    axes[i].axhline(0, color='grey', alpha=.5)
    axes[i].set_ylabel("difference")
    axes[i].set_title(f"${name}$", fontsize=10)
    axes[i].axvspan(0, 0.2, color='gray', alpha=0.1)
fig.savefig(figures_dir / f"diagonals.pdf", transparent=True)

### plot learn df x time gen correlation for all networks ###
fig, axes = plt.subplots(9, 1, figsize=(6, 12), sharex=True, layout='tight')
fig.suptitle("learn df x time gen correlation", fontsize=14)
for i, (network, name) in enumerate(zip(networks, names)):
    rhos = np.load(res_dir / network / "corr" / "rhos_learn.npy")
    pval = np.load(res_dir / network / "corr" / "pval_learn-pval.npy")
    sig = pval < threshold
    im = axes[i].imshow(
        rhos.mean(0),
        interpolation="lanczos",
        origin="lower",
        cmap=cmap1,
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=-.2,
        vmax=.2)
    if network == networks[-1]:
        axes[i].set_xlabel("Testing Time (s)")
    axes[i].set_ylabel("Training Time (s)")
    axes[i].set_title(f"{name}", style='italic')
    axes[i].axvline(0, color="k")
    axes[i].axhline(0, color="k")
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    axes[i].contour(xx, yy, sig, colors='Gray', levels=[0],
                        linestyles='solid', linewidths=1)
    cbar = plt.colorbar(im, ax=axes[i])
    cbar.set_label("accuracy")
fig.savefig(figures_dir / "learn_corr.pdf", transparent=True)

### Mean effect ###
mean_diag = []
mean_net_sess = []
mean_diag_sess = []
filt = np.where((times >= 0.2) & (times <= 0.6))[0]
mean_net = []
filter_time = np.where((times >= -0.5) & (times < 0))[0]  # Correct filtering condition
for i, net in enumerate(networks):
    contrast = all_patterns[net] - all_randoms[net]
    # Compute mean net effect
    mean_net.append(contrast[:, filter_time][:, :, filter_time].mean())
    # Mean diagonal for the specific time window with absolute values
    mean_diag.append((all_diags[net][:, filt].mean()))
    sess1, sess2 = [], []
    for sub in range(len(subjects)):
        # Mean for session 1 and 2 per subject
        sess1.append(contrast[sub, filter_time][:, filter_time].mean())
        sess2.append(all_diags[net][sub, filt].mean())
    mean_net_sess.append(np.array(sess1))
    mean_diag_sess.append(np.array(sess2))
mean_net_sess = np.array(mean_net_sess)
mean_diag_sess = np.array(mean_diag_sess)
sem_net = np.std(mean_net_sess, axis=1) / np.sqrt(len(mean_net_sess[0]))  # SEM for mean_net
sem_diag = np.std(mean_diag_sess, axis=1) / np.sqrt(len(mean_diag_sess[0]))  # SEM for mean_diag
### Plot mean effect ### 
alpha = 0.05
color_diag = '#0072B2'  # Blue
color_net = '#E69F00'  # Orange
# Perform statistical tests for mean_net (one-sample t-test against 0)
mean_net_significance = [ttest_1samp(data, 0)[1] < alpha for data in mean_net_sess]
fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(axis='y', linestyle='--', alpha=0.3)
x = np.arange(len(networks))
spacing = 0.35
rects2 = ax.bar(x + spacing/2, mean_net, spacing, label='Prediction effect', color=color_net, edgecolor='black')
rects1 = ax.bar(x - spacing/2, mean_diag, spacing, label='(Random - Pattern) contrast diagonal', color=color_diag, edgecolor='black')
# ax.errorbar(x - spacing/2, mean_diag, yerr=sem_net, fmt='none', color='black', capsize=5)
# ax.errorbar(x + spacing/2, mean_net, yerr=sem_diag, fmt='none', color='black', capsize=5)
# Annotate significant mean_net bars with an asterisk
for i, rect in enumerate(rects2):
    if mean_net_significance[i]:
        ax.annotate('*',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=20, color='black')
ax.set_ylabel('Mean effect / diagonal')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
ax.axhline(0, color='grey', linewidth=2)
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.suptitle('Mean contrast and prediction effects by network', pad=20, fontsize=16)
fig.savefig(figures_dir / f"mean_effect.pdf", transparent=True)
plt.show()

def crop_images(screenshot):
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    return cropped_screenshot

# Define subplot design layout
design = [['br11', 'br12', 'a1', 'a2', 'a3'],
          ['br21', 'br22', 'b1', 'b2', 'b3'],
          ['br31', 'br32', 'c1', 'c2', 'c3'],
          ['br41', 'br42', 'd1', 'd2', 'd3'],
          ['br51', 'br52', 'e1', 'e2', 'e3'],
          ['br61', 'br62', 'f1', 'f2', 'f3'],
          ['br71', 'br72', 'g1', 'g2', 'g3'],
          ['br81', 'br82', 'h1', 'h2', 'h3'],
          ['br91', 'br92', 'i1', 'i2', 'i3'],
          [None, None, 'j', 'j', 'j']]
vmin, vmax = 0.23, 0.27

fig, axes = plt.subplot_mosaic(design, figsize=(12, 16), sharey=False, sharex=False, layout="constrained",
                                   gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
                                                'width_ratios': [.2, .2, .5, .5, .5]})

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
# fig.suptitle("Comparison of Pattern, Random, and Contrast Accuracy Over Time", fontsize=14, fontweight='bold')
### Plot brain ###
cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9']
brain_kwargs = dict(hemi='both', background="white", cortex="low_contrast", surf='inflated', subjects_dir=subjects_dir, size=(800, 400))
for i, (label, sideA, sideB) in enumerate(zip(networks, ['br11', 'br21', 'br31', 'br41', 'br51', 'br61', 'br71', 'br81', 'br91'], ['br12', 'br22', 'br32', 'br42', 'br52', 'br62', 'br72', 'br82', 'br92'])):
    # Initialize Brain object
    # Add labels
    if label in networks[:-2]:
        brain = Brain(subject='sub01', alpha=1, **brain_kwargs) 
        for hemi in ['lh', 'rh']:
        # hemi = 'split'
            brain.add_label(f'{label}', color=cmap[i], hemi=hemi, borders=False, alpha=.85, subdir='n7')
    else:
        brain = Brain(subject='sub01', alpha=.5, **brain_kwargs) 
        labels = ['Left-Hippocampus', 'Right-Hippocampus'] if label == 'Hippocampus' else ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
        brain.add_volume_labels(aseg='aseg', labels=labels, colors=cmap[i], alpha=.85, legend=False)
    
    # brain.set_data_smoothing(50)
    # Capture snapshots for the desired views 
    brain.show_view('lateral', distance="auto")
    lateral_img = brain.screenshot('rgb')
    brain.show_view('medial', distance="auto")
    medial_img = brain.screenshot('rgb')
    brain.close()
    
    lateral_img, medial_img = crop_images(lateral_img), crop_images(medial_img)
    # lateral_img = crop_images(lateral_img)
    
    # Display images side by side using Matplotlib
    axes[sideA].imshow(lateral_img)
    axes[sideA].axis('off')
    # axd[sideA].set_title(net_name, fontsize=14)
    # Call the function to set the br-specific title
    net_name = " "
    # plot_with_br_title(fig, axes, design, net_name, row_idx=i)
    axes[sideB].imshow(medial_img)
    axes[sideB].axis('off')

### Pattern ###
for network, pattern_idx in zip(networks, ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1']):
    im = axes[pattern_idx].imshow(all_patterns[network].mean(0),
                                  interpolation="lanczos",
                                  origin="lower",
                                  cmap=cmap1,
                                  extent=times[[0, -1, 0, -1]],
                                  aspect=0.5,
                                  vmin=vmin,
                                  vmax=vmax)
    axes[pattern_idx].axvline(0, color="k", alpha=.5)
    axes[pattern_idx].axhline(0, color="k", alpha=.5)
    
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_pattern-pval.npy")
    sig = pval < threshold
    axes[pattern_idx].contour(xx, yy, sig, colors='grey', levels=[0], linestyles='-', linewidths=1)
    axes[pattern_idx].set_ylabel("Training time (s)")
    if pattern_idx == 'i1':
        axes[pattern_idx].set_xlabel("Testing time (s)")
    else:
        axes[pattern_idx].set_xticklabels([])

    if pattern_idx == 'a1':
        axes[pattern_idx].set_title("Pattern")
        cbar = fig.colorbar(im, ax=axes[pattern_idx], location='top', fraction=.1, ticks=[vmin, vmax])
        cbar.set_label('Accuracy')

    # Hide labels but keep minor ticks on the left side
    # axes[pattern_idx].tick_params(axis='y', which='both', left=True, labelleft=False)

### Random ###    
for network, random_idx in zip(networks, ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'i2']):
    im = axes[random_idx].imshow(all_randoms[network].mean(0),
                                 interpolation="lanczos",
                                 origin="lower",
                                 cmap=cmap1,
                                 extent=times[[0, -1, 0, -1]],
                                 aspect=0.5,
                                 vmin=vmin,
                                 vmax=vmax)
    axes[random_idx].axvline(0, color="k", alpha=.5)
    axes[random_idx].axhline(0, color="k", alpha=.5)
    if random_idx == 'i2':
        axes[random_idx].set_xlabel("Testing time (s)")
    else:
        axes[random_idx].set_xticklabels([])
    axes[random_idx].set_yticklabels([])
    
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_random-pval.npy")
    sig = pval < threshold
    axes[random_idx].contour(xx, yy, sig, colors='grey', levels=[0], linestyles='-', linewidths=1)
    
    if random_idx == 'a2':
        axes[random_idx].set_title("Random")
        cbar = fig.colorbar(im, ax=axes[random_idx], location='top', fraction=.1, ticks=[vmin, vmax])
        cbar.set_label('Accuracy')

### Contrast ###
for network, contrast_idx in zip(networks, ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3']):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = axes[contrast_idx].imshow(all_contrast.mean(0),
                                   interpolation="lanczos",
                                   origin="lower",
                                   cmap=cmap2,
                                   extent=times[[0, -1, 0, -1]],
                                   aspect=0.5,
                                   vmin=-0.01,
                                   vmax=0.01)
    
    axes[contrast_idx].axvline(0, color="k", alpha=.5)
    axes[contrast_idx].axhline(0, color="k", alpha=.5)
    if contrast_idx == 'i3':
        axes[contrast_idx].set_xlabel("Testing time (s)")
    else:
        axes[contrast_idx].set_xticklabels([])
    axes[contrast_idx].set_yticklabels([])
    
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = pval < threshold
    axes[contrast_idx].contour(xx, yy, sig, colors='black', levels=[0], linestyles='solid', linewidths=1)
        
    if contrast_idx == 'a3':
        axes[contrast_idx].set_title("Contrast (Pattern - Random)")
        cbar = fig.colorbar(im, ax=axes[contrast_idx], location='top', fraction=0.1, ticks=[-0.01, 0.01])
        cbar.set_label("Difference in accuracy")

mean_diag = []
mean_net_sess = []
mean_diag_sess = []
filt = np.where((times >= 0) & (times <= 0.6))[0]
mean_net = []
filter_time = np.where((times >= -0.5) & (times < 0))[0]  # Correct filtering condition
for i, net in enumerate(networks):
    contrast = all_patterns[net] - all_randoms[net]
    # Compute mean net effect
    mean_net.append(contrast[:, filter_time][:, :, filter_time].mean())
    # Mean diagonal for the specific time window with absolute values
    mean_diag.append((all_diags[net][:, filt].mean()))
    sess1, sess2 = [], []
    for sub in range(len(subjects)):
        # Mean for session 1 and 2 per subject
        sess1.append(contrast[sub, filter_time][:, filter_time].mean())
        sess2.append(all_diags[net][sub, filt].mean())
    mean_net_sess.append(np.array(sess1))
    mean_diag_sess.append(np.array(sess2))
mean_net_sess = np.array(mean_net_sess)
mean_diag_sess = np.array(mean_diag_sess)
sem_net = np.std(mean_net_sess, axis=1) / np.sqrt(len(mean_net_sess[1]))  # SEM for mean_net
sem_diag = np.std(mean_diag_sess, axis=1) / np.sqrt(len(mean_diag_sess[1]))  # SEM for mean_diag
### Plot mean effect ### 
alpha = 0.05

c1 = '#FF0000'  # Blue
c2 = '#5BBCD6'  # Orange
# Perform statistical tests for mean_net (one-sample t-test against 0)
mean_net_significance = [ttest_1samp(data, 0)[1] < alpha for data in mean_net_sess]

# axes['d'].grid(axis='y', linestyle='--', alpha=0.3)
x = np.arange(len(networks))
spacing = 0.35
rects1 = axes['j'].bar(x + spacing/2, mean_net, spacing, label='Prediction', facecolor=c1, alpha=.8, edgecolor=None)
rects2 = axes['j'].bar(x - spacing/2, mean_diag, spacing, label='Perception', facecolor=c2, alpha=.8, edgecolor=None)
# axes['d'].errorbar(x - spacing/2, mean_diag, yerr=sem_net, fmt='none', color='black', capsize=5)
# axes['d'].errorbar(x + spacing/2, mean_net, yerr=sem_diag, fmt='none', color='black', capsize=5)
# Annotate significant mean_net bars with an asterisk
for i, rect in enumerate(rects1):
    if mean_net_significance[i]:
        axes['j'].annotate('*',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 9),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=15, color='black')

axes['j'].errorbar(x - spacing/2, mean_diag, yerr=sem_net, fmt='none', color='black', capsize=3)
axes['j'].errorbar(x + spacing/2, mean_net, yerr=sem_diag, fmt='none', color='black', capsize=3)
axes['j'].set_xticks(x)
axes['j'].set_xticklabels(names, rotation=45, ha='right')
axes['j'].axhline(0, color='grey', linewidth=2)
axes['j'].set_title('Predictive coding during pre-stimulus period and perception', fontsize=12, pad=-20)
axes['j'].set_ylabel('Mean effect')
axes['j'].set_yticks([-0.01, 0, 0.01])
axes['j'].legend(frameon=False, loc='upper left', fontsize=9)
axes['j'].spines['top'].set_visible(False)
axes['j'].spines['right'].set_visible(False)

# Hide all None areas
for key, ax in axes.items():
    if key is None:
        ax.set_visible(False)

fig.savefig(figures_dir / f"pattern_random_contrast-3.pdf", transparent=True)
plt.close()