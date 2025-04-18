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
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock
ensure_dir(figures_dir)
overwrite = False

networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
times = np.linspace(-1.5, 1.5, 307)
chance = .25
threshold = .05
res_dir = TIMEG_DATA_DIR / 'results' / 'source' / 'max-power'


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
            pat = np.load(res_dir / network / 'pattern' / f"{subject}-{j}-scores.npy")
            rand = np.load(res_dir / network / 'random' / f"{subject}-{j}-scores.npy")
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
          ['br101', 'br102', 'k1', 'k2', 'k3'],
          ['l', 'l', 'j', 'j', 'j']]
vmin, vmax = 0.2, 0.3

fig, axes = plt.subplot_mosaic(design, figsize=(12, 20), sharey=False, sharex=False, layout="constrained",
                               gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
                                            'width_ratios': [.2, .2, .5, .5, .5]})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
# fig.suptitle("Comparison of Pattern, Random, and Contrast Accuracy Over Time", fontsize=14, fontweight='bold')
### Plot brain ###
cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]
sig_color = "#00BFA6"
sig_color = '#708090'
brain_kwargs = dict(hemi='both', background="white", cortex="low_contrast", surf='inflated', subjects_dir=subjects_dir, size=(800, 400))
for i, (label, sideA, sideB) in enumerate(zip(networks, \
    ['br11', 'br21', 'br31', 'br41', 'br51', 'br61', 'br71', 'br81', 'br91', 'br101'], ['br12', 'br22', 'br32', 'br42', 'br52', 'br62', 'br72', 'br82', 'br92', 'br102'])):
    # Initialize Brain object
    # Add labels
    if label in networks[:-3]:
        brain = Brain(subject='fsaverage2', alpha=1, **brain_kwargs) 
        for hemi in ['lh', 'rh']:
        # hemi = 'split'
            brain.add_label(f'{label}', color=cmap[i], hemi=hemi, borders=False, alpha=.85, subdir='n7')
    else:
        brain = Brain(subject='fsaverage2', alpha=.5, **brain_kwargs) 
        if label == 'Hippocampus':
            labels = ['Left-Hippocampus', 'Right-Hippocampus']
        elif label == 'Thalamus':
            labels = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
        else:
            labels = ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']
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
for network, pattern_idx in zip(networks, ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1', 'k1']):
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
    axes[pattern_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='-', linewidths=1)
    axes[pattern_idx].set_ylabel("Training time (s)")
    if pattern_idx == 'k1':
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
for network, random_idx in zip(networks, ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'i2', 'k2']):
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
    if random_idx == 'k2':
        axes[random_idx].set_xlabel("Testing time (s)")
    else:
        axes[random_idx].set_xticklabels([])
    axes[random_idx].set_yticklabels([])
    
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_random-pval.npy")
    sig = pval < threshold
    axes[random_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='-', linewidths=1)
    
    if random_idx == 'a2':
        axes[random_idx].set_title("Random")
        cbar = fig.colorbar(im, ax=axes[random_idx], location='top', fraction=.1, ticks=[vmin, vmax])
        cbar.set_label('Accuracy')

### Contrast ###
vminC, vmaxC = -0.05, 0.05
for network, contrast_idx in zip(networks, ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'k3']):
    all_contrast = all_patterns[network] - all_randoms[network]
    im = axes[contrast_idx].imshow(all_contrast.mean(0),
                                   interpolation="lanczos",
                                   origin="lower",
                                   cmap=cmap2,
                                   extent=times[[0, -1, 0, -1]],
                                   aspect=0.5,
                                   vmin=vminC,
                                   vmax=vmaxC)
    
    axes[contrast_idx].axvline(0, color="k", alpha=.5)
    axes[contrast_idx].axhline(0, color="k", alpha=.5)
    if contrast_idx == 'k3':
        axes[contrast_idx].set_xlabel("Testing time (s)")
    else:
        axes[contrast_idx].set_xticklabels([])
    axes[contrast_idx].set_yticklabels([])
    
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    pval = np.load(res_dir / network / "pval" / "all_contrast-pval.npy")
    sig = pval < threshold
    axes[contrast_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='solid', linewidths=1)
        
    if contrast_idx == 'a3':
        axes[contrast_idx].set_title("Contrast (Pattern - Random)")
        cbar = fig.colorbar(im, ax=axes[contrast_idx], location='top', fraction=0.1, ticks=[vminC, vmaxC])
        cbar.set_label("Difference in accuracy")

mean_diag = []
mean_net_sess = []
mean_diag_sess = []
filt = np.where((times >= 0.05) & (times <= 0.55))[0]
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
        # sess1.append(contrast[sub, filter_time][:, filter_time].mean())
        sess1.append(np.diag(contrast[sub, filter_time][:, filter_time]))
        sess2.append(all_diags[net][sub, filt].mean())
    mean_net_sess.append(np.array(sess1))
    mean_diag_sess.append(np.array(sess2))
mean_net_sess = np.array(mean_net_sess).mean(-1)
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
ymin = -0.04
ymax = 0.04
x = np.arange(len(networks))
spacing = 0.35
rects1 = axes['j'].bar(x + spacing/2, mean_net, spacing, label='Pre-activation', facecolor=c1, alpha=.8, edgecolor=None)
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
axes['j'].set_xticklabels(network_names, rotation=45, ha='right')
axes['j'].axhline(0, color='grey', linewidth=2)
axes['j'].set_title('Predictive coding during pre-stimulus period and perception', fontsize=12, pad=-20)
axes['j'].set_ylabel('Mean effect', labelpad=-20)
axes['j'].set_ylim(ymin, ymax)
axes['j'].set_yticks([ymin, ymax])
axes['j'].legend(frameon=False, loc='upper left', fontsize=9)
axes['j'].spines['top'].set_visible(False)
axes['j'].spines['right'].set_visible(False)

# example plot
axes['l'].axhline(0, color='k', alpha=.5)
axes['l'].axvline(0, color='k', alpha=.5)
axes['l'].set_xlim(times[0], times[-1])
axes['l'].set_ylim(times[0], times[-1])
axes['l'].set_yticks([-1, 0, 1])
axes['l'].plot(times[filter_time], times[filter_time], color=c1, lw=3, label='Pre-activation')
axes['l'].plot(times[filt], times[filt], color=c2, lw=3, label='Perception')
axes['l'].set_xlabel("Testing time (s)")
axes['l'].set_ylabel("Training time (s)")
axes['l'].legend(loc='lower right', fontsize=9)
axes['l'].set_aspect(0.5)

# # Hide all None areas
# for key, ax in axes.items():
#     if key is None:
#         ax.set_visible(False)

fig.savefig(figures_dir / f"test-pattern_random_contrast-5.pdf", transparent=True)
plt.close()