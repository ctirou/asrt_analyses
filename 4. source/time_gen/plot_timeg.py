import os
from base import ensured
from config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp
from tqdm.auto import tqdm
from mne.viz import Brain

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

networks = NETWORKS
network_names = NETWORK_NAMES
times = np.linspace(-1.5, 1.5, 307)
chance = .25
threshold = .05
res_dir = RESULTS_DIR / 'TIMEG' / 'source'

data_type = 'scores_blocks'

cont_blocks = {}
patterns = {}
randoms = {}
contrasts = {}
for network in tqdm(networks):
    pats_blocks, rands_blocks = [], []
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        contrasts[network] = []
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
    patterns[network] = pats_blocks
    randoms[network] = rands_blocks
    contrasts[network] = pats_blocks - rands_blocks

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
          ['br101', 'br102', 'k1', 'k2', 'k3']]

vmin, vmax = 0.2, 0.3

cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]
sig_color = "#00BFA6"
sig_color = '#708090'
plot_brains = False

fig, axes = plt.subplot_mosaic(design, figsize=(13, 18), sharey=False, sharex=False, layout="constrained",
                            #    gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
                               gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            'width_ratios': [.2, .2, .5, .5, .5]})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
### Plot brain ###
if plot_brains:
    brain_kwargs = dict(hemi='both', background="white", cortex="low_contrast", surf='inflated', subjects_dir=subjects_dir, size=(800, 400))
    for i, (label, sideA, sideB) in enumerate(zip(networks, \
        ['br11', 'br21', 'br31', 'br41', 'br51', 'br61', 'br71', 'br81', 'br91', 'br101'], ['br12', 'br22', 'br32', 'br42', 'br52', 'br62', 'br72', 'br82', 'br92', 'br102'])):
        # Initialize Brain object
        # Add labels
        if label in networks[:-3]:
            brain = Brain(subject='fsaverage2', alpha=1, **brain_kwargs) 
            for hemi in ['lh', 'rh']:
            # hemi = 'split'
                brain.add_label(f'{label}', color=cmap[i], hemi=hemi, alpha=.85, subdir='n7')
        else:
            brain = Brain(subject='fsaverage2', alpha=.5, **brain_kwargs) 
            if label == 'Hippocampus':
                labels = ['Left-Hippocampus', 'Right-Hippocampus']
            elif label == 'Thalamus':
                labels = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
            else:
                labels = ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']
            brain.add_volume_labels(aseg='aseg', labels=labels, colors=cmap[i], alpha=.85, legend=False)
        
        # Capture snapshots for the desired views
        brain.show_view('lateral', distance="auto")
        lateral_img = brain.screenshot('rgb')
        brain.show_view('medial', distance="auto")
        medial_img = brain.screenshot('rgb')
        brain.close()
        lateral_img, medial_img = crop_images(lateral_img), crop_images(medial_img)
        axes[sideA].imshow(lateral_img)
        axes[sideA].axis('off')
        net_name = " "
        axes[sideB].imshow(medial_img)
        axes[sideB].axis('off')

### Pattern ###
for i, (network, pattern_idx) in enumerate(zip(networks, ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1', 'k1'])):
    im = axes[pattern_idx].imshow(patterns[network][:, 3:].mean((0, 1)),
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
    pval = np.load(res_dir / network / data_type / "pval" / "all_pattern-pval.npy")
    sig = pval < threshold
    axes[pattern_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='-', linewidths=1, alpha=1)
    axes[pattern_idx].set_ylabel("Training time (s)")
    if pattern_idx == 'k1':
        axes[pattern_idx].set_xlabel("Testing time (s)")
    else:
        axes[pattern_idx].set_xticklabels([])
    if pattern_idx == 'a1':
        axes[pattern_idx].set_title("Pattern")
        cbar = fig.colorbar(im, ax=axes[pattern_idx], location='top', fraction=.1, ticks=[vmin, vmax])
        cbar.set_label('Accuracy')

### Random ###    
for i, (network, random_idx) in enumerate(zip(networks, ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'i2', 'k2'])):
    im = axes[random_idx].imshow(randoms[network][:, 3:].mean((0, 1)),
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
    pval = np.load(res_dir / network / data_type / "pval" / "all_random-pval.npy")
    sig = pval < threshold
    axes[random_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='-', linewidths=1, alpha=1)
    if random_idx == 'a2':
        axes[random_idx].set_title("Random")
        cbar = fig.colorbar(im, ax=axes[random_idx], location='top', fraction=.1, ticks=[vmin, vmax])
        cbar.set_label('Accuracy')

### Contrast ###
win = np.where((times >= -0.75) & (times < 0))[0]
msig = []
for network in networks:
    s = []
    for sub in range(len(subjects)):
        cont = contrasts[network][:, 3:].mean(1)[sub, win][:, win].mean()
        s.append(cont)
    sig = ttest_1samp(s, 0, axis=0)[1] < threshold
    msig.append(sig)
vminC, vmaxC = -0.03, 0.03
for i, (network, contrast_idx) in enumerate(zip(networks, ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'k3'])):
    im = axes[contrast_idx].imshow(contrasts[network][:, 3:].mean((0, 1)),
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
    pval = np.load(res_dir / network / data_type / "pval" / "all_contrast-pval.npy")
    sig = pval < threshold
    axes[contrast_idx].contour(xx, yy, sig, colors=sig_color, levels=[0], linestyles='-', linewidths=1, alpha=1)
    if contrast_idx == 'a3':
        axes[contrast_idx].set_title("Contrast (Pattern - Random)")
        cbar = fig.colorbar(im, ax=axes[contrast_idx], location='top', fraction=0.1, ticks=[vminC, vmaxC])
        cbar.set_label("Difference in accuracy")
fig.savefig(figures_dir / "timeg-matrix.pdf", transparent=True)
plt.close()