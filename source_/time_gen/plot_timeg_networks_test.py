import os
from base import ensured
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
from scipy.ndimage import gaussian_filter1d

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']
times = np.linspace(-1.5, 1.5, 307)
chance = .25
threshold = .05
res_dir = RESULTS_DIR / 'TIMEG' / 'source'

data_type = 'scores_lobotomized'

# diags = {}
# all_diags = {}
# patterns, randoms = {}, {}
# all_patterns, all_randoms = {}, {}
# for network in tqdm(networks):
#     if not network in patterns:
#         patterns[network], randoms[network] = [], []
#         all_patterns[network], all_randoms[network] = [], []
#     all_pat, all_rand, all_diag = [], [], []
#     patpat, randrand = [], []
#     ddd = []
#     for i, subject in enumerate(subjects):
#         pat, rand = [], []
#         dd = []
#         for j in [0, 1, 2, 3, 4]:
#             pat = np.load(res_dir / network / data_type2 / subject / f"pat-{j}.npy")
#             rand = np.load(res_dir / network / data_type2 / subject / f"rand-{j}.npy")
#             patpat.append(np.array(pat))
#             randrand.append(np.array(rand))
#             d = np.array(pat) - np.array(rand)
#             dd.append(np.diag(d))
#         all_pat.append(np.load(res_dir / network / data_type1 / subject / "pat-all.npy"))
#         all_rand.append(np.load(res_dir / network / data_type1 / subject / "rand-all.npy"))
#         diag = np.array(all_pat) - np.array(all_rand)
#         all_diag.append(np.diag(diag[i]))
#         ddd.append(np.array(dd))
#     all_patterns[network] = np.array(all_pat)
#     all_randoms[network] = np.array(all_rand)
#     all_diags[network] = np.array(all_diag)
#     diags[network] = np.array(ddd)
#     patterns[network] = np.array(patpat)
#     randoms[network] = np.array(randrand)

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
        #   ['l', 'l', 'j', 'j', 'j']]

vmin, vmax = 0.2, 0.3

cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]
sig_color = "#00BFA6"
sig_color = '#708090'
plot_brains = True

fig, axes = plt.subplot_mosaic(design, figsize=(12, 16), sharey=False, sharex=False, layout="constrained",
                            #    gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
                               gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            'width_ratios': [.2, .2, .5, .5, .5]})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
# fig.suptitle("Comparison of Pattern, Random, and Contrast Accuracy Over Time", fontsize=14, fontweight='bold')
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

# fig, axes = plt.subplot_mosaic(design, figsize=(12, 18), sharey=False, sharex=False, layout="constrained",
#                                gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
#                                             'width_ratios': [.2, .2, .5, .5, .5]})

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

    # Hide labels but keep minor ticks on the left side
    # axes[pattern_idx].tick_params(axis='y', which='both', left=True, labelleft=False)

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
    
    if msig[i]:
        print(f"Significant contrast for {network} at time {times[win].mean():.2f} s")
        rect = plt.Rectangle([-0.75, -0.75], 0.75, 0.75, fill=False, edgecolor="black", linestyle='--', lw=1, zorder=10)
        axes[contrast_idx].add_patch(rect)
        axes[contrast_idx].text(-0.8, -0.5, "*", fontsize=20, color="black", ha='right', va='center', weight='normal')

# percept = []
# mean_net_sess = []
# mean_diag_sess = []
# preact = []
# filt = np.where((times >= 0.25) & (times <= 0.6))[0]
# filter_time = np.where((times >= -0.5) & (times < 0))[0]  # Correct filtering condition
# for i, net in enumerate(networks):
#     contrast = contrasts[net]
#     # Compute mean net effect
#     preact.append(contrast[:, 3:, filter_time][:, 3:, :, filter_time].mean())
#     # Mean diagonal for the specific time window with absolute values
#     # percept.append((all_diags[net][:, filt].mean()))
#     diag = np.diag(contrast[:, 3:, filt]).mean()
#     percept.append(diag)
#     sess1, sess2 = [], []
#     for sub in range(len(subjects)):
#         # Mean for session 1 and 2 per subject
#         # sess1.append(contrast[sub, filter_time][:, filter_time].mean())
#         sess1.append(np.diag(contrast[sub, 3:, filter_time][3:, :, filter_time]))
#         sess2.append(all_diags[net][sub, filt].mean())
#     mean_net_sess.append(np.array(sess1))
#     mean_diag_sess.append(np.array(sess2))
# mean_net_sess = np.array(mean_net_sess).mean(-1)
# mean_diag_sess = np.array(mean_diag_sess)
# sem_net = np.std(mean_net_sess, axis=1) / np.sqrt(len(mean_net_sess[1]))  # SEM for preact
# sem_diag = np.std(mean_diag_sess, axis=1) / np.sqrt(len(mean_diag_sess[1]))  # SEM for percept

# ### Plot mean effect ### 
# alpha = 0.05
# c1 = '#FF0000'  # Blue
# c2 = '#5BBCD6'  # Orange
# # Perform statistical tests for preact (one-sample t-test against 0)
# mean_net_significance = [ttest_1samp(data, 0)[1] < alpha for data in mean_net_sess]

# # axes['d'].grid(axis='y', linestyle='--', alpha=0.3)
# ymin = -0.03
# ymax = 0.03
# x = np.arange(len(networks))
# spacing = 0.35
# # rects1 = axes['j'].bar(x + spacing/2, preact, spacing, label='Pre-activation', facecolor=c1, alpha=.8, edgecolor=None)
# # rects2 = axes['j'].bar(x - spacing/2, percept, spacing, label='Perception', facecolor=c2, alpha=.8, edgecolor=None)

# rects1 = axes['j'].bar(x, preact, spacing, label='Pre-activation', facecolor=c1, alpha=.8, edgecolor=None)
# rects2 = axes['j'].bar(x, percept, spacing, label='Perception', facecolor=c2, alpha=.8, edgecolor=None)


# # axes['d'].errorbar(x - spacing/2, percept, yerr=sem_net, fmt='none', color='black', capsize=5)
# # axes['d'].errorbar(x + spacing/2, preact, yerr=sem_diag, fmt='none', color='black', capsize=5)
# # Annotate significant preact bars with an asterisk
# # for i, rect in enumerate(rects1):
# #     if mean_net_significance[i]:
# #         axes['j'].annotate('*',
# #                 xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
# #                 xytext=(0, 9),
# #                 textcoords="offset points",
# #                 ha='center', va='bottom',
# #                 fontsize=15, color='black')

# # axes['j'].errorbar(x - spacing/2, percept, yerr=sem_net, fmt='none', color='black', capsize=3)
# # axes['j'].errorbar(x + spacing/2, preact, yerr=sem_diag, fmt='none', color='black', capsize=3)

# axes['j'].errorbar(x, percept, yerr=sem_net, fmt='none', color='black', capsize=3)
# axes['j'].errorbar(x, preact, yerr=sem_diag, fmt='none', color='black', capsize=3)

# axes['j'].set_xticks(x)
# names = [name.replace(' ', '\n') for name in network_names]
# axes['j'].set_xticklabels(names, rotation=45, ha='right')
# axes['j'].axhline(0, color='grey', linewidth=2, zorder=10)
# axes['j'].set_title('Predictive coding during pre-stimulus period and perception', fontsize=12, pad=-20)
# axes['j'].set_ylabel('Mean effect', labelpad=-20)
# # axes['j'].set_ylim(ymin, ymax)
# axes['j'].set_yticks([ymin, ymax])
# # axes['j'].legend(frameon=False, loc='upper left', fontsize=9)
# axes['j'].spines['top'].set_visible(False)
# axes['j'].spines['right'].set_visible(False)

# # example plot
# axes['l'].axhline(0, color='k', alpha=.5)
# axes['l'].axvline(0, color='k', alpha=.5)
# axes['l'].set_xlim(times[0], times[-1])
# axes['l'].set_ylim(times[0], times[-1])
# axes['l'].set_yticks([-1, 0, 1])
# axes['l'].plot(times[filter_time], times[filter_time], color=c1, lw=3, label='Pre-activation')
# axes['l'].plot(times[filt], times[filt], color=c2, lw=3, label='Perception')
# axes['l'].set_xlabel("Testing time (s)")
# axes['l'].set_ylabel("Training time (s)")
# axes['l'].legend(loc='upper left', fontsize=9)
# axes['l'].set_aspect(0.5)

# # Hide all None areas
# for key, ax in axes.items():
#     if key is None:
#         ax.set_visible(False)

fig.savefig(figures_dir / "timeg-final-test.pdf", transparent=True)
# plt.close()

fns = []
max_where = []
f = np.where((times >= 0.1) & (times <= 1))[0]
fig, axes = plt.subplots(2, 5, figsize=(20, 4), sharey=True, layout='tight')
for i, (label, ax) in enumerate(zip(networks, axes.flatten())):
    ax.set_title(network_names[i])
    ax.axhline(0, color='k', alpha=.5)
    ax.plot(times[f], all_diags[label][:, f].mean(0), color=cmap[i], alpha=1)
    smoothed_curve = gaussian_filter1d(all_diags[label][:, f].mean(0), sigma=2)
    ax.plot(times[f], smoothed_curve, color='#FF0000', lw=1.5)
    
    fnegative = np.where(smoothed_curve < 0)[0][0]
    # ax.axvline(times[f][fnegative], color='k', alpha=1)
    fns.append(times[f][fnegative])
    
    mwhre = np.where(smoothed_curve == np.max(smoothed_curve))[0][0]
    max_where.append(times[f][mwhre])
    # ax.axvline(times[f][mwhre], color='k', alpha=1)
    ax.axvline(0.6, color='k', alpha=.5)

mmax_where = np.mean(max_where)
print(f"Mean max peak: {mmax_where:.2f} s")    
    
mfns = np.mean(fns)
print(f"Mean first negative peak: {mfns:.2f} s")

percpt_f = np.where((times >= 0.25) & (times <= 0.6))[0]
preact_f = np.where((times >= -0.5) & (times < 0))[0]  # Correct filtering condition

fig, ax = plt.subplots(figsize=(10, 5), layout='tight')
for i, _ in enumerate(networks):
    ax.scatter(percept[i], preact[i], color=cmap[i])
    ax.annotate(network_names[i], (percept[i] + 0.0001, preact[i]), fontsize=14, ha='left', va='center')
# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(percept, preact)
# Plot the linear fit
x_vals = np.linspace(min(percept), max(percept), 100)
y_vals = slope * x_vals + intercept
ax.plot(x_vals, y_vals, color='black', linestyle='--', label=f'Linear fit (R={r_value:.2f}, p={p_value:.3f})')
# Add legend
ax.legend(fontsize=12)
ax.set_xlabel("Perception", fontsize=14)
ax.set_ylabel("Pre-activation", fontsize=14)

sub_percept, sub_preact = [], []
for sub, _ in enumerate(subjects):
    sub_percept.append(np.array([all_diags[net][sub, percpt_f].mean() for net in networks]))
    sub_preact.append(np.array([all_diags[net][sub, preact_f].mean() for net in networks]))
sub_percept, sub_preact = np.array(sub_percept), np.array(sub_preact)

rhos = []
for sub in range(len(subjects)):
    rho, _ = spearmanr(sub_percept[sub], sub_preact[sub])
    rhos.append(rho)
pval = ttest_1samp(rhos, 0)[1]

def fisher_tranform(r):
    return 0.5 * np.log((1 + r) / (1 - r))

from scipy.stats import pearsonr
prhos = []
for sub in range(len(subjects)):
    # Compute the Pearson correlation coefficient
    r, _ = pearsonr(sub_percept[sub], sub_preact[sub])
    r = fisher_tranform(r)
    prhos.append(r)
ppval = ttest_1samp(prhos, 0)[1]

print(f"Spearman correlation: {np.mean(rhos):.2f} ± {np.std(rhos):.2f}, p = {pval:.3f}")
print(f"Pearson correlation: {np.mean(prhos):.2f} ± {np.std(prhos):.2f}, p = {ppval:.3f}")