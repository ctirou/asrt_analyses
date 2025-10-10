# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp
import pandas as pd
from tqdm.auto import tqdm
from mne.viz import Brain

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

networks = NETWORKS
network_names = NETWORK_NAMES
times = np.linspace(-1.5, 1.5, 307)
chance = 25
threshold = .05
res_dir = RESULTS_DIR / 'TIMEG' / 'source'

data_type = 'scores_blocks'
patterns = {}
randoms = {}
contrasts = {}
win4 = np.where((times >= -1))[0]
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
            pattern.append(np.diag(np.load(pfname)))
            random.append(np.diag(np.load(rfname)))
        if subject == 'sub05':
            pat_bsl = np.diag(np.load(res_path / "pat-4.npy")) if network in networks[:-3] else np.diag(np.load(res_path / "pat-4-4.npy"))
            rand_bsl = np.diag(np.load(res_path / "rand-4.npy")) if network in networks[:-3] else np.diag(np.load(res_path / "rand-4-4.npy"))
            for i in range(3):
                pattern[i] = pat_bsl.copy()
                random[i] = rand_bsl.copy()
        pats_blocks.append(np.array(pattern))
        rands_blocks.append(np.array(random))
    pats_blocks, rands_blocks = np.array(pats_blocks), np.array(rands_blocks)
    patterns[network] = pats_blocks[:, :, win4] * 100
    randoms[network] = rands_blocks[:, :, win4] * 100
    contrasts[network] = patterns[network] - randoms[network]

times = times[win4]

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

def plot_onset(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)

# Define subplot design layout
design = [['br11', 'br12', 'A', 'B', 'C'], 
          ['br21', 'br22', 'D', 'E', 'F'],
          ['br31', 'br32', 'G', 'H', 'I'],
          ['br41', 'br42', 'J', 'K', 'L'], 
          ['br51', 'br52', 'M', 'N', 'O'],
          ['br61', 'br62', 'P', 'Q', 'R'], 
          ['br71', 'br72', 'S', 'T', 'U'],
          ['br81', 'br82', 'V', 'W', 'X'],
          ['br91', 'br92', 'Y', 'Z', 'AA'],
          ['br101', 'br102', 'AB', 'AC', 'AD']]

cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]

plot_brains = True

fig, axes = plt.subplot_mosaic(design, figsize=(13, 18), sharey=False, sharex=False, layout="tight",
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
        # Display images side by side using Matplotlib
        axes[sideA].imshow(lateral_img)
        axes[sideA].axis('off')
        # Call the function to set the br-specific title
        net_name = " "
        axes[sideB].imshow(medial_img)
        axes[sideB].axis('off')

### Pattern ###
# Sig from GAMM
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_pat_rand_source.csv")
seg_df = seg_df[seg_df['metric'] == 'PATTERN']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_pat_rand_source.csv")
sig_df = sig_df[sig_df['metric'] == 'PATTERN']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
            del sig_dict[net]

for i, (network, pattern_idx) in enumerate(zip(networks, ['A', 'D', 'G', 'J', 'M', 'P', 'S', 'V', 'Y', 'AB'])):
    plot_onset(axes[pattern_idx])
    data = patterns[network][:, 3:].mean(1)
    axes[pattern_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    axes[pattern_idx].axhline(chance, color='grey', alpha=.5)
    # Get significant clusters
    sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
    # Main plot
    axes[pattern_idx].plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axes[pattern_idx].plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(data, axis=0) / np.sqrt(len(subjects))
    axes[pattern_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axes[pattern_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    axes[pattern_idx].fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    axes[pattern_idx].set_ylabel('Acc. (%)', fontsize=11)
    axes[pattern_idx].set_ylim(23, 35)
    axes[pattern_idx].set_xticks(np.arange(-1, 2, 0.5))
    axes[pattern_idx].set_yticks(np.arange(25, 36, 5))
    axes[pattern_idx].set_yticklabels(np.arange(25, 36, 5))
    if pattern_idx == 'A':
        axes[pattern_idx].set_title('Pattern')
    elif pattern_idx == 'AB':
        axes[pattern_idx].set_xlabel('Time (s)', fontsize=11)
    sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
    if sig_level != 'ns':
        axes[pattern_idx].text(0.5, 33, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')
    

### Random ###    
# Sig from GAMM
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_pat_rand_source.csv")
seg_df = seg_df[seg_df['metric'] == 'RANDOM']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_pat_rand_source.csv")
sig_df = sig_df[sig_df['metric'] == 'RANDOM']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i+10] == 'ns':
            del sig_dict[net]

for i, (network, random_idx) in enumerate(zip(networks, ['B', 'E', 'H', 'K', 'N', 'Q', 'T', 'W', 'Z', 'AC'])):
    plot_onset(axes[random_idx])
    data = randoms[network][:, 3:].mean(1)
    axes[random_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    axes[random_idx].axhline(chance, color='grey', alpha=.5)
    # Get significant clusters
    sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
    # Main plot
    axes[random_idx].plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axes[random_idx].plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(data, axis=0) / np.sqrt(len(subjects))
    axes[random_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axes[random_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    axes[random_idx].fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    # axes[random_idx].set_ylabel('Acc. (%)', fontsize=11)
    axes[random_idx].set_ylim(23, 35)
    axes[random_idx].set_xticks(np.arange(-1, 2, 0.5))
    axes[pattern_idx].set_yticks(np.arange(25, 36, 5))
    axes[pattern_idx].set_yticklabels(np.arange(25, 36, 5))
    if random_idx == 'B':
        axes[random_idx].set_title('Random')
    if random_idx == 'AC':
        axes[random_idx].set_xlabel('Time (s)', fontsize=11)
    sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
    if sig_level != 'ns':
        axes[random_idx].text(0.5, 33, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')

### Contrast ###
win = np.where((times >= -0.5) & (times < 0))[0]
msig = []
for network in networks:
    s = []
    for sub in range(len(subjects)):
        cont = contrasts[network][sub, 3:, win].mean()
        s.append(cont)
    sig = ttest_1samp(s, 0, axis=0)[1] < threshold
    if sig:
        print(f"Significant contrast for {network} in the window {times[win][0]} to {times[win][-1]}")
    msig.append(sig)
    
# Sig from GAMM
# get significant time points from GAMM csv --- contrast
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_cont_source.csv")
seg_df = seg_df[seg_df['metric'] == 'PA']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_cont_source.csv")
sig_df = sig_df[sig_df['metric'] == 'PA']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
            del sig_dict[net]

for i, (network, contrast_idx) in enumerate(zip(networks, ['C', 'F', 'I', 'L', 'O', 'R', 'U', 'X', 'AA', 'AD'])):
    plot_onset(axes[contrast_idx])
    data = contrasts[network][:, 3:].mean(1)
    axes[contrast_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    axes[contrast_idx].axhline(0, color='grey', alpha=.5)
    # Get significant clusters
    sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
    # Main plot
    axes[contrast_idx].plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axes[contrast_idx].plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(data, axis=0) / np.sqrt(len(subjects))
    axes[contrast_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axes[contrast_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    axes[contrast_idx].fill_between(times, data.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    axes[contrast_idx].axhline(0, color='grey', alpha=.5)
    axes[contrast_idx].set_ylabel('Diff in acc. (%)', fontsize=11)
    axes[contrast_idx].set_ylim(-2, 5)
    axes[contrast_idx].set_yticks(np.arange(0, 5, 2))
    axes[contrast_idx].set_yticklabels(np.arange(0, 5, 2))
    if contrast_idx == 'C':
        axes[contrast_idx].set_title('Contrast\n(Pattern - Random)')
    elif contrast_idx == 'AD':
        axes[contrast_idx].set_xlabel('Time (s)', fontsize=11)
    sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
    if sig_level != 'ns':
        axes[contrast_idx].text(-0.5, 4, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')

fig.savefig(figures_dir / "timeg-diag.pdf", transparent=True)

# Correlation with behavior
from scipy.stats import spearmanr as spear
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})
fig, axes = plt.subplots(5, 2, figsize=(7, 9), sharey=True, sharex=True, layout="tight")
for i, ax in enumerate(axes.flatten()):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='black', alpha=1)
    ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.2)
    network = networks[i]
    all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], contrasts[network][sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
    # all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    p_values = decod_stats(all_rhos, -1)
    sig = p_values < 0.05
    # Main plot
    ax.plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
        # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        ax.plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.5, zorder=5, facecolor='C7')
    # Highlight significant regions
    ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    ax.set_title(network_names[i], fontsize=13, fontstyle='italic')
    wo = np.where((times >= -0.5) & (times < 0))[0]
    mrho = all_rhos[:, wo].mean(1)
    mrho_sig = ttest_1samp(mrho, 0)[1]
    if mrho_sig < 0.05:
        print(f"Significant correlation for {network} in the window {times[wo][0]} to {times[wo][-1]}")
        ax.axvspan(times[win][0], times[win][-1], facecolor=cmap[i], edgecolor=None, alpha=0.3, zorder=5)
        ax.text(0.4, 0.7, '*', fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')
    if ax in axes[:, 0]:
        ax.set_ylabel("Spearman's rho", fontsize=11)
    # Only set xlabel for axes in the bottom row
    if i >= (axes.shape[0] - 1) * axes.shape[1]:
        ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_xticks(np.arange(-1, 2, 0.5))

fig.savefig(figures_dir / "timeg-corr.pdf", transparent=True)
