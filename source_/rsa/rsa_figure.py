import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import *
from config import *
from mne import read_epochs
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from cycler import cycler
import seaborn as sns

lock = 'stim'
# analysis = 'usual'
analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
# analysis = 'rdm_high_rdm_low'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
subjects_dir = FREESURFER_DIR
trial_type = 'all'
# get times
times = np.load(data_path / "times.npy")

# labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)
n_parcels = 200
n_networks = 7
networks = schaefer_7[:-2] if n_networks == 7 else schaefer_17
networks += ['Hippocampus', 'Thalamus']
names_corrected = pd.read_csv(FREESURFER_DIR / 'Schaefer2018' / f'{n_networks}NetworksOrderedNames.csv', header=0)[' Network Name'].tolist()[:-2] 
names_corrected += ['Hippocampus', 'Thalamus']
label_names = schaefer_7[:-2] if n_networks == 7 else schaefer_17
label_names += ['Hippocampus', 'Thalamus']

figures_dir = FIGURES_DIR / "RSA" / "source" / lock
ensure_dir(figures_dir)

decoding = {}
all_highs, all_lows = {}, {}
diff_sess = {}
for network in networks:
    print(f"Processing {network}...")
    if not network in decoding:
        decoding[network] = []
        all_highs[network] = []
        all_lows[network] = []
        diff_sess[network] = []
    for subject in subjects:        
        score = np.load(RESULTS_DIR / "RSA" / 'source' / network / lock / 'scores' / trial_type / f"{subject}-all-scores.npy")
        decoding[network].append(score)
        # # RSA stuff
        behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject))
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / "RSA" / 'source' / network / lock / 'rdm' / subject
        high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
        all_highs[network].append(high)    
        all_lows[network].append(low)
    decoding[network] = np.array(decoding[network])
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    # plot diff session by session
    for i in range(5):
        rev_low = all_lows[network][:, :, i, :].mean(1) - all_lows[network][:, :, 0, :].mean(axis=1)
        rev_high = all_highs[network][:, :, i, :].mean(1) - all_highs[network][:, :, 0, :].mean(axis=1)
        diff_sess[network].append(rev_low - rev_high)
    diff_sess[network] = np.array(diff_sess[network]).swapaxes(0, 1)
    
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
chance = 25
threshold = .05
design = [['br11', 'br12', 'A', 'B', 'C'], 
          ['br21', 'br22', 'D', 'E', 'F'],
          ['br31', 'br32', 'G', 'H', 'I'],
          ['br41', 'br42', 'J', 'K', 'L'], 
          ['br51', 'br52', 'M', 'N', 'O'],
          ['br61', 'br62', 'P', 'Q', 'R'], 
          ['br71', 'br72', 'S', 'T', 'U']]

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
cmap = plt.cm.get_cmap('tab20', len(label_names))
cmap = sns.color_palette("colorblind", as_cmap=True)

def plot_onset(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if lock == 'stim':
        ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    else:
        ax.axvline(0, color='black', label='Button press')
        
def plot_with_br_title(fig, axd, design, title, row_idx, fontsize=14):
    """Add a central title for a specific row of 'br' plots."""
    # Extract only "br" subplot positions for the specific row
    br_axes = [axd[key] for key in design[row_idx] if "br" in key]
    # Compute the central x-position and top y-position for the row
    x_center = sum([ax.get_position().x0 + ax.get_position().x1 for ax in br_axes]) / (2 * len(br_axes))
    y_top = max([ax.get_position().y1 for ax in br_axes])
    # Place the title above the selected plots for the given row
    fig.text(x_center, y_top, title, ha='center', va='top', fontsize=fontsize, fontweight='bold')

def crop_images(screenshot):
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    return cropped_screenshot

fig, axd = plt.subplot_mosaic(
    design, 
    sharex=False, 
    figsize=(13, 16), 
    layout='tight',
    gridspec_kw={
        # 'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5, 1],  # Adjust heights
        'width_ratios': [.5, .4, .5, .5, .5]  # Adjust widths if needed
    })
### Plot brain ###
brain_kwargs = dict(hemi='both', background="white", cortex="low_contrast", surf='inflated', subjects_dir=subjects_dir, size=(800, 400))
for i, (label, name, sideA, sideB) in enumerate(zip(label_names, names_corrected, ['br11', 'br21', 'br31', 'br41', 'br51', 'br61', 'br71'], ['br12', 'br22', 'br32', 'br42', 'br52', 'br62', 'br72'])):
    # Initialize Brain object
    # Add labels
    if label in label_names[:-2]:
        brain = mne.viz.Brain(subject='sub01', alpha=1, **brain_kwargs) 
        net_name = f'{name.strip()} network'        
        for hemi in ['lh', 'rh']:
            brain.add_label(f'{label}', color=cmap[i], hemi=hemi, borders=False, alpha=.85, subdir='n7')
    else:
        brain = mne.viz.Brain(subject='sub01', alpha=.5, **brain_kwargs) 
        net_name = f'{name.strip()}'
        labels = ['Left-Hippocampus', 'Right-Hippocampus'] if label == 'Hippocampus' else ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
        brain.add_volume_labels(aseg='aseg', labels=labels, colors=cmap[i], alpha=.85, legend=False)
    # Capture snapshots for the desired views
    brain.show_view('lateral', distance="auto")
    lateral_img = brain.screenshot('rgb')
    brain.show_view('caudal', distance="auto")
    caudal_img = brain.screenshot('rgb')
    brain.close()
    
    lateral_img, caudal_img = crop_images(lateral_img), crop_images(caudal_img)
    
    # Display images side by side using Matplotlib
    axd[sideA].imshow(lateral_img)
    axd[sideA].axis('off')
    # axd[sideA].set_title(net_name, fontsize=14)
    # Call the function to set the br-specific title
    net_name = " "
    plot_with_br_title(fig, axd, design, net_name, row_idx=i)
    axd[sideB].imshow(caudal_img)
    axd[sideB].axis('off')
    # break

### Plot decoding ###
for i, (label, name, j) in enumerate(zip(label_names, names_corrected, ['A', 'D', 'G', 'J', 'M', 'P', 'S'])):
    plot_onset(axd[j])
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    # Get significant clusters
    p_values = decod_stats(score - chance, jobs)
    sig = p_values < threshold
    # Main plot
    axd[j].plot(times, score.mean(0), alpha=1, label='Random - Pattern', zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axd[j].plot(times[start:end], score.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    axd[j].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axd[j].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])    
    axd[j].fill_between(times, score.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])    
    axd[j].axhline(chance, color='grey', alpha=.5)
    axd[j].set_ylabel('Accuracy (%)', fontsize=11)
    axd[j].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    axd[j].xaxis.set_major_locator(plt.MultipleLocator(0.2))
    if j == 'A':
        axd[j].set_title('Time course decoding')
    if j == 'S':
        axd[j].set_xlabel('Time (s)', fontsize=11)
### Plot similarity index ###    
for i, (label, name, j) in enumerate(zip(label_names, names_corrected, ['B', 'E', 'H', 'K', 'N', 'Q', 'T'])):
    plot_onset(axd[j])
    axd[j].axhline(0, color='grey', alpha=.5)
    high = all_highs[label][:, :, 1:, :].mean((1, 2)) - all_highs[label][:, :, 0, :].mean(1)
    low = all_lows[label][:, :, 1:, :].mean((1, 2)) - all_lows[label][:, :, 0, :].mean(axis=1)
    diff = low - high
    p_values = decod_stats(diff, jobs)
    sig = p_values < threshold
    # Main plot
    axd[j].plot(times, diff.mean(0), alpha=1, label='Random - Pattern', zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axd[j].plot(times[start:end], diff.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(diff, axis=0) / np.sqrt(len(subjects))
    axd[j].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axd[j].fill_between(times, diff.mean(0) - sem, diff.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    axd[j].fill_between(times, diff.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    axd[j].set_ylabel('Sim. index', fontsize=11)
    axd[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axd[j].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    axd[j].xaxis.set_major_locator(plt.MultipleLocator(0.2))
    # axd[j].set_xticklabels([])
    if j == 'T':
        axd[j].set_xlabel('Time (s)', fontsize=11)
    if j == 'B':
        axd[j].set_title(f'Average similarity index time course')
### Plot subject x learning index correlation ###
for i, (label, name, j) in enumerate(zip(label_names, names_corrected, ['C', 'F', 'I', 'L', 'O', 'R', 'U'])):
    plot_onset(axd[j])
    axd[j].axhline(0, color="grey", alpha=0.5)
    all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], diff_sess[label][sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    # axd[j].plot(times, all_rhos.mean(0), color=cmap[i])
    p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(all_rhos, -1)
    sig = p_values < 0.05
    # Main plot
    axd[j].plot(times, all_rhos.mean(0), alpha=1, label='Random - Pattern', zorder=10, color='C7')
    # Plot significant regions separately
    for start, end in contiguous_regions(sig):
        axd[j].plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
    sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
    axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
    # Highlight significant regions
    axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=cmap[i], alpha=0.2)
    axd[j].fill_between(times, 0, all_rhos.mean(0) - sem, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")    
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=cmap[i], alpha=0.2)
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
    axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=.4, facecolor="#F2AD00", label='Significance - corrected')
    axd[j].set_ylabel("Spearman's rho", fontsize=11)
    axd[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    axd[j].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    axd[j].xaxis.set_major_locator(plt.MultipleLocator(0.2))
    if j == 'U':
        axd[j].set_xlabel('Time (s)', fontsize=11)
    # axd[j].legend(frameon=False, loc="lower right")
    if j == 'C':
        axd[j].set_title(f'Similarity and learning correlation')
        
fig.savefig(figures_dir / f"{lock}-rsa-tight-no-title.pdf", transparent=True)
plt.close()


# # Set parameters
# network = "Vis"
# subdir = 'n7'
# # Visualization parameters
# brain_kwargs = dict(alpha=.5, background="white", cortex="low_contrast")
# # Initialize Brain object
# brain = mne.viz.Brain(subject=subject, hemi='both', surf='inflated', 
#                        subjects_dir=subjects_dir, **brain_kwargs)
# # Add labels
# for hemi in ['lh', 'rh']:
#     brain.add_label(f'{network}', color=cmap[0], hemi=hemi, borders=False, alpha=.85, subdir='n7')
# # Add volume labels
# brain.add_volume_labels(aseg='aseg', labels=['Left-Hippocampus', 'Right-Hippocampus'], colors=cmap[i], alpha=1, legend=False)
# # Capture snapshots for the desired views
# brain.show_view('lateral')
# lateral_img = brain.screenshot()
# brain.show_view('caudal')
# caudal_img = brain.screenshot()
# brain.close()
# # Display images side by side using Matplotlib
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(lateral_img)
# axes[0].axis('off')
# axes[0].set_title('Visual\nNetwork', fontsize=14)
# axes[1].imshow(caudal_img)
# axes[1].axis('off')
# plt.show()
