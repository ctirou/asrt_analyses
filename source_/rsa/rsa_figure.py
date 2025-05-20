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
# from surfer import Brain

lock = 'stim'
# analysis = 'usual'
analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
# analysis = 'rdm_high_rdm_low'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS15, EPOCHS
subjects_dir = FREESURFER_DIR

# get times
times = np.linspace(-0.2, 0.6, 82)
timesg = np.linspace(-1.5, 1.5, 307)

# labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)
networks = NETWORKS + ['Cerebellum-Cortex']
network_names = NETWORK_NAMES + ['Cerebellum']

figures_dir = ensured(FIGURES_DIR / "RSA" / "source")
# --------- Shuffled ---------
# Load RSA data
filt = np.where((timesg >= -0.2) & (timesg <= 0.6))[0]
all_highs, all_lows = {}, {}
diff_sess = {}
pattern, random = {}, {}
for network in tqdm(networks):
    if not network in all_highs:
        all_highs[network], all_lows[network] = [], []
        diff_sess[network] = []
        pattern[network], random[network] = [], []
    for subject in subjects:        
        # RSA stuff
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / 'rdm_skf' / subject
        pats, rands = [], []
        for epoch_num in range(5):
            pats.append(np.load(res_path / f"pat-{epoch_num}.npy"))
            rands.append(np.load(res_path / f"rand-{epoch_num}.npy"))
        pats = np.array(pats)
        rands = np.array(rands)
        high, low = get_all_high_low(pats, rands, sequence, False)
        high, low = np.array(high).mean(0), np.array(low).mean(0)
        if subject == 'sub05':
            pat_b1 = np.load(res_path / "pat-b1.npy")
            high[0] = pat_b1.copy()
            rand_b1 = np.load(res_path / "rand-b1.npy")
            low[0] = rand_b1.copy()
        
        all_highs[network].append(high)
        all_lows[network].append(low)
        # Decoding stuff
        res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / 'scores_skf' / subject
        pat = np.diag(np.load(res_path / "pat-all.npy"))[filt]
        rand = np.diag(np.load(res_path / "rand-all.npy"))[filt]
        pattern[network].append(pat)
        random[network].append(rand)
    all_highs[network] = np.array(all_highs[network])
    all_lows[network] = np.array(all_lows[network])
    
    # plot diff session by session
    for i in range(5):
        low_sess = all_lows[network][:, i, :] - all_lows[network][:, 0, :]
        high_sess = all_highs[network][:, i, :] - all_highs[network][:, 0, :]
        diff_sess[network].append(low_sess - high_sess)
    diff_sess[network] = np.array(diff_sess[network]).swapaxes(0, 1)
    
    pattern[network] = np.array(pattern[network])
    random[network] = np.array(random[network])

learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)
chance = 25
threshold = .05

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

design = [['br11', 'br12', [['A1'], ['A2']], 'B', 'C'], 
          ['br21', 'br22', [['D1'], ['D2']], 'E', 'F'],
          ['br31', 'br32', [['G1'], ['G2']], 'H', 'I'],
          ['br41', 'br42', [['J1'], ['J2']], 'K', 'L'], 
          ['br51', 'br52', [['M1'], ['M2']], 'N', 'O'],
          ['br61', 'br62', [['P1'], ['P2']], 'Q', 'R'], 
          ['br71', 'br72', [['S1'], ['S2']], 'T', 'U'],
          ['br81', 'br82', [['V1'], ['V2']], 'W', 'X'],
          ['br91', 'br92', [['Y1'], ['Y2']], 'Z', 'AA'],
          ['br101', 'br102', [['AB1'], ['AB2']], 'AC', 'AD']]

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
cmap = plt.cm.get_cmap('tab20', len(network_names))
cmap = sns.color_palette("colorblind", as_cmap=True)
cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9', "#76B041"]

fig, axd = plt.subplot_mosaic(
    design, 
    sharex=False, 
    figsize=(13, 18), 
    layout='tight',
    gridspec_kw={
        # 'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5, 1],  # Adjust heights
        'width_ratios': [.2, .2, .5, .5, .5]  # Adjust widths if needed
    })
### Plot brain ###
brain_kwargs = dict(hemi='both', background="white", cortex="low_contrast", surf='inflated', subjects_dir=subjects_dir, size=(800, 400))
for i, (label, name, sideA, sideB) in enumerate(zip(networks, network_names, \
    ['br11', 'br21', 'br31', 'br41', 'br51', 'br61', 'br71', 'br81', 'br91', 'br101'], ['br12', 'br22', 'br32', 'br42', 'br52', 'br62', 'br72', 'br82', 'br92', 'br102'])):
    # Initialize Brain object
    # Add labels
    if label in networks[:-3]:
        
        brain = mne.viz.Brain(subject='fsaverage2', alpha=1, **brain_kwargs) 
        net_name = f'{name.strip()} network'        
        for hemi in ['lh', 'rh']:
        # hemi = 'split'
            brain.add_label(f'{label}', color=cmap[i], hemi=hemi, borders=False, alpha=.85, subdir='n7')
    else:
        brain = mne.viz.Brain(subject='fsaverage2', alpha=.5, **brain_kwargs) 
        net_name = f'{name.strip()}'
        
        if label == 'Hippocampus':
            labels = ['Left-Hippocampus', 'Right-Hippocampus']
        elif label == 'Thalamus':
            labels = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
        else:
            labels = ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']
                    
        brain.add_volume_labels(aseg='aparc+aseg', labels=labels, colors=cmap[i], alpha=.85, legend=False)
    
    # brain = mne.viz.Brain(subject='fsaverage2', alpha=.5, **brain_kwargs) 
    # for i, label in enumerate(['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']):
    #     if label == 'Hippocampus':
    #         labels = ['Left-Hippocampus', 'Right-Hippocampus']
    #     elif label == 'Thalamus':
    #         labels = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
    #     else:
    #         labels = ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']
    #     brain.add_volume_labels(aseg='aseg', labels=labels, colors=cmap[i], alpha=.85, legend=False)
    
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
    axd[sideA].imshow(lateral_img)
    axd[sideA].axis('off')
    # axd[sideA].set_title(net_name, fontsize=14)
    # Call the function to set the br-specific title
    net_name = " "
    plot_with_br_title(fig, axd, design, net_name, row_idx=i)
    axd[sideB].imshow(medial_img)
    axd[sideB].axis('off')

### Plot decoding ###
for i, (label, name, j, k) in enumerate(zip(networks, network_names,  \
    ['A1', 'D1', 'G1', 'J1', 'M1', 'P1', 'S1', 'V1', 'Y1', 'AB1'], ['A2', 'D2', 'G2', 'J2', 'M2', 'P2', 'S2', 'V2', 'Y2', 'AB2'])):
    for l, data in zip([j, k], [pattern, random]):
        plot_onset(axd[l])
        score = data[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        # Get significant clusters
        p_values = decod_stats(score - chance, jobs)
        sig = p_values < threshold
        # Main plot
        axd[l].plot(times, score.mean(0), alpha=1, zorder=10, color='C7')
        # Plot significant regions separately
        for start, end in contiguous_regions(sig):
            axd[l].plot(times[start:end], score.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        axd[l].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
        # Highlight significant regions
        axd[l].fill_between(times, score.mean(0) - sem, score.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])    
        axd[l].fill_between(times, score.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])    
        axd[l].axhline(chance, color='grey', alpha=.5)
        axd[l].set_ylabel('Acc. (%)', fontsize=11)
        # axd[l].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
        # axd[l].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        if l == 'A1':
            axd[l].set_title('Decoding')
            ymax = axd[l].get_ylim()[1]
        # elif l == 'A2':
        elif l == 'AB2':
            axd[l].set_xlabel('Time (s)', fontsize=11)
        if l == j:
            axd[l].set_xticklabels([])
            # axd[l].figure.subplots_adjust(hspace=0)
        if '1' in l:
            axd[l].text(times[0], ymax - 0.2 * ymax, 'Pattern', fontsize=9, ha='left', weight='normal', style='italic')
        else:
            axd[l].text(times[0], ymax - 0.2 * ymax, 'Random', fontsize=9, ha='left', weight='normal', style='italic')
        axd[l].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
        axd[l].yaxis.set_major_locator(plt.MaxNLocator(nbins=2, prune='both'))
        axd[l].set_ylim(23, 50)
        
### Plot similarity index ###    
for i, (label, name, j) in enumerate(zip(networks, network_names, ['B', 'E', 'H', 'K', 'N', 'Q', 'T', 'W', 'Z', 'AC'])):
    plot_onset(axd[j])
    axd[j].axhline(0, color='grey', alpha=.5)
    high = all_highs[label][:, 1:, :].mean(1) - all_highs[label][:, 0, :]
    low = all_lows[label][:, 1:, :].mean(1) - all_lows[label][:, 0, :]
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
    if j == 'AC':
        axd[j].set_xlabel('Time (s)', fontsize=11)
    if j == 'B':
        axd[j].set_title(f'Similarity index')
    axd[j].set_ylim(-1.5, 1.5)

### Plot subject x learning index correlation ###
for i, (label, name, j) in enumerate(zip(networks, network_names, ['C', 'F', 'I', 'L', 'O', 'R', 'U', 'X', 'AA', 'AD'])):
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
    # axd[j].fill_between(times, 0, all_rhos.mean(0) - sem, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")    
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, color=cmap[i], alpha=0.2)
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig_unc, alpha=.3, label='Significance - uncorrected', facecolor="#7294D4")
    # axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=.4, facecolor="#F2AD00", label='Significance - corrected')
    axd[j].set_ylabel("Rho", fontsize=11)
    axd[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axd[j].fill_between(times, all_rhos.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
    axd[j].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    axd[j].xaxis.set_major_locator(plt.MultipleLocator(0.2))
    if j == 'AD':
        axd[j].set_xlabel('Time (s)', fontsize=11)
    # axd[j].legend(frameon=False, loc="lower right")
    if j == 'C':
        axd[j].set_title('Similarity index\n& learning correlation')
    axd[j].set_ylim(-.5, .5)
    axd[j].set_yticks([-1, 0, 1])

fig.savefig(figures_dir / "rsa-final.pdf", transparent=True)
plt.close()