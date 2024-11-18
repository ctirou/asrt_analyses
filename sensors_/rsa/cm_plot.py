import os.path as op
import numpy as np
from base import ensure_dir, decod_stats, get_sequence, get_inout_seq
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import ttest_1samp, spearmanr
import gc
from pathlib import Path
from numba import jit
from tqdm.auto import tqdm
from matplotlib.ticker import FuncFormatter

lock = "stim"
trial_type = "pattern"
subjects = SUBJS
analysis = "pred_decoding"
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
verbose = True
hemi = 'both'
chance = .5
overwrite = False
jobs = -1

summary = False

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']

decod_in_lab = {}
decod_in_lab2 = {}
corr = {}
decoding = dict()

# get label names
label_names = SURFACE_LABELS + VOLUME_LABELS if lock == 'stim' else SURFACE_LABELS_RT + VOLUME_LABELS_RT

figures_dir = res_path / "figures_dir" / analysis / 'source' / lock / trial_type
ensure_dir(figures_dir)
gc.collect()

@jit(nopython=True)
def spearman_rank_correlation(x, y):
    # Convert input lists to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    # Get the ranks (add 1 to correct for zero-indexing)
    rank_x = np.argsort(np.argsort(x)) + 1
    rank_y = np.argsort(np.argsort(y)) + 1
    # Calculate the difference in ranks
    d_squared = np.sum((rank_x - rank_y) ** 2)
    # Spearman's rho formula
    rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
    return rho

def format_func(value, tick_number):
    return f'{value:.1f}'

for ilabel, label in enumerate(label_names):
    
    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)
    all_in_seqs, all_out_seqs = [], []
    decoding[label] = list()
    
    for subject in subjects:
        
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()
        
        behav_dir = HOME / "raw_behavs" / subject
        sequence = get_sequence(behav_dir)
                
        sub_scores, sub_rsa, sub_cms = [], [], []
        for session_id, session in enumerate(sessions):
            
            # res_dir = res_path / analysis / 'source' / lock / trial_type / label / subject / session
            # res_dir = res_path / analysis / 'source' / lock / trial_type / label / subject
                    
            # sub_scores.append(np.load(res_dir / "scores.npy"))
            
            # sub_scores.append(np.load(res_path / 'source' / lock / trial_type / label / subject / session / "scores.npy"))
            sub_cms.append(np.load(res_path / analysis / 'source' / lock / trial_type / label / subject / session / "cms.npy") )
            sub_rsa.append(np.load(res_path / analysis / 'source' / lock / trial_type / label / subject / session / "rsa.npy"))
            
        sub_scores = np.array(sub_scores)
        sub_cms = np.array(sub_cms)

        sub_rsa = np.array(sub_rsa)
        one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
        for session_id, _ in enumerate(sessions):
            for sim, sim_list in enumerate([one_four, one_three, one_two, three_four, two_four, two_three]):
                sim_list.append(sub_rsa[session_id, sim, :])
        
        for all_sims, sim_list in zip(
            [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities], 
            [one_two, one_three, one_four, two_three, two_four, three_four]):
                all_sims.append(np.array(sim_list))
                    
        similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]
        in_seq, out_seq = get_inout_seq(sequence, similarities)
        all_in_seqs.append(np.array(in_seq))
        all_out_seqs.append(np.array(out_seq))
        
        if summary:
            from sklearn.metrics import ConfusionMatrixDisplay
            ensure_dir(figures_dir / "summary" / label)
            fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7), sharey=False)
            fig.suptitle(f'{subject} / ${label}$')
            times_win = np.where((times >= 0) & (times <= 0.2))[0]
            for i, (ax1, ax2, session) in enumerate(zip(axs.flat[:5], axs.flat[5:], sessions)):
                ax1.plot(times, sub_scores[i])
                ax1.axvspan(0, 0.2, color='grey', alpha=.2)
                ax1.set_title(session)
                ax1.axhline(chance, color='white', ls='dashed', alpha=.5)
                ax1.set_ylim(0.2, 0.8)
                ax1.grid(True, color='grey', alpha=0.3)
                max_score = np.argmax(sub_scores[i][times_win]) + np.where(times==0)[0][0]
                ax1.annotate(f'Max Score: {sub_scores[i][max_score]:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
                ax1.annotate('', xy=(times[max_score], sub_scores[i][max_score]), xytext=(times[max_score], sub_scores[i][max_score] + 0.1),
                            arrowprops=dict(arrowstyle='->', color='white'))
                disp = ConfusionMatrixDisplay(sub_cms[i, max_score, :, :], display_labels=[1, 2, 3, 4])
                disp.plot(ax=ax2)
                disp.im_.set_clim(0, 1)  # Set colorbar limits
            plt.savefig(figures_dir / "summary" / label / f"{subject}.png")
            plt.close()
            
        decoding[label].append(sub_scores)
    decoding[label] = np.array(decoding[label])
        
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    decod_in_lab[label] = diff_inout
    decod_in_lab2[label] = [np.squeeze(all_in_seq).mean(axis=1), np.squeeze(all_out_seq).mean(axis=1)]
    
    if not op.exists(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy") or overwrite:
        corr_npy = []
        for sub in range(len(subjects)):
            rhos = []
            for t in range(len(times)):
                # rho = spearman_rank_correlation([0, 1, 2, 3, 4], diff_inout[sub, :, t])
                rho = spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t])
                rhos.append(rho)
            corr_npy.append(rhos)
        corr_npy = np.array(corr_npy)
        np.save(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy", corr)
    corr_npy = np.load(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy")
    corr[label] = corr_npy
        
label_names = sorted(SURFACE_LABELS + VOLUME_LABELS, key=str.casefold) if lock == 'stim' else sorted(SURFACE_LABELS_RT + VOLUME_LABELS_RT, key=str.casefold)
figures_dir = FIGURES_DIR / analysis / 'source' / lock / trial_type
ensure_dir(figures_dir)
# define parameters    
chance = 25
ncols = 4
nrows = 10 if lock == 'stim' else 9
far_left = [0] + [i for i in range(0, len(label_names), ncols*2)]
color1, color2 = ("#1982C4", "#74B3CE") if lock == 'stim' else ("#D76A03", "#EC9F05")
color3 = "C7"

# Diff in/out 
for ilabel in tqdm(range(0, len(label_names), 2)):
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    fig.subplots_adjust(hspace=0)
    label = label_names[ilabel]
    ytitle = 0.21 if lock == 'stim' else 0.11
    if label == "Cerebellum-White-Matter-lh":
        xtitle=0.6
        ha='right'
    else:
        xtitle=0.25
        ha='left' 
    axs[1].text(xtitle, ytitle, f"{label.capitalize()[:-3]}",
                fontsize=13, weight='normal', style='italic', ha=ha,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    if ilabel in range(8):
        if lock == 'stim':
            axs[0].text(0.1, 0.22, "$Stimulus$", fontsize=11, zorder=10, ha='center')
        else:
            axs[0].text(0.05, 0.11, "Button press", style='italic', fontsize=11, zorder=10, ha='center')
    for i in range(2):
        axs[i].set_ylim(-0.2, 0.2) if lock == 'stim' else axs[i].set_ylim(-0.1, 0.1)
        yticks = axs[i].get_yticks()
        yticks = yticks[1:-1]  # Remove first and last element
        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels(yticks, fontsize=11)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].axhline(0, color='black', ls='dashed', alpha=.7, zorder=-1)
        axs[i].yaxis.set_major_formatter(FuncFormatter(format_func))  # Set the formatter for y-ticks
        # Add the stimulus span or vertical line
        if lock == 'stim':
            axs[i].axvspan(0, 0.2, color='grey', lw=0, alpha=.2, label="Stimulus")
        else:
            axs[i].axvline(0, color='black', alpha=.5)
        if ilabel in far_left:
            axs[i].set_ylabel("Similarity index", fontsize=11)
        else:
            axs[i].set_yticklabels([])  # Remove y-axis labels for non-left plots
            # axs[i].spines["left"].set_visible(False)
        # if ilabel not in range(len(label_names))[-8:]:
        #     axs[i].spines["bottom"].set_visible(False)
        #     axs[i].get_xaxis().set_visible(False)
        # if ilabel not in far_left:
        #     axs[i].get_yaxis().set_visible(False)
    if ilabel in far_left:
        if lock == 'stim':
            axs[0].text(-0.20, 0.12, "Left\nhemisphere", fontsize=12, color=color1, ha='left', weight='normal', style='italic')
            axs[1].text(-0.20, 0.12, "Right\nhemisphere", fontsize=12, color=color2, ha='left', weight='normal', style='italic')
        else:    
            axs[0].text(-0.20, 0.075, "Left\nhemisphere", fontsize=12, color=color1, ha='left', weight='normal', style='italic')
            axs[1].text(-0.20, 0.075, "Right\nhemisphere", fontsize=12, color=color2, ha='left', weight='normal', style='italic')
    # Show the x-axis label only on the bottom row
    if ilabel in range(len(label_names))[-8:]:
        axs[1].get_xaxis().set_visible(True)
        axs[1].set_xlabel("Time (s)", fontsize=11)
    else:
        axs[1].set_xticklabels([])
    # First curve
    practice = np.array(decod_in_lab[label][:, 0, :], dtype=float) * (-1)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    learning = np.array(decod_in_lab[label][:, 1:, :], dtype=float) * (-1)
    diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
    diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
    diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
    diff = learning.mean(1) - practice
    p_values = decod_stats(diff, jobs)
    sig = p_values < 0.05
    axs[0].fill_between(times, prac_m1, prac_m2, facecolor=color3, alpha=.5, label='Pre-learning')
    axs[0].fill_between(times, diff_m1, diff_m2, facecolor=color1, alpha=.8, label='Learning')
    axs[0].fill_between(times, diff_m1, diff_m2, where=sig, color='black', alpha=1)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].xaxis.set_ticks_position('none')  # Remove x-ticks on the upper plot
    axs[0].xaxis.set_tick_params(labelbottom=False)  # Remove x-tick labels on the upper plot
    # Second curve
    label = label_names[ilabel+1]
    practice = np.array(decod_in_lab[label][:, 0, :], dtype=float) * (-1)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    learning = np.array(decod_in_lab[label][:, 1:, :], dtype=float) * (-1)
    diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
    diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
    diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
    axs[1].fill_between(times, prac_m1, prac_m2, facecolor=color3, alpha=.5, label='Pre-learning')
    axs[1].fill_between(times, diff_m1, diff_m2, facecolor=color2, alpha=.8, label='Learning')
    axs[1].fill_between(times, diff_m1, diff_m2, where=sig, color='black', alpha=1)
    diff = learning.mean(1) - practice
    p_values = decod_stats(diff, jobs)
    sig = p_values < 0.05
    # save figure
    plt.savefig(figures_dir / f'{ilabel}_{label}.pdf', transparent=True)
    plt.savefig(figures_dir / f'{ilabel}_{label}.png', dpi='figure', transparent=True)
    plt.close()

# Correlations
for ilabel in tqdm(range(0, len(label_names), 2)):
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    fig.subplots_adjust(hspace=0)
    label = label_names[ilabel]
    axs[1].text(0.25, 1, f"{label.capitalize()[:-3]}",
                fontsize=9, weight='normal', style='italic', ha='left',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    if ilabel in range(8):
        if lock == 'stim':
            axs[0].text(0.1, 1.1, "$Stimulus$", fontsize=9, zorder=10, ha='center')
        else:
            axs[0].text(0.05, 1.1, "Button press", style='italic', fontsize=9, zorder=10, ha='center')
    for i in range(2):
        axs[i].set_ylim(-1, 1)
        yticks = axs[i].get_yticks()
        yticks = yticks[1:-1]  # Remove first and last element
        axs[i].set_yticks(yticks)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].axhline(0, color='black', ls='dashed', alpha=.7, zorder=-1)
        # Add the stimulus span or vertical line
        if lock == 'stim':
            axs[i].axvspan(0, 0.2, color='grey', lw=0, alpha=.2, label="Stimulus")
        else:
            axs[i].axvline(0, color='black', alpha=.5)
        if ilabel in far_left:
            axs[i].set_ylabel("Spearman's rho")
        else:
            axs[i].set_yticklabels([])  # Remove y-axis labels for non-left plots
            axs[i].spines["left"].set_visible(False)
        if ilabel not in range(len(label_names))[-8:]:
            axs[i].spines["bottom"].set_visible(False)
            axs[i].get_xaxis().set_visible(False)
        if ilabel not in far_left:
            axs[i].get_yaxis().set_visible(False)            
    if ilabel in far_left:
        axs[0].text(-0.19, 0.52, "Left\nhemisphere", fontsize=9, color=color1, ha='left', weight='normal', style='italic')
        axs[1].text(-0.19, 0.52, "Right\nhemisphere", fontsize=9, color=color2, ha='left', weight='normal', style='italic')
    # Show the x-axis label only on the bottom row
    if ilabel in range(len(label_names))[-8:]:
        axs[1].get_xaxis().set_visible(True)
        axs[1].set_xlabel("Time (s)")
    else:
        axs[1].set_xticklabels([])
    # First curve
    correlations = corr[label][:, :, 0]
    corr_sem = np.std(correlations, axis = (0)) / np.sqrt(len(subjects))
    corr_m1 = np.array(correlations.mean(0) + np.array(corr_sem))
    corr_m2 = np.array(correlations.mean(0) - np.array(corr_sem))
    p_values = decod_stats(correlations, jobs)
    sig = p_values < 0.05
    axs[0].fill_between(times, corr_m1, corr_m2, facecolor=color1, alpha=.8, label='Learning')
    axs[0].fill_between(times, corr_m1, corr_m2, where=sig, color='black', alpha=1)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].xaxis.set_ticks_position('none')  # Remove x-ticks on the upper plot
    axs[0].xaxis.set_tick_params(labelbottom=False)  # Remove x-tick labels on the upper plot
    # Second curve
    label = label_names[ilabel+1]
    correlations = corr[label][:, :, 0]
    corr_sem = np.std(correlations, axis = (0)) / np.sqrt(len(subjects))
    corr_m1 = np.array(correlations.mean(0) + np.array(corr_sem))
    corr_m2 = np.array(correlations.mean(0) - np.array(corr_sem))
    p_values = decod_stats(correlations, jobs)
    sig = p_values < 0.05
    axs[1].fill_between(times, corr_m1, corr_m2, facecolor=color2, alpha=.8, label='Learning')
    axs[1].fill_between(times, corr_m1, corr_m2, where=sig, color='black', alpha=1)
    # save figure
    plt.savefig(figures_dir / f'{ilabel}_{label}_corr.pdf', transparent=True)
    plt.close()

# # decoding
# chance = 0.25
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
# fig.suptitle(f"${lock}$ / ${trial_type}$ / decoding")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     # score = decoding[label][:, 1:5, :]
#     score = decoding[label][:, :, :]
#     # ax.plot(times, score.mean((0, 1)), label='roc_auc')
#     ax.plot(times, score.mean((0, 1)), label='acc')
#     ax.axhline(chance, color='black', ls='dashed', alpha=.5)
#     ax.set_title(f"${label}$", fontsize=8)    
#     p_values = decod_stats(score.mean(1) - chance)
#     p_values_unc = ttest_1samp(score, axis=0, popmean=0)[1]
#     sig = p_values < 0.05
#     ax.fill_between(times, chance, score.mean((0, 1)), where=sig, alpha=.4)
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.savefig(figures_dir / f"decoding_{lock}_{trial_type}.pdf")
# plt.close()
