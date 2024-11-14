import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import *
from config import *
from mne import read_epochs
from scipy.stats import spearmanr, ttest_1samp
from tqdm.auto import tqdm
from matplotlib.ticker import FuncFormatter

lock = 'stim'
analysis = 'usual'
# analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
jobs = -1

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
figures_dir = FIGURES_DIR / "RSA" / "sensors" / lock / analysis
ensure_dir(figures_dir)
metric = 'mahalanobis'

# get times
epoch_fname = op.join(data_path, lock, 'sub01-0-epo.fif')
epochs = mne.read_epochs(epoch_fname)
times = epochs.times
del epochs

rsa, rsa_high, rsa_low, corr = {}, {}, {}, {}

def format_func(value, tick_number):
    return f'{value:.1f}'

labels = (SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else (SURFACE_LABELS_RT + VOLUME_LABELS_RT)

def process_label(labels):
    """Process both surface and volume labels."""
    print(f"Processing {label}...")
    for label in labels:
        all_high, all_low = [], []

        for subject in subjects:
            # Read the behav file to get the sequence 
            behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
            sequence = get_sequence(behav_dir)
            
            res_path = RESULTS_DIR / analysis / 'source' / label.name / lock / 'rdm' / subject
            sub_high, sub_low = get_all_high_low(res_path, sequence, analysis)
            
            all_high.append(sub_high)
            all_low.append(sub_low)

        all_high, all_low = np.array(all_high).mean(1), np.array(all_low).mean(1)
        diff_low_high = np.squeeze(all_high - all_low)
        rsa[label] = diff_low_high
        rsa_high[label] = all_high
        rsa_low[label] = all_low

        all_rhos = [
            [spearmanr([0, 1, 2, 3, 4], diff_low_high[sub, :, itime])[0] for itime in range(len(times))]
            for sub in range(len(subjects))
        ]
        corr[label] = np.array(all_rhos)

label_names = sorted(SURFACE_LABELS + VOLUME_LABELS, key=str.casefold) if lock == 'stim' else sorted(SURFACE_LABELS_RT + VOLUME_LABELS_RT, key=str.casefold)
# define parameters    
chance = 25
ncols = 4
nrows = 10 if lock == 'stim' else 9
far_left = [0] + [i for i in range(0, len(label_names), ncols*2)]
color1, color2 = ("#1982C4", "#74B3CE") if lock == 'stim' else ("#D76A03", "#EC9F05")
color3 = "C7"
for ilabel in tqdm(range(0, len(label_names), 2)):
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    fig.subplots_adjust(hspace=0)
    label = label_names[ilabel]
    ytitle = -0.20 if lock == 'stim' else 0.25
    if label == "Cerebellum-White-Matter-lh":
        xtitle=0.6
        ha='right'
    else:
        xtitle=0.25
        ha='left' 
    axs[0].text(xtitle, ytitle, f"{label.capitalize()[:-3]}",
                fontsize=13, weight='normal', style='italic', ha=ha,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    if ilabel in range(8):
        if lock == 'stim':
            axs[0].text(0.1, 0.32, "$Stimulus$", fontsize=11, zorder=10, ha='center')
        else:
            axs[0].text(0.05, 0.32, "Button press", style='italic', fontsize=11, zorder=10, ha='center')
    for i in range(2):
        axs[i].set_ylim(-0.3, 0.3) if lock == 'stim' else axs[i].set_ylim(-0.5, 0.3)
        yticks = axs[i].get_yticks()
        yticks = yticks[1:-1]  # Remove first and last element
        # axs[i].set_yticks(yticks)
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
            axs[i].set_ylabel("Similarity index", fontsize=12)
        else:   
            axs[i].set_yticklabels([])  # Remove y-axis labels for non-left plots
    if ilabel in far_left:
        if lock == 'stim':
            axs[0].text(-0.2, 0.15, "Left\nhemisphere", fontsize=12, color=color1, ha='left', weight='normal', style='italic')
            axs[1].text(-0.2, 0.15, "Right\nhemisphere", fontsize=12, color=color2, ha='left', weight='normal', style='italic')
        else:    
            axs[0].text(-0.2, -0.4, "Left\nhemisphere", fontsize=12, color=color1, ha='left', weight='normal', style='italic')
            axs[1].text(-0.2, -0.4, "Right\nhemisphere", fontsize=12, color=color2, ha='left', weight='normal', style='italic')
    # Show the x-axis label only on the bottom row
    if ilabel in range(len(label_names))[-8:]:
        axs[1].get_xaxis().set_visible(True)
        axs[1].set_xlabel("Time (s)", fontsize=12)
    else:
        axs[1].set_xticklabels([])
    # First curve
    practice = np.array(rsa[label][:, 0, :], dtype=float)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    learning = np.array(rsa[label][:, 1:, :], dtype=float)
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
    practice = np.array(rsa[label][:, 0, :], dtype=float)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    learning = np.array(rsa[label][:, 1:, :], dtype=float)
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
    axs[0].text(0.25, 0.75, f"{label.capitalize()[:-3]}",
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
    if ilabel in far_left:
        axs[0].text(-0.19, 0.5, "Left\nhemisphere", fontsize=9, color=color1, ha='left', weight='normal', style='italic')
        axs[1].text(-0.19, 0.5, "Right\nhemisphere", fontsize=9, color=color2, ha='left', weight='normal', style='italic')
    # Show the x-axis label only on the bottom row
    if ilabel in range(len(label_names))[-8:]:
        axs[1].get_xaxis().set_visible(True)
        axs[1].set_xlabel("Time (s)")
    else:
        axs[1].set_xticklabels([])
    # First curve
    correlations = corr[label][:, :].mean(0)
    corr_sem = np.std(corr[label][:, :], axis = (0)) / np.sqrt(len(subjects))
    corr_m1 = np.array(correlations + np.array(corr_sem))
    corr_m2 = np.array(correlations - np.array(corr_sem))
    p_values = decod_stats(corr[label][:, :], jobs)
    sig = p_values < 0.05
    axs[0].fill_between(times, corr_m1, corr_m2, facecolor=color1, alpha=.8, label='Learning')
    axs[0].fill_between(times, corr_m1, corr_m2, where=sig, color='black', alpha=1)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].xaxis.set_ticks_position('none')  # Remove x-ticks on the upper plot
    axs[0].xaxis.set_tick_params(labelbottom=False)  # Remove x-tick labels on the upper plot
    # Second curve
    label = label_names[ilabel+1]
    correlations = corr[label][:, :].mean(0)
    corr_sem = np.std(corr[label][:, :], axis = (0)) / np.sqrt(len(subjects))
    corr_m1 = np.array(correlations + np.array(corr_sem))
    corr_m2 = np.array(correlations - np.array(corr_sem))
    p_values = decod_stats(corr[label][:, :], jobs)
    sig = p_values < 0.05
    axs[1].fill_between(times, corr_m1, corr_m2, facecolor=color2, alpha=.8, label='Learning')
    axs[1].fill_between(times, corr_m1, corr_m2, where=sig, color='black', alpha=1)
    # save figure
    plt.savefig(figures_dir / f'{ilabel}_{label}_corr.pdf', transparent=True)
    plt.close()

    
# # plot average rho across subjects
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
# # fig.suptitle("correlations")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     else:
#         ax.axvline(0, color='black', alpha=.7)
#     correlations = corr[label][:, :].mean(0)
#     ax.plot(times, correlations)
#     p_values = decod_stats(corr[label][:, :], jobs)
#     # p_values_unc = ttest_1samp(corr[label][:, :, 0], axis=0, popmean=0)[1]
#     sig = p_values < 0.05
#     # sig_unc = p_values_unc < 0.05
#     # ax.fill_between(times, 0, corr, where=sig_unc, color='C2', alpha=1)
#     ax.fill_between(times, 0, corr, where=sig, alpha=0.3)
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     ax.set_title(f"${label}$", fontsize=8)   
# plt.savefig(figures_dir / "correlations" / "mean.pdf", transparent=True)
# plt.close()

# nrows, ncols = 8, 5
# # color1 = "#1f77b4"
# # color2 = "#F79256"
# ensure_dir(figures_dir / "correlations")
# ensure_dir(figures_dir / "rsa")
# # plot per subject
# for isub, subject in enumerate(subjects):
#     print(subject)
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20,7))
#     fig.suptitle(subject)
#     for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#         correlations = corr[label][isub, :]
#         ax.plot(times, correlations)
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#         ax.axhline(0, color='black', ls='dashed', alpha=.5)
#         ax.set_title(label)
#     plt.savefig(figures_dir / "correlations" / f"{subject}.png", transparent=False)
#     plt.close()
    
#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
#     for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#         practice = rsa[label][isub, 0, :] * (-1)
#         learning = rsa[label][isub, 1:5, :] * (-1)
        
#         ax.plot(times, practice, label='practice')
#         ax.plot(times, learning.mean(0).flatten(), label='learning')
#         ax.set_title(label)
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#         ax.axhline(0, color='black', ls='dashed', alpha=.5)
#         if i == 0:
#             legend = ax.legend()
#             plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
        
#         # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
#         #     tick.label.set_fontsize(8)
#         # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
#         #     tick.label.set_fontsize(8)
#     plt.savefig(figures_dir / "rsa" / f"{subject}.png", transparent=False)
#     plt.close()

# # plot rsa
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(10, 15))
# # fig.suptitle("RSA average across subjects")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     practice = rsa[label][:, 0, :].mean(0).astype(np.float64) * (-1)
#     learning = rsa[label][:, 1:5, :].mean((0, 1)).astype(np.float64) * (-1)
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     else:
#         ax.axvline(0, color='black', alpha=.7)
#     ax.plot(times, practice, label='practice')
#     ax.plot(times, learning, label='learning')
#     ax.set_title(f"${label}$", fontsize=8)     
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     if i == 0:
#         legend = ax.legend()
#         plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
#     # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
#     #     tick.label.set_fontsize(8)
#     # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
#     #     tick.label.set_fontsize(8)
#     diff = rsa[label][:, 1:5, :].mean((1)) - rsa[label][:, 0, :]
#     p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
#     sig_unc = p_values_unc < 0.05
#     ax.fill_between(times, 0, learning, where=sig_unc, color='C2', alpha=0.2)
#     p_values = decod_stats(diff, jobs)
#     sig = p_values < 0.05
#     ax.fill_between(times, 0, learning, where=sig, color='C3', alpha=0.3)
# plt.savefig(figures_dir / "rsa" / "mean.pdf", transparent=True)
# plt.close()

# # plot rsa practice vs b4
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(26, 7))
# # fig.suptitle("RSA average across subjects")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     practice = rsa[label][:, 0, :].mean(0).astype(np.float64) * (-1)
#     learning = rsa[label][:, -1, :].mean(0).astype(np.float64) * (-1)
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     else:
#         ax.axvline(0, color='black', alpha=.7)
#     ax.plot(times, practice, label='Practice')
#     ax.plot(times, learning, label='Block_4')
#     ax.set_title(f"${label}$", fontsize=8)     
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     if i == 0:
#         legend = ax.legend()
#         plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
#     # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
#     #     tick.label.set_fontsize(8)
#     # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
#     #     tick.label.set_fontsize(8)
#     diff = rsa[label][:, -1, :] - rsa[label][:, 0, :]
#     p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
#     sig_unc = p_values_unc < 0.05
#     ax.fill_between(times, 0, learning, where=sig_unc, color='C2', alpha=1)
#     p_values = decod_stats(diff, jobs)
#     sig = p_values < 0.05
#     ax.fill_between(times, 0, learning, where=sig, color='black', alpha=1)
# plt.savefig(figures_dir / "rsa" / "mean_prac_vs_b4.pdf", transparent=True)
# plt.close()

# # plot rsa b1 vs b4
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(26, 7))
# # fig.suptitle("RSA average across subjects")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     practice = rsa[label][:, 1, :].mean(0).astype(np.float64) * (-1)
#     learning = rsa[label][:, -1, :].mean(0).astype(np.float64) * (-1)
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     else:
#         ax.axvline(0, color='black', alpha=.7)
#     ax.plot(times, practice, label='Block_1')
#     ax.plot(times, learning, label='Block_4')
#     ax.set_title(f"${label}$", fontsize=8)     
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     if i == 0:
#         legend = ax.legend()
#         plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
#     # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
#     #     tick.label.set_fontsize(8)
#     # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
#     #     tick.label.set_fontsize(8)
#     diff = rsa[label][:, -1, :] - rsa[label][:, 1, :]
#     p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
#     sig_unc = p_values_unc < 0.05
#     ax.fill_between(times, 0, learning, where=sig_unc, color='C2', alpha=1)
#     p_values = decod_stats(diff, jobs)
#     sig = p_values < 0.05
#     ax.fill_between(times, 0, learning, where=sig, color='black', alpha=1)
# plt.savefig(figures_dir / "rsa" / "mean_b1_vs_b4.pdf", transparent=True)
# plt.close()