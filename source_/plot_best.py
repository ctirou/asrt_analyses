import os.path as op
import numpy as np
from base import ensure_dir, decod_stats, get_sequence, get_inout_seq
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import ttest_1samp, spearmanr
import gc
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

analysis = "RSA"
lock = "stim"
trial_type = "pattern"
subjects = SUBJS
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
verbose = "error"
hemi = 'both'
chance = 25

per_label = False

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']

decod_in_lab = {}
decod_in_lab2 = {}
corr_in_lab = {}
decoding = dict()
rsa_in_lab = {}

# get label names
best_regions = [6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 42, 43, 50, 51, 58, 59]
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for ilabel, label in enumerate(labels) if ilabel in best_regions]
del labels

figures = res_path / "figures" / analysis / 'source' / lock / trial_type
ensure_dir(figures)
gc.collect()

similarity_names = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

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
        
        behav_dir = RAW_DATA_DIR / subject / 'behav_data'
        sequence = get_sequence(behav_dir)
                
        scores_dir = res_path / 'concatenated' / 'source' / lock / trial_type / label / subject
        sub_scores = np.load(scores_dir / "scores.npy")
        
        rsa_dir = res_path / analysis / 'source' / lock / trial_type
        rsa_df = pd.read_hdf(rsa_dir / f"{subject}_rsa.h5", key='rsa')
        for session_id, session in enumerate(sessions):
            one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
            for sim, sim_list in zip(similarity_names, [one_four, one_three, one_two, three_four, two_four, two_three]):
                sim_list.append(rsa_df.loc[(label, session_id, sim), :])
            for all_sims, sim_list in zip([one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities], 
                                          [one_two, one_three, one_four, two_three, two_four, three_four]):
                    all_sims.append(np.array(sim_list))
                
        for all_sims in [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]:
            all_sims = np.array(all_sims)
            
        similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]
        
        in_seq, out_seq = get_inout_seq(sequence, similarities)
        
        all_in_seqs.append(in_seq)
        all_out_seqs.append(out_seq)
        
        decoding[label].append(sub_scores)
                
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    rsa_in_lab[label] = diff_inout

    decoding[label] = np.array(decoding[label])
    if per_label:
        # decoding
        fig, ax = plt.subplots(figsize=(7, 4))
        if lock == 'stim':
            ax.axvspan(0, 0.2, color='grey', alpha=.2, label='stimulus')
        score = decoding[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score.mean(0) + np.array(sem))
        m2 = np.array(score.mean(0) - np.array(sem))
        ax.axhline(chance, color='black', ls='dashed', alpha=.5, label='chance')
        ax.set_title(f"${label}$", fontsize=13)    
        p_values = decod_stats(score - chance)
        sig = p_values < 0.05
        ax.fill_between(times, m1, m2, color='0.6')
        ax.fill_between(times, m1, m2, color='#1f77b4', where=sig, alpha=1)
        ax.fill_between(times, chance, m2, where=sig, color='#1f77b4', alpha=0.7, label='significant')
        ax.set_ylim(20, 45)
        yticks = ax.get_yticks()
        yticks = yticks[1:-1]  # Remove first and last element
        ax.set_yticks(yticks)  # Set new y-ticks
        if ilabel == 0:
            legend = ax.legend(loc='best')
            plt.setp(legend.get_texts(), fontsize=9)  # Adjust legend size
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlabel("Time (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(figures / f"{label}_decoding.pdf", transparent=True)
        plt.close()

        # rsa
        fig, ax = plt.subplots(figsize=(7, 4))
        if lock == 'stim':
            ax.axvspan(0, 0.2, color='grey', alpha=.2, label='stimulus')
        practice = np.array(rsa_in_lab[label][:, 0, :], dtype=float) * (-1)
        prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
        prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
        prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
        learning = np.array(rsa_in_lab[label][:, 1:, :], dtype=float) * (-1)
        diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
        diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
        diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
        ax.fill_between(times, prac_m1, prac_m2, color='C7', alpha=.5, label='Pre-learning')
        ax.fill_between(times, diff_m1, diff_m2, color='#d627', alpha=.7, label='Learning')
        diff = learning.mean(1) - practice
        p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < .05
        # pv2 = decod_stats(diff)
        # sig2 = pv2 < .05
        # ax.fill_between(times, diff_m1, diff_m2, where=sig2, color='#d62728', alpha=1)
        ax.fill_between(times, diff_m1, diff_m2, where=sig_unc, color='black', alpha=0.3)
        ax.set_title(f"${label}$", fontsize=13)
        ax.axhline(0, color='black', ls='dashed', alpha=.3)
        ax.set_ylim(-0.2, 0.2)
        yticks = ax.get_yticks()
        yticks = yticks[1:-1]  # Remove first and last element
        ax.set_yticks(yticks)  # Set new y-ticks
        if ilabel == 0:
            legend = ax.legend(loc='best')
            plt.setp(legend.get_texts(), fontsize=9)  # Adjust legend size
            ax.set_ylabel("Similarity index")
            ax.set_xlabel("Time (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(figures / f"{label}_rsa.pdf", transparent=True)
        plt.close()
    
# #1f77b4 #b45c1f
# #084887 #F9AB55 
# #e4572e #29335c #F3A712 #A8C686 #669BBC
# #F79256 #00B2CA #1D4E89

color1 = "#1f77b4"
color2 = "#F79256"

color3 = "C7"

for ilabel in tqdm(range(len(label_names))):
    if ilabel % 2 == 0:
        # decoding
        fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True)
        fig.subplots_adjust(hspace=0)
        label = label_names[ilabel]
        fig.suptitle(f"{label.capitalize()[:-3]}", y=0.95)
        axs[0].set_title("$Left$\n$hemi.$", fontsize=10, x=0.15, y=0.70)
        axs[0].axhline(chance, color='black', ls='dashed', alpha=.7, label='Chance', zorder=-1)
        if lock == 'stim':
            axs[0].axvspan(0, 0.2, facecolor='grey', alpha=.2, label='Stimulus', lw=0, zorder=1)
            axs[1].axvspan(0, 0.2, facecolor='grey', alpha=.2, lw=0, zorder=1)
        score = decoding[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score.mean(0) + np.array(sem))
        m2 = np.array(score.mean(0) - np.array(sem))
        p_values = decod_stats(score - chance)
        sig = p_values < 0.05
        axs[0].fill_between(times, m1, m2, facecolor='0.6')
        axs[0].fill_between(times, m1, m2, facecolor=color1, where=sig, alpha=1, label='Significance')
        axs[0].fill_between(times, chance, m2, facecolor=color1, where=sig, alpha=0.7)
        axs[0].spines["bottom"].set_visible(False)
        axs[0].xaxis.set_ticks_position('none')  # New line: remove x-ticks of the upper plot
        axs[0].xaxis.set_tick_params(labelbottom=False)  # New line: remove x-tick labels of the upper plot
        label = label_names[ilabel+1]
        score2 = decoding[label] * 100
        sem = np.std(score2, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score2.mean(0) + np.array(sem))
        m2 = np.array(score2.mean(0) - np.array(sem))
        p_values = decod_stats(score2 - chance)
        sig = p_values < 0.05
        axs[1].set_title("$Right$\n$hemi.$", fontsize=10, x=0.15, y=0.70)
        axs[1].axhline(chance, color='black', ls='dashed', alpha=.7, zorder=-1)
        axs[1].fill_between(times, m1, m2, facecolor='0.6')
        axs[1].fill_between(times, m1, m2, facecolor=color2, where=sig, alpha=1, label='Significance')
        axs[1].fill_between(times, chance, m2, facecolor=color2, where=sig, alpha=0.7)
        axs[1].set_xlabel("Time (s)")
        for i in range(2):
            axs[i].set_ylim(20, 45)
            yticks = axs[i].get_yticks()
            yticks = yticks[1:-1]  # Remove first and last element
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks(yticks)  # Set new y-ticks
            axs[i].set_ylabel("Accuracy (%)")
            if ilabel == 0:
                legend = axs[i].legend(loc='best', frameon=False)
                plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
        plt.savefig(figures / f"{label[:-3]}_decoding.pdf", transparent=True)
        plt.close()
        
for ilabel in tqdm(range(len(label_names))):
    if ilabel % 2 == 0:
        # RSA
        fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True)
        fig.subplots_adjust(hspace=0)
        label = label_names[ilabel]
        fig.suptitle(f"{label.capitalize()[:-3]}", y=0.95)
        axs[0].set_title("$Left$\n$hemi.$", fontsize=10, x=0.15, y=0.70)
        axs[0].axhline(0, color='black', ls='dashed', alpha=.7, zorder=-1)
        if lock == 'stim':
            axs[0].axvspan(0, 0.2, facecolor='grey', alpha=.2, label='Stimulus', lw=0, zorder=1)
            axs[1].axvspan(0, 0.2, facecolor='grey', alpha=.2, lw=0, zorder=1)    
        practice = np.array(rsa_in_lab[label][:, 0, :], dtype=float) * (-1)
        prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
        prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
        prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
        learning = np.array(rsa_in_lab[label][:, 1:, :], dtype=float) * (-1)
        diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
        diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
        diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
        axs[0].fill_between(times, prac_m1, prac_m2, facecolor=color3, alpha=.5, label='Pre-learning')
        axs[0].fill_between(times, diff_m1, diff_m2, facecolor=color1, alpha=.8, label='Learning')
        diff = learning.mean(1) - practice
        p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < .05
        pv2 = decod_stats(diff)
        sig2 = pv2 < .05
        axs[0].fill_between(times, diff_m1, diff_m2, where=sig2, color='#d62728', alpha=1)
        axs[0].fill_between(times, diff_m1, diff_m2, where=sig_unc, color='black', alpha=1)
        axs[0].spines["bottom"].set_visible(False)
        axs[0].xaxis.set_ticks_position('none')  # New line: remove x-ticks of the upper plot
        axs[0].xaxis.set_tick_params(labelbottom=False)  # New line: remove x-tick labels of the upper plot    
        label = label_names[ilabel+1]
        practice = np.array(rsa_in_lab[label][:, 0, :], dtype=float) * (-1)
        prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
        prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
        prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
        learning = np.array(rsa_in_lab[label][:, 1:, :], dtype=float) * (-1)
        diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
        diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
        diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
        axs[1].fill_between(times, prac_m1, prac_m2, facecolor=color3, alpha=.5)
        axs[1].fill_between(times, diff_m1, diff_m2, facecolor=color2, alpha=.8, label='Learning')
        diff = learning.mean(1) - practice
        p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < .05
        pv2 = decod_stats(diff)
        sig2 = pv2 < .05
        axs[1].fill_between(times, diff_m1, diff_m2, where=sig2, color='#d62728', alpha=1)
        axs[1].fill_between(times, diff_m1, diff_m2, where=sig_unc, color='black', alpha=1)
        axs[1].set_title("$Right$\n$hemi.$", fontsize=10, x=0.15, y=0.70)
        axs[1].axhline(0, color='black', ls='dashed', alpha=.7, zorder=-1)        
        axs[1].set_xlabel("Time (s)")
        for i in range(2):
            axs[i].set_ylim(-0.3, 0.2)
            yticks = axs[i].get_yticks()
            yticks = yticks[1:-1]  # Remove first and last element
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks(yticks)  # Set new y-ticks
            axs[i].set_ylabel("Similarity index")
        if ilabel == 0:
            legend = axs[0].legend(loc='best', frameon=False)
            plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size            
            legend = axs[1].legend(loc='lower right', frameon=False)
            plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
        plt.savefig(figures / f"{label[:-3]}_rsa.pdf", transparent=True)
        plt.close()

nrows, ncols = 3, 6
# decoding
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
# fig.suptitle(f"${lock}$ / ${trial_type}$ / decoding")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    score = decoding[label] * 100
    sem = np.std(score, axis=0) / np.sqrt(len(subjects))
    m1 = np.array(score.mean(0) + np.array(sem))
    m2 = np.array(score.mean(0) - np.array(sem))
    ax.axhline(chance, color='black', ls='dashed', alpha=.5)
    ax.set_title(f"${label}$", fontsize=8)    
    p_values = decod_stats(score - chance)
    sig = p_values < 0.05
    ax.fill_between(times, m1, m2, color='0.6')
    ax.fill_between(times, m1, m2, color=color1, where=sig, alpha=1)
    ax.fill_between(times, chance, m2, where=sig, color=color1, alpha=0.7, label='significant')
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
plt.savefig(figures / f"mean_best_decoding.pdf", transparent=True)
plt.close()

# plot diff in/out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
# fig.suptitle(f"${lock}$ / ${trial_type}$ / diff_in_out")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)        
    
    practice = np.array(rsa_in_lab[label][:, 0, :], dtype=float) * (-1)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    
    learning = np.array(rsa_in_lab[label][:, 1:, :], dtype=float) * (-1)
    diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
    diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
    diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
        
    ax.fill_between(times, prac_m1, prac_m2, color='C7', alpha=.5, label='pre-learning')
    ax.fill_between(times, diff_m1, diff_m2, color=color2, alpha=.7, label='learning')

    diff = learning.mean(1) - practice
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < .05
    # pv2 = decod_stats(diff)
    # sig2 = pv2 < .05
    # ax.fill_between(times, diff_m1, diff_m2, where=sig2, color='#d62728', alpha=1)
    ax.fill_between(times, diff_m1, diff_m2, where=sig_unc, color='black', alpha=0.3)
    
    ax.set_title(f"${label}$", fontsize=8)
    ax.axhline(0, color='black', ls='dashed', alpha=.3)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
plt.savefig(figures / f"mean_best_rsa.pdf", transparent=True)
plt.close()


# # correlations
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
# fig.suptitle(f"${lock}$ / ${trial_type}$ / correlations")
# for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
#     rho = corr_in_lab[label][:, :, 0]
#     ax.plot(times, rho.mean(0), label='rho')
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     ax.set_title(f"${label}$", fontsize=8)
#     if i == 0:
#         legend = ax.legend()
#         plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
#     p_values = decod_stats(rho)
#     p_values_unc = ttest_1samp(rho, axis=0, popmean=0)[1]
#     sig = p_values < 0.05
#     sig_unc = p_values_unc < 0.05
#     ax.fill_between(times, 0, rho.mean(0), where=sig_unc, color='C2', alpha=1)
#     ax.fill_between(times, 0, rho.mean(0), where=sig, alpha=0.3)
#     if lock == 'stim':
#         ax.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.savefig(figures / "correlations.pdf")
# plt.close()