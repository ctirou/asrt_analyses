import os
import numpy as np
import pandas as pd
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import spearmanr, ttest_1samp
from tqdm.auto import tqdm

loca = "subcx"
trial_type = "pattern"
subjects = SUBJS
lock = "stim"
analysis = "rsa"
sessions = EPOCHS
res_path = RESULTS_DIR
# res dir
res_dir = res_path / analysis / 'source' / lock / trial_type
figures_dir = HOME / 'figures' / analysis / 'sourcs' / lock / trial_type / loca
subjects_dir = FREESURFER_DIR
verbose = True
hemi = 'both'
parc='aparc'
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

blocks = ['prac', 'b1', 'b2', 'b3', 'b4']
similarity_names = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']        

in_seqs, out_seqs = [], []
rsa_in_lab = {}
corr_in_lab = {}

# get labels
labels = read_labels_from_annot(subject='sub01', parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]
del labels

label_names = VOLUME_LABELS

for ilabel, label in enumerate(label_names):
        
    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)

    all_in_seqs, all_out_seqs = [], []
    
    for subject in subjects:
                
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()

        behav_dir = HOME / "raw_behavs" / subject
        sequence = get_sequence(behav_dir)

        rsa_df = pd.read_hdf(res_dir / f"{subject}_rsa-subcx.h5", key='rsa')
        
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
                
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    rsa_in_lab[label] = diff_inout
    
    all_rhos = []
    for sub in range(len(subjects)):
        rhos = []
        for itime in range(len(times)):
            rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, itime]))
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)
    corr_in_lab[label] = all_rhos
    
nrows, ncols = 3, 7
color1 = "#1f77b4"
color2 = "#F08700"
color3 = "#00A6A6"
ensure_dir(figures_dir / "correlations")
ensure_dir(figures_dir / "rsa")
# plot per subject
for isub, subject in enumerate(subjects):
    print(subject)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20,7))
    fig.suptitle(subject)
    for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
        corr = corr_in_lab[label][isub, :, 0]
        ax.plot(times, corr)
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
        ax.axhline(0, color='black', ls='dashed', alpha=.5)
        ax.set_title(label)
    plt.savefig(figures_dir / "correlations" / f"{subject}.png", transparent=False)
    plt.close()
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
    for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
        practice = rsa_in_lab[label][isub, 0, :] * (-1)
        learning = rsa_in_lab[label][isub, 1:5, :] * (-1)
        
        ax.plot(times, practice, label='practice')
        ax.plot(times, learning.mean(0).flatten(), label='learning')
        ax.set_title(label)
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
        ax.axhline(0, color='black', ls='dashed', alpha=.5)
        if i == 0:
            legend = ax.legend()
            plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
        
        # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
        #     tick.label.set_fontsize(8)
        # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
        #     tick.label.set_fontsize(8)
    plt.savefig(figures_dir / "rsa" / f"{subject}.png", transparent=False)
    plt.close()
    
# plot average rho across subjects
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
# fig.suptitle("correlations")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    print(label)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black', alpha=.7)
    corr = corr_in_lab[label][:, :, 0].mean(0)
    ax.plot(times, corr, color=color2, label='rho')
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)
        yticks = ax.get_yticks()
    ax.axhline(0, color='black', alpha=.7, linewidth=.5)
    p_values_unc = ttest_1samp(corr_in_lab[label][:, :, 0], axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    # ax.fill_between(times, 0, corr, where=sig_unc, color='C2', alpha=1)
    for idx, sig in enumerate(sig_unc):
        if sig:
            # ax.plot(times[idx], yticks[0], 'o', color=color3, markersize=4)
            ax.plot(times[idx], 0.45, 'o', color=color3, markersize=4)
    p_values = decod_stats(corr_in_lab[label][:, :, 0], jobs)
    sig = p_values < 0.05
    ax.fill_between(times, 0, corr, where=sig, color='black', alpha=1)
    ax.set_title(f"${label}$", fontsize=8)   
plt.savefig(figures_dir / "correlations" / "mean.pdf", transparent=True)
plt.close()

# plot rsa
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20, 7))
# fig.suptitle("RSA average across subjects")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    print(label)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black', alpha=.7)
    practice = np.array(rsa_in_lab[label][:, 0, :], dtype=float) * (-1)
    prac_sem = np.std(practice, axis=0) / np.sqrt(len(subjects))
    prac_m1 = np.array(practice.mean(0) + np.array(prac_sem))
    prac_m2 = np.array(practice.mean(0) - np.array(prac_sem))
    learning = np.array(rsa_in_lab[label][:, 1:, :], dtype=float) * (-1)
    diff_sem = np.std(learning, axis = (0, 1)) / np.sqrt(len(subjects))
    diff_m1 = np.array(learning.mean((0, 1)) + np.array(diff_sem))
    diff_m2 = np.array(learning.mean((0, 1)) - np.array(diff_sem))
    ax.fill_between(times, prac_m1, prac_m2, facecolor="grey", alpha=.7, label='Practice')
    ax.fill_between(times, diff_m1, diff_m2, facecolor=color2, alpha=.8, label='Learning')
    ax.set_title(f"${label}$", fontsize=8)     
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)
        yticks = ax.get_yticks()
    # Adjust legend size
    # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
    #     tick.label.set_fontsize(8)
    # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
    #     tick.label.set_fontsize(8)
    ax.axhline(0, color='black', alpha=.7, linewidth=.5)
    diff = learning.mean(1) - practice
    p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    for idx, sig in enumerate(sig_unc):
        if sig:
            ax.plot(times[idx], 0.2, 'o', color=color3, markersize=4)
    # ax.fill_between(times, 0, learning, where=sig_unc, color='C2', alpha=0.2)
    p_values = decod_stats(diff, jobs)
    sig = p_values < 0.05
    ax.fill_between(times, diff_m1, diff_m2, where=sig, color='black', alpha=1)
plt.savefig(figures_dir / "rsa" / "mean.pdf", transparent=True)
plt.close()

# plot rsa practice vs b4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(26, 7))
# fig.suptitle("RSA average across subjects")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    practice = rsa_in_lab[label][:, 0, :].mean(0).astype(np.float64) * (-1)
    learning = rsa_in_lab[label][:, -1, :].mean(0).astype(np.float64) * (-1)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black', alpha=.7)
    ax.plot(times, practice, label='Practice', color='grey', alpha=.7)
    ax.plot(times, learning, label='Block_4', color=color2, alpha=.8)
    ax.set_title(f"${label}$", fontsize=8)     
    ax.axhline(0, color='black', alpha=.7, linewidth=.5)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    diff = rsa_in_lab[label][:, -1, :] - rsa_in_lab[label][:, 0, :]
    p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    for idx, sig in enumerate(sig_unc):
        if sig:
            ax.plot(times[idx], 0.15, 'o', color=color3, markersize=4)
    p_values = decod_stats(diff, jobs)
    sig = p_values < 0.05
    ax.fill_between(times, 0, learning, where=sig, color='black', alpha=1)
plt.savefig(figures_dir / "rsa" / "mean_prac_vs_b4.pdf", transparent=True)
plt.close()

# plot rsa b1 vs b4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(26, 7))
# fig.suptitle("RSA average across subjects")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    practice = rsa_in_lab[label][:, 1, :].mean(0).astype(np.float64) * (-1)
    learning = rsa_in_lab[label][:, -1, :].mean(0).astype(np.float64) * (-1)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        ax.axvline(0, color='black', alpha=.7)
    ax.plot(times, practice, label='Block_1', color='grey', alpha=.7)
    ax.plot(times, learning, label='Block_4', color=color2, alpha=.8)
    ax.set_title(f"${label}$", fontsize=8)     
    ax.axhline(0, color='black', alpha=.7, linewidth=.5)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    # for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
    #     tick.label.set_fontsize(8)
    # for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
    #     tick.label.set_fontsize(8)
    diff = rsa_in_lab[label][:, -1, :] - rsa_in_lab[label][:, 1, :]
    p_values_unc = ttest_1samp(diff.astype(np.float64), axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    for idx, sig in enumerate(sig_unc):
        if sig:
            ax.plot(times[idx], 0.10, 'o', color=color3, markersize=4)
    p_values = decod_stats(diff, jobs)
    sig = p_values < 0.05
    ax.fill_between(times, 0, learning, where=sig, color='black', alpha=1)
plt.savefig(figures_dir / "rsa" / "mean_b1_vs_b4.pdf", transparent=True)
plt.close()