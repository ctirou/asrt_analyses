import os
import numpy as np
import pandas as pd
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import spearmanr, ttest_1samp

trial_type = "pattern"
subjects = SUBJS
lock = "stim"
params = "RSA"
sessions = EPOCHS
res_dir = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type
subjects_dir = FREESURFER_DIR
verbose = 'error'
hemi = 'both'
parc='aparc'

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
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

        behav_dir = RAW_DATA_DIR / subject / 'behav_data'
        sequence = get_sequence(behav_dir)
                

        rsa_df = pd.read_hdf(res_dir / f"{subject}_rsa.h5", key='rsa')
        
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
    
nrows, ncols = 7, 10
    
ensure_dir(res_dir / "correlations")
ensure_dir(res_dir / "rsa_plots")
# plot per subject
for isub, sub in enumerate(subjects):
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained', figsize=(40, 13))
    fig.suptitle(sub)
    for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
        corr = corr_in_lab[label][isub, :, 0]
        ax.plot(times, corr)
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
        ax.axhline(0, color='black', ls='dashed', alpha=.5)
        ax.set_title(label)
    plt.savefig(res_dir / "correlations" / f"{subject}.png")
    plt.close()
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained', figsize=(40, 13))
    for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
        practice = rsa_in_lab[label][isub, 0, :]
        learning = rsa_in_lab[label][isub, 1:5, :]
        
        ax.plot(times, practice, label='practice')
        ax.plot(times, learning.mean(0).flatten(), label='learning')
        ax.set_title(label)
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
        ax.axhline(0, color='black', ls='dashed', alpha=.5)
        if i == 0:
            legend = ax.legend()
            plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
        
        for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
            tick.label.set_fontsize(8)
    plt.savefig(res_dir / "rsa_plots" / f"{subject}.png")
    plt.close()
    
# plot average rho across subjects
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained', figsize=(40, 13))
fig.suptitle("correlations")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    corr = corr_in_lab[label][:, :, 0].mean(0)
    ax.plot(times, corr)

    p_values = decod_stats(corr_in_lab[label][:, :, 0])
    p_values_unc = ttest_1samp(corr_in_lab[label][:, :, 0], axis=0, popmean=0)[1]
    sig = p_values < 0.05
    sig_unc = p_values_unc < 0.05
    
    ax.fill_between(times, 0, corr, where=sig_unc, color='C2', alpha=1)
    ax.fill_between(times, 0, corr, where=sig, alpha=0.3)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    ax.set_title(label)
plt.savefig(res_dir / "correlations" / "mean.png")
plt.close()

# plot rsa
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained', figsize=(40, 13))
fig.suptitle("RSA average across subjects")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    practice = rsa_in_lab[label][:, 0, :].mean(0)
    learning = rsa_in_lab[label][:, 1:5, :].mean((0, 1))
    
    ax.plot(times, practice, label='practice')
    ax.plot(times, learning, label='learning')
    ax.set_title(label)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    
    for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
        tick.label.set_fontsize(8)
        
    # diff = rsa_in_lab[label][:, 1:5, :].mean((1)) - rsa_in_lab[label][:, 0, :]
    # # p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    # # sig_unc = p_values_unc < 0.05
    # p_values = decod_stats(diff)
    # sig = p_values < 0.05
    # # ax.fill_between(times, 0, learning, where=sig_unc, color='C2', alpha=0.2)
    # ax.fill_between(times, 0, learning, where=sig, color='C3', alpha=0.3)
    
plt.savefig(res_dir / "rsa_plots" / "mean-rsa.png")
plt.close()