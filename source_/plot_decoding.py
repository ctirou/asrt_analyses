import os
import numpy as np
import pandas as pd
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import spearmanr

trial_type = "pattern"
subjects = SUBJS
subjects = ['sub01', 'sub02', 'sub04']
lock = "stim"
params = "new_decoding"
sessions = EPOCHS
res_dir = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type
subjects_dir = FREESURFER_DIR
verbose = 'error'
hemi = 'both'
parc='aparc'
chance = .25

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

blocks = ['prac', 'b1', 'b2', 'b3', 'b4']
similarity_names = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']        

in_seqs, out_seqs = [], []
decod_in_lab = {}
rsa_in_lab = {}
corr_in_lab = {}

# get labels
labels = read_labels_from_annot(subject='sub01', parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]
del labels

for ilabel, label in enumerate(label_names):
        
    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)

    all_in_seqs, all_out_seqs = [], []
    subs_decod = []
    
    for subject in subjects:
                
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()

        behav_dir = RAW_DATA_DIR / subject / 'behav_data'
        sequence = get_sequence(behav_dir)
        
        scores_df = pd.read_hdf(res_dir / f"{subject}_scores.h5", key='scores')
        
        ave_score = []

        rsa_df = pd.read_hdf(res_dir / f"{subject}_rsa.h5", key='rsa')
        
        for session_id, session in enumerate(sessions):
            
            one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
            for sim, sim_list in zip(similarity_names, [one_four, one_three, one_two, three_four, two_four, two_three]):
                sim_list.append(rsa_df.loc[(label, session_id, sim), :])
            for all_sims, sim_list in zip([one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities], 
                                          [one_two, one_three, one_four, two_three, two_four, three_four]):
                    all_sims.append(np.array(sim_list))

            tstore = []
            for itime in range(len(times)):        
                tstore.append(scores_df.loc[(label, session_id), itime])
                
            tstore = np.array(tstore)
            ave_score.append(tstore)
                    
        
        for all_sims in [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]:
            all_sims = np.array(all_sims)
            
        similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]
        in_seq, out_seq = get_inout_seq(sequence, similarities)
        all_in_seqs.append(in_seq)
        all_out_seqs.append(out_seq)
        
        ave_score = np.array(ave_score)
        subs_decod.append(ave_score.mean(0))
        
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    # np.save(res_dir / f"{label}.npy", diff_inout)
    rsa_in_lab[label] = diff_inout
    
    decod_in_lab[label] = np.array(subs_decod)
    
    
    # all_rhos = []
    # for sub in range(len(subjects)):
    #     rhos = []
    #     for t in range(len(times)):
    #         rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
    #     all_rhos.append(rhos)
    # all_rhos = np.array(all_rhos)
    # corr_in_lab[label] = all_rhos
    
nrows, ncols = 7, 10

# plot rsa
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for ax, label in zip(axs.flat, label_names):
    practice = rsa_in_lab[label][:, 0, :].mean(0)
    learning = rsa_in_lab[label][:, 1:5, :].mean((0, 1))
    
    ax.plot(times, practice, label='practice')
    ax.plot(times, learning, label='learning')
    ax.set_title(label)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    ax.legend()
plt.show()

# plot decoding
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for ax, label in zip(axs.flat, label_names):
    ax.plot(times, decod_in_lab[label][0])
    ax.set_title(label)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(chance, color='black', ls='dashed', alpha=.5)
plt.show()