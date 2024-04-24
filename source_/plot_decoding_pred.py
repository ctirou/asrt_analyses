import os
import numpy as np
import pandas as pd
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import ttest_1samp, spearmanr
from tqdm.auto import tqdm

trial_type = "pattern"
subjects = SUBJS
lock = "stim"
params = "pred_decoding"
sessions = EPOCHS
res_dir = RESULTS_DIR / 'figures' / lock / 'old_decoding' / params / 'source' / trial_type
subjects_dir = FREESURFER_DIR
verbose = "error"
hemi = 'both'
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
decod_in_lab2 = {}
corr_in_lab = {}

# get label names
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]

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
        
        scores_df = pd.read_hdf(res_dir / f"{subject}_pred.h5", key='pred')
        scores_df = scores_df.groupby(level=['label', 'session', 'similarities']).first()
        
        new_index = pd.MultiIndex.from_product([scores_df.index.levels[0], scores_df.index.levels[1], similarity_names], 
                                       names=scores_df.index.names)   
        scores_df = scores_df.reindex(new_index)
                    
        for session_id, session in enumerate(sessions):
            
            one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
        
            for sim, sim_list in zip(similarity_names, [one_four, one_three, one_two, three_four, two_four, two_three]):
                sim_list.append(scores_df.loc[(label, session_id, sim), :])

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
    decod_in_lab[label] = diff_inout
    
    decod_in_lab2[label] = [np.squeeze(all_in_seq).mean(axis=1), np.squeeze(all_out_seq).mean(axis=1)]
    
    # all_rhos = []
    # for sub in range(len(subjects)):
    #     rhos = []
    #     for t in range(len(times)):
    #         rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
    #     all_rhos.append(rhos)
    # all_rhos = np.array(all_rhos)
    # corr_in_lab[label] = all_rhos
    
nrows, ncols = 7, 10

# plot diff in/out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for ax, label in zip(axs.flat, label_names):
    practice = decod_in_lab[label][:, 0, :].mean(0)
    learning = decod_in_lab[label][:, 1:, :].mean((0, 1))
    
    ax.plot(times, practice, label='practice')
    ax.plot(times, learning, label='learning')
    ax.set_title(label)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    ax.legend()
plt.show()

# plot in vs out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for ax, label in zip(axs.flat, label_names):
    ins = decod_in_lab2[label][0].mean(axis=(0, 1))
    outs = decod_in_lab2[label][1].mean(axis=(0, 1))
    ax.plot(times, ins, label='in_seq') 
    ax.plot(times, outs, label='out_seq')
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.set_title(label)
    ax.legend()
plt.show()


# correlations
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
# for ax, label in zip(axs.flat, label_names):
#     rho = corr_in_lab[label][:, :, 0].mean(0)
#     ax.plot(times, rho, label='rho')
#     ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     ax.set_title(label)
# plt.show()