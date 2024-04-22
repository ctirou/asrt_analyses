import os
import numpy as np
import pandas as pd
from jr.gat import scorer_spearman
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs

trial_type = "pattern"
subjects = SUBJS
lock = "stim"
params = "pred_decoding"
sessions = EPOCHS
res_dir = RESULTS_DIR / 'figures' / lock / 'decoding' / params / 'source' / trial_type
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
label_d = {}

# get label names
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]

for ilabel, label in enumerate(label_names):
    
    print(f"{ilabel+1}/{len(label_names)}", label)
    
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
        
        # max_value = scores_df.max().max()
        # min_value = scores_df.min().min()
        
        # nrows=10
        # ncols=16
        
        # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
        #                         sharey=True, 
        #                         sharex=False,
        #                         squeeze=False,
        #                         layout='constrained')
        # axs = axs.flatten()
        
        # in_labels = []
        # for ilab, label in enumerate(label_names[:1]):
            
        for session_id, session in enumerate(sessions):
            
            one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
            # similarities_list = [one_two, one_three, one_four, two_three, two_four, three_four]
        
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
        
        #     per_sess = []
                
        #     for session_id, session in enumerate(sessions):
        #         per_sess.append(np.array(scores_df.loc[(label, session_id), :]))
        
        #     per_sess = np.array(per_sess)

        #     for i, block in zip(range(5), blocks):
        #         axs[ilab].plot(times, per_sess[i].T, )
            
        #     axs[ilab].set_title(label)
        #     axs[ilab].axvspan(0, 0.2, color='grey', alpha=.2)
        #     axs[ilab].axhline(chance, color='black', ls='dashed', alpha=.5)
        #     axs[ilab].set_ylim(round(min_value, 2)-0.015, round(max_value, 2)+0.015)

        # in_labels.append(per_sess)
        # in_labels = np.array(in_labels)
            
        # for j in range(ilab+1, nrows*ncols):
        #     axs[j].axis('off')
        # plt.show()
    
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    # np.save(res_dir / f"{label}.npy", diff_inout)
    
    label_d[label] = diff_inout