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
params = "new_decoding"
sessions = EPOCHS
res_dir = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type / 'test'
subjects_dir = FREESURFER_DIR
verbose = True
hemi = 'lh'
chance = .25

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

for subject in subjects[:1]:
    
    scores_df = pd.read_csv(res_dir / f"{subject}_scores.csv", sep='\t', index_col=[0, 1])
    labels = read_labels_from_annot(subject=subject, parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    label_names = [label.name for label in labels]
    
    max_value = scores_df.max().max()
    min_value = scores_df.min().min()
    
    nrows=5
    ncols=8
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=False, layout='constrained')
    axs = axs.flatten()
    
    in_labels = []
    for ilab, label in enumerate(label_names):
        
        per_sess = []
            
        for session_id, session in enumerate(sessions[:3]):
            per_sess.append(np.array(scores_df.loc[(label, session_id), :]))
    
        per_sess = np.array(per_sess)

        # axs[ilab].plot(times, per_sess.flatten())
        # axs[ilab].set_title(label)
        # axs[ilab].axvspan(0, 0.2, color='grey', alpha=.2)
        # axs[ilab].axhline(chance, color='black', ls='dashed', alpha=.5)
        # axs[ilab].set_ylim(round(min_value, 2)-0.015, round(max_value, 2)+0.015)

    in_labels.append(per_sess)
    in_labels = np.array(in_labels)
        
    # for j in range(ilab+1, nrows*ncols):
    #     axs[j].axis('off')
    # plt.show()