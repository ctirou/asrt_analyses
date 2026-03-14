
# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import sys
from config import *
from joblib import Parallel, delayed
import os.path as op
import pandas as pd
import numpy as np
import gc
from base import ensured
from mne import set_log_level, read_epochs, concatenate_epochs
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Levenshtein import editops

data_path = DATA_DIR / 'for_timeg_reordered'
subjects = SUBJS15
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

analysis = "scores_blocks_reordered"

set_log_level(verbose)

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def int_to_unicode(array):
        return ''.join([str(chr(int(ii))) for ii in array]) # permet de convertir int en unicode (pour editops)

def process_subject(subject, jobs):
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    res_path = ensured(RESULTS_DIR / 'TIMEG' / 'sensors' / analysis / subject)
    
    all_epochs, all_behavs = [], []
    for epoch_num in range(1, 5):
        # read behav        
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = read_epochs(epoch_fname, verbose=verbose)
        all_epochs.append(epoch)
    
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = concatenate_epochs(all_epochs)
    
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav = behav[behav.trialtypes == 1].reset_index(drop=True)  # keep only pattern trials
    
    changes = editops(int_to_unicode(behav.positions), int_to_unicode(np.array(epochs.events[:, 2])))
    
    if len(changes) !=0:
        del_from_behav = list()
        for change in changes:
            if change[0] == 'delete':
                del_from_behav.append(change[1])
        behav.drop(behav.index[del_from_behav], inplace=True)
    behav = behav.reset_index(drop=True)
    assert len(behav) == len(np.array(epochs.events[:, 2])), "Length of behav and events do not match after editops"

    data = epochs.get_data(picks='mag', copy=True)
    
    del all_epochs, epochs, all_behavs
    gc.collect()
    
    blocks = np.unique(behav.blocks)
    
    for block in blocks:
        block = int(block)
        
        this_block = behav.blocks == block
        out_blocks = behav.blocks != block

        pat = behav.trialtypes == 1
        pat_out_blocks = pat & out_blocks
        pat_this_block = pat & this_block
        
        yob = behav[pat_out_blocks]
        ytb = behav[pat_this_block]
        
        Xtrain = data[yob.index]
        ytrain = yob.positions
        
        Xtest = data[ytb.index]
        ytest = ytb.positions
        
        assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
        assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
        print(f"Training samples: {len(ytrain)}, Test samples: {len(ytest)}")
        
        if not op.exists(res_path / f"scores-{block}.npy") or overwrite:
            clf.fit(Xtrain, ytrain)
            acc_matrix = clf.score(Xtest, ytest)
            np.save(res_path / f"scores-{block}.npy", acc_matrix)
        else:
            print(f"Data for{subject} block {block} already exists")

    del data, behav
    gc.collect()

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 20))
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)