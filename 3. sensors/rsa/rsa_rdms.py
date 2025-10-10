# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_rsa'
subjects = SUBJS15
overwrite = True
verbose = True

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, jobs, verbose):
    
    print(f"Processing {subject} - {epoch_num}")
    
    res_path = ensured(RESULTS_DIR / 'RSA' / 'sensors' / "rdm_blocks_new" / subject)
    
    # read behav        
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname).reset_index(drop=True)
    behav['trials'] = behav.index
    
    # read epochs
    epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    assert len(data) == len(behav), "Data and behavior lengths do not match"
    
    blocks = np.unique(behav["blocks"])
        
    for block in blocks:
        block = int(block)
        
        # pattern trials
        pat = behav.trialtypes == 1
        this_block = behav.blocks == block
        out_blocks = behav.blocks != block
        pat_this_block = pat & this_block
        pat_out_blocks = pat & out_blocks
        yob = behav[pat_out_blocks]
        ytb = behav[pat_this_block]
        Xtrain = data[yob.trials.values]
        ytrain = yob.positions
        Xtest = data[ytb.trials.values]
        ytest = ytb.positions
        assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
        assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
        if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
            print(f"Computing Mahalanobis for {subject} epoch {epoch_num} block {block} pattern")
            rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
            np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
        else:
            print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} pattern already exists")
        
        # random trials        
        rand = behav.trialtypes == 2
        this_block = behav.blocks == block
        out_blocks = behav.blocks != block
        rand_this_block = rand & this_block
        rand_out_blocks = rand & out_blocks
        yob = behav[rand_out_blocks]
        ytb = behav[rand_this_block]
        Xtrain = data[yob.trials.values]
        ytrain = yob.positions
        Xtest = data[ytb.trials.values]
        ytest = ytb.positions
        assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
        assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
        if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
            print(f"Computing Mahalanobis for {subject} epoch {epoch_num} block {block} random")
            rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
            np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
        else:
            print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} random already exists")
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = sys.argv[1]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, epoch_num, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, jobs, verbose) for subject in subjects for epoch_num in range(5))