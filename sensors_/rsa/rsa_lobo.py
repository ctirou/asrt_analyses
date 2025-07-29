import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed
import gc

data_path = DATA_DIR / 'for_rsa'
subjects = SUBJS15
overwrite = True
verbose = True

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs, verbose):
        
    res_path = ensured(RESULTS_DIR / 'RSA' / 'sensors' / "rdm_lobo" / subject)
    
    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        # read behav
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        all_epochs.append(epoch)
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    data = mne.concatenate_epochs(all_epochs).get_data(picks='mag', copy=True)
    assert len(data) == len(behav)
    
    del all_epochs, all_behavs
    gc.collect()
    
    # rename blocks columns
    behav.loc[behav.sessions != 0, 'blocks'] += 3
                
    blocks = np.unique(behav.blocks)
            
    for block in blocks:
        block = int(block)
        
        this_block = behav.blocks == block
        if block in blocks[:3]:
            rand_blocks = np.random.choice(blocks[3:], size=19, replace=False)
            out_blocks = behav.blocks.isin(rand_blocks)
        else:
            out_blocks = (behav.blocks != block) & (behav.sessions != 0)

        # pattern trials
        pat = behav.trialtypes == 1
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
        if not op.exists(res_path / f"pat-{block}.npy") or overwrite:
            print(f"Computing Mahalanobis for {subject} block {block} pattern")
            rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
            np.save(res_path / f"pat-{block}.npy", rdm_pat)
        else:
            print(f"Mahalanobis for {subject} block {block} pattern already exists")
        
        # random trials        
        rand = behav.trialtypes == 2
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
        if not op.exists(res_path / f"rand-{block}.npy") or overwrite:
            print(f"Computing Mahalanobis for {subject} block {block} random")
            rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
            np.save(res_path / f"rand-{block}.npy", rdm_rand)
        else:
            print(f"Mahalanobis for {subject} block {block} random already exists")

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs, verbose) for subject in subjects)