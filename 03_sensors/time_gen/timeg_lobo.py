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

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

analysis = "scores_blocks"

set_log_level(verbose)

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    res_path = ensured(RESULTS_DIR / 'TIMEG' / 'sensors' / analysis / subject)
    
    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        # read behav        
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = read_epochs(epoch_fname, verbose=verbose)
        all_epochs.append(epoch)
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    data = concatenate_epochs(all_epochs).get_data(picks='mag', copy=True)
    assert len(data) == len(behav)
    
    del all_epochs, all_behavs
    gc.collect()
    
    behav.loc[behav.sessions != 0, 'blocks'] += 3
    
    blocks = np.unique(behav.blocks)
    rng = np.random.default_rng(seed=42)
    
    for block in blocks:
        block = int(block)
        this_block = behav.blocks == block
        
        if block in blocks[:3]:
            rand_blocks = rng.choice(blocks[3:], size=19, replace=False)
            out_blocks = behav.blocks.isin(rand_blocks) & (behav.blocks != block)
        else:
            out_blocks = (behav.blocks != block) & (behav.sessions != 0)

        for b, w in zip(behav.blocks, out_blocks):
            print(f"Block: {b}, Out: {w}")

        # pattern trials
        pat = behav.trialtypes == 1
        pat_out_blocks = pat & out_blocks
        pat_this_block = pat & this_block
        yob = behav[pat_out_blocks]
        ytb = behav[pat_this_block]
        Xtrain = data[yob.trials.values]
        ytrain = yob.positions
        Xtest = data[ytb.trials.values]
        ytest = ytb.positions
        assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
        assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
        print(f"Training samples: {len(ytrain)}, Test samples: {len(ytest)}")
        
        if not op.exists(res_path / f"pat-{block}.npy") or overwrite:
            clf.fit(Xtrain, ytrain)
            acc_matrix = clf.score(Xtest, ytest)
            np.save(res_path / f"pat-{block}.npy", acc_matrix)
        else:
            print(f"Pattern for {subject} block {block} already exists")

        # random trials        
        rand = behav.trialtypes == 2
        rand_out_blocks = rand & out_blocks
        rand_this_block = rand & this_block
        yob = behav[rand_out_blocks]
        ytb = behav[rand_this_block]
        Xtrain = data[yob.trials.values]
        ytrain = yob.positions
        Xtest = data[ytb.trials.values]
        ytest = ytb.positions
        assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
        assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
        print(f"Training samples: {len(ytrain)}, Test samples: {len(ytest)}")
        
        if not op.exists(res_path / f"rand-{block}.npy") or overwrite:
            clf.fit(Xtrain, ytrain)
            acc_matrix = clf.score(Xtest, ytest)
            np.save(res_path / f"rand-{block}.npy", acc_matrix)
        else:
            print(f"Random for {subject} block {block} already exists")

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