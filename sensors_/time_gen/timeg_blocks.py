import os
import sys
from config import *
from joblib import Parallel, delayed
import os.path as op
import pandas as pd
import numpy as np
import gc
from base import ensured
from mne import read_epochs
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    res_path = ensured(RESULTS_DIR / 'TIMEG' / 'sensors' / "scores_blocks" / subject)
    
    for epoch_num in [0, 1, 2, 3, 4]:
        
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl')).reset_index(drop=True)
        behav['trials'] = behav.index
        
        # read epoch
        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = read_epochs(epoch_fname, verbose=verbose, preload=True)
        
        data = epoch.get_data(picks='mag', copy=True)
        assert len(behav) == len(data)
        
        del epoch
        gc.collect()
        
        blocks = np.unique(behav["blocks"])        
        
        for block in blocks:
            block = int(block)
            this_block = behav.blocks == block
            out_blocks = behav.blocks != block
            
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
            
            clf.fit(Xtrain, ytrain)
            if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
                ypred = clf.predict(Xtest)
                print(f"Scoring pattern for {subject} epoch {epoch_num} block {block}")
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                np.save(res_path / f"pat-{epoch_num}-{block}.npy", acc_matrix)
            else:
                print(f"Pattern for {subject} epoch {epoch_num} block {block} already exists")

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
            
            clf.fit(Xtrain, ytrain)
            if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
                ypred = clf.predict(Xtest)
                print(f"Scoring random for {subject} epoch {epoch_num} block {block}")
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                np.save(res_path / f"rand-{epoch_num}-{block}.npy", acc_matrix)
            else:
                print(f"Random for {subject} epoch {epoch_num} block {block} already exists")

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