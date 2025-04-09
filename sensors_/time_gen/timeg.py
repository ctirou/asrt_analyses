import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import gc
from base import ensure_dir
from config import *
from mne import read_epochs, concatenate_epochs
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from joblib import Parallel, delayed

data_path = TIMEG_DATA_DIR / 'gen44'
subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):    
    # define classifier
    # clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
        
    all_behavs = list()
    all_epochs = list()
    
    for epoch_num in [0, 1, 2, 3, 4]:
        
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch_gen = read_epochs(epoch_fname, verbose=verbose, preload=True)
        
        times = epoch_gen.times
        win = np.where((times >= -1.5) & (times <= 3))[0]        
        
        for trial_type in ['pattern', 'random']:
            res_dir = TIMEG_DATA_DIR / 'results' / 'sensors' / lock / f'{trial_type}_logRegCV'
            ensure_dir(res_dir)
        
            if not op.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
                print(f"Processing {subject} - session {epoch_num} - {trial_type}...")
                if trial_type == 'pattern':
                    pattern = behav.trialtypes == 1
                    X = epoch_gen.get_data()[pattern][:, :, win]
                    y = behav.positions[pattern]
                elif trial_type == 'random':
                    random = behav.trialtypes == 2
                    X = epoch_gen.get_data()[random][:, :, win]
                    y = behav.positions[random]
                y = y.reset_index(drop=True)            
                assert X.shape[0] == y.shape[0]
                gc.collect()
                
                cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
                scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
                np.save(res_dir / f"{subject}-{epoch_num}-scores.npy", scores.mean(0))
            else:
                print(f"Skipping {subject} - session {epoch_num} - {trial_type}...")
        
        if epoch_num != 0:
            all_epochs.append(epoch_gen)
            all_behavs.append(behav)
        
    # concatenate epochs and behav
    for epoch in all_epochs:
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = concatenate_epochs(all_epochs)
    
    behav_df = pd.concat(all_behavs)
    behav_data = behav_df.reset_index(drop=True)
    
    del all_epochs, all_behavs, epoch_gen, behav, behav_df
    gc.collect()
    
    for trial_type in ['pattern', 'random']:
        res_dir = TIMEG_DATA_DIR / 'results' / 'sensors' / lock / f'{trial_type}_logRegCV'
        ensure_dir(res_dir)
        
        if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
            print(f"Processing {subject} - all - {trial_type}...") 
            if trial_type == 'pattern':
                pattern = behav_data.trialtypes == 1
                X = epochs.get_data()[pattern][:, :, win]
                y = behav_data.positions[pattern]
            elif trial_type == 'random':
                random = behav_data.trialtypes == 2
                X = epochs.get_data()[random][:, :, win]
                y = behav_data.positions[random]
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]
            
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
            np.save(res_dir / f"{subject}-all-scores.npy", scores.mean(0))
            del X, y, scores
            gc.collect()
        else:
            print(f"Skipping {subject} - all - {trial_type}...")
    
    del epochs, behav_data
    gc.collect()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 20
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)