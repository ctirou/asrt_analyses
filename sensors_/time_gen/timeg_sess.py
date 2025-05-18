import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import gc
from base import ensure_dir, ensured
from config import *
from mne import read_epochs, concatenate_epochs
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
        
    all_behavs = list()
    all_epochs = list()
    
    for epoch_num in [0, 1, 2, 3, 4]:
        
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
        epoch = read_epochs(epoch_fname, verbose=verbose, preload=True)
        
        res_dir = ensured(DATA_DIR / 'TIMEG' / 'sensors' / 'scores_skf' / subject)
    
        print(f"Processing {subject} - session {epoch_num} - pattern...")
        if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:
            pattern = behav.trialtypes == 1
            X = epoch.get_data()[pattern]
            y = behav.positions[pattern]
            y = y.reset_index(drop=True)
            assert X.shape[0] == y.shape[0]
            gc.collect()
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
            np.save(res_dir / f"pat-{epoch_num}.npy", scores.mean(0))

        print(f"Processing {subject} - session {epoch_num} - random...")                
        if not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:
            random = behav.trialtypes == 2
            X = epoch.get_data()[random]
            y = behav.positions[random]
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]
            gc.collect()
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
            np.save(res_dir / f"rand-{epoch_num}.npy", scores.mean(0))
        
        if epoch_num != 0:
            all_epochs.append(epoch)
            all_behavs.append(behav)
        
    # concatenate epochs and behav
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = concatenate_epochs(all_epochs)
    
    behav_df = pd.concat(all_behavs)
    behav_data = behav_df.reset_index(drop=True)
    
    del all_epochs, all_behavs, epoch, behav, behav_df
    gc.collect()
    
    res_dir = ensured(DATA_DIR / 'TIMEG' / 'sensors' / 'scores_skf' / subject)
    
    print(f"Processing {subject} - session all - pattern...")
    if not op.exists(res_dir / "pat-all.npy") or overwrite:
        pattern = behav_data.trialtypes == 1
        X = epochs.get_data()[pattern]
        y = behav_data.positions[pattern]
        y = y.reset_index(drop=True)
        assert X.shape[0] == y.shape[0]
        cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
        scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
        np.save(res_dir / "pat-all.npy", scores.mean(0))
        del X, y, scores
        gc.collect()
    
    print(f"Processing {subject} - session all - random...")
    if not op.exists(res_dir / "rand-all.npy") or overwrite:
        random = behav_data.trialtypes == 2
        X = epochs.get_data()[random]
        y = behav_data.positions[random]
        y = y.reset_index(drop=True)            
        assert X.shape[0] == y.shape[0]
        cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
        scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose, n_jobs=jobs)
        np.save(res_dir / "rand-all.npy", scores.mean(0))
        del X, y, scores
        gc.collect()

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