import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from base import ensure_dir
from config import *
import gc
import os
import sys

# params
subjects = SUBJS

lock = "stim"
data_path = TIMEG_DATA_DIR / "rdm_bsling"
subjects_dir = FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
overwrite = False
verbose = True

def process_subject(subject, trial_type, jobs):
    # set-up the classifier and cv structure
    # clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
                
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f and not f.startswith('.') and ('pat' in f if trial_type == 'pattern' else 'rand' in f)]
    
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epo)
    
    times = epoch.times
    win = np.where((times >= -0.2) & (times <= 0.6))[0]

    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f and not f.startswith('.')]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)
    
    # res_dir = RESULTS_DIR / 'decoding' / 'sensors' / lock / trial_type / 'max-power'
    res_dir = RESULTS_DIR / 'decoding' / 'sensors' / lock / trial_type / 'max-power-cv'
    ensure_dir(res_dir)
    
    if not op.exists(res_dir / f"{subject}-scores.npy") or overwrite:
        print("Processing", subject, trial_type)
        X = epoch.get_data(copy=False)[:, :, win]
        filter = behav.trialtypes == 1 if trial_type == 'pattern' else behav.trialtypes == 2
        y = behav.positions[filter].reset_index(drop=True)
        assert X.shape[0] == y.shape[0], "Shape mismatch"
        
        del epoch, behav
        gc.collect()
        
        cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
        scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=verbose)
        np.save(res_dir / f"{subject}-scores.npy", scores.mean(0))
            
        del X, y, scores
        gc.collect()
    
    else:
        print("Skipping", subject, trial_type)
        del epoch, behav
        gc.collect()
        

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        trial_type = sys.argv[1]
        process_subject(subject, trial_type, jobs=20)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        for trial_type in ['pattern', 'random']:
            process_subject(subject, trial_type, jobs=-1)