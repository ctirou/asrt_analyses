import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import ensure_dir, get_volume_estimate_time_course
from config import *
import gc
import os
from tqdm.auto import tqdm

# params
subjects = SUBJS

analysis = "decoding"
lock = "button" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
jobs = -1

overwrite = False


# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
cv = StratifiedKFold(folds, shuffle=True)

for subject in subjects:
    
    for trial_type in ['random', 'all']:
        
        res_path = RESULTS_DIR / 'decoding' / 'sensors' / lock / trial_type
        ensure_dir(res_path)

        print(f"Processing {subject}")
                
        epo_dir = data_path / lock
        epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
        all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
        for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
            epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
        epoch = mne.concatenate_epochs(all_epo)

        beh_dir = data_path / 'behav'
        beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
        all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
        behav = pd.concat(all_beh)

        if lock == 'button': 
            bsl_data = data_path / "bsl"
            epoch_bsl_fnames = [bsl_data / f"{f}" for f in sorted(os.listdir(bsl_data)) if ".fif" in f and subject in f]
            all_bsl = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epoch_bsl_fnames]
            for epo in all_bsl:
                epo.info['dev_head_t'] = all_epo[0].info['dev_head_t']
            epoch_bsl = mne.concatenate_epochs(all_bsl)
            
        if not op.exists(res_path / f"{subject}-all-scores.npy") or overwrite:
            if trial_type == 'pattern':
                pattern = behav.trialtypes == 1
                X = epoch.get_data()[pattern]
                y = behav.positions[pattern]
            elif trial_type == 'random':
                random = behav.trialtypes == 2
                X = epoch.get_data()[random]
                y = behav.positions[random]
            else:
                X = epoch.get_data()
                y = behav.positions
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]

            del epoch
            gc.collect()

            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
            np.save(res_path / f"{subject}-all-scores.npy", scores.mean(0))
                
            del X, y, scores
            gc.collect()