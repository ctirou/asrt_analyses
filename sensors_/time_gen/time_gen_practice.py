import mne
import os
import os.path as op
import numpy as np
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from base import ensure_dir
from config import *
import gc
# stim disp = 500 ms
# RSI = 750 ms in task
data_path = TIMEG_DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
jobs = -1

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
cv = StratifiedKFold(folds, shuffle=True)

for subject in subjects:

    for trial_type in ['pattern', 'random']:
        
        print(f"Processing {subject} - {trial_type}...")

        res_dir = data_path / 'results' / 'sensors' / lock
        ensure_dir(res_dir)

        epoch_num, epo = 0, epochs_list[0]
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch_gen = mne.read_epochs(epoch_fname, verbose="error", preload=False)
        
        # run time generalization decoding on unique epoch
        if trial_type == 'pattern':
            pattern = behav.trialtypes == 1
            X = epoch_gen.get_data()[pattern]
            y = behav.positions[pattern]
        elif trial_type == 'random':
            random = behav.trialtypes == 2
            X = epoch_gen.get_data()[random]
            y = behav.positions[random]
        else:
            X = epoch_gen.get_data()
            y = behav.positions    
        y = y.reset_index(drop=True)            
        assert X.shape[0] == y.shape[0]
        gc.collect()
        scores = cross_val_multiscore(clf, X, y, cv=cv)
        np.save(res_dir / f"{subject}-epoch0-{trial_type}-scores.npy", scores.mean(0))