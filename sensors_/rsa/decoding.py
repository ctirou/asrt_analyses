import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config import *
import gc
import os
import matplotlib.pyplot as plt

# params
subjects = SUBJS + ['sub03', 'sub06']
lock = "stim"
data_path = DATA_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
jobs = -1
verbose = True

subject = 'sub05'

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
skf = StratifiedKFold(folds, shuffle=True, random_state=42)
loo = LeaveOneOut()
            
epo_dir = data_path / lock
epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f and not f.startswith('.')]

all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
    epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']

beh_dir = data_path / 'behav'
beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f and not f.startswith('.')]
all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
        
trial_type = 'random'  # 'pattern' or 'random'
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 5))

for i, ax in enumerate(axes.flatten()[:-1]):
        
    epoch = all_epo[i].copy()
    behav = all_beh[i].copy()
    
    print("Processing", subject, trial_type)
    filter = behav.trialtypes == 1 if trial_type == 'pattern' else behav.trialtypes == 2
    X = epoch.get_data(copy=False)[filter]
    y = behav.positions[filter].reset_index(drop=True)
    assert X.shape[0] == y.shape[0], "Shape mismatch"
    
    cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
    scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=verbose)
    
    times = np.linspace(-0.2, 0.6, X.shape[2])
    
    ax.axvspan(0, 0.2, color='grey', alpha=0.1)
    ax.axhline(0.25, color='k', linestyle='--')
    ax.plot(times, scores.mean(0))
    ax.set_title(f"{subject} {trial_type} Epoch {i}", fontstyle='italic')
    
    del X, y, scores
    gc.collect()
            