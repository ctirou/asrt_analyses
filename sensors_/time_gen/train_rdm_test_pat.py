import os
import os.path as op
import numpy as np
import mne
from mne.decoding import CSP
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegressionCV, LogisticRegression
import matplotlib.pyplot as plt
from jr.gat import scorer_spearman
from sklearn.metrics import make_scorer
from base import *
from config import *
import pandas as pd

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = PRED_PATH
analysis = 'time_generalization'
subjects, epochs_list, subjects_dir = SUBJS, EPOCHS, FREESURFER_DIR
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
jobs = 10
verbose = True
res_path = data_path / 'results' / 'source'
ensure_dir(res_path)

# lock = str(sys.argv[1])
# subject_num = int(sys.argv[2])
# subject = subjects[subject_num]

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
cv = StratifiedKFold(folds, shuffle=True)

# ensure_dir(figures / trial_type)
# practice
scores_0 = []
# blocks 
scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []

for subject in subjects:
    
    all_epochs, all_behavs = [], []
    
    for epoch_num, epo, score in zip([1, 2, 3, 4], epochs_list[1:], [scores_1, scores_2, scores_3, scores_4]):
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
        
        pattern = behav.trialtypes == 1
        random = behav.trialtypes == 2
        
        Xrand = epoch.get_data()[random]
        yrand = np.array(behav.positions[random])
        
        Xpat = epoch.get_data()[pattern]
        ypat = np.array(behav.positions[pattern])
        
        for train, test in cv.split():
            continue