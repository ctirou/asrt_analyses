import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'
folds = 10
solver = 'lbfgs'
scoring = "roc_auc"
verbose = True
overwrite = False

all_in_seqs, all_out_seqs = [], []

def process_subject(subject, lock, scoring, solver, folds, jobs, verbose):
    from mne.decoding import SlidingEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from base import get_cm
    
    # get times
    epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
    epochs = mne.read_epochs(epoch_fname, verbose=verbose)
    times = epochs.times
    
    # set-up the classifier and cv structure
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, multi_class="ovr", max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True)
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cm" / subject
    ensure_dir(res_path)
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
                    
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname)
        
        if not op.exists(res_path / f"pat-cm-{epoch_num}.npy") or overwrite:
            X_pat = epoch[np.where(behav["trialtypes"]==1)].get_data(copy=False).mean(axis=0)
            y_pat = behav[behav["trialtypes"]==1].positions.reset_index(drop=True)
            assert len(X_pat) == len(y_pat)
            cm_pat, score_pat = get_cm(clf, cv, X_pat, y_pat, times)
            np.save(res_path / f"pat-cm-{epoch_num}.npy", cm_pat)
            np.save(res_path / f"pat-score-{epoch_num}.npy", score_pat)

        
        if not op.exists(res_path / f"rand-cm-{epoch_num}.npy") or overwrite:
            X_rand = epoch[np.where(behav["trialtypes"]==2)].get_data(copy=False).mean(axis=0)
            y_rand = behav[behav["trialtypes"]==2].positions.reset_index(drop=True)
            assert len(X_rand) == len(y_rand)
            cm_rand, score_rand = get_cm(clf, cv, X_rand, y_rand, times)
            np.save(res_path / f"rand-cm-{epoch_num}.npy", cm_rand)
            np.save(res_path / f"rand-score-{epoch_num}.npy", score_rand)

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 10
        lock = sys.argv[1]
        process_subject(subject, lock, scoring, solver, folds, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    lock = 'stim'
    for subject in subjects:
        process_subject(subject, lock, scoring, solver, folds, jobs, verbose)