import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

overwrite = True
verbose = 'error'

data_path = DATA_DIR
subjects = SUBJS
subjects = ALL_SUBJS
lock = 'stim'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, jobs, verbose):
    
    kf = KFold(n_splits=2, shuffle=False)
    
    res_path = ensured(RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_no_shfl" / subject)
        
    print(f"Processing {subject} - {lock} - {epoch_num}")
                
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    blocks = np.unique(behav["blocks"])
    Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat = [], [], [], []
    Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand = [], [], [], []
    
    for block in blocks:
        block = int(block)
        this_block = behav.blocks == block
        X = data[this_block]
        y = behav[this_block].reset_index(drop=True)
        assert len(X) == len(y), "Data and behavior lengths do not match"

        # Fix: Compute trialtype indices within this block only
        pattern_idx = np.where(y.trialtypes == 1)[0]
        random_idx = np.where(y.trialtypes == 2)[0]

        for train_index, test_index in kf.split(X):

            # Pattern trials
            trainxpat = np.intersect1d(train_index, pattern_idx)
            testxpat = np.intersect1d(test_index, pattern_idx)

            X_train, X_test = X[trainxpat], X[testxpat]
            y_train = y.iloc[trainxpat].positions
            y_test = y.iloc[testxpat].positions

            Xtraining_pat.append(X_train)
            Xtesting_pat.append(X_test)
            ytraining_pat.append(y_train)
            ytesting_pat.append(y_test)

            # Random trials
            trainxrand = np.intersect1d(train_index, random_idx)
            testxrand = np.intersect1d(test_index, random_idx)

            X_train, X_test = X[trainxrand], X[testxrand]
            y_train = y.iloc[trainxrand].positions
            y_test = y.iloc[testxrand].positions

            Xtraining_rand.append(X_train)
            Xtesting_rand.append(X_test)
            ytraining_rand.append(y_train)
            ytesting_rand.append(y_test)
                
    # Pattern per session
    for i, _ in enumerate(Xtesting_pat):
        if not op.exists(res_path / f"pat-{epoch_num}-{i+1}.npy") or overwrite:
            print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} pattern")
            rdm_pat = train_test_mahalanobis_fast(Xtraining_pat[i], Xtesting_pat[i], ytraining_pat[i], ytesting_pat[i], jobs, verbose)
            np.save(res_path / f"pat-{epoch_num}-{i+1}.npy", rdm_pat)
        else:
            print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} pattern already exists")
    
    # Random per session
    for i, _ in enumerate(Xtesting_rand):
        if not op.exists(res_path / f"rand-{epoch_num}-{i+1}.npy") or overwrite:
            print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} random")
            rdm_rand = train_test_mahalanobis_fast(Xtraining_rand[i], Xtesting_rand[i], ytraining_rand[i], ytesting_rand[i], jobs, verbose)
            np.save(res_path / f"rand-{epoch_num}-{i+1}.npy", rdm_rand)
        else:
            print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} random already exists")
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = sys.argv[1]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, epoch_num, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, jobs, verbose) for subject in subjects for epoch_num in range(5))