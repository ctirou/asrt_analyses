import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys

overwrite = True
verbose = 'error'

data_path = DATA_DIR
subjects = SUBJS

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "k10_rdm" / subject
    ensure_dir(res_path)
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
        
        print(f"Processing {subject} - {lock} - {epoch_num} - K10")
                    
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        data = epoch.get_data(picks='mag', copy=True)
        
        if not op.exists(res_path / f"pat-{epoch_num}.npy") or overwrite:
            X_pat = data[np.where(behav["trialtypes"]==1)]
            y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions
            assert len(X_pat) == len(y_pat)
            rdm_pat = cv_mahalanobis(X_pat, y_pat)
            np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
        else:
            rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
        
        if not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
            X_rand = data[np.where(behav["trialtypes"]==2)]
            y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions
            assert len(X_rand) == len(y_rand)
            rdm_rand = cv_mahalanobis(X_rand, y_rand)
            np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        else:
            rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")
            
def process_subject2(subject, lock, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "loocv_rdm" / subject
    ensure_dir(res_path)
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
        
        print(f"Processing {subject} - {lock} - {epoch_num} - LOOCV")
                    
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        data = epoch.get_data(picks='mag', copy=True)
        
        if not op.exists(res_path / f"pat-{epoch_num}.npy") or overwrite:
            X_pat = data[np.where(behav["trialtypes"]==1)]
            y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions
            assert len(X_pat) == len(y_pat)
            rdm_pat = loocv_mahalanobis(X_pat, y_pat)
            np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
        else:
            rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
        
        if not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
            X_rand = data[np.where(behav["trialtypes"]==2)]
            y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions
            assert len(X_rand) == len(y_rand)
            rdm_rand = loocv_mahalanobis(X_rand, y_rand)
            np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        else:
            rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        # lock = sys.argv[1]
        for lock in ['stim', 'button']:
            process_subject(subject, lock, verbose)
            process_subject2(subject, lock, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for lock in ['stim', 'button']:
        for subject in subjects:
            # process_subject(subject, lock, verbose)
            process_subject2(subject, lock, verbose)