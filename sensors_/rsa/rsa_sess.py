import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed
import gc

overwrite = True
verbose = 'error'

data_path = DATA_DIR / 'for_rsa_new'
subjects = SUBJS15

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / "rdm_skf_new" / subject
    ensure_dir(res_path)
        
    print(f"Processing {subject} - {epoch_num}")
                
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    if not op.exists(res_path / f"pat-{epoch_num}.npy") or overwrite:
        X_pat = data[np.where(behav["trialtypes"]==1)]
        y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions
        assert len(X_pat) == len(y_pat), "X_pat and y_pat lengths do not match"
        rdm_pat = cv_mahalanobis_parallel(X_pat, y_pat, shuffle=True)
        np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
    
    if not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
        X_rand = data[np.where(behav["trialtypes"]==2)]
        y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions
        assert len(X_rand) == len(y_rand), "X_rand and y_rand lengths do not match"
        rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, shuffle=True)
        np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = sys.argv[1]
        process_subject(subject, epoch_num, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, verbose) for subject in subjects for epoch_num in range(5))

data_path = DATA_DIR / 'for_rsa'
subjects = SUBJS15

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / "rdm_skf2" / subject
    ensure_dir(res_path)

    # Practice first
    behav_fname = op.join(data_path, "behav/%s-0.pkl" % (subject))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "epochs", "%s-0-epo.fif" % (subject))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    if not op.exists(res_path / "pat-prac.npy") or overwrite:
        X_pat = data[np.where(behav["trialtypes"]==1)]
        y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions
        assert len(X_pat) == len(y_pat), "X_pat and y_pat lengths do not match"
        rdm_pat = cv_mahalanobis_parallel(X_pat, y_pat, shuffle=True)
        np.save(res_path / "pat-prac.npy", rdm_pat)
    
    if not op.exists(res_path / "rand-prac.npy") or overwrite:
        X_rand = data[np.where(behav["trialtypes"]==2)]
        y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions
        assert len(X_rand) == len(y_rand), "X_rand and y_rand lengths do not match"
        rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, shuffle=True)
        np.save(res_path / "rand-prac.npy", rdm_rand)
        
    # Now the learning epochs
    all_behavs = []
    all_epochs = []
    for epoch_num in range(1, 5):
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        all_epochs.append(epoch)
    
    behavs = pd.concat(all_behavs, ignore_index=True)
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    data = mne.concatenate_epochs(all_epochs).get_data(picks='mag', copy=True)
    
    del all_behavs, all_epochs
    gc.collect()
        
    if not op.exists(res_path / "pat-learn.npy") or overwrite:
        X_pat = data[np.where(behavs["trialtypes"]==1)]
        y_pat = behavs[behavs["trialtypes"]==1].reset_index(drop=True).positions
        assert len(X_pat) == len(y_pat), "X_pat and y_pat lengths do not match"
        rdm_pat = cv_mahalanobis_parallel(X_pat, y_pat, shuffle=True)
        np.save(res_path / "pat-learn.npy", rdm_pat)
    
    if not op.exists(res_path / "rand-learn.npy") or overwrite:
        X_rand = data[np.where(behavs["trialtypes"]==2)]
        y_rand = behavs[behavs["trialtypes"]==2].reset_index(drop=True).positions
        assert len(X_rand) == len(y_rand), "X_rand and y_rand lengths do not match"
        rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, shuffle=True)
        np.save(res_path / "rand-learn.npy", rdm_rand)
        
        
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = sys.argv[1]
        process_subject(subject, epoch_num, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    # Parallel(-1)(delayed(process_subject)(subject, epoch_num, verbose) for subject in subjects for epoch_num in range(5))
    Parallel(-1)(delayed(process_subject)(subject, verbose) for subject in subjects)
    