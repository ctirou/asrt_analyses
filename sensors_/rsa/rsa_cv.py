import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed

overwrite = True
verbose = 'error'

data_path = DATA_DIR
subjects = SUBJS
subjects = ['sub03', 'sub06']

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, epoch_num, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm" / subject
    ensure_dir(res_path)
        
    print(f"Processing {subject} - {lock} - {epoch_num}")
                
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
    
    if not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
        X_rand = data[np.where(behav["trialtypes"]==2)]
        y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions
        assert len(X_rand) == len(y_rand)
        rdm_rand = cv_mahalanobis(X_rand, y_rand)
        np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        # lock = sys.argv[1]
        for lock in ['stim', 'button']:
            process_subject(subject, lock, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    Parallel(-1)(delayed(process_subject)(subject, lock, epoch_num, verbose) for subject in subjects for epoch_num in range(5))