import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed

data_path = DATA_DIR
subjects = ALL_SUBJS
lock = 'stim'

overwrite = False
verbose = 'error'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, jobs, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "loocv_rdm_blocks_fixed" / subject
    ensure_dir(res_path)
        
    # read behav
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    blocks = np.unique(behav["blocks"])

    for block in blocks:
        block = int(block)
        print(f"Processing {subject} - session {epoch_num} - block {block}")

        if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 1) & (behav["blocks"] == block)        
            X_pat = data[filter]
            y_pat = behav[filter].reset_index(drop=True).positions
            assert len(X_pat) == len(y_pat)
            rdm_pat = loocv_mahalanobis_parallel(X_pat, y_pat, jobs)
            np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
        
        if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 2) & (behav["blocks"] == block)            
            X_rand = data[filter]
            y_rand = behav[filter].reset_index(drop=True).positions
            assert len(X_rand) == len(y_rand)
            rdm_rand = loocv_mahalanobis_parallel(X_rand, y_rand, jobs)
            np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = int(sys.argv[1])
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, epoch_num, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, jobs, verbose) for subject in subjects for epoch_num in range(5))