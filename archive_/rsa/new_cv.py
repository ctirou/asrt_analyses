import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import ensure_dir
from config import *
import sys
from dissimilarity import *
from sklearn.discriminant_analysis import _cov
import matplotlib.pyplot as plt

lock = 'stim'
verbose = "error"
data_type = 'new_k10_rdm'

data_path = DATA_DIR
subjects = SUBJS

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock):

    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / data_type / subject
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

        X_pat = data[np.where(behav["trialtypes"]==1)]
        y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions

        X_rand = data[np.where(behav["trialtypes"]==2)]
        y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions

        rdm_pat = cv_mahalanobis(X_pat, y_pat)
        rdm_rand = cv_mahalanobis(X_rand, y_rand)

        np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
        np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        lock = str(sys.argv[1])
        process_subject(subject, lock)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock)