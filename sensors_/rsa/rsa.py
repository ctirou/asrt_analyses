import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys

lock = 'stim'
overwrite = True

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "rdm" / subject
    ensure_dir(res_path)
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
                    
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname)
        
        if not op.exists(res_path / f"pat-{epoch_num}.npy") or overwrite:
            epoch_pat = epoch[np.where(behav["trialtypes"]==1)].get_data(copy=False).mean(axis=0)
            behav_pat = behav[behav["trialtypes"]==1]
            assert len(epoch_pat) == len(behav_pat)
            rdm_pat = get_rdm(epoch_pat, behav_pat)
            np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
        else:
            rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
        
        if not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
            epoch_rand = epoch[np.where(behav["triplets"]==34)].get_data(copy=False).mean(axis=0)
            behav_rand = behav[behav["triplets"]==34]
            assert len(epoch_rand) == len(behav_rand)
            rdm_rand = get_rdm(epoch_rand, behav_rand)
            np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        else:
            rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject)