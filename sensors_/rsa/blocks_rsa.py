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
lock = 'stim'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm_blocks" / subject
    ensure_dir(res_path)
        
    # read behav
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    blocks = np.unique(behav.blocks)
    
    for block in blocks:
        
        block = int(block)
        
        print(f"Processing {subject} - session {epoch_num} - block {block}")

        if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 1) & (behav["blocks"] == block)        
            X_pat = data[filter]
            y_pat = behav[filter].reset_index(drop=True).positions
            
            max_split = min(np.unique(y_pat, return_counts=True)[1])
            splits = min(10, max_split) if max_split > 1 else 2

            assert len(X_pat) == len(y_pat)
            rdm_pat = cv_mahalanobis(X_pat, y_pat, n_splits=splits)
            np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
        
        if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 2) & (behav["blocks"] == block)            
            X_rand = data[filter]
            y_rand = behav[filter].reset_index(drop=True).positions
            
            max_split = min(np.unique(y_rand, return_counts=True)[1])
            splits = min(10, max_split) if max_split > 1 else 2

            assert len(X_rand) == len(y_rand)
            rdm_rand = cv_mahalanobis(X_rand, y_rand, n_splits=splits)
            np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        # lock = sys.argv[1]
        process_subject(subject, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, verbose) for subject in subjects for epoch_num in range(5))

from tqdm.auto import tqdm

analysis = 'pat_high_rdm_high'

all_highs, all_lows = [], []
patterns, randoms = [], []

for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"cv_rdm_blocks" / subject
    ensure_dir(res_path)
        
    # RSA stuff
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    high, low = get_all_high_low(res_path, sequence, analysis, cv=True)    
    all_highs.append(high)
    all_lows.append(low)
    
all_highs = np.array(all_highs)
all_lows = np.array(all_lows)

high = all_highs[:, :, 1:, :].mean((1, 2)) - all_highs[:, :, 0, :].mean(axis=1)
low = all_lows[:, :, 1:, :].mean((1, 2)) - all_lows[:, :, 0, :].mean(axis=1)
diff = low - high

diff_sess = list()   
for i in range(5):
    rev_low = all_lows[:, :, i, :].mean(1) - all_lows[:, :, 0, :].mean(axis=1)
    rev_high = all_highs[:, :, i, :].mean(1) - all_highs[:, :, 0, :].mean(axis=1)
    diff_sess.append(rev_low - rev_high)
diff_sess = np.array(diff_sess).swapaxes(0, 1)
