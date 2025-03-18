import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed

overwrite = False
verbose = 'error'

data_path = DATA_DIR
subjects = SUBJS
lock = 'stim'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, verbose):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "loocv_rdm_blocks" / subject
    ensure_dir(res_path)
        
    # read behav
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    blocks = np.unique(behav["blocks"])
    all_pats = []
    all_rands = []
    
    for block in blocks:
        block = int(block)
        print(f"Processing {subject} - session {epoch_num} - block {block}")

        if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 1) & (behav["blocks"] == block)        
            X_pat = data[filter]
            y_pat = behav[filter].reset_index(drop=True).positions
            assert len(X_pat) == len(y_pat)
            rdm_pat = loocv_mahalanobis(X_pat, y_pat)
            np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
        else:
            rdm_pat = np.load(res_path / f"pat-{epoch_num}-{block}.npy")
        all_pats.append(rdm_pat)
        
        if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
            filter = (behav["trialtypes"] == 2) & (behav["blocks"] == block)            
            X_rand = data[filter]
            y_rand = behav[filter].reset_index(drop=True).positions
            assert len(X_rand) == len(y_rand)
            rdm_rand = loocv_mahalanobis(X_rand, y_rand)
            np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
        else:
            rdm_rand = np.load(res_path / f"rand-{epoch_num}-{block}.npy")
        all_rands.append(rdm_rand)
            
    all_pats, nan_pat = interpolate_rdm_nan(np.array(all_pats))
    if nan_pat:
        print(subject, "has pattern nans interpolated in session", epoch_num)
    all_rands, nan_rand = interpolate_rdm_nan(np.array(all_rands))
    if nan_rand:
        print(subject, "has random nans interpolated in session", epoch_num)
    
    np.save(res_path / f"pat-{epoch_num}.npy", all_pats)
    np.save(res_path / f"rand-{epoch_num}.npy", all_rands)

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, verbose) for subject in subjects for epoch_num in range(5))


# plot it
from tqdm.auto import tqdm
all_highs, all_lows = {}, {}
times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.28) & (times <= 0.51))[0]

liste = ['01', '02', '03'] + [str(j) for j in range(1, 21)]
for i in liste:
    all_highs[i] = []
    all_lows[i] = []

for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"cv_rdm_blocks" / subject        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    high, low = new_get_all_high_low(res_path, sequence)
    
    for i in range(1, 5):
        if np.isnan(np.sum(high[i])):
            print(subject, 'has a high nan in sess', i)
        if np.isnan(np.sum(low[i])):
            print(subject, 'has a low nan in sess', i)
            
                
    for sess in range(5):
        if sess == 0:
            for block, blockin in zip(range(4), ['01','02','03']):
                all_highs[blockin].append(high[sess][block, win].mean())
                all_lows[blockin].append(low[sess][block, win].mean())
        else:
            for blockin, block in zip(range(1 + 5 * (sess - 1), 6 + 5 * (sess - 1)), range(5)):
                all_highs[str(blockin)].append(high[sess][block, win].mean())
                all_lows[str(blockin)].append(low[sess][block, win].mean())

diff = []
l_bsl, h_bsl = [], []
for i in liste:
    if i in ['01', '02', '03']:
        diff.append(np.mean(all_lows[i]) - np.mean(all_highs[i]))
        l_bsl.append(np.mean(all_lows[i]))
        h_bsl.append(np.mean(all_highs[i]))
    else:
        l = all_lows[i] - np.nanmean(l_bsl)
        h = all_highs[i] - np.nanmean(h_bsl)
        diff.append(np.mean(l) - np.mean(h))
diff = np.array(diff)

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
