import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc
import sys
from joblib import Parallel, delayed

# params
subjects = SUBJS
lock = 'stim'
analysis = 'RSA'
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR

verbose = 'error'
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num):
    
    # read volume source space
    vol_src_fname =  TIMEG_DATA_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
                                     
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work

    del vol_src
    gc.collect()

    # read behav
    behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
    # read epoch
    epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)

    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    # conpute rank
    rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-htc-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    # compute source estimates
    filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori='max-power', weight_norm="unit-noise-gain",
                        rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    # get data from volume source space
    label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
    
    del noise_cov, data_cov, fwd, filters, epoch, stcs
    gc.collect()
    
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_dir = RESULTS_DIR / "RSA" / 'source' / region / lock / 'loocv_rdm_blocks' / subject
        ensure_dir(res_dir)
        
        # get data from regions of interest
        labels = [label for label in label_tc.keys() if region in label]
        stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
        assert len(stcs_data) == len(behav), "Length mismatch"
            
        blocks = np.unique(behav.blocks)
        all_pats, all_rands = [], []

        for block in blocks:
            
            block = int(block)
            print(f"Processing {subject} - session {epoch_num} - block {block} - {region}")

            if not op.exists(res_dir / f"pat-{epoch_num}-{block}.npy") or overwrite:
                filter = (behav["trialtypes"] == 1) & (behav["blocks"] == block)        
                X_pat = stcs_data[filter]
                y_pat = behav[filter].reset_index(drop=True).positions
                assert len(X_pat) == len(y_pat)
                rdm_pat = loocv_mahalanobis(X_pat, y_pat)
                np.save(res_dir / f"pat-{epoch_num}-{block}.npy", rdm_pat)
            else:
                rdm_pat = np.load(res_dir / f"pat-{epoch_num}-{block}.npy")
            all_pats.append(rdm_pat)
            
            if not op.exists(res_dir / f"rand-{epoch_num}-{block}.npy") or overwrite:
                filter = (behav["trialtypes"] == 2) & (behav["blocks"] == block)            
                X_rand = stcs_data[filter]
                y_rand = behav[filter].reset_index(drop=True).positions
                assert len(X_rand) == len(y_rand)
                rdm_rand = loocv_mahalanobis(X_rand, y_rand)
                np.save(res_dir / f"rand-{epoch_num}-{block}.npy", rdm_rand)
            else:
                rdm_rand = np.load(res_dir / f"rand-{epoch_num}-{block}.npy")
            all_rands.append(rdm_rand)
                
        all_pats, nan_pat = interpolate_rdm_nan(np.array(all_pats))
        if nan_pat:
            print(subject, "has pattern nans interpolated in session", epoch_num, region)
        all_rands, nan_rand = interpolate_rdm_nan(np.array(all_rands))
        if nan_rand:
            print(subject, "has random nans interpolated in session", epoch_num, region)
        
        np.save(res_dir / f"pat-{epoch_num}.npy", all_pats)
        np.save(res_dir / f"rand-{epoch_num}.npy", all_rands)

        del labels, stcs_data
        gc.collect()
        
    del label_tc
    gc.collect()
    
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = str(sys.argv[1])
        process_subject(subject, epoch_num)
    
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    # for epoch_num in range(5):
    Parallel(-1)(delayed(process_subject)(subject, epoch_num) for subject in subjects for epoch_num in range(5))