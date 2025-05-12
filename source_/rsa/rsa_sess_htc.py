import mne
import os
import os.path as op
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import pandas as pd
from base import ensure_dir, get_volume_estimate_tc, cv_mahalanobis_parallel
from config import *
import gc
import sys
from joblib import Parallel, delayed

data_path = DATA_DIR
subjects, subjects_dir = SUBJS15, FREESURFER_DIR

verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = RESULTS_DIR / 'RSA' / 'source'
ensure_dir(res_path)

def process_subject(subject, epoch_num, jobs):    
    # read volume source space
    vol_src_fname =  DATA_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    
    del vol_src
    gc.collect()
            
    # read behav and epoch
    behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
    epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
    
    # compute noise covariance
    noise_cov = mne.compute_covariance(epoch, tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)        
    # compute data covariance matrix
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=0.6, method="empirical", rank="info", verbose=verbose)
    # conpute rank
    rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)

    # compute forward solution
    fwd_fname = RESULTS_DIR / "fwd" / f"{subject}-htc-{epoch_num}-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    # compute source estimates
    filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori='max-power', weight_norm="unit-noise-gain",
                        rank=rank, reduce_rank=True, verbose=verbose)
            
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    # get data from volume source space
    label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
    
    del epoch, noise_cov, data_cov, fwd, filters, stcs
    gc.collect()
    
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_dir = res_path / region / 'rdm_sess' / subject
        ensure_dir(res_dir)

        # get data from regions of interest
        labels = [label for label in label_tc.keys() if region in label]
        stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)                
        
        if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:

            pattern = behav.trialtypes == 1
            X_pat = stcs_data[pattern]
            y_pat = behav.positions[pattern].reset_index(drop=True)
            assert X_pat.shape[0] == y_pat.shape[0], "Length mismatch"
            
            _, counts = np.unique(y_pat, return_counts=True)
            if any(counts < 10):
                print(f"Skipping {subject} {epoch_num} due to low counts")
            else:
                print("Processing", subject, "epoch", epoch_num, region, 'pattern')
                rdm_pat = cv_mahalanobis_parallel(X_pat, y_pat, jobs)
                np.save(res_dir / f"pat-{epoch_num}.npy", rdm_pat)
                del X_pat, y_pat
                gc.collect()
        
        if not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:    
            
            random = behav.trialtypes == 2
            X_rand = stcs_data[random]
            y_rand = behav.positions[random].reset_index(drop=True)
            assert X_rand.shape[0] == y_rand.shape[0], "Length mismatch"
            
            _, counts = np.unique(y_rand, return_counts=True)
            if any(counts < 10):
                print(f"Skipping {subject} {epoch_num} due to low counts")
            else:
                print("Processing", subject, "epoch", epoch_num, region, 'random')
                rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, jobs)
                np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)
                del X_rand, y_rand
                gc.collect()
                
        del labels, stcs_data
        gc.collect()
                
if is_cluster:
    jobs = 20
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = str(sys.argv[1])
        process_subject(subject, epoch_num, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, 1) for subject in subjects for epoch_num in range(5))