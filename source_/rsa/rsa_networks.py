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
analysis = 'RSA'
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
parc = 'aparc'
hemi = 'both'
lock = 'stim'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
verbose = True
overwrite = False

def process_subject(subject, epoch_num, lock):
    
    print("Processing", subject, epoch_num, '!!!')
    
    # network and custom label_names
    n_parcels = 200
    n_networks = 7
    # networks = (NEW_LABELS + schaefer_7) if n_networks == 7 else (NEW_LABELS + schaefer_17)
    # networks = schaefer_7 if n_networks == 7 else schaefer_17
    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / f'networks_{n_parcels}_{n_networks}' / subject
    
    # read stim epoch
    epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
    epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
    # read behav
    behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
    behav = pd.read_pickle(behav_fname).reset_index()
    # get session behav and epoch
    if lock == 'button':
        epoch_bsl_fname = data_path / "bsl" / f"{subject}-{epoch_num}-epo.fif"
        epoch_bsl = mne.read_epochs(epoch_bsl_fname, preload=True, verbose=verbose)
        # compute noise covariance
        noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
    else:
        noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    # read forward solution    
    fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    # compute data covariance matrix on evoked data
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    info = epoch.info
    # conpute rank
    rank = mne.compute_rank(data_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)
    # compute source estimates
    filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov, reg=0.05,
                    pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    del epoch, epoch_fname, behav_fname, fwd, data_cov, noise_cov, rank, info, filters
    gc.collect()

    for network in networks[-2:]:
        
        res_dir = RESULTS_DIR / "RSA" / 'source' / network / lock / 'rdm' / subject
        ensure_dir(res_dir)

        print("Processing", subject, epoch_num, network)
                
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')        
        stcs_data = np.array([stc.in_label(lh_label + rh_label).data for stc in stcs])
        
        if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:

            pattern = behav.trialtypes == 1
            X_pat = stcs_data[pattern]
            y_pat = behav.positions[pattern].reset_index(drop=True)
            assert X_pat.shape[0] == y_pat.shape[0]
            rdm_pat = cv_mahalanobis(X_pat, y_pat)
            np.save(res_dir / f"pat-{epoch_num}.npy", rdm_pat)
            del X_pat, y_pat
            gc.collect()
        
        if not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:    
            random = behav.trialtypes == 2
            X_rand = stcs_data[random]
            y_rand = behav.positions[random].reset_index(drop=True)
            assert X_rand.shape[0] == y_rand.shape[0]
            rdm_rand = cv_mahalanobis(X_rand, y_rand)
            np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)
            del X_rand, y_rand
            gc.collect()
        
        del stcs_data, lh_label, rh_label
        gc.collect()

    del stcs
    gc.collect()
            
if is_cluster:
    lock = str(sys.argv[1])
    epoch_num = str(sys.argv[2])
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, epoch_num, lock)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    # for subject in subjects:
    #     for epoch_num in range(5):
    #         process_subject(subject, epoch_num, lock)
    lock = 'stim'
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, lock) for subject in subjects for epoch_num in range(5))