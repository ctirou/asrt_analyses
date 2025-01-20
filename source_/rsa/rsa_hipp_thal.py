import mne
import os
import os.path as op
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import pandas as pd
from base import ensure_dir, get_volume_estimate_tc, cv_mahalanobis
from config import *
import gc
import sys

data_path = DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
lock = 'stim'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
overwrite = True
verbose = True

def process_subject(subject, lock):
    # read volume source space
    vol_src_fname =  RESULTS_DIR / 'src' / f"{subject}-hipp-thal-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)

    for region in ['Hippocampus', 'Thalamus']:
        # define results path
        res_dir = RESULTS_DIR / 'RSA' / 'source' / region / lock / 'rdm' / subject
        ensure_dir(res_dir)
            
        for epoch_num in [0, 1, 2, 3, 4]:
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            # read epoch
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
            if lock == 'button': 
                epoch_bsl_fname = data_path / 'bsl' / f'{subject}_{epoch_num}_bl-epo.fif'
                epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose, preload=False)
                # compute noise covariance
                noise_cov = mne.compute_covariance(epoch_bsl, method="oas", rank="info", verbose=verbose)
            else:
                noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="oas", rank="info", verbose=verbose)
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="oas", rank="info", verbose=verbose)
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)    
            # compute forward solution
            fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-hipp-thal-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            # compute source estimates
            filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            # get data from volume source space
            offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
            label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
            # get data from regions of interest
            labels = [label for label in label_tc.keys() if region in label]
            stcs_data = np.concatenate([label_tc[label] for label in labels], axis=1) # this works
            
            del noise_cov, data_cov, filters, fwd
            gc.collect()
            
            print(f"Processing {subject} - {epoch_num} - {region}...")
            if not op.exists(res_dir / f"pat-{epoch_num}.npy") or not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:

                pattern = behav.trialtypes == 1
                X_pat = stcs_data[pattern]
                y_pat = behav.positions[pattern].reset_index(drop=True)
                assert X_pat.shape[0] == y_pat.shape[0]
                rdm_pat = cv_mahalanobis(X_pat, y_pat)
                np.save(res_dir / f"pat-{epoch_num}.npy", rdm_pat)
                
                random = behav.trialtypes == 2
                X_rand = stcs_data[random]
                y_rand = behav.positions[random].reset_index(drop=True)
                assert X_rand.shape[0] == y_rand.shape[0]
                rdm_rand = cv_mahalanobis(X_rand, y_rand)
                np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)
                
                del X_pat, y_pat, X_rand, y_rand
                gc.collect()
            
            del epoch, behav, stcs, stcs_data
            if lock == 'button':
                del epoch_bsl
            gc.collect()
            
if is_cluster:
    lock = str(sys.argv[1])
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock)