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
subjects = SUBJS15
data_path = DATA_DIR / 'for_rsa'
subjects_dir = FREESURFER_DIR

verbose = 'error'
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, jobs):

    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / 'networks_200_7' / subject
            
    # read behav
    behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
    # read epoch
    epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)

    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    # conpute rank
    rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / 'for_rsa' / f"{subject}-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    # compute source estimates
    filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori='max-power', weight_norm="unit-noise-gain",
                        rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    del noise_cov, data_cov, fwd, filters, epoch
    gc.collect()
    
    for network in networks:
        
        res_dir = ensured(RESULTS_DIR / "RSA" / 'source' / network / 'rdm_skf' / subject)
        
        if not op.exists(res_dir / f"pat-{epoch_num}.npy") or not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:
            # read labels
            lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
            stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs])
            assert len(stcs_data) == len(behav), "Length mismatch"
            
            if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:

                pattern = behav.trialtypes == 1
                X_pat = stcs_data[pattern]
                y_pat = behav.positions[pattern].reset_index(drop=True)
                assert X_pat.shape[0] == y_pat.shape[0], "Length mismatch"
                
                _, counts = np.unique(y_pat, return_counts=True)
                if any(counts < 10):
                    print(f"Skipping {subject} {epoch_num} due to low counts")
                else:
                    print("Processing", subject, "epoch", epoch_num, network, 'pattern')
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
                    print("Processing", subject, "epoch", epoch_num, network, 'random')
                    rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, jobs)
                    np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)
                    del X_rand, y_rand
                    gc.collect()
            
            del stcs_data, lh_label, rh_label
            gc.collect()
        
        else:
            print("Files already exist, skipping", subject, epoch_num, network)

    del stcs
    gc.collect()
    
if is_cluster:
    epoch_num = str(sys.argv[1])
    jobs = 20
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, epoch_num, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, 1) for subject in subjects for epoch_num in range(5))