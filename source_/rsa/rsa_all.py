import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc
import sys

# params
subjects = SUBJS

analysis = 'RSA'
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
parc = 'aparc'
hemi = 'both'
verbose = True
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs, rsync):
    # # get label names
    # best_labels = SURFACE_LABELS if lock == 'stim' else SURFACE_LABELS_RT

    # # get labels
    # all_labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    # labels = [label for label in all_labels if label.name in best_labels]
    
    # del all_labels
    # gc.collect()
    
    res_path = RESULTS_DIR / analysis / 'source' / 'all-nopca' / lock / 'rdm' / subject
    ensure_dir(res_path)
    
    for epoch_num in [0, 1, 2, 3, 4]:
        
        # read stim epoch
        epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
        epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
        # read behav
        behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
        behav = pd.read_pickle(behav_fname).reset_index()
        # get session behav and epoch
        if lock == 'button': 
            epoch_bsl_fname = data_path / "bsl" / f"{subject}-{epoch_num}-epo.fif"
            epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
            # compute noise covariance
            noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
        else:
            noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
        # compute data covariance matrix
        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
        # conpute rank
        rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        # compute source estimates
        filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                        pick_ori=None, rank='info', reg=0.05, reduce_rank=True, verbose=verbose)    
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        del epoch, epoch_fname, behav_fname, fwd, data_cov, noise_cov, rank, filters
        gc.collect()
        
        stcs_data = np.array([stc.data for stc in stcs])
        assert len(stcs_data) == len(behav)

        pattern = behav.trialtypes == 1
        X_pat = stcs_data[pattern]
        y_pat = behav.positions[pattern].reset_index(drop=True)
        assert X_pat.shape[0] == y_pat.shape[0]
        rdm_pat = cv_mahalanobis(X_pat, y_pat)
        np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)

        random = behav.trialtypes == 2
        X_rand = stcs_data[random]
        y_rand = behav.positions[random].reset_index(drop=True)
        assert X_rand.shape[0] == y_rand.shape[0]
        rdm_rand = cv_mahalanobis(X_rand, y_rand)
        np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        
        del stcs, stcs_data, X_pat, y_pat, X_rand, y_rand
        gc.collect()
    

    #     # loop across labels
    #     for ilabel, label in enumerate(labels):
    #         print(f"{str(ilabel+1).zfill(2)}/{len(labels)}", subject, epoch_num, label.name)
            
    #         res_path = RESULTS_DIR / analysis / 'source' / label.name / lock / 'rdm' / subject

    #         if not op.exists(res_path / f"pat-{epoch_num}.npy") or not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
    #             ensure_dir(res_path)

    #             # get stcs in label
    #             stcs_data = [stc.in_label(label).data for stc in stcs]
    #             stcs_data = np.array(stcs_data)
    #             assert len(stcs_data) == len(behav)

    #             pattern = behav.trialtypes == 1
    #             X_pat = stcs_data[pattern]
    #             y_pat = behav.positions[pattern]
    #             assert X_pat.shape[0] == y_pat.shape[0]
    #             rdm_pat = get_rdm(X_pat, y_pat)
    #             np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)

    #             random = behav.trialtypes == 2
    #             X_rand = stcs_data[random]
    #             y_rand = behav.positions[random]
    #             assert X_rand.shape[0] == y_rand.shape[0]
    #             rdm_rand = get_rdm(X_rand, y_rand)
    #             np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
            
    #         del stcs_data, X_pat, y_pat, X_rand, y_rand
    #         gc.collect()
    
    # del labels, stcs
    # gc.collect()
    
    # if rsync:
    #     source = RESULTS_DIR / analysis / 'source'
    #     destination = "coum@crnl-dycog369.crnl.local:/Users/coum/Desktop/asrt/results/RSA/"
    #     options = "-av"
    #     rsync_files(source, destination, options)
    
if is_cluster:
    lock = str(sys.argv[1])
    jobs = 10
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock, jobs, rsync=True)
    
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    for subject in subjects:
        process_subject(subject, lock, jobs=-1, rsync=False)