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
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs, rsync):
    
    # path to bem file
    bem_fname = op.join(RESULTS_DIR, "bem", "%s-bem-sol.fif" % (subject))
    # read volume and surface source space
    src_fname = op.join(RESULTS_DIR, "src", "%s-src.fif" % (subject))
    src = mne.read_source_spaces(src_fname, verbose=verbose)        
    vol_src_fname = op.join(RESULTS_DIR, "src", "%s-vol-src.fif" % (subject))
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    mixed_src = src + vol_src
            
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
            
        # compute data covariance matrix on evoked data
        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)

        info = epoch.info
        # conpute rank
        rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)
        trans_fname = os.path.join(RESULTS_DIR, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
        fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                    src=mixed_src, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
        # compute source estimates
        filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                        pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])
        label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
        # subcortex labels
        labels_subcx = list(label_tc.keys())
        
        del epoch, epoch_fname, behav_fname, fwd, data_cov, noise_cov, rank, info, filters
        gc.collect()

        # loop across labels
        for ilabel, label in enumerate(labels_subcx):
            print(f"{str(ilabel+1).zfill(2)}/{len(labels_subcx)}", subject, epoch_num, label)
            
            res_path = RESULTS_DIR / analysis / 'source' / label / lock / 'rdm' / subject

            if not op.exists(res_path / f"pat-{epoch_num}.npy") or not op.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
                ensure_dir(res_path)

                # get stcs in label
                stcs_data = label_tc[label]
                assert len(stcs_data) == len(behav)

                pattern = behav.trialtypes == 1
                X_pat = stcs_data[pattern]
                y_pat = behav.positions[pattern]
                assert X_pat.shape[0] == y_pat.shape[0]
                rdm_pat = get_rdm(X_pat, y_pat)
                np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)

                random = behav.trialtypes == 2
                X_rand = stcs_data[random]
                y_rand = behav.positions[random]
                assert X_rand.shape[0] == y_rand.shape[0]
                rdm_rand = get_rdm(X_rand, y_rand)
                np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
            
                del stcs_data, X_pat, y_pat, X_rand, y_rand
                gc.collect()
    
        del stcs
        gc.collect()
    
    if rsync:
        source = RESULTS_DIR / analysis / 'source'
        destination = "coum@crnl-dycog369.crnl.local:/Users/coum/Desktop/asrt/results/RSA/"
        options = "-av"
        rsync_files(source, destination, options)
    
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