import os
import os.path as op
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
lock = 'stim'
analysis = 'RSA'
data_path = DATA_DIR / 'for_rsa'
subjects_dir = FREESURFER_DIR

verbose = 'error'
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

networks = NETWORKS[:-2]

def process_subject(subject, jobs, verbose):

    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    for network in networks:
    
        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / network / "rdm_blocks" / subject)
        
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        
        for epoch_num in range(5):
        
            # read behav
            behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
            behav = pd.read_pickle(behav_fname).reset_index(drop=True)
            behav['trials'] = behav.index            
            # read epoch
            epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)

            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
            noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
            # conpute rank
            rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
            # read forward solution
            fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            # compute source estimates
            filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='max-power', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            
            data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs])
            assert len(data) == len(behav), "Length mismatch"

            del stcs, noise_cov, data_cov, fwd, filters, epoch
            gc.collect()
            
            blocks = np.unique(behav["blocks"])
                
            for block in blocks:
                block = int(block)
                
                # pattern trials
                pat = behav.trialtypes == 1
                this_block = behav.blocks == block
                out_blocks = behav.blocks != block
                pat_this_block = pat & this_block
                pat_out_blocks = pat & out_blocks
                yob = behav[pat_out_blocks]
                ytb = behav[pat_this_block]
                Xtrain = data[yob.trials.values]
                ytrain = yob.positions
                Xtest = data[ytb.trials.values]
                ytest = ytb.positions
                assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
                assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
                if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
                    print(f"Computing Mahalanobis for {subject} epoch {epoch_num} block {block} pattern")
                    rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
                else:
                    print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} pattern already exists")
                
                # random trials        
                rand = behav.trialtypes == 2
                this_block = behav.blocks == block
                out_blocks = behav.blocks != block
                rand_this_block = rand & this_block
                rand_out_blocks = rand & out_blocks
                yob = behav[rand_out_blocks]
                ytb = behav[rand_this_block]
                Xtrain = data[yob.trials.values]
                ytrain = yob.positions
                Xtest = data[ytb.trials.values]
                ytest = ytb.positions
                assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
                assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
                if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
                    print(f"Computing Mahalanobis for {subject} epoch {epoch_num} block {block} random")
                    rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
                else:
                    print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} random already exists")

            del data, behav
            gc.collect()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs, verbose) for subject in subjects)