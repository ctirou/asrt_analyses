import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import KFold
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

networks = NETWORKS[:-2]

def process_subject(subject, jobs, verbose):

    kf = KFold(n_splits=4, shuffle=False)
    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    for network in networks:
    
        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / network / lock)
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        
        all_Xtraining_pat, all_Xtesting_pat = [], []
        all_ytraining_pat, all_ytesting_pat = [], []

        all_Xtraining_rand, all_Xtesting_rand = [], []
        all_ytraining_rand, all_ytesting_rand = [], []
    
        for epoch_num in range(5):
        
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
            fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
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
            Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat = [], [], [], []
            Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand = [], [], [], []

            for block in blocks:
                block = int(block)
                
                this_block = behav.blocks == block
                X = data[this_block]
                y = behav[this_block].reset_index(drop=True)
                assert len(X) == len(y), "Data and behavior lengths do not match"

                # Fix: Compute trialtype indices within this block only
                pattern_idx = np.where(y.trialtypes == 1)[0]
                random_idx = np.where(y.trialtypes == 2)[0]

                for train_index, test_index in kf.split(X):

                    # Pattern trials
                    trainxpat = np.intersect1d(train_index, pattern_idx)
                    testxpat = np.intersect1d(test_index, pattern_idx)

                    X_train, X_test = X[trainxpat], X[testxpat]
                    y_train = y.iloc[trainxpat].positions
                    y_test = y.iloc[testxpat].positions

                    Xtraining_pat.append(X_train)
                    Xtesting_pat.append(X_test)
                    ytraining_pat.append(y_train)
                    ytesting_pat.append(y_test)

                    if epoch_num != 0:
                        all_Xtraining_pat.append(X_train)
                        all_Xtesting_pat.append(X_test)
                        all_ytraining_pat.append(y_train)
                        all_ytesting_pat.append(y_test)

                    # Random trials
                    trainxrand = np.intersect1d(train_index, random_idx)
                    testxrand = np.intersect1d(test_index, random_idx)

                    X_train, X_test = X[trainxrand], X[testxrand]
                    y_train = y.iloc[trainxrand].positions
                    y_test = y.iloc[testxrand].positions

                    Xtraining_rand.append(X_train)
                    Xtesting_rand.append(X_test)
                    ytraining_rand.append(y_train)
                    ytesting_rand.append(y_test)

                    if epoch_num != 0:
                        all_Xtraining_rand.append(X_train)
                        all_Xtesting_rand.append(X_test)
                        all_ytraining_rand.append(y_train)
                        all_ytesting_rand.append(y_test)
                        
                del X, y
                gc.collect()
            
            del behav, data
            gc.collect()
            
            # Pattern per session
            res_dir = ensured(res_path / 'split_20s_pattern' / subject)
            for i, _ in enumerate(Xtesting_pat):
                if not op.exists(res_dir / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} pattern")
                    rdm_pat = train_test_mahalanobis_fast(Xtraining_pat[i], Xtesting_pat[i], ytraining_pat[i], ytesting_pat[i], jobs)
                    np.save(res_dir / f"{subject}-{epoch_num}-{i+1}.npy", rdm_pat)
                else:
                    print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} pattern already exists")
            
            # Random per session
            res_dir = ensured(res_path / 'split_20s_random' / subject)
            for i, _ in enumerate(Xtesting_rand):
                if not op.exists(res_dir / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} random")
                    rdm_rand = train_test_mahalanobis_fast(Xtraining_rand[i], Xtesting_rand[i], ytraining_rand[i], ytesting_rand[i], jobs)
                    np.save(res_dir / f"{subject}-{epoch_num}-{i+1}.npy", rdm_rand)
                else:
                    print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} random already exists")
        
        # Pattern all sessions
        res_dir = ensured(res_path / "split_20s_all_pattern" / subject)
        for i, _ in enumerate(all_Xtesting_pat):
            if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
                print(f"Computing Mahalanobis for quarter {i+1} for {subject} all pattern")
                rdm_pat = train_test_mahalanobis_fast(all_Xtraining_pat[i], all_Xtesting_pat[i], all_ytraining_pat[i], all_ytesting_pat[i], jobs)
                np.save(res_dir / f"{subject}-{i+1}.npy", rdm_pat)
            else:
                print(f"Mahalanobis for quarter {i+1} for {subject} all pattern already exists")
        
        # Random all sessions
        res_dir = ensured(res_path / "split_20s_all_random" / subject)
        for i, _ in enumerate(all_Xtesting_rand):
            if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
                print(f"Computing Mahalanobis for quarter {i+1} for {subject} all random")
                rdm_rand = train_test_mahalanobis_fast(all_Xtraining_rand[i], all_Xtesting_rand[i], all_ytraining_rand[i], all_ytesting_rand[i], jobs)
                np.save(res_dir / f"{subject}-{i+1}.npy", rdm_rand)
            else:
                print(f"Mahalanobis for quarter {i+1} for {subject} all random already exists")
                                
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