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

def process_subject(subject, jobs, verbose):
    
    # read volume source space
    vol_src_fname =  TIMEG_DATA_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)                     
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    del vol_src
    gc.collect()

    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / region / lock)
        
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
                                        
            # get data from regions of interest
            labels = [label for label in label_tc.keys() if region in label]
            data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
            assert len(data) == len(behav), "Length mismatch"
                
            blocks = np.unique(behav.blocks)
            Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat = [], [], [], []
            Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand = [], [], [], []

            for block in blocks:
                block = int(block)
                
                this_block = behav.blocks == block
                out_blocks = behav.blocks != block

                pattern = behav.trialtypes == 1        
                X_train = data[out_blocks & pattern]
                y_train = behav[out_blocks & pattern].reset_index(drop=True).positions            
                X_test = data[this_block & pattern]
                y_test = behav[this_block & pattern].reset_index(drop=True).positions
                
                Xtraining_pat.append(X_train)
                Xtesting_pat.append(X_test)
                ytraining_pat.append(y_train)
                ytesting_pat.append(y_test)
                
                if epoch_num != 0:
                    all_Xtraining_pat.append(X_train)
                    all_Xtesting_pat.append(X_test)
                    all_ytraining_pat.append(y_train)
                    all_ytesting_pat.append(y_test)
                
                random = behav.trialtypes == 2
                X_train = data[out_blocks & random]
                y_train = behav[out_blocks & random].reset_index(drop=True).positions
                X_test = data[this_block & random]
                y_test = behav[this_block & random].reset_index(drop=True).positions
                
                Xtraining_rand.append(X_train)
                Xtesting_rand.append(X_test)
                ytraining_rand.append(y_train)
                ytesting_rand.append(y_test)
                
                if epoch_num != 0:
                    all_Xtraining_rand.append(X_train)
                    all_Xtesting_rand.append(X_test)
                    all_ytraining_rand.append(y_train)
                    all_ytesting_rand.append(y_test)
                            
            del data
            gc.collect()
            
        res_dir = ensured(res_path / "split_pattern")
        for i, _ in enumerate(Xtesting_pat):
            if not op.exists(res_dir / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                print(f"Processing {subject} - session {epoch_num} - block {i+1}")
                rdm_pat = train_test_mahalanobis_fast(Xtraining_pat[i], Xtesting_pat[i], ytraining_pat[i], ytesting_pat[i], n_jobs=jobs)
                np.save(res_dir / f"{subject}-{epoch_num}-{i+1}.npy", rdm_pat)
            else:
                print(f"File {res_dir / f'{subject}-{epoch_num}-{i+1}.npy'} already exists, skipping.")
        del Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat
        gc.collect()
                
        res_dir = ensured(res_path / "split_random")
        for i, _ in enumerate(Xtesting_rand):
            if not op.exists(res_path / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                print(f"Processing {subject} - session {epoch_num} - block {i+1}")
                rdm_rand = train_test_mahalanobis_fast(Xtraining_rand[i], Xtesting_rand[i], ytraining_rand[i], ytesting_rand[i], n_jobs=jobs)
                np.save(res_path / f"{subject}-{epoch_num}-{i+1}.npy", rdm_rand)
            else:
                print(f"File {f'{subject}-{epoch_num}-{i+1}.npy'} already exists, skipping.")
        del Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand
        gc.collect()
    
    res_dir = ensured(res_path / "split_all_pattern")
    for i, _ in enumerate(all_Xtesting_pat):
        if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
            print(f"Processing {subject} - block {i+1}")
            rdm_pat = train_test_mahalanobis_fast(all_Xtraining_pat[i], all_Xtesting_pat[i], all_ytraining_pat[i], all_ytesting_pat[i], n_jobs=jobs)
            np.save(res_dir / f"{subject}-{i+1}.npy", rdm_pat)
        else:
            print(f"File {f'{subject}-{i+1}.npy'} already exists, skipping.")
    del all_Xtraining_pat, all_Xtesting_pat, all_ytraining_pat, all_ytesting_pat
    gc.collect()
    
    res_dir = ensured(res_path / "split_all_random")
    for i, _ in enumerate(all_Xtesting_rand):
        if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
            print(f"Processing {subject} - block {i+1}")
            rdm_rand = train_test_mahalanobis_fast(all_Xtraining_rand[i], all_Xtesting_rand[i], all_ytraining_rand[i], all_ytesting_rand[i], n_jobs=jobs)
            np.save(res_dir / f"{subject}-{i+1}.npy", rdm_rand)
        else:
            print(f"File {f'{subject}-{i+1}.npy'} already exists, skipping.")
    del all_Xtraining_rand, all_Xtesting_rand, all_ytraining_rand, all_ytesting_rand
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
    # run on local machine
    jobs = 1    
    Parallel(-1)(delayed(process_subject)(subject, jobs, verbose) for subject in subjects)