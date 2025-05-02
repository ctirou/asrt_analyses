import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed
import gc

data_path = DATA_DIR
subjects = ALL_SUBJS
lock = 'stim'

overwrite = False
verbose = 'error'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs, verbose):
    
    res_path = ensured(RESULTS_DIR / 'RSA' / 'sensors' / lock)
    
    all_Xtraining_pat, all_Xtesting_pat = [], []
    all_ytraining_pat, all_ytesting_pat = [], []

    all_Xtraining_rand, all_Xtesting_rand = [], []
    all_ytraining_rand, all_ytesting_rand = [], []
    
    for epoch_num in range(5):
        
        # read behav
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        data = epoch.get_data(picks='mag', copy=True)
        
        del epoch
        gc.collect()
        
        blocks = np.unique(behav["blocks"])
        
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
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs, verbose) for subject in subjects)