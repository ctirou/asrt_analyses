import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

data_path = DATA_DIR / 'for_rsa'
subjects = SUBJS15
lock = 'stim'
overwrite = True
verbose = 'error'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, epoch_num, jobs, verbose):
    
    print(f"Processing {subject} - {lock} - {epoch_num}")
    
    res_path = ensured(RESULTS_DIR / 'RSA' / 'sensors' / "rdm_40s" / subject)
    kf = KFold(n_splits=2, shuffle=False)    
                
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname).reset_index(drop=True)
    behav['trials'] = behav.index
    # read epochs
    epoch_fname = op.join(data_path / "epochs" / f"{subject}-{epoch_num}-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    assert len(data) == len(behav), "Data and behavior lengths do not match"
    
    blocks = np.unique(behav["blocks"])
        
    for block in blocks:
        block = int(block)

        pat = behav.trialtypes == 1
        this_block = behav.blocks == block
        pat_this_block = pat & this_block
        ypat = behav[pat_this_block]
        
        for i, (_, test_index) in enumerate(kf.split(ypat)):
            
            test_in_ypat = ypat.iloc[test_index].trials.values
                            
            test_idx = [i for i in behav.trials.values if i in test_in_ypat]
            train_idx = [i for i in behav.trials.values if i not in test_in_ypat]
            
            ytrain = [behav.iloc[i].positions for i in train_idx]
            ytest = [behav.iloc[i].positions for i in test_idx]
            
            Xtraining = data[train_idx]
            Xtesting = data[test_idx]
            
            assert len(Xtraining) == len(ytrain), "Xtraining and ytrain lengths do not match"
            assert len(Xtesting) == len(ytest), "Xtesting and ytest lengths do not match"
    
            if not op.exists(res_path / f"pat-{epoch_num}-{block}-{i+1}.npy") or overwrite:
                print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} block {block} pattern")
                rdm_pat = train_test_mahalanobis_fast(Xtraining, Xtesting, ytrain, ytest, jobs, verbose)
                np.save(res_path / f"pat-{epoch_num}-{block}-{i+1}.npy", rdm_pat)
            else:
                print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} block {block} pattern already exists")
        
        rand = behav.trialtypes == 2
        rand_this_block = rand & this_block
        yrand = behav[rand_this_block]
        
        for i, (_, test_index) in enumerate(kf.split(yrand)):
            test_in_yrand = yrand.iloc[test_index].trials.values
            
            test_idx = [i for i in behav.trials.values if i in test_in_yrand]
            train_idx = [i for i in behav.trials.values if i not in test_in_yrand]
            
            ytrain = [behav.iloc[i].positions for i in train_idx]
            ytest = [behav.iloc[i].positions for i in test_idx]
            
            Xtraining = data[train_idx]
            Xtesting = data[test_idx]
            
            assert len(Xtraining) == len(ytrain), "Xtraining and ytrain lengths do not match"
            assert len(Xtesting) == len(ytest), "Xtesting and ytest lengths do not match"
            
            if not op.exists(res_path / f"rand-{epoch_num}-{block}-{i+1}.npy") or overwrite:
                print(f"Computing Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} block {block} random")
                rdm_rand = train_test_mahalanobis_fast(Xtraining, Xtesting, ytrain, ytest, jobs, verbose)
                np.save(res_path / f"rand-{epoch_num}-{block}-{i+1}.npy", rdm_rand)
            else:
                print(f"Mahalanobis for quarter {i+1} for {subject} epoch {epoch_num} block {block} random already exists")
            
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = sys.argv[1]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 20))
        process_subject(subject, epoch_num, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, jobs, verbose) for subject in subjects for epoch_num in range(5))