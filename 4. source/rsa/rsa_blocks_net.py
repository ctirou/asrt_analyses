import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
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
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

networks = NETWORKS[:-3]  # Exclude 'Hippocampus', 'Thalamus', 'Cerebellum-Cortex' for this analysis

pick_ori = 'vector'
analysis = 'rdm_blocks'

def process_subject(subject, jobs, verbose):

    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    for network in networks:
        
        print(f"Processing subject {subject} in network {network}")

        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / network / analysis / subject)
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        
        for epoch_num in range(5):
        
            # read behav
            behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
            behav = pd.read_pickle(behav_fname).reset_index(drop=True)
            behav['trials'] = behav.index            
            # read epoch
            epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)

            # read forward solution
            fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            
            blocks = np.unique(behav["blocks"])
                
            for block in blocks:
                block = int(block)
                
                # random trials
                if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} block {block} random")
                    Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                        'random', block, verbose=verbose)
                    rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_path / f"rand-{epoch_num}-{block}.npy", rdm_rand)
                    del Xtrain, ytrain, Xtest, ytest, rdm_rand
                    gc.collect()
                else:
                    print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} random already exists")

                # pattern trials
                if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} block {block} pattern")
                    Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                        'pattern', block, verbose=verbose)
                    rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_path / f"pat-{epoch_num}-{block}.npy", rdm_pat)
                    del Xtrain, ytrain, Xtest, ytest, rdm_pat
                    gc.collect()
                else:
                    print(f"Mahalanobis for {subject} epoch {epoch_num} block {block} pattern already exists")
                
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