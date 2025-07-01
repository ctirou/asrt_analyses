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

analysis = 'rdm_blocks_vect_0200'

def process_subject(subject, jobs, verbose):

    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    for network in networks:
        
        print(f"Processing subject {subject} in network {network}")

        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / network / "rdm_blocks_vect_0200" / subject)
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
                
                Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(data_path, subject, epoch, fwd, behav, 'vector', lh_label, rh_label, \
                    'random', block, False, verbose=verbose)

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