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
data_path = DATA_DIR / 'for_rsa_new'
subjects_dir = FREESURFER_DIR

verbose = 'error'
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

# pick_ori = 'max-power'
pick_ori = 'vector'
analysis = 'rdm_lobo_vect' if pick_ori == 'vector' else 'rdm_lobo_maxpower'
analysis += "_new"

def process_subject(subject, jobs, verbose):
    
    # read volume source space
    vol_src_fname = RESULTS_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)                     
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    del vol_src
    gc.collect()

    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        # read behav
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        all_epochs.append(epoch)
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epochs)
    assert len(epoch) == len(behav)
    
    del all_epochs, all_behavs
    gc.collect()

    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-htc-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    behav.loc[behav.sessions != 0, 'blocks'] += 3                
    blocks = np.unique(behav["blocks"])
                
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_path = ensured(RESULTS_DIR / "RSA" / 'source' / region / analysis / subject)

        for block in blocks:
            block = int(block)
            
            # random trials
            if not op.exists(res_path / f"rand-{block}.npy") or overwrite:
                print(f"Computing Mahalanobis for {subject} block {block} random")
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(epoch, fwd, behav, pick_ori, 'random', block, blocks, verbose=verbose)
                label_tc_train, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc_train.keys() if region in label]
                Xtrain = np.concatenate([np.real(label_tc_train[label]) for label in labels], axis=1)
                if pick_ori == 'vector':
                    Xtrain = svd(Xtrain)

                label_tc_test, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)
                Xtest = np.concatenate([np.real(label_tc_test[label]) for label in labels], axis=1)
                if pick_ori == 'vector':
                    Xtest = svd(Xtest)
                rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                np.save(res_path / f"rand-{block}.npy", rdm_rand)
            else:
                print(f"Mahalanobis for {subject} block {block} random already exists")

            # pattern trials
            if not op.exists(res_path / f"pat-{block}.npy") or overwrite:
                print(f"Computing Mahalanobis for {subject} block {block} pattern")
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(epoch, fwd, behav, pick_ori, 'pattern', block, blocks, verbose=verbose)
                label_tc_train, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc_train.keys() if region in label]
                Xtrain = np.concatenate([np.real(label_tc_train[label]) for label in labels], axis=1)
                if pick_ori == 'vector':
                    Xtrain = svd(Xtrain)

                label_tc_test, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)
                Xtest = np.concatenate([np.real(label_tc_test[label]) for label in labels], axis=1)
                if pick_ori == 'vector':
                    Xtest = svd(Xtest)
                rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                np.save(res_path / f"pat-{block}.npy", rdm_pat)
            else:
                print(f"Mahalanobis for {subject} block {block} pattern already exists")

if is_cluster:
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