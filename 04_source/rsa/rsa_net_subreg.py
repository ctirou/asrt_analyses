# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import mne
from base import *
from config import *
from joblib import Parallel, delayed
import gc

data_path = DATA_DIR / 'for_rsa'
subjects = SUBJS15
# subjects = ['sub09', 'sub10']
lock = 'stim'
verbose = 'error'
overwrite = False

networks = ['SomMot', 'DorsAttn']
parc = "Schaefer2018_200Parcels_7Networks"
hemi = 'both'
subjects_dir = FREESURFER_DIR
pick_ori = 'vector'
weight_norm = "unit-noise-gain-invariant"

pick_ori = 'vector'
analysis = 'rdm_blocks_subreg'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

# subject = subjects[0]
# jobs = -1

def process_subject(subject, jobs):

    # Read labels per network (list of Label objects per network)
    labels_per_network = {
        network: mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, regexp=network, verbose=verbose)
        for network in networks
    }

    # Flat list of (label, network) tuples — one entry per sub-region parcel
    all_labels = [
        (label, network)
        for network in networks
        for label in labels_per_network[network]
    ]

    # Cache output paths per sub-region (one subdirectory per parcel label)
    res_paths = {
        label.name: ensured(RESULTS_DIR / "RSA" / 'source' / network / analysis / subject / label.name)
        for label, network in all_labels
    }

    for epoch_num in [1, 2, 3, 4]:
        # read behav
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        behav['trials'] = behav.index
        # read epochs
        epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

        blocks = np.unique(behav["blocks"])

        for block in blocks:
            block = int(block)

            # RANDOM: only compute source estimates if at least one output is missing.
            random_pending = [
                (label, network)
                for label, network in all_labels
                if overwrite or not op.exists(res_paths[label.name] / f"rand-{epoch_num}-{block}.npy")]

            if random_pending:
                stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'random', block, blocks, rsa=True, verbose=verbose)

                for label, network in random_pending:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} | {label.name} (network {network}) block {block} random")
                    Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                    Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)
                    rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_paths[label.name] / f"rand-{epoch_num}-{block}.npy", rdm_rand)
                    del Xtrain, Xtest, rdm_rand
                del stcs_train, stcs_test, ytrain, ytest
                gc.collect()

            else:
                print(f"Random outputs already exist for subject {subject}, epoch {epoch_num}, block {block}")

            # PATTERN: only compute source estimates if at least one output is missing.
            pattern_pending = [
                (label, network)
                for label, network in all_labels
                if overwrite or not op.exists(res_paths[label.name] / f"pat-{epoch_num}-{block}.npy")]

            if pattern_pending:
                stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'pattern', block, blocks, rsa=True, verbose=verbose)

                for label, network in pattern_pending:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} | {label.name} (network {network}) block {block} pattern")
                    Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                    Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)
                    rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_paths[label.name] / f"pat-{epoch_num}-{block}.npy", rdm_pat)
                    del Xtrain, Xtest, rdm_pat
                del stcs_train, stcs_test, ytrain, ytest
                gc.collect()

            else:
                print(f"Pattern outputs already exist for subject {subject}, epoch {epoch_num}, block {block}")

        del epoch, behav, fwd
        gc.collect()

    del labels_per_network, all_labels, res_paths
    gc.collect()

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    for subject in subjects:
        process_subject(subject, jobs)
