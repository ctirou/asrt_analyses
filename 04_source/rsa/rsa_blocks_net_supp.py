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

pick_ori = 'vector'
analysis = 'rdm_blocks'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):

    label_path = RESULTS_DIR / 'networks_200_7' / subject

    networks = ['SomMot-precentral', 'SomMot-postcentral', 'SomMot-paracentral', \
        'DorsAttn-superiorparietal', 'DorsAttn-caudalmiddlefrontal', \
        'Cont-rostralmiddlefrontal', 'Cont-superiorfrontal', 'Cont-parsopercularis', 'Cont-parstriangularis', 'Cont-supramarginal']

    labels = {
        network: (
            mne.read_label(label_path / f'{network}-lh.label')
            + mne.read_label(label_path / f'{network}-rh.label')
        )
        for network in networks
    }

    # Cache output paths once per subject/network.
    res_paths = {
        network: ensured(RESULTS_DIR / "RSA" / 'source' / network / analysis / subject)
        for network in networks
    }

    for epoch_num in range(5):
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
                network
                for network in networks
                if overwrite or not op.exists(res_paths[network] / f"rand-{epoch_num}-{block}.npy")]

            if random_pending:
                stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'random', block, blocks, verbose)

                for network in random_pending:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} in network {network} block {block} random")
                    label = labels[network]
                    Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                    Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)
                    rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_paths[network] / f"rand-{epoch_num}-{block}.npy", rdm_rand)
                    del Xtrain, Xtest, rdm_rand
                del stcs_train, stcs_test, ytrain, ytest
                gc.collect()
                 
            else:
                print(f"Random outputs already exist for subject {subject}, epoch {epoch_num}, block {block}")

            # PATTERN: only compute source estimates if at least one output is missing.
            pattern_pending = [
                network
                for network in networks
                if overwrite or not op.exists(res_paths[network] / f"pat-{epoch_num}-{block}.npy")]

            if pattern_pending:
                stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'pattern', block, blocks, verbose)

                for network in pattern_pending:
                    print(f"Processing Mahalanobis for {subject} epoch {epoch_num} in network {network} block {block} pattern")
                    label = labels[network]
                    Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                    Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)
                    rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, ytrain, ytest, jobs, verbose)
                    np.save(res_paths[network] / f"pat-{epoch_num}-{block}.npy", rdm_pat)
                    del Xtrain, Xtest, rdm_pat
                del stcs_train, stcs_test, ytrain, ytest
                gc.collect()

            else:
                print(f"Pattern outputs already exist for subject {subject}, epoch {epoch_num}, block {block}")

        del epoch, behav, fwd
        gc.collect()

    del labels, res_paths
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
    jobs = 1
    for subject in subjects:
        process_subject(subject, jobs)
