# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import *
from config import *
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

pick_ori = 'vector'
weight_norm = "unit-noise-gain-invariant"
networks = NETWORKS[:-3]  # Exclude 'Hippocampus', 'Thalamus', 'Cerebellum-Cortex' for this analysis

analysis = 'scores_blocks'
# crop1, crop2 = -1.5, 1.5
crop1, crop2 = -3, 1.5
if crop1 != -1.5 or crop2 != 1.5:
    analysis += f'_{str(crop1)}_{str(crop2)}'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
mne.set_log_level(verbose)

def process_subject(subject, jobs):

    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject
        
    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        # read behav        
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose).crop(crop1, crop2)
        all_epochs.append(epoch)
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epochs, verbose=verbose)
    assert len(epoch) == len(behav)

    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

    # rename blocks columns
    behav.loc[behav.sessions != 0, 'blocks'] += 3
    blocks = np.unique(behav["blocks"])

    # Cache output paths and labels once per subject/network.
    res_paths = {
        network: ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)
        for network in networks
    }
    labels = {
        network: (
            mne.read_label(label_path / f'{network}-lh.label')
            + mne.read_label(label_path / f'{network}-rh.label')
        )
        for network in networks
    }

    for block in blocks:
        block = int(block)

        # RANDOM: only compute source estimates if at least one output is missing.
        random_pending = [
            network
            for network in networks
            if overwrite or not op.exists(res_paths[network] / f"rand-{block}.npy")]
        
        if random_pending:
            stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net( epoch, fwd, behav, pick_ori, 'random', block, blocks, verbose=verbose)
            
            for network in random_pending:
                print(f"Processing subject {subject} in network {network} with pick_ori={pick_ori}")
                label = labels[network]
                Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                if pick_ori == 'vector':
                    Xtrain = svd_fast(Xtrain)
                    Xtest = svd_fast(Xtest)
                assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_paths[network] / f"rand-{block}.npy", acc_matrix)
        
        else:
            print(f"Random outputs already exist for subject {subject}, block {block}")

        # PATTERN: only compute source estimates if at least one output is missing.
        pattern_pending = [
            network
            for network in networks
            if overwrite or not op.exists(res_paths[network] / f"pat-{block}.npy")]
        
        if pattern_pending:
            
            stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'pattern', block, blocks, verbose=verbose)
            
            for network in pattern_pending:
                print(f"Processing subject {subject} in network {network} with pick_ori={pick_ori}")
                label = labels[network]
                Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                if pick_ori == 'vector':
                    Xtrain = svd_fast(Xtrain)
                    Xtest = svd_fast(Xtest)
                assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_paths[network] / f"pat-{block}.npy", acc_matrix)
        
        else:
            print(f"Pattern outputs already exist for subject {subject}, block {block}")
            
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
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)