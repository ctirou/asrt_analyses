# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import mne
from collections import defaultdict
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import *
from config import *
from joblib import Parallel, delayed
import gc

def get_base_name(label_name):
    """Strip hemisphere prefix (LH_/RH_) and suffix (-lh/-rh) to get bilateral base name.
    e.g. '7Networks_LH_SomMot_1-lh' -> '7Networks_SomMot_1'
    """
    name = label_name.replace('_LH_', '_').replace('_RH_', '_')
    return name.replace('-lh', '').replace('-rh', '')

data_path = DATA_DIR / 'for_timeg'
# subjects = SUBJS15
subjects = [
    # 'sub01',
    # 'sub02',
    'sub03',
    # 'sub04',
    'sub05',
    'sub06',
    # 'sub07',
    'sub08',
    'sub09',
    'sub10',
    'sub11',
    # 'sub12',
    'sub13',
    # 'sub14',
    'sub15'
    ]

lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

networks = ['SomMot', 'DorsAttn']
parc = "Schaefer2018_200Parcels_7Networks"
hemi = 'both'
subjects_dir = FREESURFER_DIR
pick_ori = 'vector'
weight_norm = "unit-noise-gain-invariant"

analysis = 'scores_blocks_subreg_merged'
crop1, crop2 = -1, 1.5

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
mne.set_log_level(verbose)

subject = subjects[0]
jobs = -1

def process_subject(subject, jobs):

    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs)
        
    # concatenate all epochs and behavs
    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose).crop(crop1, crop2)
        all_epochs.append(epoch)
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epochs, verbose=verbose)
    del all_epochs, all_behavs
    assert len(epoch) == len(behav)

    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

    # rename blocks columns
    behav.loc[behav.sessions != 0, 'blocks'] += 3
    blocks = np.unique(behav["blocks"])

    # Read labels per network (list of Label objects per network)
    labels_per_network = {
        network: mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, regexp=network, verbose=verbose)
        for network in networks
    }

    # Group lh/rh labels by bilateral base name to find mergeable pairs
    label_groups = defaultdict(list)
    for network in networks:
        for label in labels_per_network[network]:
            base = get_base_name(label.name)
            label_groups[base].append((label, network))

    # Merge lh+rh pairs into a BiHemiLabel; keep singletons (one hemi only) as-is
    all_labels = []
    for base_name, group in label_groups.items():
        network = group[0][1]
        if len(group) == 2:
            lh = next(l for l, _ in group if l.name.endswith('-lh'))
            rh = next(l for l, _ in group if l.name.endswith('-rh'))
            merged = lh + rh   # MNE BiHemiLabel covering both hemispheres
            merged.name = base_name
        else:
            merged = group[0][0]  # only one hemisphere available, use as-is
        all_labels.append((merged, network))

    # Cache output paths per merged label (one subdirectory per label name)
    res_paths = {
        label.name: ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject / label.name)
        for label, network in all_labels
    }

    for block in blocks[3:]:
        block = int(block)

        random_pending = [
            (label, network)
            for label, network in all_labels
            if overwrite or not op.exists(res_paths[label.name] / f"rand-{block}.npy")]

        if random_pending:
            stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'random', block, blocks, verbose=verbose)

            for label, network in random_pending:
                print(f"Processing subject {subject} | block {block} | {label.name} (network {network})")
                Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                if pick_ori == 'vector':
                    Xtrain = svd_fast(Xtrain)
                    Xtest = svd_fast(Xtest)
                assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_paths[label.name] / f"rand-{block}.npy", acc_matrix)
                del Xtrain, Xtest, acc_matrix
            del stcs_train, stcs_test, ytrain, ytest
            gc.collect()

        else:
            print(f"Random outputs already exist for subject {subject}, block {block}")

        pattern_pending = [
            (label, network)
            for label, network in all_labels
            if overwrite or not op.exists(res_paths[label.name] / f"pat-{block}.npy")]

        if pattern_pending:
            stcs_train, stcs_test, ytrain, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, 'pattern', block, blocks, verbose=verbose)

            for label, network in pattern_pending:
                print(f"Processing subject {subject} | block {block} | {label.name} (network {network})")
                Xtrain = np.array([np.real(stc.in_label(label).data) for stc in stcs_train])
                Xtest = np.array([np.real(stc.in_label(label).data) for stc in stcs_test])
                if pick_ori == 'vector':
                    Xtrain = svd_fast(Xtrain)
                    Xtest = svd_fast(Xtest)
                assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_paths[label.name] / f"pat-{block}.npy", acc_matrix)
                del Xtrain, Xtest, acc_matrix
            del stcs_train, stcs_test, ytrain, ytest
            gc.collect()

        else:
            print(f"Pattern outputs already exist for subject {subject}, block {block}")

        if 'stcs_train' in locals():
            del stcs_train, stcs_test, ytrain, ytest
            gc.collect()

    del epoch, behav, fwd, labels_per_network, all_labels, res_paths
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