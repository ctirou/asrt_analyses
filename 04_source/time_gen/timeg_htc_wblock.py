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
import gc

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
subjects_dir = FREESURFER_DIR

lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False
mne.set_log_level(verbose)

pick_ori = 'vector'
regions = ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']
analysis = 'scores_blocks'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None


def process_subject(subject, jobs):
    print(f"Processing subject {subject} with pick_ori={pick_ori}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=100000,
            solver=solver,
            class_weight="balanced",
            random_state=42,
        ),
    )
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    vol_src_fname = RESULTS_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])

    res_paths = {
        region: ensured(RESULTS_DIR / 'TIMEG' / 'source' / region / analysis / subject)
        for region in regions
    }

    for epoch_num in [0, 1, 2, 3, 4]:
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl')).reset_index(drop=True)
        behav['trials'] = behav.index
        blocks = np.unique(behav["blocks"])

        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)

        fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-htc-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

        for block in blocks:
            block = int(block)

            random_pending = [
                region
                for region in regions
                if overwrite or not op.exists(res_paths[region] / f"rand-{epoch_num}-{block}.npy")
            ]
            if random_pending:
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(
                    epoch, fwd, behav, pick_ori, 'random', block, blocks, verbose=verbose
                )
                label_tc_train, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                label_tc_test, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)

                for region in random_pending:
                    labels = [label for label in label_tc_train.keys() if region in label]
                    if not labels:
                        print(
                            f"No labels found for region {region} in subject {subject}; "
                            f"skipping random epoch {epoch_num} block {block}"
                        )
                        continue

                    Xtrain = np.concatenate([np.real(label_tc_train[label]) for label in labels], axis=1)
                    Xtest = np.concatenate([np.real(label_tc_test[label]) for label in labels], axis=1)
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)

                    assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                    assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                    clf.fit(Xtrain, ytrain)
                    acc_matrix = clf.score(Xtest, ytest)
                    np.save(res_paths[region] / f"rand-{epoch_num}-{block}.npy", acc_matrix)
                    del Xtrain, Xtest, acc_matrix
                del label_tc_train, label_tc_test, stcs_train, stcs_test, ytrain, ytest
                gc.collect()
            else:
                print(f"Random outputs already exist for {subject} epoch {epoch_num} block {block}")

            pattern_pending = [
                region
                for region in regions
                if overwrite or not op.exists(res_paths[region] / f"pat-{epoch_num}-{block}.npy")
            ]
            if pattern_pending:
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(
                    epoch, fwd, behav, pick_ori, 'pattern', block, blocks, verbose=verbose
                )
                label_tc_train, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                label_tc_test, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)

                for region in pattern_pending:
                    labels = [label for label in label_tc_train.keys() if region in label]
                    if not labels:
                        print(
                            f"No labels found for region {region} in subject {subject}; "
                            f"skipping pattern epoch {epoch_num} block {block}"
                        )
                        continue

                    Xtrain = np.concatenate([np.real(label_tc_train[label]) for label in labels], axis=1)
                    Xtest = np.concatenate([np.real(label_tc_test[label]) for label in labels], axis=1)
                    if pick_ori == 'vector':
                        Xtrain = svd_fast(Xtrain)
                        Xtest = svd_fast(Xtest)

                    assert len(Xtrain) == len(ytrain), "Length mismatch in training data"
                    assert len(Xtest) == len(ytest), "Length mismatch in testing data"

                    clf.fit(Xtrain, ytrain)
                    acc_matrix = clf.score(Xtest, ytest)
                    np.save(res_paths[region] / f"pat-{epoch_num}-{block}.npy", acc_matrix)
                    del Xtrain, Xtest, acc_matrix
                del label_tc_train, label_tc_test, stcs_train, stcs_test, ytrain, ytest
                gc.collect()
            else:
                print(f"Pattern outputs already exist for {subject} epoch {epoch_num} block {block}")

            if 'stcs_train' in locals():
                del stcs_train, ytrain, stcs_test, ytest, label_tc_train, label_tc_test
                gc.collect()

        del epoch, fwd, behav
        gc.collect()

    del vol_src, offsets, res_paths
    gc.collect()


if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
        process_subject(subject, jobs)
    except (IndexError, ValueError):
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
