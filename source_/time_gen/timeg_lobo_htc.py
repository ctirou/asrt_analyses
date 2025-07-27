import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import gc
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import *
from config import *
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg_new'
subjects = SUBJS15
subjects_dir = FREESURFER_DIR

lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False
mne.use_log_level(verbose)

pick_ori = 'vector'
# pick_ori = 'max-power'
analysis = 'scores_lobo_vector' if pick_ori == 'vector' else 'scores_lobo_maxpower'
analysis += '_new'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    
    print(f"Processing subject {subject} with pick_ori={pick_ori}")
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

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
        epoch = mne.read_epochs(epoch_fname, verbose=verbose).crop(-1.5, 1.5)
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
    fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-htc-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

    # rename blocks columns
    behav.loc[behav.sessions != 0, 'blocks'] += 3
    blocks = np.unique(behav["blocks"])        
            
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:

        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / region / analysis / subject)
        
        for block in blocks:
            block = int(block)
            
            # random trials
            if not op.exists(res_path / f"rand-{block}.npy") or overwrite:
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
                
                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_path / f"rand-{block}.npy", acc_matrix)

                del stcs_train, ytrain, stcs_test, ytest, label_tc_train, label_tc_test, Xtrain, Xtest
                gc.collect()

            else:
                print(f"Random for {subject} block {block} already exists")
                
            
            # pattern trials
            if not op.exists(res_path / f"pat-{block}.npy") or overwrite:
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

                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_path / f"pat-{block}.npy", acc_matrix)

                del stcs_train, ytrain, stcs_test, ytest, label_tc_train, label_tc_test, Xtrain, Xtest
                gc.collect()

            else:
                print(f"Pattern for {subject} block {block} already exists")
                
    del epoch, fwd
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
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
        