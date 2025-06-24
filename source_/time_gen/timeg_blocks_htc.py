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
from sklearn.metrics import accuracy_score as acc
from base import ensured, get_volume_estimate_tc, get_train_test_blocks_htc
from config import *
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
subjects_dir = FREESURFER_DIR

lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = True

pick_ori = sys.argv[1]
# pick_ori = 'vector'
# pick_ori = 'max-power'

analysis = 'scores_blocks' + '_maxp' if pick_ori == 'max-power' else 'scores_blocks' + '_vect'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    # read volume source space
    vol_src_fname = RESULTS_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)

    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    
    del vol_src
    gc.collect()

    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:

        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / region / analysis / subject)
         
        for epoch_num in [0, 1, 2, 3, 4]:
            
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl')).reset_index(drop=True)
            behav['trials'] = behav.index
            
            # read epoch
            epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
                            
            # compute forward solution
            fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-htc-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            
            blocks = np.unique(behav["blocks"])        
            
            for block in blocks:
                block = int(block)
                this_block = behav.blocks == block
                out_blocks = behav.blocks != block
                
                # random trials
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(epoch, fwd, behav, pick_ori, 'random', this_block, out_blocks, verbose=verbose)
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
                if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
                    ypred = clf.predict(Xtest)
                    print(f"Scoring random for {subject} epoch {epoch_num} block {block}")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    np.save(res_path / f"rand-{epoch_num}-{block}.npy", acc_matrix)
                else:
                    print(f"Random for {subject} epoch {epoch_num} block {block} already exists")
                    
                del stcs_train, ytrain, stcs_test, ytest, label_tc_train, label_tc_test, Xtrain, Xtest
                gc.collect()
                
                # pattern trials
                stcs_train, ytrain, stcs_test, ytest = get_train_test_blocks_htc(epoch, fwd, behav, pick_ori, 'pattern', this_block, out_blocks, verbose=verbose)
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
                if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
                    ypred = clf.predict(Xtest)
                    print(f"Scoring pattern for {subject} epoch {epoch_num} block {block}")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    np.save(res_path / f"pat-{epoch_num}-{block}.npy", acc_matrix)
                else:
                    print(f"Pattern for {subject} epoch {epoch_num} block {block} already exists")
                    
                del stcs_train, ytrain, stcs_test, ytest, label_tc_train, label_tc_test, Xtrain, Xtest
                gc.collect()
            
            del epoch, fwd
            gc.collect()
            
if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 20
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
        