import mne
import os
import os.path as op
import numpy as np
from mne.decoding import GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from base import *
from config import *
import gc
import sys
from sklearn.metrics import accuracy_score as acc

data_path = DATA_DIR / 'for_timeg_new'
subjects, subjects_dir = SUBJS15, FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

use_vector = True

analysis = 'scores_skf_vect_0200_new'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    
    print(f"Processing subject {subject} with analysis {analysis}...")
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    
    # read volume source space
    vol_src_fname =  RESULTS_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work

    del vol_src
    gc.collect()

    # compute forward solution
    fwd_fname = RESULTS_DIR / "fwd" / 'for_timeg' / f"{subject}-htc-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    all_behavs = []
    all_epochs = []
    
    for epoch_num in [1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
        all_behavs.append(behav)
        all_epochs.append(epoch)

    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = mne.concatenate_epochs(all_epochs)
    behavs = pd.concat(all_behavs, ignore_index=True)
    assert len(epochs) == len(behavs), "Length mismatch between epochs and behavs"
    
    del all_epochs, all_behavs
    gc.collect()
    
    pick_ori = 'vector' if use_vector == 'True' else 'max-power'
        
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
            
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / region / analysis / subject)
        ensure_dir(res_path / 'noise_cov')
        
        random = behavs[behavs.trialtypes == 2].reset_index(drop=True)
        random_epochs = epochs[random.index]
        
        ensure_dir(res_path / 'noise_cov')
        
        if not op.exists(res_path / "rand-all.npy") or overwrite:
            print("Processing", subject, 'all', "random", region)
            
            acc_matrices = list()
            for i, (train_idx, test_idx) in enumerate(skf.split(random_epochs, random.positions)):
                
                # training data
                noise_cov = mne.compute_covariance(random_epochs[train_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                mne.write_cov(res_path / 'noise_cov' / f'{epoch_num}-{i+1}-noise-cov.fif', noise_cov, overwrite=True, verbose=verbose)
                data_cov = mne.compute_covariance(random_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=random_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(random_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs = apply_lcmv_epochs(random_epochs[train_idx], filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                Xtrain = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
                Xtrain = svd(Xtrain)
                ytrain = random.positions[train_idx]
                assert Xtrain.shape[0] == ytrain.shape[0], "Shape mismatch between training data and labels"

                # testing data
                stcs = apply_lcmv_epochs(random_epochs[test_idx], filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                Xtest = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
                Xtest = svd(Xtest)
                ytest = random.positions[test_idx]
                assert Xtest.shape[0] == ytest.shape[0], "Shape mismatch between testing data and labels"
                
                clf.fit(Xtrain, ytrain)
                ypred = clf.predict(Xtest)
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                acc_matrices.append(acc_matrix)
                
                del Xtrain, ytrain, Xtest, ytest, stcs, label_tc
                gc.collect()

            np.save(res_path / "rand-all.npy", np.array(acc_matrices).mean(0))
            
            del acc_matrices
            gc.collect()
        else:
            print("Skipping", subject, 'all', "pattern", region)
            
        pattern = behavs[behavs.trialtypes == 1].reset_index(drop=True)
        pattern_epochs = epochs[pattern.index]
        
        if not op.exists(res_path / "pat-all.npy") or overwrite:
            print("Processing", subject, 'all', "pattern", region)
            
            acc_matrices = list()
            for i, (train_idx, test_idx) in enumerate(skf.split(pattern_epochs, pattern.positions)):
                
                # training data
                ensure_dir(res_path / 'noise_cov')
                noise_cov = mne.read_cov(res_path / 'noise_cov' / f'{epoch_num}-{i+1}-noise-cov.fif', verbose=verbose)
                data_cov = mne.compute_covariance(pattern_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=pattern_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(pattern_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs = apply_lcmv_epochs(pattern_epochs[train_idx], filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                Xtrain = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
                Xtrain = svd(Xtrain)
                ytrain = pattern.positions[train_idx]
                assert Xtrain.shape[0] == ytrain.shape[0], "Shape mismatch between training data and labels"

                # testing data
                stcs = apply_lcmv_epochs(pattern_epochs[test_idx], filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                Xtest = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
                Xtest = svd(Xtest)
                ytest = pattern.positions[test_idx]
                assert Xtest.shape[0] == ytest.shape[0], "Shape mismatch between testing data and labels"

                clf.fit(Xtrain, ytrain)
                ypred = clf.predict(Xtest)
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                acc_matrices.append(acc_matrix)

                del Xtrain, ytrain, Xtest, ytest, stcs, label_tc
                gc.collect()

            np.save(res_path / "pat-all.npy", np.array(acc_matrices).mean(0))

            del acc_matrices
            gc.collect()
        else:
            print("Skipping", subject, 'all', "pattern", region)

if is_cluster:
    jobs = 20
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    for subject in subjects:
        process_subject(subject, jobs)