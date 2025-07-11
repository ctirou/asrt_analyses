import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys

# params
subjects = SUBJS15
data_path = DATA_DIR / 'for_timeg_new'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10
verbose = 'error'
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
mne.use_log_level(verbose)

# pick_ori = 'vector'
pick_ori = 'max-power'
weight_norm = "unit-noise-gain-invariant" if pick_ori == 'vector' else "unit-noise-gain"
analysis = 'scores_skf_vect' if pick_ori == 'vector' else 'scores_skf_maxpower'
    
networks = NETWORKS[:-2]

def process_subject(subject, jobs):
    
    print(f"Processing subject {subject} with analysis {analysis}...")
    
    # define classifier'
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)

    all_epochs = []
    all_behavs = []
    
    for epoch_num in [1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
        
        all_epochs.append(epoch)                
        all_behavs.append(behav)
                
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']  # ensure all epochs have the same dev_head_t
    epochs = mne.concatenate_epochs(all_epochs)
    behavs = pd.concat(all_behavs, ignore_index=True)
    assert len(epochs) == len(behavs), "Length mismatch between epochs and behavs"
    
    del all_epochs, all_behavs
    gc.collect()    

    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / 'for_timeg' / f"{subject}-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    for network in networks:
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)
        
        random = behavs[behavs.trialtypes == 2]
        random_epochs = epochs[random.index]
        random = random.reset_index(drop=True)
        
        # random trials
        if not os.path.exists(res_path / "rand-all.npy") or overwrite:
        
            acc_matrices = list()
            for i, (train_idx, test_idx) in enumerate(skf.split(random_epochs, random.positions)):

                print(f"Processing {subject} random {network} split {i+1}")
                
                # training data
                noise_cov = mne.compute_covariance(random_epochs[train_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                data_cov = mne.compute_covariance(random_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=random_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(random_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori=pick_ori, weight_norm=weight_norm,
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(random_epochs[train_idx], filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                # Xtrain = svd(Xtrain)
                ytrain = random.positions[train_idx]
                assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                                
                # testing data
                stcs_test = apply_lcmv_epochs(random_epochs[test_idx], filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                # Xtest = svd(Xtest)
                ytest = random.positions[test_idx]
                assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"
                
                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                acc_matrices.append(acc_matrix)

            np.save(res_path / "rand-all.npy", np.array(acc_matrices).mean(0))
            
            del acc_matrices, Xtrain, ytrain, Xtest, ytest, stcs_train, stcs_test
            gc.collect()
        
        pattern = behavs[behavs.trialtypes == 1]
        pattern_epochs = epochs[pattern.index]
        pattern = pattern.reset_index(drop=True)

        # pattern trials
        if not os.path.exists(res_path / "pat-all.npy") or overwrite:
            
            acc_matrices = list()
            for i, (train_idx, test_idx) in enumerate(skf.split(pattern_epochs, pattern.positions)):

                print(f"Processing {subject} pattern {network} split {i+1}")
                
                # get training data
                # noise covariance computed on random trials
                for j, (tidx, _) in enumerate(skf.split(random_epochs, random.positions)):
                    if j == i:
                        noise_cov = mne.compute_covariance(random_epochs[tidx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                data_cov = mne.compute_covariance(pattern_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=pattern_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(pattern_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori=pick_ori, weight_norm=weight_norm,
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(pattern_epochs[train_idx], filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                if pick_ori == 'vector':
                    Xtrain = svd(Xtrain)
                ytrain = pattern.positions[train_idx]
                assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                                
                # get testing data
                stcs_test = apply_lcmv_epochs(pattern_epochs[test_idx], filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                if pick_ori == 'vector':
                    Xtest = svd(Xtest)
                ytest = pattern.positions[test_idx]
                assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"
                
                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                acc_matrices.append(acc_matrix)

            np.save(res_path / "pat-all.npy", np.array(acc_matrices).mean(0))
                            
            del acc_matrices, Xtrain, ytrain, Xtest, ytest, stcs_train, stcs_test
            gc.collect()
        
        del pattern_epochs, pattern, random_epochs, random
        gc.collect()
            
    print(f"Analysis {analysis} completed for subject {subject}.")
        
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
    for subject in subjects[::-1]:
        process_subject(subject, jobs)