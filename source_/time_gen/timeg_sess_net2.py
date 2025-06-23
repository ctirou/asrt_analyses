import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys
from sklearn.metrics import accuracy_score as acc

# params
subjects = SUBJS15
data_path = DATA_DIR / 'for_timeg'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10
verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

analysis = 'scores_sss'

use_resting = sys.argv[1]
use_vector = sys.argv[2]
# use_resting = False
# use_vector = False

analysis = analysis + '_rs' if use_resting else analysis + '_0200'
analysis = analysis + '_vect' if use_vector else analysis + '_maxp'
    
networks = NETWORKS[:-2]

def process_subject(subject, jobs):
    
    print(f"Processing subject {subject} with analysis {analysis}...")
    
    # define classifier'
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state=42)

    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject
        
    rand_train, rand_test = dict(), dict()
    pat_train, pat_test = dict(), dict()
    
    if use_resting:
        noise_cov = mne.read_cov(data_path / 'noise_cov' / f"{subject}-cov.fif", verbose=verbose)
    else:
        pass
    
    pick_ori = 'vector' if use_vector else 'max-power'
    
    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
                        
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / 'for_timeg' / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        
        pattern = behav[behav.trialtypes == 1].reset_index(drop=True)
        pattern_epochs = epoch[pattern.index]
        
        random = behav[behav.trialtypes == 2].reset_index(drop=True)
        random_epochs = epoch[random.index]
                
        for network in networks:
            # read labels
            lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
            res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)
            
            if network not in rand_train:
                rand_train[network] = list()
                rand_test[network] = list()
            if network not in pat_train:
                pat_train[network] = list()
                pat_test[network] = list()
            
            # random trials
            if not os.path.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
            
                acc_matrices = list()
                for i, (train_idx, test_idx) in enumerate(sss.split(random_epochs, random.positions)):

                    print(f"Processing {subject} epoch {epoch_num} random {network} split {i+1}")
                    
                    # get training data
                    if not use_resting:
                        noise_cov = mne.compute_covariance(random_epochs[train_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                    else:
                        pass
                    data_cov = mne.compute_covariance(random_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                    rank = mne.compute_rank(data_cov, info=random_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                    filters = make_lcmv(random_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                        pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                        rank=rank, reduce_rank=True, verbose=verbose)
                    stcs_train = apply_lcmv_epochs(random_epochs[train_idx], filters=filters, verbose=verbose)
                    Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                    
                    if use_vector:
                        Xtrain = svd(Xtrain)
                    else:
                        pass
                    
                    ytrain = random.positions[train_idx].reset_index(drop=True)
                    assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                                    
                    # get testing data
                    if not use_resting:
                        noise_cov = mne.compute_covariance(random_epochs[test_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                    else:
                        pass
                    data_cov = mne.compute_covariance(random_epochs[test_idx], method="empirical", rank="info", verbose=verbose)
                    rank = mne.compute_rank(data_cov, info=random_epochs[test_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                    filters = make_lcmv(random_epochs[test_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                        pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                        rank=rank, reduce_rank=True, verbose=verbose)
                    stcs_test = apply_lcmv_epochs(random_epochs[test_idx], filters=filters, verbose=verbose)
                    Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                    
                    if use_vector:
                        Xtest = svd(Xtest)
                    else:
                        pass
                    
                    ytest = random.positions[test_idx].reset_index(drop=True)
                    assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"                
                    
                    clf.fit(Xtrain, ytrain)
                    ypred = clf.predict(Xtest)
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    acc_matrices.append(acc_matrix)

                np.save(res_path / f"rand-{epoch_num}.npy", acc_matrices.mean(0))
                
                if epoch_num == 0:
                    rand_train[network].append(Xtrain)
                    rand_test[network].append(Xtest)
                
                del acc_matrices, Xtrain, ytrain, Xtest, ytest, stcs_train, stcs_test
                gc.collect()
                
            # pattern trials
            if not os.path.exists(res_path / f"pat-{epoch_num}.npy"):
                
                acc_matrices = list()
                for i, (train_idx, test_idx) in enumerate(sss.split(pattern_epochs, pattern.positions)):

                    print(f"Processing {subject} epoch {epoch_num} pattern {network} split {i+1}")
                    
                    # get training data - pattern trials
                    if use_resting:
                        noise_cov = mne.compute_covariance(pattern_epochs[train_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                    else:
                        pass
                    data_cov = mne.compute_covariance(pattern_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                    rank = mne.compute_rank(data_cov, info=pattern_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                    filters = make_lcmv(pattern_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                        pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                        rank=rank, reduce_rank=True, verbose=verbose)
                    stcs_train = apply_lcmv_epochs(pattern_epochs[train_idx], filters=filters, verbose=verbose)
                    Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                    
                    if use_vector:
                        Xtrain = svd(Xtrain)
                    else: 
                        pass
                    
                    ytrain = pattern.positions[train_idx].reset_index(drop=True)
                    assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                                    
                    # get testing data - pattern trials
                    if use_resting:
                        noise_cov = mne.compute_covariance(pattern_epochs[test_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                    else:
                        pass
                    data_cov = mne.compute_covariance(pattern_epochs[test_idx], method="empirical", rank="info", verbose=verbose)
                    rank = mne.compute_rank(data_cov, info=pattern_epochs[test_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                    filters = make_lcmv(pattern_epochs[test_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                        pick_ori=pick_ori, weight_norm="unit-noise-gain",
                                        rank=rank, reduce_rank=True, verbose=verbose)
                    stcs_test = apply_lcmv_epochs(pattern_epochs[test_idx], filters=filters, verbose=verbose)
                    Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                    
                    if use_vector:
                        Xtest = svd(Xtest)
                    else:
                        pass
                    
                    ytest = pattern.positions[test_idx].reset_index(drop=True)
                    assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"
                    
                    clf.fit(Xtrain, ytrain)
                    ypred = clf.predict(Xtest)
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    acc_matrices.append(acc_matrix)

                np.save(res_path / f"pat-{epoch_num}.npy", acc_matrices.mean(0))
                
                if epoch_num == 0:
                    pat_train[network].append(Xtrain)
                    pat_test[network].append(Xtest)
                
                del acc_matrices, Xtrain, ytrain, Xtest, ytest, stcs_train, stcs_test
                gc.collect()
            
    for network in networks:
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)

        if not op.exists(res_path / "rand-all.npy") or overwrite:
            acc_matrices = list()
            for i in range(folds):
                Xtrain = np.concatenate([rand_train[network][j] for j in range(folds) if j != i], axis=0)
                ytrain = pd.concat([rand_test[network][j] for j in range(folds) if j != i], axis=0).reset_index(drop=True)
                assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                
                Xtest = rand_test[network][i]
                ytest = rand_test[network][i].reset_index(drop=True)
                assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"
                
                clf.fit(Xtrain, ytrain)
                ypred = clf.predict(Xtest)
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                acc_matrices.append(acc_matrix)
            
            np.save(res_path / "rand-all.npy", acc_matrices.mean(0))
        
            del acc_matrices, Xtrain, ytrain, Xtest, ytest
            gc.collect()
        
        if not op.exists(res_path / "pat-all.npy") or overwrite:
            acc_matrices = list()
            for i in range(folds):
                Xtrain = np.concatenate([pat_train[network][j] for j in range(folds) if j != i], axis=0)
                ytrain = pd.concat([pat_train[network][j] for j in range(folds) if j != i], axis=0).reset_index(drop=True)
                assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                
                Xtest = pat_test[network][i]
                ytest = pat_test[network][i].reset_index(drop=True)
                assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"
                
                clf.fit(Xtrain, ytrain)
                ypred = clf.predict(Xtest)
                acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                acc_matrices.append(acc_matrix)

            np.save(res_path / "pat-all.npy", acc_matrices.mean(0))

            del acc_matrices, Xtrain, ytrain, Xtest, ytest
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
    for subject in subjects:
        process_subject(subject, jobs)