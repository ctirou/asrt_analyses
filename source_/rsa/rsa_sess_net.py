import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc
import sys
from joblib import Parallel, delayed

# params
subjects = SUBJS15
data_path = DATA_DIR / 'for_rsa'
subjects_dir = FREESURFER_DIR

verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
jobs = -1

def process_subject(subject, jobs):

    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    
    # practice first
    behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-0.pkl'))
    epoch_fname = op.join(data_path, 'epochs', f"{subject}-0-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
    random = behav[behav.trialtypes == 2].reset_index(drop=True)
    random_epochs = epoch[random.index]    # read forward solution
    pattern = behav[behav.trialtypes == 1].reset_index(drop=True)
    pattern_epochs = epoch[pattern.index]
    
    fwd_fname = RESULTS_DIR / "fwd" / 'for_rsa' / f"{subject}-0-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

    for network in networks:
        res_dir = ensured(RESULTS_DIR / "RSA" / 'source' / network / 'rdm_skf' / subject)
        ensure_dir(res_dir / "noise_cov")
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        cvMD_rand = list()
        if not op.exists(res_dir / "rand-prac.npy") or overwrite:
            for i, (train_idx, test_idx) in enumerate(skf.split(random_epochs, random.positions)):
                X_train, X_test = random_epochs[train_idx], random_epochs[test_idx]
                y_train, y_test = random.positions.iloc[train_idx], random.positions.iloc[test_idx]
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.compute_covariance(X_train, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                mne.write_cov(res_dir / 'noise_cov' / f'{subject}-prac-{i}-noise-cov.fif', noise_cov, overwrite=True, verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='vector', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                Xtrain = svd(Xtrain)
                
                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                Xtest = svd(Xtest)
                
                dist = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_rand.append(dist)
            cvMD = np.array(cvMD_rand).mean(0)
            np.save(res_dir / "rand-prac.npy", cvMD)
            print("Saved random RDM for", subject, network)
            del cvMD_rand, Xtrain, Xtest, y_train, y_test, stcs_train, stcs_test
            gc.collect()
        else:
            print("Random RDM already exists for", subject, network)

        cvMD_pat = list()
        if not op.exists(res_dir / "pat-prac.npy") or overwrite:
            for i, (train_idx, test_idx) in enumerate(skf.split(pattern_epochs, pattern.positions)):
                X_train, X_test = pattern_epochs[train_idx], pattern_epochs[test_idx]
                y_train, y_test = pattern.positions.iloc[train_idx], pattern.positions.iloc[test_idx]
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.read_cov(res_dir / 'noise_cov' / f'{subject}-prac-{i}-noise-cov.fif', verbose=verbose)
                
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='vector', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                Xtrain = svd(Xtrain)
                
                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                Xtest = svd(Xtest)
                
                dist = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_pat.append(dist)
            cvMD = np.array(cvMD_pat).mean(0)
            np.save(res_dir / "pat-prac.npy", cvMD)
            print("Saved pattern RDM for", subject, network)
            del cvMD_pat, Xtrain, Xtest, y_train, y_test, stcs_train, stcs_test
            gc.collect()
        else:
            print("Pattern RDM already exists for", subject, network)


    # now the learning epochs
    all_epochs = []
    all_behavs = []
    for epoch_num in range(1, 5):
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        all_behavs.append(behav)
        # read epoch
        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
        all_epochs.append(epoch)
        
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']  # ensure all epochs have the same head transformation
    epoch = mne.concatenate_epochs(all_epochs, verbose=verbose)
    behav = pd.concat(all_behavs, ignore_index=True)    
    random = behav[behav.trialtypes == 2].reset_index(drop=True)
    random_epochs = epoch[random.index]
    pattern = behav[behav.trialtypes == 1].reset_index(drop=True)
    pattern_epochs = epoch[pattern.index]

    fwd_fname = RESULTS_DIR / "fwd" / 'for_rsa' / f"{subject}-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    # random
    for network in networks:
        res_dir = ensured(RESULTS_DIR / "RSA" / 'source' / network / 'rdm_skf' / subject)
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        cvMD_rand = list()
        
        if not op.exists(res_dir / "rand-learn.npy") or overwrite:
            for i, (train_idx, test_idx) in enumerate(skf.split(random_epochs, random.positions)):
                X_train, X_test = random_epochs[train_idx], random_epochs[test_idx]
                y_train, y_test = random.positions.iloc[train_idx], random.positions.iloc[test_idx]
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.compute_covariance(X_train, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                mne.write_cov(res_dir / 'noise_cov' / f'{subject}-learn-{i}-noise-cov.fif', noise_cov, overwrite=True, verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='vector', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                Xtrain = svd(Xtrain)
                
                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                Xtest = svd(Xtest)
                
                dist = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_rand.append(dist)
            cvMD = np.array(cvMD_rand).mean(0)
            np.save(res_dir / "rand-learn.npy", cvMD)
            print("Saved random RDM for", subject, network)
            del cvMD_rand, Xtrain, Xtest, y_train, y_test, stcs_train, stcs_test
            gc.collect()
        else:
            print("Random RDM already exists for", subject, network)


    # pattern
    for network in networks:
        res_dir = ensured(RESULTS_DIR / "RSA" / 'source' / network / 'rdm_skf' / subject)
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        cvMD_pat = list()
        
        if not op.exists(res_dir / "pat-learn.npy") or overwrite:
            for i, (train_idx, test_idx) in enumerate(skf.split(pattern_epochs, pattern.positions)):
                X_train, X_test = pattern_epochs[train_idx], pattern_epochs[test_idx]
                y_train, y_test = pattern.positions.iloc[train_idx], pattern.positions.iloc[test_idx]
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.read_cov(res_dir / 'noise_cov' / f'{subject}-learn-{i}-noise-cov.fif', verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='vector', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                Xtrain = svd(Xtrain)
                
                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                Xtest = svd(Xtest)
                
                dist = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_pat.append(dist)
            cvMD = np.array(cvMD_pat).mean(0)
            np.save(res_dir / "pat-learn.npy", cvMD)
            print("Saved pattern RDM for", subject, network)
            del cvMD_pat, Xtrain, Xtest, y_train, y_test, stcs_train, stcs_test
            gc.collect()
        else:
            print("Pattern RDM already exists for", subject, network)


    del fwd, epoch, behav
    gc.collect()
    
if is_cluster:
    jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, 1) for subject in subjects)
    # jobs = -1
    # for subject in subjects:
    #     process_subject(subject, jobs)