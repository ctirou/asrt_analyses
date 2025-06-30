import mne
import os
import os.path as op
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import pandas as pd
from base import get_volume_estimate_tc, cv_mahalanobis_parallel, ensured
from config import *
import gc
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

data_path = DATA_DIR / 'for_rsa'
subjects, subjects_dir = SUBJS15, FREESURFER_DIR

verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = ensured(RESULTS_DIR / 'RSA' / 'source')

def process_subject(subject, epoch_num, jobs):
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # read volume source space
    vol_src_fname =  RESULTS_DIR / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    
    del vol_src
    gc.collect()
    
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
    
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:

        res_dir = ensured(res_path / region / 'rdm_skf' / subject)
        
        if not op.exists(res_dir / "rand-0.npy") or overwrite:
            cvMD_rand = []
            for train_idx, test_idx in skf.split(random_epochs, random.positions):
                X_train, X_test = random_epochs[train_idx], random_epochs[test_idx]
                y_train, y_test = random.positions.iloc[train_idx].reset_index(drop=True), random.positions.iloc[test_idx].reset_index(drop=True)
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.compute_covariance(X_train, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='vector', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)     
                Xtrain = svd(stcs_data)

                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)     
                Xtest = svd(stcs_data)
                
                rdm_rand = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_rand.append(rdm_rand)
            
            cvMD_rand = np.array(cvMD_rand).mean(0)
            np.save(res_dir / "rand-0.npy", cvMD_rand)
            del cvMD_rand, X_train, X_test, y_train, y_test, data_cov, noise_cov, rank, filters, stcs_train, label_tc, stcs_data
            gc.collect()
        
        if not op.exists(res_dir / "pat-0.npy") or overwrite:
            cvMD_pat = []
            for train_idx, test_idx in skf.split(pattern_epochs, pattern.positions):
                X_train, X_test = pattern_epochs[train_idx], pattern_epochs[test_idx]
                y_train, y_test = pattern.positions.iloc[train_idx].reset_index(drop=True), pattern.positions.iloc[test_idx].reset_index(drop=True)
                data_cov = mne.compute_covariance(X_train, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
                noise_cov = mne.compute_covariance(X_train, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(data_cov, info=X_train.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute source estimates
                filters = make_lcmv(X_train.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori='vector', weight_norm="unit-noise-gain",
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(X_train, filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs_train, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region in label]
                stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)     
                Xtrain = svd(stcs_data) 
                
                stcs_test = apply_lcmv_epochs(X_test, filters=filters, verbose=verbose)
                label_tc, _ = get_volume_estimate_tc(stcs_test, fwd, offsets, subject, subjects_dir)
                labels = [label for label in label_tc.keys() if region  in label]
                stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)     
                Xtest = svd(stcs_data)
                
                rdm_pat = train_test_mahalanobis_fast(Xtrain, Xtest, y_train, y_test, n_jobs=jobs)
                cvMD_pat.append(rdm_pat)
            
            cvMD_pat = np.array(cvMD_pat).mean(0)
            np.save(res_dir / "pat-0.npy", cvMD_pat)
            del cvMD_pat, X_train, X_test, y_train, y_test, data_cov, noise_cov, rank, filters, stcs_train, label_tc, stcs_data
            gc.collect()
    
    del epoch, random_epochs, pattern_epochs, fwd
    gc.collect()
    
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

    # compute noise covariance
    noise_cov = mne.compute_covariance(epoch, tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)        
    # compute data covariance matrix
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=0.6, method="empirical", rank="info", verbose=verbose)
    # conpute rank
    rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)

    # compute forward solution
    fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-htc-{epoch_num}-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
    
    # compute source estimates
    filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori='max-power', weight_norm="unit-noise-gain",
                        rank=rank, reduce_rank=True, verbose=verbose)
            
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    # get data from volume source space
    label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
    
    del epoch, noise_cov, data_cov, fwd, filters, stcs
    gc.collect()
    
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_dir = ensured(res_path / region / 'rdm_skf' / subject)

        # get data from regions of interest
        labels = [label for label in label_tc.keys() if region in label]
        stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1)                
        
        if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:

            pattern = behav.trialtypes == 1
            X_pat = stcs_data[pattern]
            y_pat = behav.positions[pattern].reset_index(drop=True)
            assert X_pat.shape[0] == y_pat.shape[0], "Length mismatch"
            
            _, counts = np.unique(y_pat, return_counts=True)
            if any(counts < 10):
                print(f"Skipping {subject} {epoch_num} due to low counts")
            else:
                print("Processing", subject, "epoch", epoch_num, region, 'pattern')
                rdm_pat = cv_mahalanobis_parallel(X_pat, y_pat, jobs)
                np.save(res_dir / f"pat-{epoch_num}.npy", rdm_pat)
                del X_pat, y_pat
                gc.collect()
        
        if not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:    
            
            random = behav.trialtypes == 2
            X_rand = stcs_data[random]
            y_rand = behav.positions[random].reset_index(drop=True)
            assert X_rand.shape[0] == y_rand.shape[0], "Length mismatch"
            
            _, counts = np.unique(y_rand, return_counts=True)
            if any(counts < 10):
                print(f"Skipping {subject} {epoch_num} due to low counts")
            else:
                print("Processing", subject, "epoch", epoch_num, region, 'random')
                rdm_rand = cv_mahalanobis_parallel(X_rand, y_rand, jobs)
                np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)
                del X_rand, y_rand
                gc.collect()
                
        del labels, stcs_data
        gc.collect()
                
if is_cluster:
    jobs = 20
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        epoch_num = str(sys.argv[1])
        process_subject(subject, epoch_num, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    Parallel(-1)(delayed(process_subject)(subject, epoch_num, 1) for subject in subjects for epoch_num in range(5))