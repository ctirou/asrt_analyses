import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
import sys
from joblib import Parallel, delayed

# params
subjects = SUBJS
lock = 'stim'
data_path = TIMEG_DATA_DIR / 'gen44'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

# res_path = TIMEG_DATA_DIR / 'results' / 'source' / 'max-power'
res_path = TIMEG_DATA_DIR / 'results' / 'source' / 'max-power-cv'
ensure_dir(res_path)

def process_subject(subject, jobs):
    # define classifier'
    # clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    # network and custom label_names
    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / 'networks_200_7' / subject    
    
    all_behavs = list()
    all_stcs = list()
        
    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        big_epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
                        
        filter = behav.trialtypes == 2
        noise_epoch = big_epoch[filter]
        noise_cov = mne.compute_covariance(noise_epoch, tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
        epoch = big_epoch.copy().crop(-1.5, 1.5)
        
        del big_epoch, noise_epoch
        gc.collect()
        
        # compute data covariance matrix
        data_cov = mne.compute_covariance(epoch, method="empirical", rank="info", verbose=verbose)
        # conpute rank
        rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
        
        # read forward solution
        fwd_fname = TIMEG_DATA_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
                
        # compute source estimates
        filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                            pick_ori='max-power', weight_norm="unit-noise-gain",
                            rank=rank, reduce_rank=True, verbose=verbose)
                
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

        del epoch, noise_cov, data_cov, fwd, filters
        gc.collect()

        for network in networks:
            # read labels
            lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
            stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs])
            assert len(stcs_data) == len(behav), "Length mismatch"
            
            for trial_type in ['pattern', 'random']:
                res_dir = res_path / network / trial_type
                ensure_dir(res_dir)
                
                if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
                    print("Processing", subject, epoch_num, trial_type, network)
                    
                    if trial_type == 'pattern':
                        pattern = behav.trialtypes == 1
                        X = stcs_data[pattern]
                        y = behav.positions[pattern]

                    elif trial_type == 'random':
                        random = behav.trialtypes == 2
                        X = stcs_data[random]
                        y = behav.positions[random]
                    
                    y = y.reset_index(drop=True)
                    assert X.shape[0] == y.shape[0], "Length mismatch"
                    
                    cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
                    scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=verbose)                    
                    np.save(op.join(res_dir, f"{subject}-{epoch_num}-scores.npy"), scores.mean(0))
                    
                    del X, y, scores
                    gc.collect()
                
                else:
                    print("Skipping", subject, epoch_num, trial_type, network)
                
            del stcs_data
            gc.collect()
        
        if epoch_num != 0:
            all_behavs.append(behav)
            all_stcs.extend(stcs)
        
        del stcs, behav
        gc.collect()
    
    behav_df = pd.concat(all_behavs)
    del all_behavs
    gc.collect()
    
    for network in networks:
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
        behav_data = behav_df.reset_index(drop=True)
        assert len(stcs_data) == len(behav_data), "Shape mismatch"
    
        for trial_type in ['pattern', 'random']:
            res_dir = res_path / network / trial_type
            ensure_dir(res_dir)

            if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
                print("Processing", subject, 'all', trial_type, network)
                if trial_type == 'pattern':
                    pattern = behav_data.trialtypes == 1
                    X = stcs_data[pattern]
                    y = behav_data.positions[pattern]
                elif trial_type == 'random':
                    random = behav_data.trialtypes == 2
                    X = stcs_data[random]
                    y = behav_data.positions[random]
                y = y.reset_index(drop=True)
                assert X.shape[0] == y.shape[0]
                
                cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
                scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=True)
                np.save(op.join(res_dir, f"{subject}-all-scores.npy"), scores.mean(0))
                
                del X, y, scores
                gc.collect()
            
            else:
                print("Skipping", subject, 'all', trial_type, network)            

        del stcs_data, behav_data
        gc.collect()
        
    del all_stcs, behav_df
    gc.collect()
        
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
    jobs = 15
    # Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
    for subject in subjects:
        process_subject(subject, jobs)