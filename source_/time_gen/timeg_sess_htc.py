import mne
import os
import os.path as op
import numpy as np
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import pandas as pd
from base import ensure_dir, get_volume_estimate_tc
from config import *
import gc
import sys

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"

lock = 'stim'

verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = data_path / 'results' / 'source' / 'max-power'
ensure_dir(res_path)

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    # read volume source space
    vol_src_fname =  data_path / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work

    all_behavs = list()
    all_stcs = list()
    
    del vol_src
    gc.collect()
    
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

        # compute forward solution
        fwd_fname = data_path / "fwd" / f"{subject}-htc-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        
        # compute source estimates
        filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                            pick_ori='max-power', weight_norm="unit-noise-gain",
                            rank=rank, reduce_rank=True, verbose=verbose)
                
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        # get data from volume source space
        label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
        
        all_stcs.extend(stcs)
        
        del epoch, noise_cov, data_cov, fwd, filters, stcs
        gc.collect()
        
        for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:

            # get data from regions of interest
            labels = [label for label in label_tc.keys() if region in label]
            stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
            
            for trial_type in ['pattern', 'random']:
                
                res_dir = res_path / region / trial_type
                ensure_dir(res_dir)
                
                if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
                    print("Processing", subject, epoch_num, trial_type, region)
                    
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
                    print("Skipping", subject, epoch_num, trial_type, region)
            
            del labels, stcs_data
            gc.collect()
                
        all_behavs.append(behav)
                    
    behav_data = pd.concat(all_behavs)
    all_stcs = np.array(all_stcs)
    
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose) # fwd only needed to extract source space, so can be any one of the epochs
    label_tc, _ = get_volume_estimate_tc(all_stcs, fwd, offsets, subject, subjects_dir)
    
    del fwd, all_stcs, all_behavs
    gc.collect()
    
    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
    
        labels = [label for label in label_tc.keys() if region in label]
        stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works
    
        for trial_type in ['pattern', 'random']:
            res_dir = res_path / region / trial_type
            ensure_dir(res_dir)

            if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
                print("Processing", subject, 'all', trial_type, region)
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
                print("Skipping", subject, 'all', trial_type, region)
                
        del labels, stcs_data
        gc.collect()
                
if is_cluster:
    jobs = 20
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, jobs=-1)