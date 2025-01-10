import mne
import os
import os.path as op
import numpy as np
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from base import ensure_dir
from config import *
import gc
import sys

data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
verbose = True
res_dir = data_path / 'results' / 'source'
ensure_dir(res_dir)
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
lock = 'stim'
overwrite = False

def process_subject(subject, lock, jobs, rsync):    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    cv = StratifiedKFold(folds, shuffle=True)
    # network and custom label_names
    n_parcels = 200
    n_networks = 7
    # networks = (NEW_LABELS + schaefer_7) if n_networks == 7 else (NEW_LABELS + schaefer_17)
    networks = schaefer_7 if n_networks == 7 else schaefer_17
    label_path = RESULTS_DIR / f'networks_{n_parcels}_{n_networks}' / subject
    
    for trial_type in ['pattern', 'random']:
        all_behavs = list()
        all_stcs = list()
        
        for epoch_num in [0, 1, 2, 3, 4]:
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            # read epoch
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
            
            times = epoch.times
            win = np.where((times >= -1.5) & (times <= 1.5))[0]
            
            if lock == 'button': 
                epoch_bsl_fname = data_path / 'bsl' / f'{subject}_{epoch_num}_bl-epo.fif'
                epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose, preload=False)
                # compute noise covariance
                noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
            else:
                noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
            # path to trans file
            fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            # compute source estimates
            filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

            del noise_cov, data_cov, fwd, filters
            gc.collect()

            for network in networks[:-2]:                
                print("Processing", subject, epoch_num, trial_type, network)
                res_path = res_dir / lock / f'networks_{n_parcels}_{n_networks}' / network / trial_type
                ensure_dir(res_path)
                lh_label = mne.read_label(label_path / f'{network}-lh.label')
                rh_label = mne.read_label(label_path / f'{network}-rh.label')
                stcs_data = [stc.in_label(lh_label + rh_label).data for stc in stcs]
                stcs_data = np.array(stcs_data)
                assert len(stcs_data) == len(behav)
                
                # run time generalization decoding on unique epoch
                if not os.path.exists(res_path / f"{subject}-{epoch_num}-scores.npy") or overwrite:
                    if trial_type == 'pattern':
                        pattern = behav.trialtypes == 1
                        X = stcs_data[pattern][:, :, win]
                        y = behav.positions[pattern]
                    elif trial_type == 'random':
                        random = behav.trialtypes == 2
                        X = stcs_data[random][:, :, win]
                        y = behav.positions[random]
                    else:
                        X = stcs_data
                        y = behav.positions    
                    y = y.reset_index(drop=True)            
                    assert X.shape[0] == y.shape[0]
                    scores = cross_val_multiscore(clf, X, y, cv=cv)
                    np.save(op.join(res_path, f"{subject}-{epoch_num}-scores.npy"), scores.mean(0))

                    del stcs_data, X, y, scores
                    gc.collect()
            
            # append epochs
            all_behavs.append(behav)
            all_stcs.extend(stcs)
            
            del epoch, behav, stcs
            if trial_type == 'button':
                del epoch_bsl
            gc.collect()
        
        behav_df = pd.concat(all_behavs)
        all_stcs = np.array(all_stcs)
        del all_behavs
        gc.collect()
        
        for network in networks[:-2]:
            print("Processing", subject, 'all', trial_type, network)
            res_path = res_dir / lock / f'networks_{n_parcels}_{n_networks}' / network / trial_type
            ensure_dir(res_path)

            if not op.exists(res_path / f"{subject}-all-scores.npy") or overwrite:
                stcs_data = [stc.in_label(lh_label + rh_label).data for stc in all_stcs]
                stcs_data = np.array(stcs_data)
                behav_data = behav_df.reset_index(drop=True)
                assert len(stcs_data) == len(behav_data)
                if trial_type == 'pattern':
                    pattern = behav_data.trialtypes == 1
                    X = stcs_data[pattern][:, :, win]
                    y = behav_data.positions[pattern]
                elif trial_type == 'random':
                    random = behav_data.trialtypes == 2
                    X = stcs_data[random][:, :, win]
                    y = behav_data.positions[random]
                else:
                    X = stcs_data
                    y = behav_data.positions    
                y = y.reset_index(drop=True)
                assert X.shape[0] == y.shape[0]
                del stcs_data, behav_data
                gc.collect()
                scores = cross_val_multiscore(clf, X, y, cv=cv)
                np.save(op.join(res_path, f"{subject}-all-scores.npy"), scores.mean(0))
                del X, y, scores
                gc.collect()
        
        del behav_df, all_stcs
        gc.collect()
        
if is_cluster:
    lock = str(sys.argv[1])
    jobs = 20
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock, jobs, rsync=True)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock, jobs=-1, rsync=False)