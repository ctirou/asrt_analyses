import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys
from joblib import Parallel, delayed

# params
subjects = SUBJS
lock = 'stim'
analysis = 'RSA'
# data_path = DATA_DIR
data_path = TIMEG_DATA_DIR / 'rdm_bsling'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)  
    # define networks labels path
    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / f'networks_200_7' / subject
    
    # for network in networks:
    all_behavs = list()
    all_stcs = list()

    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-0.2, 0.6)

        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
        noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
        # conpute rank
        rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif" # this fwd was not generated on the rdm_bsling data
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        # compute source estimates
        filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                            pick_ori='max-power', weight_norm="unit-noise-gain",
                            rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            
        # append epochs
        all_behavs.append(behav)
        all_stcs.extend(stcs)
        
    behav_df = pd.concat(all_behavs)
    behav_data = behav_df.reset_index(drop=True)
    
    del all_behavs, behav_df
    gc.collect()
    
    for network in networks:
    
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
        
        assert len(stcs_data) == len(behav_data), "Length mismatch"
        
        for trial_type in ['pattern', 'random']:
            res_dir = RESULTS_DIR / "RSA" / "source" / network / lock / "power_scores_cv" / trial_type
            ensure_dir(res_dir)
            
            if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
                print("Processing", subject, network, trial_type)
                if trial_type == 'pattern':
                    pattern = behav_data.trialtypes == 1
                    X = stcs_data[pattern]
                    y = behav_data.positions[pattern]
                elif trial_type == 'random':
                    random = behav_data.trialtypes == 2
                    X = stcs_data[random]
                    y = behav_data.positions[random]
                else:
                    X = stcs_data
                    y = behav_data.positions
                y = y.reset_index(drop=True)
                assert X.shape[0] == y.shape[0], "Length mismatch"
                
                scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=verbose)
                np.save(op.join(res_dir, f"{subject}-all-scores.npy"), scores.mean(0))
                
                del X, y, scores
                gc.collect()
            
            else:
                print("Skipping", subject, network, trial_type)
    
        del stcs_data
        gc.collect()

    del all_stcs, behav_data
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
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)