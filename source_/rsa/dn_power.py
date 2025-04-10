import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import SlidingEstimator, cross_val_multiscore
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
analysis = 'RSA'
subjects_dir = FREESURFER_DIR
# data_path = DATA_DIR
data_path = TIMEG_DATA_DIR / 'gen44'
# data_path = TIMEG_DATA_DIR / 'rdm_bsling'

# analysis = "no-bls-noise-0200-data-0006"
# analysis = "no-bls-noise-1715-data-0006"
analysis = "no-bls-noise-1715-data-0206"

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = 'error'
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, trial_type, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    # clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000000, solver=solver, class_weight="balanced", random_state=42, n_jobs=jobs))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    # define networks labels path
    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / 'networks_200_7' / subject
    
    # for network in networks:
    all_behavs = list()
    all_stcs = list()

    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
        # epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-0.2, 0.6)
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
        # compute data and noise covariance
        # data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
        # noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
        data_cov = mne.compute_covariance(epoch, tmin=-0.2, tmax=.6, method="empirical", rank="info", verbose=verbose)
        noise_cov = mne.compute_covariance(epoch, tmin=-1.7, tmax=-1.5, method="empirical", rank="info", verbose=verbose)
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
        
    times = epoch.times
    win = np.where((times >= -0.2) & (times <= 0.6))[0]

    behav_data = pd.concat(all_behavs).reset_index(drop=True)
    
    del all_behavs, epoch
    gc.collect()
    
    for network in networks:
    
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
        
        res_dir = RESULTS_DIR / "RSA" / "source" / network / lock / analysis / trial_type
        ensure_dir(res_dir)
        
        if not op.exists(res_dir / f"{subject}-scores.npy") or overwrite:
            print("Processing", subject, network, trial_type)
            
            filter = behav_data.trialtypes == 1 if trial_type == 'pattern' else behav_data.trialtypes == 2
            # X = stcs_data.copy()[filter]
            X = stcs_data.copy()[filter][:, :, win]
            y = behav_data.positions[filter]
            y = y.reset_index(drop=True)
            assert X.shape[0] == y.shape[0], "Length mismatch"
            
            del stcs_data
            gc.collect()
            
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=verbose)
            np.save(op.join(res_dir, f"{subject}-scores.npy"), scores.mean(0))
            
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
        trial_type = sys.argv[1]
        process_subject(subject, trial_type, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, trial_type, jobs) for subject in subjects for trial_type in ['pattern', 'random']) 