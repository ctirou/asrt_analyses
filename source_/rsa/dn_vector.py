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
from dn_power import process_subject_power


# params
subjects = SUBJS
lock = 'stim'
# trial_type = 'all' # "all", "pattern", or "random"
analysis = 'RSA'
# data_path = DATA_DIR
data_path = TIMEG_DATA_DIR / 'rdm_bsling'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = True
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)  
    # define networks labels path
    n_parcels = 200
    n_networks = 7
    networks = NETWORKS[:-2]
    label_path = RESULTS_DIR / f'networks_{n_parcels}_{n_networks}' / subject
    
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
                            pick_ori='vector', weight_norm="unit-noise-gain",
                            rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        # del noise_cov, data_cov, fwd, filters
        # gc.collect()

        # print("Processing", subject, epoch_num, trial_type, network)
        
        # lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        # stcs_data = np.array([stc.in_label(lh_label + rh_label).data for stc in stcs])
        # assert len(stcs_data) == len(behav)
        
        # if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
        #     if trial_type == 'pattern':
        #         pattern = behav.trialtypes == 1
        #         X = stcs_data[pattern]
        #         y = behav.positions[pattern]
        #     elif trial_type == 'random':
        #         random = behav.trialtypes == 2
        #         X = stcs_data[random]
        #         y = behav.positions[random]
        #     else:
        #         X = stcs_data
        #         y = behav.positions    
        #     y = y.reset_index(drop=True)            
        #     assert X.shape[0] == y.shape[0]
        #     scores = cross_val_multiscore(clf, X, y, cv=cv)   
        #     np.save(op.join(res_dir, f"{subject}-{epoch_num}-scores.npy"), scores.mean(0))
            
        #     del stcs_data, X, y, scores
        #     gc.collect()
    
        # append epochs
        all_behavs.append(behav)
        all_stcs.extend(stcs)
        
        # del epoch, behav, stcs
        # if trial_type == 'button':
        #     del epoch_bsl
        # gc.collect()

    behav_df = pd.concat(all_behavs)
    behav_data = behav_df.reset_index(drop=True)
    del all_behavs, behav_df
    gc.collect()
    
    for network in networks:
    
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
        stcs_data = svd(stcs_data)
        
        assert len(stcs_data) == len(behav_data), "Length mismatch"
        
        for trial_type in ['pattern', 'random']:
            
            print("Processing", subject, 'all', network, trial_type)
            
            res_dir = RESULTS_DIR / "RSA" / "source" / network / lock / "vector_scores" / trial_type
            ensure_dir(res_dir)
            
            if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
                
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
                assert X.shape[0] == y.shape[0]
                
                scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs)
                np.save(op.join(res_dir, f"{subject}-all-scores.npy"), scores.mean(0))
                
                del X, y, scores
                gc.collect()
    
        del stcs_data
        gc.collect()

    del all_stcs, behav_data
    gc.collect()
    
if is_cluster:
    lock = str(sys.argv[1])
    trial_type = str(sys.argv[2])
    jobs = 20
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock, trial_type, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    # Parallel(-1)(delayed(process_subject)(subject, lock, jobs) for subject in subjects)
    for subject in subjects:
        process_subject(subject, lock, jobs)
    for subject in subjects:
        process_subject_power(subject, lock, jobs)