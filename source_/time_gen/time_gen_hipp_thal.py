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
verbose = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

lock = 'stim'
jobs = -1
overwrite = False

res_path = data_path / 'results' / 'source'
ensure_dir(res_path)

def process_subject(subject, lock, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    # read volume source space
    vol_src_fname =  RESULTS_DIR / 'src' / f"{subject}-hipp-thal-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)

    for region in ['Hippocampus', 'Thalamus']:

        for trial_type in ['pattern', 'random']:
            all_behavs = list()
            all_stcs = list()
            # results dir
            res_dir = res_path / lock / region / trial_type
            ensure_dir(res_dir)
            
            for epoch_num in [0, 1, 2, 3, 4]:
                # read behav
                behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
                # read epoch
                epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
                epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
                # compute data covariance matrix on evoked data
                data_cov = mne.compute_covariance(epoch, tmin=epoch.times[0], tmax=epoch.times[-1], method="auto", rank="info", verbose=verbose)
                # read noise cov computed on resting state
                noise_cov = mne.read_cov(data_path / 'noise_cov' / f"{subject}-rs2-cov.fif", verbose=verbose)
                # conpute rank
                rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
                # compute forward solution
                fwd_fname = data_path / "fwd" / lock / f"{subject}-hipp-thal-{epoch_num}-fwd.fif"
                fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
                # compute source estimates
                filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                                pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
                stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
                # get data from volume source space
                offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
                label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                # get data from regions of interest
                labels = [label for label in label_tc.keys() if region in label]
                stcs_data = np.concatenate([label_tc[label] for label in labels], axis=1) # this works
                
                del noise_cov, data_cov, filters
                gc.collect()
                
                print(f"Processing {subject} - {epoch_num} - {region} - {trial_type}...")
                
                if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
                    if trial_type == 'pattern':    
                        pattern = behav.trialtypes == 1
                        X = stcs_data[pattern]
                        y = behav.positions[pattern]
                    elif trial_type == 'random':
                        random = behav.trialtypes == 2
                        X = stcs_data[random]
                        y = behav.positions[random]
                    else:
                        X = stcs_data
                        y = behav.positions
                    y = y.reset_index(drop=True)            
                    assert X.shape[0] == y.shape[0]        
                    scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
                    np.save(res_dir / f"{subject}-{epoch_num}-scores.npy", scores.mean(0))
            
                    del X, y, scores
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
            
            label_tc, _ = get_volume_estimate_tc(all_stcs, fwd, offsets, subject, subjects_dir)
            labels = [label for label in label_tc.keys() if region in label]
            stcs_data = np.concatenate([label_tc[label] for label in labels], axis=1) # this works
            
            print(f"Processing {subject} - all - {region} - {trial_type}...")
            
            if not os.path.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
                if trial_type == 'pattern':    
                    pattern = behav_df.trialtypes == 1
                    X = stcs_data[pattern]
                    y = behav_df.positions[pattern]
                elif trial_type == 'random':
                    random = behav_df.trialtypes == 2
                    X = stcs_data[random]
                    y = behav_df.positions[random]
                else:
                    X = stcs_data
                    y = behav_df.positions
                y = y.reset_index(drop=True)            
                assert X.shape[0] == y.shape[0]
                scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
                np.save(res_dir / f"{subject}-all-scores.npy", scores.mean(0))
                
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
        process_subject(subject, lock, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock, jobs=-1)