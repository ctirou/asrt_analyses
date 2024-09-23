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
from base import ensure_dir, get_volume_estimate_tc, rsync_files
from config import *
import gc
import sys

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = PRED_PATH
analysis = 'time_generalization'
subjects, epochs_list, subjects_dir = SUBJS, EPOCHS, FREESURFER_DIR
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
jobs = -1
verbose = True
overwrite = False

res_path = data_path / 'results' / 'source'
src_path = RESULTS_DIR / 'src'
ensure_dir(res_path)

# lock = str(sys.argv[1])
# subject_num = int(sys.argv[2])
# subject = subjects[subject_num]

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
cv = StratifiedKFold(folds, shuffle=True)

for lock in ['stim', 'button']:

    for subject in subjects:
        # read source space
        src_fname = src_path / f"{subject}-src.fif"
        src = mne.read_source_spaces(src_fname, verbose=verbose)
        # path to bem file
        bem_fname = op.join(data_path, "bem", "%s-bem-sol.fif" % (subject))
        
        for hemi in ['lh', 'rh', 'others']:
            # create mixed source space
            vol_src_fname = src_path / f"{subject}-{hemi}-vol-src.fif"
            vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
            mixed_src = src + vol_src

            for trial_type in ['pattern', 'random']:
                all_behavs = list()
                all_stcs = list()
                
                for epoch_num, epo in zip([1, 2, 3, 4], epochs_list[1:]):
                    # read behav
                    behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
                    # read epoch
                    epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
                    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
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
                    trans_fname = os.path.join(data_path, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))            
                    # compute forward solution
                    fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                                src=mixed_src, bem=bem_fname,
                                                meg=True, eeg=False,
                                                mindist=5.0,
                                                n_jobs=jobs,
                                                verbose=verbose)
                    # compute source estimates
                    filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                                    pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
                    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

                    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])
                    label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
                    # subcortex labels
                    labels = [label for label in label_tc.keys() if label not in BAD_VOLUME_LABELS]
                    del noise_cov, data_cov, filters
                    gc.collect()
                    for ilabel, label in enumerate(labels):
                        print(subject, lock, trial_type, hemi, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label) 
                        # results dir
                        res_dir = res_path / 'source' / lock / label / trial_type
                        ensure_dir(res_dir)
                        if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy"):
                            if trial_type == 'pattern':    
                                pattern = behav.trialtypes == 1
                                X = label_tc[label][pattern]
                                y = behav.positions[pattern]
                            elif trial_type == 'random':
                                random = behav.trialtypes == 2
                                X = label_tc[label][random]
                                y = behav.positions[random]
                            else:
                                X = label_tc[label]
                                y = behav.positions
                            y = y.reset_index(drop=True)            
                            assert X.shape[0] == y.shape[0]
                            # if X.shape[1] < 1:
                            #     continue            
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
                # subcortex labels
                labels = [label for label in label_tc.keys() if label not in BAD_VOLUME_LABELS]
                for ilabel, label in enumerate(labels):
                    # results dir
                    res_dir = res_path / 'source' / lock / label / trial_type
                    ensure_dir(res_dir)
                    if not os.path.exists(res_dir / f"{subject}-all-scores.npy"):
                        print(subject, lock, trial_type, hemi, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label) 
                        if trial_type == 'pattern':    
                            pattern = behav_df.trialtypes == 1
                            X = label_tc[label][pattern]
                            y = behav_df.positions[pattern]
                        elif trial_type == 'random':
                            random = behav_df.trialtypes == 2
                            X = label_tc[label][random]
                            y = behav_df.positions[random]
                        else:
                            X = label_tc[label]
                            y = behav_df.positions
                        y = y.reset_index(drop=True)            
                        assert X.shape[0] == y.shape[0]
                        if X.shape[1] < 1:
                            continue            
                        scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
                        np.save(res_dir / f"{subject}-all-scores.npy", scores.mean(0))
                        del X, y, scores
                        gc.collect()
                del behav_df, all_stcs
                gc.collect()
                
                source = "/Users/coum/Desktop/pred_asrt/results/source"
                destination = "/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results/time_generalization"
                options = "-av" 
                rsync_files(source, destination, options)