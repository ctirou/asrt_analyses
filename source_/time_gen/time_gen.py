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

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = TIMEG_DATA_DIR
analysis = 'time_generalization'
subjects, epochs_list, subjects_dir = SUBJS, EPOCHS, FREESURFER_DIR
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
jobs = 10
verbose = True
res_path = data_path / 'results' / 'source'
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
        src_fname = op.join(data_path, "src", "%s-src.fif" % (subject))
        src = mne.read_source_spaces(src_fname, verbose=verbose)
        # path to bem file
        bem_fname = op.join(data_path, "bem", "%s-bem-sol.fif" % (subject))
        # get labels
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)

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
                                            src=src, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=verbose)
                # compute source estimates
                filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                                pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
                stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

                del noise_cov, data_cov, fwd, filters
                gc.collect()

                for ilabel, label in enumerate(labels):                
                    print(subject, lock, trial_type, epo, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label.name)
                    # results dir
                    res_dir = res_path / lock / label.name / trial_type
                    ensure_dir(res_dir)
                    if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy"):
                        # get stcs in label
                        stcs_data = [stc.in_label(label).data for stc in stcs]
                        stcs_data = np.array(stcs_data)
                        assert len(stcs_data) == len(behav)
                        # run time generalization decoding on unique epoch
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
                        scores = cross_val_multiscore(clf, X, y, cv=cv)
                        np.save(op.join(res_dir, f"{subject}-{epoch_num}-scores.npy"), scores.mean(0))

                        del stcs_data, X, y
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
            
            for ilabel, label in enumerate(labels):
                print(subject, lock, trial_type, "all", f"{str(ilabel+1).zfill(2)}/{len(labels)}", label.name)
                # results dir
                res_dir = res_path / lock / label.name / trial_type
                ensure_dir(res_dir)

                if not os.path.exists(res_dir / f"{subject}-all-scores.npy"):
                    # get stcs in label
                    stcs_data = [stc.in_label(label).data for stc in all_stcs] # stc.in_label() doesn't work anymore for volume source space            
                    stcs_data = np.array(stcs_data)
                    behav_data = behav_df.reset_index(drop=True)
                    assert len(stcs_data) == len(behav_data)
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
                    del stcs_data, behav_data
                    gc.collect()
                    scores = cross_val_multiscore(clf, X, y, cv=cv)
                    np.save(op.join(res_dir, f"{subject}-all-scores.npy"), scores.mean(0))
                    del X, y, scores
                    gc.collect()
            
            del behav_df, all_stcs
            gc.collect()