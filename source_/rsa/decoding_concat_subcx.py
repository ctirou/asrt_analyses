import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import ensure_dir, get_volume_estimate_time_course
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc
import os

# params
subjects = SUBJS

lock = "stim" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR / "concatenated"
ensure_dir(res_path)
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

vertices_df = dict()

for subject in subjects:
            
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epo)

    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)

    if lock == 'button': 
        bsl_data = data_path / "bsl"
        epoch_bsl_fnames = [bsl_data / f"{f}" for f in sorted(os.listdir(bsl_data)) if ".fif" in f and subject in f]
        all_bsl = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epoch_bsl_fnames]
        for epoch in all_bsl:
            epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
        epoch_bsl = mne.concatenate_epochs(all_bsl)

    # compute data covariance matrix on evoked data
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    # compute noise covariance
    if lock == 'button':
        noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
    else:
        noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    info = epoch.info
    # conpute rank
    rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)

    sub_vertices_info = dict()
    
    for hemi in ['lh', 'rh', 'others']:

        # read forward solution    
        fwd_fname = res_path / "fwd" / lock / f"{subject}-mixed-{hemi}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, ordered=False, verbose=verbose)
        # compute source estimates
        filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                        pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        # set-up the classifier and cv structure
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
        clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
        cv = StratifiedKFold(folds, shuffle=True)
        
        label_tc, vertices_info = get_volume_estimate_time_course(stcs, fwd, subject, subjects_dir)
        labels_list = list(label_tc.keys())
        sub_vertices_info.update(vertices_info)
        
        del fwd, fwd_fname, filters, stcs
        gc.collect()

        for ilabel, label in enumerate(labels_list): 
            print(subject, f"{str(ilabel+1).zfill(2)}/{len(labels_list)}", label)
            # results dir
            res_dir = res_path / 'source' / lock / trial_type / label.name
            ensure_dir(res_dir)
            
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
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
            np.save(res_dir / f"{subject}-scores.npy", scores.mean(0))
                    
            del X, y, scores
            gc.collect()
        
        del label_tc, labels_list
        gc.collect()
            
    vertices_df[subject] = dict(sorted(sub_vertices_info.items()))
    
info_df = pd.DataFrame() 
for subject, vertices_info in vertices_df.items():
    for key, value in vertices_info.items():
        # If the key is not in the DataFrame, add it with the current subject's value
        if key not in info_df.index:
            info_df.loc[key, subject] = value
        else:
            info_df.at[key, subject] = value
info_df.replace(0, np.nan, inplace=True)
# Export the DataFrame to a CSV file
info_df.to_csv(res_path / 'subcx_vertices_info.csv', sep="\t")