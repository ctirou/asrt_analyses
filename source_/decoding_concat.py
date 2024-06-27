import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from base import ensure_dir
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc
import os

# params
subjects = SUBJS

analysis = "concatenated"
lock = "stim" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
ensure_dir(res_path)
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
parc='aparc'
# parc='aseg'
hemi = 'both'
verbose = True
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

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

    # read forward solution    
    fwd_fname = res_path / analysis / "fwd" / lock / f"{subject}-mixed-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
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
    # compute source estimates
    filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                    pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    # set-up the classifier and cv structure
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True)

    labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    
    del epoch, fwd, fwd_fname, data_cov, noise_cov, rank, info, filters
    gc.collect()

    for ilabel, label in enumerate(labels):
        
        print(subject, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label.name)
        # results dir
        res_dir = res_path / analysis / 'source' / lock / trial_type / label.name / subject
        ensure_dir(res_dir)
        
        # get stcs in label
        stcs_data = [stc.in_label(label).data for stc in stcs] # stc.in_label() doesn't work anymore for volume source space    
        
        stcs_data = np.array(stcs_data)
        assert len(stcs_data) == len(behav)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, layout="constrained")
        # axes.plot(times, stcs_data[0][6, :].T, "b", label="L_cuneus")
        # axes.plot(times, stcs_data[0][7, :].T, "g", label="R_cuneus")
        
        # axes.plot(times, stcs_data[0][-1, :].T, "r", label="R_HPC")
        # axes.plot(times, stcs_data[0][-2, :].T, "k", label="L_HPC")
        # # axes.plot(times, stcs_data[0][0, :], "k", label="bankssts-lh")
        # axes.set(xlabel="Time (ms)", ylabel="MNE current (nAm)")
        # axes.legend()
        # plt.show()

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

        del stcs_data
        gc.collect()

        scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
        np.save(res_dir / "scores.npy", scores.mean(0))
            
        del X, y, scores
        gc.collect()
    
    labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
        
    stcs_data = mne.extract_label_time_course(stcs, labels, src=fwd['src'], mode=None, allow_empty=True, verbose=verbose)
    stcs_data = np.array(stcs_data, dtype=object)
    print(stcs_data.shape)        
    assert len(stcs_data) == len(behav)
    
    volume_labels = ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
                    "Left-Thalamus-Proper", "Right-Thalamus-Proper",
                    "Left-Hippocampus", "Right-Hippocampus"]
    labels += volume_labels
    
    for ilabel, label in enumerate(labels):
        if ilabel > len(labels) - len(volume_labels) - 1:
            label_fname = label
        else:
            label_fname = label.name

        print(subject, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label_fname)
        
        if ilabel > 67:
            
            # results dir
            res_dir = res_path / analysis / 'source' / lock / trial_type / label_fname / subject
            ensure_dir(res_dir)
            
            if trial_type == 'pattern':
                pattern = behav.trialtypes == 1
                X = stcs_data[pattern][:, ilabel, :]
                y = behav.positions[pattern]
            elif trial_type == 'random':
                random = behav.trialtypes == 2
                X = stcs_data[random][:, ilabel, :]
                y = behav.positions[random]
            else:
                X = stcs_data[:, ilabel, :]
                y = behav.positions
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]

            del stcs_data
            gc.collect()
            
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
            np.save(res_dir / "scores.npy", scores.mean(0))
                
            del X, y, scores
            gc.collect()
