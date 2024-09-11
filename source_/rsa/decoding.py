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
from tqdm.auto import tqdm

# params
subjects = SUBJS

analysis = "decoding"
lock = "button" # "stim", "button"
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
verbose = 'error'
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
cv = StratifiedKFold(folds, shuffle=True)

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
        for epo in all_bsl:
            epo.info['dev_head_t'] = all_epo[0].info['dev_head_t']
        epoch_bsl = mne.concatenate_epochs(all_bsl)

    # read forward solution    
    # fwd_fname = res_path / analysis / "fwd" / lock / f"{subject}-mixed-lh-fwd.fif"
    # fwd = mne.read_forward_solution(fwd_fname, ordered=False, verbose=verbose)
    # labels = mne.get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
    
    src_fname = res_path / "src" / f"{subject}-src.fif"
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    bem_fname = res_path / "bem" / f"{subject}-bem-sol.fif"    

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
    trans_fname = os.path.join(res_path, "trans", lock, "%s-all-trans.fif" % (subject))

    # compute forward solution
    fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                src=src, bem=bem_fname,
                                meg=True, eeg=False,
                                mindist=5.0,
                                n_jobs=jobs,
                                verbose=verbose)

    # compute source estimates
    filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                    pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
    
    all_labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    labels = [label for label in all_labels if label.name in SURFACE_LABELS_RT]
    
    del epoch, fwd, src, data_cov, noise_cov, rank, info, filters
    gc.collect()
    
    for ilabel, label in enumerate(labels):
        
        print(subject, f"{str(ilabel+1).zfill(2)}/{len(labels)}", label.name)
        # results dir
        res_dir = res_path / analysis / 'source' / lock / trial_type / label.name
        ensure_dir(res_dir)
        
        # get stcs in label
        stcs_data = [stc.in_label(label).data for stc in stcs] # stc.in_label() doesn't work anymore for volume source space            
        stcs_data = np.array(stcs_data)
        assert len(stcs_data) == len(behav)

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
        np.save(res_dir / f"{subject}-scores.npy", scores.mean(0))
            
        del X, y, scores
        gc.collect()
    
labels = mne.get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
label_tc = get_volume_estimate_time_course(stcs, fwd, subject, subjects_dir)

labels_list = list(label_tc.keys())
scores_df = dict()
for label in labels_list: 
    print(label)
    pattern = behav.trialtypes == 1
    X = label_tc[label][pattern]
    y = behav.positions[pattern]
    scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
    scores_df[label] = scores.mean(0).T

import matplotlib.pyplot as plt
nrows, ncols = 4, 4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
for i, (ax, label) in enumerate(zip(axs.flat, labels_list)):
    ax.plot(times, scores_df[label])
    ax.axhline(.25, color='black', ls='dashed', alpha=.5)
    ax.set_title(f"${label}$", fontsize=8)    
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
plt.show()
fig.savefig(f"/Users/coum/Desktop/test_sub_decoding-lh.pdf")
plt.close()

plt.plot(times, scores.mean(0).T)
plt.title(label.name)
plt.axvline(x=0, color='black', linestyle='--')
plt.axhline(y=0.25, color='black', linestyle='--')
plt.show()
# plt.save(f"/Users/coum/Desktop/test_sub_decoding/{label.name}.pdf")
# plt.close()