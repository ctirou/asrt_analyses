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

subjects_dir = FREESURFER_DIR
fname_fs_src = RESULTS_DIR / "src" / "fsaverage2-src.fif"

data_path = DATA_DIR
# subject = 'sub01'
lock = 'stim'
verbose = True

for subject in SUBJS:
    
    fname_src = RESULTS_DIR / 'src' / f'{subject}-src.fif'
    # fname_fwd = RESULTS_DIR / 'fwd' / lock / f'{subject}-all-fwd.fif'
    # fwd = mne.read_forward_solution(fname_fwd, verbose=verbose)

    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
        
        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
        noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
        # conpute rank
        rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        # compute source estimates
        filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                            pick_ori="max-power", weight_norm="unit-noise-gain",
                            rank=rank, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        morph = mne.compute_source_morph(
            fwd['src'], # src_to=fwd["src"] needs to be passed, bc vertices were likely excluded during forward computation
            subject_from=subject,
            subject_to='fsaverage2',
            subjects_dir=subjects_dir,
            verbose=verbose
        )

        res_dir = RESULTS_DIR / 'power_stc' / lock
        ensure_dir(res_dir)
        morphed_stcs = [morph.apply(stc) for stc in stcs]
        morphed_stcs = np.array(morphed_stcs)
        np.save(res_dir / f"{subject}-morphed-stcs-{epoch_num}.npy", morphed_stcs)

        # stcs = np.array(stcs)
        # np.save(RESULTS_DIR / 'stc' / lock / f"{subject}-stcs-{epoch_num}.npy", stcs)

    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f and not f.startswith('.')]
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epo)

    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f and not f.startswith('.')]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)

    noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    # compute data covariance matrix on evoked data
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    # conpute rank
    rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
    # read forward solution
    # compute source estimates
    filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                    pick_ori=None, rank=rank, reduce_rank=True, reg=0.05, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

    src_fs = mne.read_source_spaces(fname_fs_src)
    src = mne.read_source_spaces(fname_src)

    morph = mne.compute_source_morph(
        fwd['src'], # src_to=fwd["src"] needs to be passed, bc vertices were likely excluded during forward computation
        subject_from=subject,
        subject_to='fsaverage2',
        subjects_dir=subjects_dir,
        verbose=verbose
    )

    ensure_dir(RESULTS_DIR / 'stc' / lock)
    morphed_stcs = [morph.apply(stc) for stc in stcs]
    morphed_stcs = np.array(morphed_stcs)
    np.save(RESULTS_DIR / 'stc' / lock / f"{subject}-morphed-stcs.npy", morphed_stcs)

    # stcs = np.array(stcs)
    # np.save(RESULTS_DIR / 'stc' / lock / f"{subject}-stcs.npy", stcs)

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver='lbfgs', class_weight="balanced", random_state=42))
clf = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=verbose)
cv = StratifiedKFold(10, shuffle=True, random_state=42)  

network = NETWORKS[1]

subjects = SUBJS
fs_label_path = RESULTS_DIR / 'networks_200_7' / 'fsaverage2'
lh_label, rh_label = mne.read_label(fs_label_path / f'{network}-lh.label'), mne.read_label(fs_label_path / f'{network}-rh.label')

for subject in subjects: 
    
    fname_stcs = RESULTS_DIR / 'stc' / lock / f"{subject}-stcs.npy"
    data = np.load(fname_stcs, allow_pickle=True)
    morphed_stcs_data = np.array([stc.in_label(lh_label + rh_label).data for stc in data])
    
    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f and not f.startswith('.')]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)
    behav_data = behav.reset_index(drop=True)
    assert len(morphed_stcs_data) == len(behav_data)

    for trial_type in ['pattern', 'random']:
        if trial_type == 'pattern':
            pattern = behav_data.trialtypes == 1
            X = morphed_stcs_data[pattern]
            y = behav_data.positions[pattern]
        elif trial_type == 'random':
            random = behav_data.trialtypes == 2
            X = morphed_stcs_data[random]
            y = behav_data.positions[random]
        else:
            X = morphed_stcs_data
            y = behav_data.positions
        y = y.reset_index(drop=True)

        assert X.shape[0] == y.shape[0]
        scores_morphed = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=-1)
        
        ensure_dir(RESULTS_DIR / 'RSA' / 'source' / network / lock / 'morphed_scores' / trial_type)
        np.save(RESULTS_DIR / 'RSA' / 'source' / network / lock / 'morphed_scores' / trial_type / f"{subject}-all-scores.npy", scores_morphed)


morphed_scores_dir = RESULTS_DIR / 'RSA' / 'source' / network / lock / 'morphed_scores' / trial_type
scores_dir = RESULTS_DIR / 'RSA' / 'source' / network / lock / 'scores' / trial_type

all_morphed_scores = [np.load(morphed_scores_dir / f"{subject}-all-scores.npy").mean(0) for subject in subjects]
all_scores = [np.load(scores_dir / f"{subject}-all-scores.npy") for subject in subjects]
all_morphed_scores, all_scores = np.array(all_morphed_scores), np.array(all_scores)

times = np.linspace(-.2, .6, all_morphed_scores.shape[-1])

trial_type = 'pattern'
import matplotlib.pyplot as plt

plt.plot(times, all_scores.mean(0), label='source')
pval = decod_stats(all_scores - .25, -1)
sig = pval < .05
plt.fill_between(times, all_scores.mean(0), 0.25, where=sig, alpha=.5)

plt.plot(times, all_morphed_scores.mean(0), label='morphed')
pval = decod_stats(all_morphed_scores - .25, -1)
sigm = pval < .05
plt.fill_between(times, all_morphed_scores.mean(0), 0.25, where=sigm, alpha=.5)

plt.axhline(.25, color='k', linestyle='--')
plt.legend()
plt.show()