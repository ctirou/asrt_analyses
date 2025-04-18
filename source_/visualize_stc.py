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
from nilearn import plotting

# params
subjects = SUBJS

analysis = 'RSA'
lock = 'stim'
trial_type = 'all' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
parc = 'aparc'
hemi = 'both'

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = True
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

subject = 'sub01'
epoch_num = 1
jobs = -1

src_fname = RESULTS_DIR / "src" / f"{subject}-src.fif"
src = mne.read_source_spaces(src_fname, verbose=verbose)
bem_fname = RESULTS_DIR / "bem" / f"{subject}-bem-sol.fif"

vol_src_fname = RESULTS_DIR / 'vol_src' / f"{subject}-htc-vol-src.fif"
vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)    

mixed = src + vol_src
print(
    f"The source space contains {len(mixed)} spaces and "
    f"{sum(s['nuse'] for s in mixed)} vertices"
)

# src.plot(subjects_dir=subjects_dir)
# vol_src.plot(subjects_dir=subjects_dir)
# mixed.plot(subjects_dir=subjects_dir)

# nii_fname = RESULTS_DIR / 'src' / f"{subject}-mixed-src.nii"
# mixed.export_volume(nii_fname, mri_resolution=True, overwrite=True)
# plotting.plot_img(str(nii_fname), cmap="nipy_spectral")

# get session behav and epoch
epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
behav = pd.read_pickle(behav_fname).reset_index()

noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)

info = epoch.info

rank = mne.compute_rank(data_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)

trans_fname = os.path.join(RESULTS_DIR, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))

fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                            src=mixed, bem=bem_fname,
                            meg=True, eeg=False,
                            mindist=5.0,
                            n_jobs=jobs,
                            verbose=verbose)

# compute source estimates
filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

stc = stcs[150]

# initial_time = 0.1
# brain = stc.plot(
#     src=fwd['src'],
#     surface='white',
#     subjects_dir=subjects_dir,
#     initial_time=initial_time,
#     clim=dict(kind="value", lims=[3, 6, 9]),
#     smoothing_steps=7,
# )

# fig = stc.volume().plot(initial_time=0.5, src=mixed, subjects_dir=subjects_dir)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)

label_ts = mne.extract_label_time_course(
    [stc], labels_parc, mixed, mode='auto', allow_empty=True
)

import matplotlib.pyplot as plt
# plot the times series of 2 labels
fig, axes = plt.subplots(1,1, figsize=(14, 5), layout="constrained")
axes.axvspan(0, 200, color='grey', alpha=0.1, zorder=-1)
# axes.plot(1e3 * stc.times, label_ts[0][6, :], label="cuneus-lh")
# axes.plot(1e3 * stc.times, label_ts[0][-1, :].T, label="cerebellum-rh")
# axes.plot(1e3 * stc.times, label_ts[0][-2, :].T, label="cerebellum-lh")
axes.plot(1e3 * stc.times, label_ts[0][-3, :].T, label="thalamus-rh")
# axes.plot(1e3 * stc.times, label_ts[0][-4, :].T, label="thalamus-lh")
axes.plot(1e3 * stc.times, label_ts[0][-5, :].T, label="hippocampus-rh")
# axes.plot(1e3 * stc.times, label_ts[0][-6, :].T, label="hippocampus-lh")
axes.set(xlabel="Time (ms)", ylabel="MNE current (nAm)")
axes.legend()
plt.show()

fname_aseg = subjects_dir / subject / "mri" / "aparc+aseg.mgz"
label_names = mne.get_volume_labels_from_aseg(fname_aseg)
stc = stcs[0]

label_tc = stc.extract_label_time_course(label_names, src=src)
