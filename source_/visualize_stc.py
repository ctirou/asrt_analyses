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

# read stim epoch
epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
# read behav
behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
behav = pd.read_pickle(behav_fname).reset_index()
# get session behav and epoch
if lock == 'button': 
    epoch_bsl_fname = data_path / "bsl" / f"{subject}-{epoch_num}-epo.fif"
    epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
    # compute noise covariance
    noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
else:
    noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    
# read forward solution    
# fwd_fname = RESULTS_DIR / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
# fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
# compute data covariance matrix on evoked data
data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)

info = epoch.info
# conpute rank
rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)
trans_fname = os.path.join(RESULTS_DIR, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
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

initial_time = 0.1
brain = stcs[0].plot(
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

fname_aseg = subjects_dir / subject / "mri" / "aparc+aseg.mgz"
label_names = mne.get_volume_labels_from_aseg(fname_aseg)
stc = stcs[0]

label_tc = stc.extract_label_time_course(label_names, src=src)
