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

# params
subject = SUBJS[0]

analysis = "decoding_cvm"
lock = "stim" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 3
solver = 'lbfgs'
scoring = "accuracy"
parc='aparc'
hemi = 'both'
verbose = True
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times

# get label index    
ilabel = 0
# get labels
labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label = labels[ilabel]

del epochs, epoch_fname, labels
gc.collect()

for session_id, session in enumerate(sessions):
        
    # results dir
    res_dir = res_path / analysis / 'source' / lock / trial_type / label.name / subject / session
    ensure_dir(res_dir)    
    # read stim epoch
    epoch_fname = data_path / lock / f"{subject}-{session_id}-epo.fif"
    epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
    # read behav
    behav_fname = data_path / "behav" / f"{subject}-{session_id}.pkl"
    behav = pd.read_pickle(behav_fname).reset_index()
    # get session behav and epoch
    if session_id == 0:
        session = 'prac'
    else:
        session = 'sess-%s' % (str(session_id).zfill(2))
    if lock == 'button': 
        epoch_bsl_fname = data_path / "bsl" / f"{subject}-{session_id}-epo.fif"
        epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
    # read forward solution    
    fwd_fname = res_path / "fwd" / lock / f"{subject}-{session_id}-fwd.fif"
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
        
    # get stcs in label
    stcs_data = [stc.in_label(label).data for stc in stcs]
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
    
    del epoch, epoch_fname, behav_fname, fwd, fwd_fname, data_cov, noise_cov, rank, info, filters, stcs, stcs_data
    gc.collect()
    
    # set-up the classifier and cv structure
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True)
    
    scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
    np.save(res_dir / "scores.npy", scores.mean(0))
        
    del X, y, clf, cv, scores
    gc.collect()