import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from base import ensure_dir
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc

# params
subject = SUBJS[1] # done : none

trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
lock = "stim" 
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 3
scoring = "roc_auc"
parc='aparc'
hemi = 'both'
analysis = "pred_decoding"
verbose = True
jobs = 10

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times

# get label index    
ilabel = 0
    
# get labels
labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label = labels[ilabel]

del epochs, epoch_fname, labels
gc.collect()

# for session_id, session in enumerate(sessions):
for session_id, session in zip([3, 4], sessions[-2:]):
    
    # results dir
    res_dir = res_path / analysis / 'source' / lock / trial_type / label.name / subject / session
    ensure_dir(res_dir)    
    # read stim epoch
    epoch_fname = data_path / lock / f"{subject}_{session_id}_s-epo.fif"
    epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
    # read behav
    behav_fname = data_path / "behav" / f"{subject}_{session_id}.pkl"
    behav = pd.read_pickle(behav_fname).reset_index()
    # get session behav and epoch
    if session_id == 0:
        session = 'prac'
    else:
        session = 'sess-%s' % (str(session_id).zfill(2))
    if lock == 'button': 
        epoch_bsl_fname = data_path / "bsl" / f"{subject}_{session_id}_bl-epo.fif"
        epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
    # read forward solution    
    fwd_fname = res_path / "fwd" / lock / f"{subject}-fwd-{session_id}.fif"
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
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(multi_class="ovr", max_iter=100000, solver='lbfgs', class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True)
    
    pred = np.zeros((len(y), X.shape[-1]))
    pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        pred[test] = np.array(clf.predict(X[test]))
        pred_rock[test] = np.array(clf.predict_proba(X[test]))
                    
    cms, scores = list(), list()
    for itime in range(len(times)):
        cms.append(confusion_matrix(y[:], pred[:, itime], normalize='true', labels=[1, 2, 3, 4]))
        scores.append(roc_auc_score(y[:], pred_rock[:, itime, :], multi_class='ovr'))
    
    cms_arr = np.array(cms)
    np.save(res_dir / "cms.npy", cms_arr)
    scores = np.array(scores)
    np.save(res_dir / "scores.npy", scores)
    
    one_two_similarity = list()
    one_three_similarity = list()
    one_four_similarity = list() 
    two_three_similarity = list()
    two_four_similarity = list()
    three_four_similarity = list()
    for itime in range(len(times)):
        one_two_similarity.append(cms_arr[itime, 0, 1])
        one_three_similarity.append(cms_arr[itime, 0, 2])
        one_four_similarity.append(cms_arr[itime, 0, 3])
        two_three_similarity.append(cms_arr[itime, 1, 2])
        two_four_similarity.append(cms_arr[itime, 1, 3])
        three_four_similarity.append(cms_arr[itime, 2, 3])
                                    
    similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                    two_three_similarity, two_four_similarity, three_four_similarity]
    similarities = np.array(similarities)
    np.save(res_dir / 'rsa.npy', similarities)
    
    del X, y, clf, cv, train, test, pred, pred_rock, cms, cms_arr, scores, similarities
    del one_two_similarity, one_three_similarity, one_four_similarity, two_three_similarity, two_four_similarity, three_four_similarity
    gc.collect()