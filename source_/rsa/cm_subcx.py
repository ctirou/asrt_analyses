import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from base import ensure_dir, get_labels_from_vol_src, get_volume_estimate_tc
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc

# params
subject = SUBJS[0]

analysis = "pred_decoding"
lock = "button" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 10
solver = 'lbfgs'
scoring = "roc_auc"
parc='aparc'
hemi = 'both'
verbose = "error"
jobs = 10

subjects = SUBJS
# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

# get label index
subcx_labels = VOLUME_LABELS

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, multi_class="ovr", max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
cv = StratifiedKFold(folds, shuffle=True)


for subject in subjects:
    # read source space file
    src_fname = op.join(res_path, "src", "%s-src.fif" % (subject))
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    # create mixed source space
    vol_src_fname = op.join(res_path, "src", "%s-vol-src.fif" % (subject))
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    mixed_src = src + vol_src
    # path to bem file
    bem_fname = op.join(res_path, "bem", "%s-bem-sol.fif" % (subject))
    # offsets    
    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])
    labels = get_labels_from_vol_src(vol_src, subject, subjects_dir)

    for session_id, session in enumerate(sessions):
        # read stim epoch
        epoch_fname = data_path / lock / f"{subject}-{session_id}-epo.fif"
        epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
        # read behav
        behav_fname = data_path / "behav" / f"{subject}-{session_id}.pkl"
        behav = pd.read_pickle(behav_fname).reset_index()
        if lock == 'button': 
            epoch_bsl_fname = data_path / "bsl" / f"{subject}-{session_id}-epo.fif"
            epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
        # path to trans file
        trans_fname = op.join(res_path, "trans", lock, "%s-%i-trans.fif" % (subject, session_id))
        # compute forward solution
        fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                    src=mixed_src, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
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
            
        label_tc = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
        
        for ilabel, label in enumerate(labels):
            print(f"{str(ilabel+1).zfill(2)}/{len(labels)}", subject, lock, session, label.name)
            # results dir
            res_dir = res_path / analysis / 'source' / lock / trial_type / label.name / subject / session
            ensure_dir(res_dir)
            
            if not op.exists(res_dir / "rsa.npy"):
                    
                stcs_data = label_tc[label.name]
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
                
                del X, y, train, test, pred, pred_rock, cms, cms_arr, scores, similarities, stcs_data
                del one_two_similarity, one_three_similarity, one_four_similarity, two_three_similarity, two_four_similarity, three_four_similarity
                gc.collect()
        
        del epoch, epoch_fname, behav_fname, fwd, data_cov, noise_cov, rank, info, filters, stcs
        gc.collect()
