import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict

# params
trial_types = ["all", "pattern", "random"]
trial_type = 'pattern'
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 10
chance = 0.25
threshold = 0.05
scoring = "accuracy"
hemi = 'both'
params = "pred_decoding"
verbose = "error"
jobs = -1
# figures dir
figures = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type
ensure_dir(figures)
# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs
# get len label names
labels = mne.read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]
del labels
combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

for subject in subjects:
    # to store dissimilarity distances
    pred_decod_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # read epochs
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose=verbose) for fname in epo_fnames]
    # read behav files
    beh_dir = data_path / "behav"
    beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
    all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]
    # get labels
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    
    for session_id, session in enumerate(sessions):
        # get session behav and epoch
        if session_id == 0:
            session = 'prac'
        else:
            session = 'sess-%s' % (str(session_id).zfill(2))
        behav = all_beh[session_id]
        epoch = all_epo[session_id]
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
        # compute source estimate
        filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                        pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        
        for ilabel, label in enumerate(labels):
            print(f"{ilabel+1}/{len(labels)}", subject, session, label.name)
            # set-up the classifier and cv structure
            clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
            clf = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=verbose)
            cv = StratifiedKFold(folds, shuffle=True)
            # get stcs in label
            stcs_data = list()
            for stc in stcs:
                stcs_data.append(stc.in_label(label).data)
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
            
            pred = np.zeros((len(y), X.shape[-1]))
            # there is only randoms in practice sessions
            for train, test in cv.split(X, y):
                clf.fit(X[train], y[train])
                pred[test] = np.array(clf.predict(X[test]))
            cms = list()
            for t in range(X.shape[-1]):
                cms.append(confusion_matrix(y, pred[:, t], normalize='true', labels=clf.classes_))
            
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list()
            three_four_similarity = list()
            
            c = np.array(cms)
            for itime in range(len(times)):
                one_two_similarity.append(c[itime, 0, 1])
                one_three_similarity.append(c[itime, 0, 2])
                one_four_similarity.append(c[itime, 0, 3])
                two_three_similarity.append(c[itime, 1, 2])
                two_four_similarity.append(c[itime, 1, 3])
                three_four_similarity.append(c[itime, 2, 3])
                                
            similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                            two_three_similarity, two_four_similarity, three_four_similarity]
            
            for combi, similarity in zip(combinations, similarities):
                pred_decod_dict[label.name][session_id][combi].append(similarity)

    ##### Pred decoding dataframe #####
    time_points = range(len(times))
    index = pd.MultiIndex.from_product([label_names, range(len(sessions)), combinations], names=['label', 'session', 'similarities'])
    pred_df = pd.DataFrame(index=index, columns=time_points)
    for label in labels:
        for session_id in range(len(sessions)):
            for isim, similarity in enumerate(combinations):
                scores_list = pred_decod_dict[label.name][session_id][similarity]
                if scores_list:
                    scores = np.mean(scores_list, axis=0)
                    pred_df.loc[(label.name, session_id, similarity), :] = scores.flatten()
    pred_df.to_csv(figures / f"{subject}_pred.csv", sep="\t")
    pred_df.to_hdf(figures / f"{subject}_pred.h5", key='pred', mode='w')