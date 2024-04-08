import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from jr.gat import scorer_spearman
from sklearn.metrics import make_scorer
from base import *
from config import *
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_1samp, zscore
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict
from tqdm.auto import tqdm

# params
trial_types = ["all", "pattern", "random"]
trial_type = 'pattern'
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 2
chance = 0.25
threshold = 0.05
scoring = "accuracy"
hemi = 'lh'
params = "step_decoding"
verbose = True
# figures dir
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params / 'source'
ensure_dir(figures)
# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)
# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs
# get len label names
labels = mne.read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]
del labels
# to store cross-val multiscore
scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
# to store dissimilarity distances
rsa_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

for subject in subjects[:1]:
    # read epochs
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose=verbose) for fname in epo_fnames]
    # read behav files
    beh_dir = data_path / "behav"
    beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
    all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]
    # get bem and src files
    bem_fname = RESULTS_DIR / "bem" / f"{subject}-bem.fif"
    src_fname = RESULTS_DIR / "src" / f"{subject}-src.fif"
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    # get labels
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    # get subject's repeated sequence
    raw_beh_dir = RAW_DATA_DIR / subject / 'behav_data'
    sequence = get_sequence(raw_beh_dir)
    
    for trial_type in trial_types[:1]:
        
        ensure_dir(figures / trial_type)
        all_in_seqs, all_out_seqs = [], []
        true_pred_means, false_pred_means = [], []

        for session_id, epo_fname in enumerate(sessions[:1]):
            # get session behav and epoch
            if session_id == 0:
                epo_fname = 'prac'
            else:
                epo_fname = 'sess-%s' % (str(session_id).zfill(2))
            behav = all_beh[session_id]
            epoch = all_epo[session_id]
            if lock == 'button': 
                epoch_bsl_fname = data_path / "bsl" / f"{subject}_{session_id}_bl-epo.fif"
                epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
            # make forward solution    
            trans_fname = res_path / "trans" / lock / f"{subject}-trans-{session_id}.fif"
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=src, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=1,
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
            # compute source estimate
            filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            
            for ilabel, label in enumerate(labels[:1]):
                print(subject, trial_type, epo_fname, label.name)            
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
                assert X.shape[0] == y.shape[0]
                
                # cross-val multiscore decoding
                scores = cross_val_multiscore(clf, X, y, cv=cv)
                scores_dict[label.name][trial_type][session_id].append(scores.mean(0).flatten())
                
                # prepare design matrix
                ntrials = len(X)
                nconditions = len(set(y))
                design_matrix = np.zeros((ntrials, nconditions))
                
                for icondi, condi in enumerate(y):
                    design_matrix[icondi, condi-1] = 1
                assert np.sum(design_matrix.sum(axis=1) == 1) == len(X)
                
                data = X.copy()
                _, verticies, ntimes = data.shape       
                data = zscore(data, axis=0)
                
                coefs = np.zeros((nconditions, verticies, ntimes))
                resids = np.zeros_like(data)
                for vertex in tqdm(range(verticies)):
                    for itime in range(ntimes):
                        Y = data[:, vertex, itime]                    
                        model = sm.OLS(endog=Y, exog=design_matrix, missing="raise")
                        results = model.fit()
                        coefs[:, vertex, itime] = results.params
                        resids[:, vertex, itime] = results.resid

                rdm_times = np.zeros((nconditions, nconditions, ntimes))
                for itime in tqdm(range(ntimes)):
                    response = coefs[:, :, itime]
                    residuals = resids[:, :, itime]
                    
                    # Estimate covariance from residuals
                    lw_shrinkage = LedoitWolf(assume_centered=True)
                    cov = lw_shrinkage.fit(residuals)
                    
                    # Compute pairwise mahalanobis distances
                    VI = np.linalg.inv(cov.covariance_) # covariance matrix needed for mahalonobis
                    rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
                    assert ~np.isnan(rdm).all()
                    rdm_times[:, :, itime] = rdm

                    rdmx = rdm_times.copy()
                    one_two_similarity = list()
                    one_three_similarity = list()
                    one_four_similarity = list() 
                    two_three_similarity = list()
                    two_four_similarity = list()
                    three_four_similarity = list()

                    for itime in range(rdmx.shape[2]):
                        one_two_similarity.append(rdmx[0, 1, itime])
                        one_three_similarity.append(rdmx[0, 2, itime])
                        one_four_similarity.append(rdmx[0, 3, itime])
                        two_three_similarity.append(rdmx[1, 2, itime])
                        two_four_similarity.append(rdmx[1, 3, itime])
                        three_four_similarity.append(rdmx[2, 3, itime])
                                    
                similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                                two_three_similarity, two_four_similarity, three_four_similarity]
                
                for combi, similarity in zip(combinations, similarities):
                    rsa_dict[label.name][trial_type][session_id][combi].append(similarity)

    ##### Decoding dataframe #####
    time_points = range(len(times))
    index = pd.MultiIndex.from_product([label_names, trial_types, range(5)], names=['label', 'trial_type', 'session'])
    scores_df = pd.DataFrame(index=index, columns=time_points)
    for label in labels:
            for trial_type in trial_types:
                for session_id in range(len(sessions)):
                    scores_list = scores_dict[label.name][trial_type][session_id]
                    if scores_list:
                        average_scores = np.mean(scores_list, axis=0)
                        scores_df.loc[(label.name, trial_type, session_id), :] = average_scores.flatten()
    scores_df.to_csv(figures / f"{subject}_scores.csv", sep="\t")

    ##### RSA dataframe #####
    index = pd.MultiIndex.from_product([label_names, trial_types, range(5), combinations], names=['label', 'trial_type', 'session', 'similarities'])
    rsa_df = pd.DataFrame(index=index, columns=time_points)
    for label in labels:
            for trial_type in trial_types:
                for session_id in range(len(sessions)):
                    for isim, similarity in enumerate(combinations):
                        rsa_list = rsa_dict[label.name][trial_type][session_id][similarity]
                        if rsa_list:
                            rsa_scores = np.mean(rsa_list, axis=0)
                            rsa_df.loc[(label.name, trial_type, session_id, similarity), :] = rsa_scores.flatten()
    rsa_df.to_csv(figures / f"{subject}_rsa.csv", sep="\t")

###### plot decoding scores #######
max_value = scores_df.max().max()
min_value = scores_df.min().min()
sco = list()
for sub in subjects[:2]:
    for i in range(1):
        sco.append(np.array(scores_df.loc[(label_names[0], sub, 'all', i), :]))
        # plt.plot(times, scores_df.loc[(label_names[0], sub, 'all', i), :], label=sub)
sco = np.array(sco)
pval = decod_stats(sco)
sig = pval - threshold
plt.subplots(1, 1, figsize=(10, 5))
plt.plot(times, sco.mean(0).flatten(), label='mean')
plt.fill_between(times, chance, sco.mean(0).flatten(), where=sig)
plt.title(label_names[0])
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.axhline(chance, color='black', ls='dashed', alpha=.5)
plt.ylim(round(min_value, 2)-0.01, round(max_value, 2)+0.01)
plt.legend()
plt.show()