import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from jr.gat import scorer_spearman
from sklearn.metrics import make_scorer
from base import *
from config import *
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_1samp
from sklearn.covariance import LedoitWolf
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
folds = 2
chance = 0.25
threshold = 0.05
scoring = "accuracy"
hemi = 'lh'
params = "step_decoding"
verbose = "error"
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
# cross-val multiscore results df
index = pd.MultiIndex.from_product([label_names, trial_types], names=['label', 'trial_type'])
columns = range(5)
scores_df = pd.DataFrame(index=index, columns=columns) 

true_pred_means, false_pred_means = [], []

scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

for subject in subjects[:2]:
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
    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()
    
    for trial_type in trial_types[:1]:
        
        ensure_dir(figures / trial_type)
        all_in_seqs, all_out_seqs = [], []
        true_pred_means, false_pred_means = [], []

        for session_id, epo_fname in zip(range(1), sessions[:1]):
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
                scores_dict[label.name][subject][trial_type][session_id].append(scores.mean(0).flatten())
                # scores_df.at[(label.name, trial_type), session_id] = scores.mean(0)
                    
            pred = np.zeros((len(y), X.shape[-1]))
            for train, test in cv.split(X, y):
                clf.fit(X[train], y[train])
                pred[test] = np.array(clf.predict(X[test]))
            cms = list()
            for t in range(X.shape[-1]):
                cms.append(confusion_matrix(y, pred[:, t], normalize='true', labels=clf.classes_))

            true_pred, false_pred = np.zeros((len(times), len(clf.classes_), len(clf.classes_))), np.zeros((len(times), len(clf.classes_), len(clf.classes_)))
            for t in range(len(times)):
                for i in range(len(clf.classes_)):
                    for j in range(len(clf.classes_)):
                        if i == j:
                            if cms[t][i, j] not in true_pred[t]:
                                true_pred[t][i, j] = cms[t][i, j] 
                        else:
                            if cms[t][i, j] not in true_pred[t]:
                                false_pred[t][i, j] = cms[t][i, j]

            true_pred_means.append(np.array([np.mean(true_pred[t][true_pred[t] != 0]) for t in range(len(times))]))
            false_pred_means.append(np.array([np.mean(false_pred[t][false_pred[t] != 0]) for t in range(len(times))]))
                        
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

            one_two_similarity = np.array(one_two_similarity)
            one_three_similarity = np.array(one_three_similarity)
            one_four_similarity = np.array(one_four_similarity) 
            two_three_similarity = np.array(two_three_similarity)
            two_four_similarity = np.array(two_four_similarity) 
            three_four_similarity = np.array(three_four_similarity)
            
            one_two_similarities.append(one_two_similarity)
            one_three_similarities.append(one_three_similarity)
            one_four_similarities.append(one_four_similarity) 
            two_three_similarities.append(two_three_similarity)
            two_four_similarities.append(two_four_similarity) 
            three_four_similarities.append(three_four_similarity)
                                
        one_two_similarities = np.array(one_two_similarities)
        one_three_similarities = np.array(one_three_similarities)  
        one_four_similarities = np.array(one_four_similarities)   
        two_three_similarities = np.array(two_three_similarities)  
        two_four_similarities = np.array(two_four_similarities)   
        three_four_similarities = np.array(three_four_similarities)

        similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]

        in_seq, out_seq = get_inout_seq(sequence, similarities)
        all_in_seqs.append(in_seq)
        all_out_seqs.append(out_seq)

time_points = range(len(times))
index = pd.MultiIndex.from_product([label_names, subjects, trial_types, range(5)], names=['label', 'subject', 'trial_type', 'session'])
ave_scores_df = pd.DataFrame(index=index, columns=time_points)

for label in labels:
    for subject in subjects:
        for trial_type in trial_types:
            for session_id in range(len(sessions)):
                scores_list = scores_dict[label.name][subject][trial_type][session_id]
                if scores_list:
                    average_scores = np.mean(scores_list, axis=0)
                    ave_scores_df.loc[(label.name, subject, trial_type, session_id), :] = average_scores.flatten()

max_value = ave_scores_df.max().max()
min_value = ave_scores_df.min().min()

sco = list()
for sub in subjects[:2]:
    for i in range(1):
        sco.append(np.array(ave_scores_df.loc[(label_names[0], sub, 'all', i), :]))
        # plt.plot(times, ave_scores_df.loc[(label_names[0], sub, 'all', i), :], label=sub)
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

# true vs false predictions
true_pred_means = np.array(true_pred_means).mean(axis=0).T
false_pred_means = np.array(false_pred_means).mean(axis=0).T
mean_pred = true_pred_means - false_pred_means

plt.subplots(1, 1, figsize=(16, 7))
plt.plot(times, true_pred_means, label="true_pred")
plt.plot(times, false_pred_means, label="false_pred")
plt.plot(times, mean_pred, label="diff")
plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
plt.axvline(0, color='grey')
plt.axhline(0.25, color='black', ls='dashed')
plt.legend()
plt.title('mean_true_vs_false_pred')
plt.savefig(figures / 'mean_true_vs_false_pred.png')
plt.close()

# in vs out sequence decoding performance
# all_in_seq = np.array(all_in_seqs).mean(axis=(0, 1)).T
# all_out_seq = np.array(all_out_seqs).mean(axis=(0, 1)).T

all_in_seq = np.array(all_in_seqs)
all_out_seq = np.array(all_out_seqs)

np.save(figures / 'all_in.npy', all_in_seq)
np.save(figures / 'all_out.npy', all_out_seq)

all_in_seq = np.load(figures / 'all_in.npy').mean(axis=(1))
all_out_seq = np.load(figures / 'all_out.npy').mean(axis=(1))

diff_inout = all_in_seq - all_out_seq

for i in range(1, 5):
    plt.subplots(1, 1, figsize=(16, 7))
    d = diff_inout[:, i, :] - diff_inout[:, 0, :]
    pval = decod_stats(d)
    sig = pval < threshold
    plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    plt.plot(times, d.mean(0).T, label=f"block_{i}")
    plt.fill_between(times, chance, d.mean(0).T, where=sig, color='C3', alpha=0.3)
    plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
    plt.axvline(0, color='grey')
    plt.axhline(0, color='black', ls='dashed')
    plt.legend()
    plt.title(f"block_{i}")
    plt.ylim(-0.10, 0.10)
    plt.savefig(figures / f"block_{i}.png")

plt.subplots(1, 1, figsize=(16, 7))
for i in range(1, 5):
    d = diff_inout[:, i, :] - diff_inout[:, 0, :]
    pval = decod_stats(d)
    sig = pval < threshold
    plt.plot(times, d.mean(0).T, label=f"block_{i}")
    plt.fill_between(times, chance, d.mean(0).T, where=sig, color='C3', alpha=0.3)
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
plt.axvline(0, color='grey')
plt.axhline(0, color='black', ls='dashed')
plt.legend()
plt.title("all_blocks")
plt.savefig(figures / "all_blocks.png")

# plot diff: in - out
diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
plt.subplots(1, 1, figsize=(16, 7))
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(diff)
sig = p_values < 0.05
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C2', alpha=0.2)
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C3', alpha=0.3)
plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
plt.axvline(0, color='grey')
plt.axhline(0, color='black', ls='dashed')
plt.legend()
plt.show()
plt.title('mean_in_vs_out_decoding')
plt.savefig(figures / 'mean_in_vs_out_decod.png')
plt.close()