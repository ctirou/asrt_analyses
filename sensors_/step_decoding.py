import os
import os.path as op
import numpy as np
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

trial_types = ["all", "pattern", "random"]
data_path = DATA_DIR
lock = "stim"
subject = "sub01"
subjects = SUBJS
folds = 10
chance = 0.25
threshold = 0.05
scoring = "accuracy"

params = "step_decoding"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

true_pred_means, false_pred_means = [], []
all_in_seqs, all_out_seqs = [], []

for subject in subjects:

    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=False, verbose="error") for fname in epo_fnames]
    times = all_epo[0].times

    beh_dir = data_path / "behav"
    beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
    all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]

    for epoch in all_epo:  # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info["dev_head_t"] = all_epo[0].info["dev_head_t"]

    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()

    beh = pd.concat(all_beh)
    epochs = mne.concatenate_epochs(all_epo)
    
    for i, epo_fname in zip(range(5), ['practice', 'b1', 'b2', 'b3', 'b4']): 
    
        X = all_epo[i].get_data()
        y = np.array(all_beh[i].positions)
        assert X.shape[0] == y.shape[0]

        pred = np.zeros((len(y), X.shape[-1]))
        # there is only randoms in practice sessions
        for train, test in cv.split(X, y):
            clf.fit(X[train], y[train])
            pred[test] = np.array(clf.predict(X[test]))
        cms = list()
        for t in range(X.shape[-1]):
            cms.append(confusion_matrix(y, pred[:, t], normalize='true', labels=clf.classes_))

        # disp = ConfusionMatrixDisplay(confusion_matrix=cms[0], display_labels=clf.classes_)
        # disp.plot()

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
        
        # get sequence
        raw_beh_dir = RAW_DATA_DIR / subject / 'behav_data'
        sequence = get_sequence(raw_beh_dir)
        
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
