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
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_1samp

trial_types = ["all", "pattern", "random"]
trial_type = "pattern"
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
folds = 10
chance = 0.5
threshold = 0.05
scoring = "accuracy"
verbose = "error"
sessions = EPOCHS
params = "pred_decoding"
jobs = 15

figures = RESULTS_DIR / 'figures' / lock / params / 'sensors' / trial_type
ensure_dir(figures)

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

    # get sequence
    raw_beh_dir = RAW_DATA_DIR / subject / 'behav_data'
    sequence = get_sequence(raw_beh_dir)
    
    all_session_cms, all_session_scores = [], []

    for i, epo_fname in zip(range(5), ['practice', 'b1', 'b2', 'b3', 'b4']): 
    
        X = all_epo[i].get_data()
        y = np.array(all_beh[i].positions)
        assert X.shape[0] == y.shape[0]

        # set-up the classifier and cv structure
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
        clf = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=verbose)
        cv = StratifiedKFold(folds, shuffle=True)
        
        pred = np.zeros((len(y), X.shape[-1]))
        pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
        # there is only randoms in practice sessions
        for train, test in cv.split(X, y):
            clf.fit(X[train], y[train])
            pred[test] = np.array(clf.predict(X[test]))
            pred_rock[test] = np.array(clf.predict_proba(X[test]))
            
        cms, scores = list(), list()
        for t in range(X.shape[-1]):
            cms.append(confusion_matrix(y[test], pred[test, t], normalize='true', labels=clf.classes_))
            scores.append(roc_auc_score(y[test], pred_rock[test, t, :], multi_class='ovr'))
            
        scores = np.array(scores)
        all_session_scores.append(scores)

        c = np.array(cms).T
        all_session_cms.append(c)
        
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()
        
        for itime in range(len(times)):
            one_two_similarity.append(c[0, 1, itime])
            one_three_similarity.append(c[0, 2, itime])
            one_four_similarity.append(c[0, 3, itime])
            two_three_similarity.append(c[1, 2, itime])
            two_four_similarity.append(c[1, 3, itime])
            three_four_similarity.append(c[2, 3, itime])

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

    similarities = [one_two_similarities, one_three_similarities, one_four_similarities, 
                    two_three_similarities, two_four_similarities, three_four_similarities]

    in_seq, out_seq = get_inout_seq(sequence, similarities)
    all_in_seqs.append(in_seq)
    all_out_seqs.append(out_seq)
    
    all_session_cms = np.array(all_session_cms)
    all_session_scores = np.array(all_session_scores)
    
    fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7))
    fig.suptitle(f'{subject}')
    for i, (ax, session) in enumerate(zip(axs.flat[:5], sessions)):
        ax.plot(times, all_session_scores[i])
        ax.set_title(session)
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
        ax.axhline(chance, color='black', ls='dashed', alpha=.5)
        ax.set_ylim(0, 1)
        ax.grid(True)
    
    for i, ax in zip(range(5), axs.flat[5:]):
        # cax = ax.imshow(all_session_cms[i].mean(-1), cmap='viridis')
        # ax.set_xticks(np.arange(len(set(y))), labels=set(y))
        # ax.set_yticks(np.arange(len(set(y))), labels=set(y))
        # cax.set_clim(0, 1)
        # for i in range(len(set(y))):
        #     for j in range(len(set(y))):
        #         text = ax.text(j, i, round(c[i, j, :].mean(-1), 2),
        #                        ha='center', va='center', color='w')
        # ax.set_ylabel("True label")
        # ax.set_xlabel("Predicted label")
        
        disp = ConfusionMatrixDisplay(all_session_cms[i, :, :, 40:80].mean(-1), display_labels=set(y))
        disp.plot(ax=ax)
        disp.im_.set_clim(0, 1)  # Set colorbar limits

    plt.savefig(figures / f"{subject}-cm.png")
    plt.close()

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
