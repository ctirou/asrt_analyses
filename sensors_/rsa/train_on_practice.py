import os
import os.path as op
import numpy as np
import mne
from mne.decoding import CSP
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegressionCV, LogisticRegression
import matplotlib.pyplot as plt
from jr.gat import scorer_spearman
from sklearn.metrics import make_scorer
from base import *
from config import *
import pandas as pd

trial_types = ["all", "pattern", "random"]
data_path = DATA_DIR
lock = "stim"
subject = "sub01"
subjects = SUBJS
folds = 10
chance = 0.25
scoring = 'accuracy'
threshold = 0.05

params = "train_on_practice_filt"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

for trial_type in trial_types[1:]:
    
    ensure_dir(figures / trial_type)
    # practice
    scores_0 = []
    # blocks 
    scores_1 = []
    scores_2 = []
    scores_3 = []
    scores_4 = []

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

        # train on practice data
        if trial_type == 'pattern':
            pattern = all_beh[0].trialtypes == 1
            X_0 = all_epo[0].get_data()[pattern]
            y_0 = np.array(all_beh[0].positions[pattern])
        elif trial_type == 'random':
            random = all_beh[0].trialtypes == 2
            X_0 = all_epo[0].get_data()[random]
            y_0 = np.array(all_beh[0].positions[random])
        else:
            X_0 = all_epo[0].get_data()
            y_0 = all_beh[0].positions
        assert X_0.shape[0] == y_0.shape[0]
        # scores = cross_val_multiscore(clf, X, y, cv=cv)
        # plt.subplots(1, 1, figsize=(16, 7))
        # plt.plot(times, scores.mean(0))
        # plt.axhline(y=0.25, ls="dashed", color="k")
        # plt.axvline(x=0, ls="dashed", color="k")
        # pval = decod_stats(scores - chance)
        # sig = pval < .05
        # plt.fill_between(times, chance, scores.mean(0), where=sig, alpha=1)
        # plt.title(f'{subject}')
        # plt.savefig(op.join(figures, '%s.png' % subject))
        # plt.close()
        # scores_0.append(scores)
        # clf.fit(X, y)
        # pred = list()
        # there is only randoms in practice sessions
        for train_0, test_0 in cv.split(X_0, y_0):
            clf.fit(X_0[train_0], y_0[train_0])
            scores_0.append(np.array(clf.score(X_0[test_0], y_0[test_0])))
            # pred.append(np.array(clf.predict_proba(X_0[test_0])))
        # pred = np.array(pred)
        # pred = pred.mean(axis=(0, 1))
        # plt.plot(times, pred, label=np.arange(1, 5))
        # plt.legend()
        # plt.show()
        
        for i, score_list in zip(range(1, len(all_epo)), [scores_1, scores_2, scores_3, scores_4]):
            X = all_epo[i].get_data()
            y = all_beh[i].positions
            assert len(X) == len(y)
            # all
            if trial_type == 'all':
                for train, test in cv.split(X, y):
                    score_list.append(np.array(clf.score(X[test], y[test])))
            elif trial_type == 'pattern':
                # pattern
                pattern = all_beh[i].trialtypes == 1
                X_pat = X[pattern]
                y_pat = np.array(y[pattern])
                for train, test in cv.split(X_pat, y_pat):
                    score_list.append(np.array(clf.score(X_pat[test], y_pat[test])))
            else:
                # random
                random = all_beh[i].trialtypes == 2
                X_rdm = X[random]
                y_rdm = np.array(y[random])
                for train, test in cv.split(X_rdm, y_rdm):
                    score_list.append(np.array(clf.score(X_rdm[test], y_rdm[test])))

    scores_0 = np.array(scores_0)
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    scores_3 = np.array(scores_3)
    scores_4 = np.array(scores_4)
            
    for score, label in zip([scores_0, scores_1, scores_2, scores_3, scores_4], ['train_prac', 'test_b1', 'test_b2', 'test_b3', 'test_b4']): 
        plt.subplots(1, 1, figsize=(16, 7))
        plt.plot(times, score.mean(0).T, label=label)
        pval = decod_stats(score - chance)
        sig = pval < threshold
        plt.fill_between(times, chance, score.mean(0).T, where=sig, alpha=.3)
        plt.legend()
        plt.title("train on practice")
        plt.axhline(y=chance, ls="dashed", color="k", alpha=.2)
        plt.axvline(x=0, ls="dashed", color="k", alpha=.2)
        plt.savefig(op.join(figures, trial_type, "%s.png" % label))
        plt.close()
        
    plt.subplots(1, 1, figsize=(16, 7))
    plt.plot(times, scores_0.mean(0).T, label='practice')
    plt.plot(times, scores_1.mean(0).T, label='block_1')
    plt.plot(times, scores_2.mean(0).T, label='block_2')
    plt.plot(times, scores_3.mean(0).T, label='block_3')
    plt.plot(times, scores_4.mean(0).T, label='block_4')
    for score in [scores_0, scores_1, scores_2, scores_3, scores_4]:
        pval = decod_stats(score - chance)
        sig = pval < threshold
        plt.fill_between(times, chance, score.mean(0).T, where=sig, alpha=.3)
    plt.legend()
    plt.title("train on practice_combined")
    plt.axhline(y=chance, ls="dashed", color="k", alpha=.2)
    plt.axvline(x=0, ls="dashed", color="k", alpha=.2)
    plt.savefig(op.join(figures, trial_type, "combined.png"))
    plt.close()