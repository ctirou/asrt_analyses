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

data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
folds = 3
chance = 0.5
scoring = 'accuracy'

params = "trialtype"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

all_scores = list()

for subject in subjects[:2]:
    
    res_dir = figures / subject
    ensure_dir(res_dir)

    scores_sub = list()

    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=False, verbose="error") for fname in epo_fnames]
    times = all_epo[0].times

    beh_dir = data_path / "behav"
    beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
    all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]

    for epoch in all_epo:  # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info["dev_head_t"] = all_epo[0].info["dev_head_t"]

    for fname, beh, epo in zip(['practice', 'b1', 'b2', 'b3', 'b4'], all_beh, all_epo):
        
        # per session
        X = epo.get_data()
        y = beh.trialtypes
        assert X.shape[0] == y.shape[0]

        scores = cross_val_multiscore(clf, X, y, cv=cv)
        pval = decod_stats(scores - chance)
        sig = pval < .05
        plt.subplots(1, 1, figsize=(16, 7))
        plt.plot(times, scores.mean(0).T)
        plt.fill_between(times, chance, scores.mean(0).T, alpha=0.2)
        plt.title(f"{fname}.png")
        plt.savefig(res_dir / f"{fname}.png")
        plt.close()
        
        scores_sub.append(scores)
        
    scores_sub = np.array(scores_sub)
    pval = decod_stats(scores_sub.mean(1) - chance)
    sig = pval < .05
    plt.subplots(1, 1, figsize=(16, 7))
    plt.plot(times, scores_sub.mean(axis=(0, 1)).T)
    plt.fill_between(times, chance, scores_sub.mean(axis=(0, 1)).T, alpha=0.2)
    plt.title(f"{subject}.png")
    plt.savefig(figures / f"{subject}.png")
    plt.close()
    
    all_scores.append(scores_sub)

all_scores = np.array(all_scores)
pval = decod_stats(scores_sub - chance)
sig = pval < .05
plt.subplots(1, 1, figsize=(16, 7))
plt.plot(times, scores_sub.mean(0).T)
plt.fill_between(times, chance, all_scores.mean(0).T, alpha=0.2)
plt.title("mean.png")
plt.savefig(figures / "mean.png")
plt.close()
