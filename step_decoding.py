import os
import os.path as op
import numpy as np
import mne
from mne.decoding import CSP
from base import *
from mne.decoding import cross_val_multiscore, SlidingEstimator, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    RidgeCV,
    Ridge,
    LogisticRegressionCV,
    LogisticRegression,
)
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

params = "train_on_random"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

all_scores = list()

for subject in subjects:

    epo_dir = data_path / lock
    epo_fnames = [
        epo_dir / f"{f}"
        for f in sorted(os.listdir(epo_dir))
        if ".fif" in f and subject in f
    ]
    all_epo = [
        mne.read_epochs(fname, preload=False, verbose="error") for fname in epo_fnames
    ]
    times = all_epo[0].times

    beh_dir = data_path / "behav"
    beh_fnames = [
        beh_dir / f"{f}"
        for f in sorted(os.listdir(beh_dir))
        if ".pkl" in f and subject in f
    ]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]

    for (
        epoch
    ) in (
        all_epo
    ):  # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info["dev_head_t"] = all_epo[0].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epo)
    behav_df = pd.concat(all_beh)

    # set-up the classifier and cv structure
    clf = make_pipeline(
        StandardScaler(), LogisticRegressionCV(max_iter=10000)
    )
    clf = SlidingEstimator(clf, n_jobs=-1, scoring="accuracy", verbose=True)
    cv = StratifiedKFold(10, shuffle=True)

    # train on practice data
    X = mne.read_epochs(epo_fnames[0], preload=True, verbose="error").get_data()
    y = all_beh[0].positions
    assert X.shape[0] == y.shape[0]

    # scores_p, scores_p1, scores_p2, scores_p3, scores_p4 = list(), list(), list(), list(), list()
    # for train, test in cv.split(X, y):
    #     clf.fit(X[train], y[train])
    #     scores_p.append(clf.score(X[test], y[test]))
    # scores_p = np.array(scores_p)
    # plt.plot(times, scores_p.mean(0))

    scores = cross_val_multiscore(clf, X, y, cv=cv)
    plt.subplots(1, 1, figsize=(16, 7))
    plt.plot(times, scores.mean(0))
    plt.axhline(y=0.25, ls="dashed", color="k")
    plt.axvline(x=0, ls="dashed", color="k")
    plt.title(f'{subject}')
    plt.savefig(op.join(figures, '%s.png' % subject))
    plt.close()
    
    all_scores.append(scores)

all_scores = np.array(all_scores)
plt.subplots(1, 1, figsize=(16, 7))
plt.plot(times, all_scores.mean((0, 1)))
plt.axhline(y=0.25, ls="dashed", color="k")
plt.axvline(x=0, ls="dashed", color="k")
plt.title('mean')
plt.savefig(op.join(figures, 'mean.png'))
plt.close()
