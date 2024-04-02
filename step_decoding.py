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

trial_types = ["all", "pattern", "random"]
data_path = DATA_DIR
lock = "stim"
subject = "sub01"
subjects = SUBJS
folds = 3
chance = 0.25
scoring = "accuracy"

params = "step_decoding"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

epo_dir = data_path / lock
epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
all_epo = [mne.read_epochs(fname, preload=False, verbose="error") for fname in epo_fnames]
times = all_epo[0].times

beh_dir = data_path / "behav"
beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]

for epoch in all_epo:  # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
    epoch.info["dev_head_t"] = all_epo[0].info["dev_head_t"]

beh = pd.concat(all_beh)
epochs = mne.concatenate_epochs(all_epo)

X_0 = all_epo[0].get_data()
y_0 = np.array(all_beh[0].positions)

# scores_0 = list()
# accuracy = list()
# for train_0, test_0 in cv.split(X_0, y_0):
#     clf.fit(X_0[train_0], y_0[train_0])
#     scores_0.append(np.array(clf.score(X_0[test_0], y_0[test_0])))
#     y_true = y_0[test_0]
#     y_pred = clf.predict(X_0[test_0])
#     for t in range(len(times)):
#         accuracy.append(np.array(accuracy_score(y_true, y_pred[:, t])))
# accuracy = np.array(accuracy)

# confuse = list()
# for t in range(len(times)):
#     y_pred = clf.predict(X_0[test_0])
#     confuse.append(ConfusionMatrixDisplay.from_estimator(clf, y_pred, y_0[test_0]))

pred = list()
# there is only randoms in practice sessions
for train_0, test_0 in cv.split(X_0, y_0):
    clf.fit(X_0[train_0], y_0[train_0])
    pred.append(np.array(clf.predict_proba(X_0[test_0])))
pred = np.array(pred)

plt.plot(times, pred)