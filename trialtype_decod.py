import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from base import *
from config import *
import pandas as pd

data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
folds = 5
chance = 0.5
threshold = 0.05
scoring = 'accuracy'

params = "trialtype"
figures = RESULTS_DIR / 'figures' / lock / 'decoding' / params
ensure_dir(figures)

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000))
clf = SlidingEstimator(clf, n_jobs=-1, scoring=scoring, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

all_scores = list()

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
        
    beh = pd.concat(all_beh)
    epochs = mne.concatenate_epochs(all_epo)
    
    X = epochs.get_data()
    y = beh.trialtypes
        
    scores = cross_val_multiscore(clf, X, y, cv=cv)
    all_scores.append(scores)
    
    plt.subplots(1, 1, figsize=(16, 7))
    plt.plot(times, scores.mean(0).T)
    plt.title(f"{subject}")
    plt.savefig(figures / f"{subject}.png")

scores = all_scores.copy()
scores = np.array(scores).mean(axis=(0,1)).T
pval = decod_stats(scores - chance)
sig = pval < threshold

plt.subplots(1, 1, figsize=(16, 7))
plt.plot(times, scores)
plt.fill_between(times, chance, scores, where=sig)
plt.title('mean')
plt.savefig(figures, 'mean.png')
plt.close()