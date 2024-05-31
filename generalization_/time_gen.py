import mne
import os.path as op
import os
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore, CSP, GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
import pandas as pd
from base import ensure_dir
from config import *
import gc

# stim disp = 500 ms
# RSI = 750 ms in task

analysis = 'time_generalization'
data_path = PRED_PATH
res_path = RESULTS_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
trial_type = 'random'

folds = 10
solver = 'lbfgs'
scoring = "accuracy"
parc='aparc'
hemi = 'both'
verbose = True
jobs = -1

res_dir = res_path / analysis / 'sensors' / lock / trial_type
ensure_dir(res_dir)

# 1 ---------- Test clasic sliding estimators
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
cv = StratifiedKFold(folds, shuffle=True)

for subject in subjects:
    
    all_epochs = list()
    all_behavs = list()
    
    print(subject)
    
    for epoch_num, epo in enumerate(epochs_list):

        behav = pd.read_pickle(op.join(data_path / 'behav' / f'{subject}-{epoch_num}.pkl'))
        epoch_fname = op.join(data_path / lock / f"{subject}-{epoch_num}-epo.fif")
        epoch_gen = mne.read_epochs(epoch_fname, verbose="error", preload=False)
        times = epoch_gen.times
                    
        all_epochs.append(epoch_gen)
        all_behavs.append(behav)
    
    for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    
    epochs = mne.concatenate_epochs(all_epochs)
    behav_df = pd.concat(all_behavs)
            
    meg_data = epochs.get_data()    
    behav_data = behav_df.reset_index(drop=True)
    
    if trial_type == 'pattern':
        pattern = behav_data.trialtypes == 1
        X = meg_data[pattern]
        y = behav_data.positions[pattern]
    elif trial_type == 'random':
        random = behav_data.trialtypes == 2
        X = meg_data[random]
        y = behav_data.positions[random]
    else:
        X = meg_data
        y = behav_data.positions    
    y = y.reset_index(drop=True)            
    assert X.shape[0] == y.shape[0]
    
    del all_epochs, all_behavs, behav, epoch_fname, epoch_gen, epochs, behav_df, meg_data, behav_data
    gc.collect()
    
    scores = cross_val_multiscore(clf, X, y, cv=cv)
    np.save(res_dir / f"{subject}_scores.npy", scores.mean(0))
    
    del X, y, scores
    gc.collect()

#     # mean_score.append(scores.mean(0))
#     res_path = op.join(figures, "big_gen", trial_type, "K10")
#     ensure_dir(res_path)
#     fig, ax = plt.subplots(1, 1)
#     im = ax.imshow(
#         scores.mean(axis=0),
#         # score,
#         interpolation="lanczos",
#         origin="lower",
#         cmap="RdBu_r",
#         extent=times[[0, -1, 0, -1]],
#         aspect=0.5)
    
#     ax.set_xlabel("Testing Time (s)")
#     ax.set_ylabel("Training Time (s)")
#     ax.set_title("Temporal generalization")
#     ax.axvline(0, color="k")
#     ax.axhline(0, color="k")
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label("accuracy")
#     fig.savefig(op.join(res_path, "%s.png" % (subject)))
    
# mean_score = np.array(mean_score)
# score_f = mean_score.copy().mean(axis=0)

# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(
#     score_f,
#     interpolation="lanczos",
#     origin="lower",
#     cmap="RdBu_r",
#     extent=times[[0, -1, 0, -1]],
#     aspect=0.5)

# ax.set_xlabel("Testing Time (s)")
# ax.set_ylabel("Training Time (s)")
# ax.set_title("Temporal generalization")
# ax.axvline(0, color="k")
# ax.axhline(0, color="k")
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("accuracy")
# fig.savefig(op.join(res_path, "mean.png"))