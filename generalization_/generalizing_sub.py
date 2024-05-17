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
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from mne.decoding import UnsupervisedSpatialFilter
from base import ensure_dir
from config import *

# stim disp = 500 ms
# RSI = 750 ms in task

do_pca = False

trial_types = ['all', 'random', 'pattern']

data_path = '/Volumes/Ultra_Touch/pred_asrt'
# data_path = '/Users/coum/Desktop/pred_asrt'
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
figures = op.join(RESULTS_DIR, 'figures', lock, 'generalizing')
ensure_dir(figures)

mean_score = list()

for trial_type in trial_types[-1:]:
     
    for subject in subjects[:1]:
        
        all_epochs = list()
        all_behavs = list()
        
        for epoch_num, epo in enumerate(epochs_list):

            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}_{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
            epoch_gen = mne.read_epochs(epoch_fname, verbose="error", preload=False)
            times = epoch_gen.times
                        
            # keep only pattern or random trials
            if trial_type == 'pattern':
                behav = behav[behav['trialtypes'] == 1]
                epoch = epoch_gen[np.where(behav['trialtypes']==1)[0]]
            elif trial_type == 'random':
                behav = behav[behav['trialtypes'] == 2]
                epoch = epoch_gen[np.where(behav['trialtypes']==2)[0]]
            else:
                epoch = epoch_gen.copy()
            assert len(epoch) == len(behav)
                        
            all_epochs.append(epoch)
            all_behavs.append(behav)
        
        for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
            epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
        
        epochs = mne.concatenate_epochs(all_epochs)
        behav_df = pd.concat(all_behavs)
                
        y = np.array(behav_df['positions'])
        X = epochs.get_data()
            
        # 1 ---------- Test clasic sliding estimators
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
        clf = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=True)
        scores = cross_val_multiscore(clf, X, y, cv=KFold(5))
        # mean_score.append(scores.mean(axis=0))

        # clf.fit(X, y)
        # score = clf.score(X, y)    
        # mean_score.append(score)

        # mean_score.append(scores.mean(0))
        res_path = op.join(figures, "big_gen", trial_type, "K10")
        ensure_dir(res_path)
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(
            scores.mean(axis=0),
            # score,
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.5)
        
        ax.set_xlabel("Testing Time (s)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title("Temporal generalization")
        ax.axvline(0, color="k")
        ax.axhline(0, color="k")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")
        fig.savefig(op.join(res_path, "%s.png" % (subject)))
        
    mean_score = np.array(mean_score)
    score_f = mean_score.copy().mean(axis=0)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        score_f,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5)

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(op.join(res_path, "mean.png"))