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

# stim disp= 500 ms
# RSI = 750 ms in task

do_pca = False

trial_types = ['all', 'pattern', 'random']

data_path = '/Volumes/Ultra_Touch/pred_asrt'
# data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
figures = op.join(RESULTS_DIR, 'figures', lock, 'generalizing')
ensure_dir(figures)

mean_score = list()

for trial_type in trial_types:
     
    for subject in subjects[:1]:
        
        all_epochs = list()
        all_behavs = list()
        
        for epoch_num, epo in enumerate(epochs_list):

            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}_{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
            epoch = mne.read_epochs(epoch_fname, verbose="error", preload=False)
            times = epoch.times
            
            # keep only pattern or random trials
            if trial_type == 'pattern':
                behav = behav[behav['trialtypes'] == 1]
                epoch = epoch[np.where(behav['trialtypes']==1)[0]]
            elif trial_type == 'random':
                behav = behav[behav['trialtypes'] == 1]
                epoch = epoch[np.where(behav['trialtypes']==1)[0]]
            else:
                pass
            assert len(epoch) == len(behav)
            
            if do_pca:
                n_component = 30    
                pca = UnsupervisedSpatialFilter(PCA(n_component), average=False)
                pca_data = pca.fit_transform(epoch.get_data())
                sampling_freq = epoch.info['sfreq']
                info = mne.create_info(n_component, ch_types='mag', sfreq=sampling_freq)
                epoch = mne.EpochsArray(pca_data, info=info, events=epoch.events, event_id=epoch.event_id)
            
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
        clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
        # scores = cross_val_multiscore(clf, X, y, cv=KFold(5))
        # mean_score.append(scores.mean(axis=0))

        clf.fit(X, y)
        score = clf.score(X, y)    
        mean_score.append(score)

        # mean_score.append(scores.mean(0))
        res_path_1 = op.join(figures, "big_gen", "noK")
        ensure_dir(res_path_1)
        ymin, ymax = -1, 4
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        im = ax.imshow(
            score,
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.7)
        
        ax.set_xlabel("Testing Time (s)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title("Temporal generalization")
        ax.axvline(0, color="k")
        ax.axhline(0, color="k")
        ax.set_ylim(ymin, ymax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")
        fig.savefig(op.join(res_path_1, "%s.png" % (subject)))
        
    mean_score = np.array(mean_score)
    score_f = mean_score.copy().mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    im = ax.imshow(
        score_f,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.7)

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    ax.set_ylim(ymin, ymax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(op.join(res_path_1, "mean.png"))