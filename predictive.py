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
from lazypredict.Supervised import LazyClassifier

# stim disp= 500 ms
# RSI = 750 ms in task

do_pca = True

data_path = '/Volumes/Ultra_Touch/pred_asrt'
# data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
figures = op.join(RESULTS_DIR, 'figures', lock, 'decoding')
ensure_dir(figures)

mean_score = list()

for subject in subjects[:1]:
    
    all_epochs = list()
    all_behavs = list()
    
    for epoch_num, epo in enumerate(epochs_list):

        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}_{epoch_num}.pkl'))
        epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname)
        times = epoch.times
        
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
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    # scores = cross_val_multiscore(clf, X, y, cv=KFold(10))
    clf.fit(X, y)
    score = clf.score(X, y)    
    
    # mean_score.append(score)
    # mean_score.append(scores.mean(axis=0))

    
    # mean_score.append(scores.mean(0))
    
    # fig, ax = plt.subplots(1, 1)
    # im = ax.imshow(
    #     mean_score,
    #     interpolation="lanczos",
    #     origin="lower",
    #     cmap="RdBu_r",
    #     extent=epochs.times[[0, -1, 0, -1]],
    #     vmin=0.0,
    #     vmax=1.0,
    # )
    # ax.set_xlabel("Testing Time (s)")
    # ax.set_ylabel("Training Time (s)")
    # ax.set_title("Temporal generalization")
    # ax.axvline(0, color="k")
    # ax.axhline(0, color="k")
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("accuracy")
    
mean_score = np.array(mean_score)
score = mean_score.copy().mean(axis=0)

fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    score,
    interpolation="gaussian",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]])

ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")