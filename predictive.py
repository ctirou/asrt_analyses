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


do_pca = False

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
figures = op.join(RESULTS_DIR, 'figures', lock, 'decoding')
ensure_dir(figures)

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

        # positions = behav['positions']
        # cv = StratifiedKFold(5, shuffle=True)
        # clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
        # clf = SlidingEstimator(clf, scoring='accuracy', n_jobs=2)
        # X = epoch.get_data()
        # y = positions
        # scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=False)
        # plt.plot(epoch.times, scores.mean(0))
        # plt.show()
        # plt.close()
    
    for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    
    epochs = mne.concatenate_epochs(all_epochs)
    behav_df = pd.concat(all_behavs)
            
    y = np.array(behav_df['positions'])
    X = epochs.get_data()
    
    # 0 ---------- Lazy Classifier
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=None)
    # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    # models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    # 1 ---------- Test clasic sliding estimators
    # Define the type of decoder
    clf = make_pipeline(LinearDiscriminantAnalysis()) # essayer avec non lineaire
    # Decod the stim (left, right, top, bottom) with SlidingEstimator
    # slide = SlidingEstimator(clf, scoring='accuracy', n_jobs=4)
    clf = GeneralizingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
    clf.fit(X, y)
    scores = clf.score(X, y)    
    
    # scores = cross_val_multiscore(slide, X, y, cv=KFold(10))
    # mean_score = scores.mean(0)
    # plt.plot(times, mean_score)
    
