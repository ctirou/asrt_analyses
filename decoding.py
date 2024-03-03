import mne
import os.path as op
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore, CSP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
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

for subject in subjects:
    
    # print(subject)
    
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
    
    # 1 ---------- Test classic sliding estimators
    res_path = op.join(figures, 'classic_stim', 'NN')
    ensure_dir(res_path)
    # Define the type of decoder
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                  'logisticregression__penalty': ['l1', 'l2', 'elasticnet', 'None'],
                  'logisticregression__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) # LR > LDA, lSVC, rbf-SVC, Perceptron, RF, GaussianNB, KNN
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    # Use the best estimator for further predictions or analysis
    best_model = grid_search.best_estimator_

    # Decod the stim (left, right, top, bottom) with SlidingEstimator
    slide = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1)
    scores = cross_val_multiscore(slide, X, y, cv=KFold(3))
    mean_score = scores.mean(0)
    print(max(mean_score))
    plt.plot(times, mean_score)
    plt.title("%s" % max(mean_score))
    plt.savefig(op.join(res_path, "%s" % subject))
    plt.close()
    
    # 2 ---------- Test decoding during sliding windows with PCA + flattened
    res_path = op.join(figures, 'slide_stim', 'LR')
    ensure_dir(res_path)
    window_length = 0.1  # time window in s
    spacing = 0.05  # sliding period in s
    X = epochs.pick_types(meg=True, stim=False, ref_meg=False)._data
    pca = UnsupervisedSpatialFilter(PCA(100), average=False)
    X_pca = pca.fit_transform(X)
    times = list()
    scores = list()
    best_scores = list()
    best_params = list()
    for time in np.arange(epochs.tmin, epochs.tmax - window_length, spacing):
        tt = np.where((epochs.times >= time) & (epochs.times < time + window_length))[0]
        xx = X_pca[:, :, tt]
        xx = xx.reshape(xx.shape[0], xx.shape[1]*xx.shape[2])
        score = cross_val_multiscore(clf, xx, y, cv=KFold(3)).mean()
        times.append(time + window_length/2.)
        scores.append(score)
    plt.plot(times, scores)
    plt.title(max(scores))
    plt.savefig(op.join(res_path, "%s" % subject))
    plt.close()


    clf = make_pipeline(CSP(n_components=20, reg=None, log=True, norm_trace=False),
                        LogisticRegression())
    for time in np.arange(epochs.tmin, epochs.tmax - window_length, spacing):
        tt = np.where((epochs.times >= time) & (epochs.times < time + window_length))[0]
        xx = X[:, :, tt]
        score = cross_val_multiscore(clf, xx, y, cv=KFold(3)).mean()
        times.append(time + window_length/2.)
        scores.append(score)
    scores3 = np.mean(scores, axis=0)

