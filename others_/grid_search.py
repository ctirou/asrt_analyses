import mne
import os.path as op
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore, CSP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from mne.decoding import UnsupervisedSpatialFilter
from base import ensure_dir
from config import *

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
figures = op.join(RESULTS_DIR, 'figures', lock, 'decoding')
ensure_dir(figures)

# res_path = op.join(figures, 'slide_stim', 'grid+LR')
# ensure_dir(res_path)

gdf_fname = op.join(figures, "grid_search_results")
ensure_dir(gdf_fname)

gen_df = pd.DataFrame()

for subject in subjects[:1]:
        
    all_epochs = list()
    all_behavs = list()
    
    for epoch_num, epo in enumerate(epochs_list):

        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}_{epoch_num}.pkl'))
        epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose="error")
        times = epoch.times
                
        all_epochs.append(epoch)
        all_behavs.append(behav)
    
    for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    
    epochs = mne.concatenate_epochs(all_epochs)
    behav_df = pd.concat(all_behavs)
            
        
    # 1 ---------- Perform grid search on entire dataset ---------------------
        
    X = epochs.pick_types(meg=True, stim=False, ref_meg=False)._data
    y = np.array(behav_df['positions'])

    # no PCA
    # pca = UnsupervisedSpatialFilter(PCA(100), average=False)
    # X_pca = pca.fit_transform(X)
    X_reshaped = X_pca.reshape(X_pca.shape[0], -1)
    
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                  'logisticregression__solver': ['liblinear', 'lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                  'logisticregression__penalty': ['l2', 'l1', 'elasticnet']}

    # try with LRCV
    clf_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)) # LR > LDA, lSVC, rbf-SVC
    grid_search = GridSearchCV(clf_pipeline, param_grid, cv=KFold(5), scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_reshaped, y)
    
    grid_df = pd.DataFrame(grid_search.cv_results_).sort_values(by=["rank_test_score"])
    grid_df.to_csv(op.join(gdf_fname, "%s.csv" % subject ), index=False)
    
    best = grid_df.iloc[0]
    best["subject"] = subject
    gen_df = pd.concat([gen_df, best], ignore_index=False)
    
    # 2 ---------- Use best params for sliding window ---------------------
    
    # window_length = 0.1  # time window in s
    # spacing = 0.05  # sliding period in s
    # times, scores = list(), list()
    
    # best_params = {key.replace('logisticregression__', ''): value for key, value in best_params.items()}
    # clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **best_params)) # LR > LDA, lSVC, rbf-SVC

    # for time in np.arange(epochs.tmin, epochs.tmax - window_length, spacing):
    #     tt = np.where((epochs.times >= time) & (epochs.times < time + window_length))[0]
    #     xx = X_pca[:, :, tt]
    #     xx = xx.reshape(xx.shape[0], xx.shape[1]*xx.shape[2])
    #     score = cross_val_multiscore(clf_pipeline, xx, y, cv=KFold(10)).mean()
    #     times.append(time + window_length/2.)
    #     scores.append(score)
        
    # plt.plot(times, scores)
    # plt.title(max(scores))
    # plt.savefig(op.join(res_path, "%s" % subject))
    # plt.close()

gen_df = gen_df.T
gen_df.to_csv(op.join(gdf_fname, "gen_grid.csv"), index=False)