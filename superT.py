import mne
import os.path as op
import os
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore, CSP, GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from config import *

trial_types = ['all', 'pattern', 'random']
data_path = DATA_DIR
lock = 'stim'
subject = 'sub01'

epo_dir = data_path / lock
epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
all_epo = [mne.read_epochs(fname, preload=False, verbose="error") for fname in epo_fnames]
times = all_epo[0].times

beh_dir = data_path / 'behav'
beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
all_beh = [pd.read_pickle(fname) for fname in beh_fnames]

for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
    epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']

epochs = mne.concatenate_epochs(all_epo)
beh_df = pd.concat(all_beh)

one = beh_df[beh_df.positions == 1].index.to_list()
two = beh_df[beh_df.positions == 2].index.to_list()
three = beh_df[beh_df.positions == 3].index.to_list()
four = beh_df[beh_df.positions == 4].index.to_list()

fold = 10
averaging_percent = 20
resampling = 1
conditions = [cond for cond in range(1, 5)]
evoked = []

for cond, num in zip(conditions, [one, two, three, four]):
    
    # Define number of trials to average
    cond_idx = [idx for idx, _ in enumerate(epochs.events[:, 2])]
    trials_to_ave = int(len(cond_idx) * (averaging_percent / 100.0))
    
    # Randomly choose trials to average
    new_beh = beh_df.copy().reset_index()
    new_beh.drop('index', axis=1, inplace=True)
    selected_idx = np.random.choice(cond_idx, trials_to_ave, replace=False)
    selected_epo = epochs[selected_idx]
    selected_beh = new_beh.loc[selected_idx]
    assert len(selected_epo) == len(selected_beh)
    
    new_epochs = epochs.copy()
    new_epochs.drop(indices=selected_idx)
    new_beh.drop(selected_idx, inplace=True)
    assert len(new_epochs) == len(new_beh)
    
    # Resampling (here rate = 1)
    evoked = selected_epo.average()

clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=1000))
clf = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=True)

X, y = new_epochs.get_data(), new_beh.positions
scores = cross_val_multiscore(clf, X, y, cv=KFold(fold, shuffle=True))
plt.plot(times, scores.mean(0))