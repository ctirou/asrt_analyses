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
from pathlib import Path

trial_types = ['all', 'pattern', 'random']
data_path = Path('/Volumes/Ultra_Touch/pred_asrt')
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
behav_df = pd.concat(all_beh)

# # decod on all trials     
y = np.array(behav_df['positions'])
X = epochs.get_data()[:, :, 800:1200]
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=1000))
clf = GeneralizingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=True)
scores = cross_val_multiscore(clf, X, y, cv=KFold(5))
plt.plot(times[800:1200], scores.mean(0), label='all')

# decod on pattern trials
pattern = behav_df['trialtypes'] == 1
y = np.array(behav_df['positions'][pattern])
X = epochs.get_data()[pattern, :, 800:1200]
scores = cross_val_multiscore(clf, X, y, cv=KFold(5))
plt.plot(times[800:1200], scores.mean(0), label='pattern')

# # decod on random trials
random = behav_df['trialtypes'] == 2
y = np.array(behav_df['positions'][random])
X = epochs.get_data()[random, :, 800:1200]
scores = cross_val_multiscore(clf, X, y, cv=KFold(5))
plt.plot(times[800:1200], scores.mean(0), label='random')
plt.legend()
