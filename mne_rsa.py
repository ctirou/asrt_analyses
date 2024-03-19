import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import mne
from mne.io import concatenate_raws, read_raw_fif
import pandas as pd
from base import *
from config import *
import os

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
behav_df = pd.concat(all_beh)

clf = make_pipeline(
    StandardScaler(), LogisticRegression(C=1, solver="liblinear", multi_class="auto"))
pattern = behav_df.trialtypes == 1
y = np.array(behav_df.positions[pattern])
X = epochs.get_data()[pattern, :, :].mean(axis=2)

classes = set(y)
cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

# Compute confusion matrix for each cross-validation fold
y_pred = np.zeros((len(y), len(classes)))
for train, test in cv.split(X, y):
    # Fit
    clf.fit(X[train], y[train])
    # Probabilistic prediction (necessary for ROC-AUC scoring metric)
    y_pred[test] = clf.predict_proba(X[test])

confusion = np.zeros((len(classes), len(classes)))
for ii, train_class in enumerate(classes):
    for jj in range(ii, len(classes)):
        confusion[ii, jj] = roc_auc_score(y == train_class, y_pred[:, jj])
        confusion[jj, ii] = confusion[ii, jj]

mds = MDS(4, random_state=0, dissimilarity="euclidean")
chance = 0.25
summary = mds.fit_transform(chance - confusion)

plt.matshow(summary)
plt.colorbar()