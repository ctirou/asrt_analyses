import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import ttest_1samp
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from base import *
from config import *
from mne.decoding import SlidingEstimator, cross_val_multiscore, GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
import random

lock = 'stim'
data_path = DATA_DIR
figures = RESULTS_DIR / 'figures' / lock / 'similarity'

subjects = SUBJS
epochs_list = EPOCHS

fold = 5

for subject in subjects[:1]:
    # Read the behav file and get the sequence 
    behav_dir = RAW_DATA_DIR / f'{subject}/behav_data/'
    sequence = get_sequence(behav_dir)
        
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
    
    pattern = beh_df.trialtypes == 1
    
    df_pat = beh_df[beh_df.trialtypes == 1]
    df_rdm = beh_df[beh_df.trialtypes == 2]

    one = df_rdm[df_rdm.positions == 1].index.to_list()
    two = df_rdm[df_rdm.positions == 2].index.to_list()
    three = df_rdm[df_rdm.positions == 3].index.to_list()
    four = df_rdm[df_rdm.positions == 4].index.to_list()

    pat_events = df_pat.positions.to_list()
    
    reord_idx = list()    
    for eve in pat_events:
        if eve == 1:
            reord_idx.append(random.choice(one))
        elif eve == 2:
            reord_idx.append(random.choice(two))
        elif eve == 3:
            reord_idx.append(random.choice(three))
        else:
            reord_idx.append(random.choice(four))

    reord_epos = list()
    for idx in reord_idx:
        reord_epos.append(epochs[idx])
    reord_epo = mne.concatenate_epochs(reord_epos)
    
    # Decoding
    X = reord_epo.get_data()
    y = np.array(pat_events)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=True)
    scores = cross_val_multiscore(clf, X, y, cv=KFold(fold))
    plt.plot(times, scores.mean(0))
    plt.title("pattern reordered")