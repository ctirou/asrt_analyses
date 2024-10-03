import mne
import os
import os.path as op
import numpy as np
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from base import ensure_dir
from config import *
import gc
import sys

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = PRED_PATH
analysis = 'time_generalization'
subjects, epochs_list, subjects_dir = SUBJS, EPOCHS, FREESURFER_DIR
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
jobs = 10
verbose = True
res_path = data_path / 'results' / 'source'
ensure_dir(res_path)


subject = SUBJS[0]

for epoch_num, epo in zip([1, 2, 3, 4], epochs_list[1:]):

    # read epoch
    epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
    epoch_filt = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
    evk_filt_ave = epoch_filt.average()
    evk_spectrum = evk_filt_ave.compute_psd()
    evk_spectrum.plot(picks="meg", exclude="bads", amplitude=False)
    
    epo_ps = epoch_filt.compute_psd(method='multitaper', tmin=0, tmax=0.6, picks='meg')
    epo_ps.plot()
        
    epoch_fname = data_path / 'no_filter' / lock / f"{subject}-{epoch_num}-epo.fif"
    epoch_no_filt = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
    evk = epoch_no_filt.average()
    evk_spectrum = evk.compute_psd()
    evk_spectrum.plot(picks="meg", exclude="bads", amplitude=False)
    
    epo_ps = epoch_no_filt.compute_psd(method='multitaper', tmin=0, tmax=0.6, picks='meg')
    epo_ps.plot()
    
events_stim = None    

interval = list()
for i, j in enumerate(events_stim[:, 0]):
    if i == len(events_stim) - 1:
        break
    interval.append(events_stim[i+1, 0] - j)

interval = np.array(interval)
filt = interval < 6000
interval = interval[filt]

import matplotlib.pyplot as plt
plt.plot(interval)
# plt.scatter(np.newaxis, interval)
plt.show()