import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
from scipy.stats import zscore
import sys
import rsatoolbox
# from rsatoolbox.vis.timecourse import plot_timecourse

lock = 'stim'
overwrite = True
verbose = True

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'
subject = subjects[0]
times = np.load(data_path / "times.npy")

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

# def process_subject(subject):
    
res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "rdm" / subject
ensure_dir(res_path)

# loop across sessions
for epoch_num in [0, 1, 2, 3, 4]:
        
    behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
    behav = pd.read_pickle(behav_fname)
    # read epochs
    epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
    epoch = mne.read_epochs(epoch_fname, verbose=verbose)
    data = epoch.get_data(picks='mag', copy=True)
    
    # Get channel names of specific types
    mag_channels = epoch.pick_types(meg='mag').ch_names
    
    epoch_pat = data[np.where(behav["trialtypes"]==1)]
    epoch_pat = zscore(epoch_pat, axis=0)
    behav_pat = behav[behav["trialtypes"]==1]

    epoch_rand = data[np.where(behav["trialtypes"]==2)]
    epoch_rand = zscore(epoch_rand, axis=0)
    behav_rand = behav[behav["trialtypes"]==2]
    
    pat_data = rsatoolbox.data.TemporalDataset(
        epoch_pat, 
        descriptors={'sub':subject},
        channel_descriptors={'channel': mag_channels},  # or grad_channels, eeg_channels as needed
        obs_descriptors={'conds':behav_pat}, 
        time_descriptors={'time':times})
    
    rand_data = rsatoolbox.data.TemporalDataset(
        epoch_rand,
        descriptors={'sub':subject},
        channel_descriptors={'channel': mag_channels},  # or grad_channels, eeg_channels as needed
        obs_descriptors={'conds':behav_rand},
        time_descriptors={'time':times})

    # COVARIANCE MATRICES FOR PATTERN DATA
    split_data_pat = pat_data.split_time('time')
    time_new_pat = pat_data.time_descriptors['time']
    tp_for_covmat_pat = []
    for tp in split_data_pat:
        tp_single = tp.convert_to_dataset('time')
        tp_for_covmat_pat.append(rsatoolbox.data.noise.cov_from_unbalanced(tp_single, obs_desc='conds', dof=None, method='shrinkage_diag'))
    tp_for_covmat_pat = np.stack(tp_for_covmat_pat)
    vars()[f'cov_mat_pat_n{epoch_num}'] = np.mean(tp_for_covmat_pat, axis=0)
    