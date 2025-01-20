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
from rsatoolbox.rdm import calc_rdm_unbalanced
from rsatoolbox.rdm.rdms import concat

lock = 'stim'
verbose = "error"

data_path = DATA_DIR
subjects = SUBJS

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "cv_rdm" / subject
    ensure_dir(res_path)

    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
        
        print(f"Processing {subject} - {epoch_num}")
            
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        data = epoch.get_data(picks='mag', copy=True)
        times = epoch.times
        
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
            descriptors={'sub':subject, 'sess':epoch_num, 'trialtype':'pattern'},
            channel_descriptors={'channel': mag_channels},  # or grad_channels, eeg_channels as needed
            obs_descriptors={'conds':behav_pat.positions}, 
            time_descriptors={'time':times})
        
        rand_data = rsatoolbox.data.TemporalDataset(
            epoch_rand,
            descriptors={'sub':subject, 'sess':epoch_num, 'trialtype':'random'},
            channel_descriptors={'channel': mag_channels},  # or grad_channels, eeg_channels as needed
            obs_descriptors={'conds':behav_rand.positions},
            time_descriptors={'time':times})

        # COVARIANCE MATRICES FOR PATTERN DATA
        split_data_pat = pat_data.split_time('time') # split data into timepoints
        time_new_pat = pat_data.time_descriptors['time']
        tp_for_covmat_pat = []
        for tp in split_data_pat:
            tp_single = tp.convert_to_dataset('time')
            tp_for_covmat_pat.append(rsatoolbox.data.noise.cov_from_unbalanced(tp_single, obs_desc='conds', dof=None, method='shrinkage_diag'))
        tp_for_covmat_pat = np.stack(tp_for_covmat_pat)
        cov_pat = np.mean(tp_for_covmat_pat, axis=0) # average across timepoints
        
        # GET RDMs FOR PATTERN DATA
        rdms_pat = list()
        for dat in split_data_pat:
            dat_single = dat.convert_to_dataset('time')
            rdms_pat.append(calc_rdm_unbalanced(dat_single, method='crossnobis', descriptor='conds', noise=cov_pat, cv_descriptor=None))
        rdms_data = concat(rdms_pat)
        rdms_data.rdm_descriptors['time'] = time_new_pat
        dis_pat = rdms_data.get_matrices()
        np.save(res_path / f'pat-{epoch_num}.npy', dis_pat)
        
        # COVARIANCE MATRICES FOR RANDOM DATA
        split_data_rand = rand_data.split_time('time')
        time_new_rand = rand_data.time_descriptors['time']
        tp_for_covmat_rand = []
        for tp in split_data_rand:
            tp_single = tp.convert_to_dataset('time')
            tp_for_covmat_rand.append(rsatoolbox.data.noise.cov_from_unbalanced(tp_single, obs_desc='conds', dof=None, method='shrinkage_diag'))
        tp_for_covmat_rand = np.stack(tp_for_covmat_rand)
        cov_rand = np.mean(tp_for_covmat_rand, axis=0)
        
        # GET RDMs FOR RANDOM DATA
        rdms_rand = list()
        for dat in split_data_rand:
            dat_single = dat.convert_to_dataset('time')
            rdms_rand.append(calc_rdm_unbalanced(dat_single, method='crossnobis', descriptor='conds', noise=cov_rand, cv_descriptor=None))
        rdms_data = concat(rdms_rand)
        rdms_data.rdm_descriptors['time'] = time_new_rand
        dis_rand = rdms_data.get_matrices()
        np.save(res_path / f'rand-{epoch_num}.npy', dis_rand)

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        lock = str(sys.argv[1])
        process_subject(subject, lock)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock)