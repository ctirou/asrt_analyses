import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import ensure_dir, get_volume_estimate_tc
from config import *
from mne.beamformer import apply_dics_tfr_epochs, make_dics, apply_lcmv_cov, make_lcmv, apply_lcmv_epochs
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.time_frequency import csd_tfr, csd_morlet, tfr_morlet, tfr_array_morlet, tfr_array_multitaper, AverageTFRArray
from mne import get_volume_labels_from_src
import gc

# params
subjects, epochs_list = SUBJS, EPOCHS

lock = "stim" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR

ensure_dir(res_path)
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
jobs = -1
parc = 'aparc'
overwrite = False

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

theta = np.logspace(np.log10(1), np.log10(8), 80)
gamma = np.logspace(np.log10(30), np.log10(100), 80)

theta_lin = np.linspace(1, 8, 80)
gamma_lin = np.linspace(30, 100, 80)

for subject in subjects:
    # read behav file
    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)
    # read source space file
    src_fname = op.join(res_path, "src", "%s-src.fif" % (subject))
    src = mne.read_source_spaces(src_fname, verbose=verbose)    
    # create mixed source space
    vol_src_fname = op.join(res_path, "src", "%s-vol-src.fif" % (subject))
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
    mixed_src = src + vol_src
    all_stcs = []
    # path to bem file
    bem_fname = op.join(res_path, "bem", "%s-bem-sol.fif" % (subject))    
    
    for epoch_num, epo in enumerate(epochs_list):
        # read epoch
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
        if lock == 'button': 
            epoch_bsl_fname = data_path / 'bsl' / f'{subject}-{epoch_num}-epo.fif'
            epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose, preload=False)
            # compute noise covariance
            noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
        else:
            noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)        
        
        # compute data covariance matrix on evoked data
        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
        # conpute rank
        rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
        # path to trans file
        trans_fname = os.path.join(res_path, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
        # compute forward solution
        fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                    src=mixed_src, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
        
        filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori='max-power', rank=rank, reduce_rank=True, verbose=verbose)
        stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
        # inv = make_inverse_operator(epoch.info, fwd, data_cov)

        stc_base = apply_lcmv_cov(noise_cov, filters)
        stc_act = apply_lcmv_cov(data_cov, filters)
        stc_act /= stc_base

        offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])
        labels = get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
        label_time_courses = {}
        stc_data = stc_act.data
        for ilabel, label in enumerate(labels):
            tc = stc_data[offsets[ilabel]:offsets[ilabel+1]]
            if label.name not in label_time_courses:
                label_time_courses[label.name] = []
            label_time_courses[label.name].append(tc)
        # Convert to numpy arrays
        for label in label_time_courses:
            label_time_courses[label] = np.array(label_time_courses[label])  # shape: (n_trials, n_vertices_in_label, n_times)
        
        # sensor level
        # Time-Frequency Analysis using Morlet wavelets
        theta_freqs = np.arange(4, 8, 1)  # Theta band frequencies
        gamma_freqs = np.arange(30, 100, 5)  # Gamma band frequencies
        # Define number of cycles
        n_cycles_theta = theta_freqs / 2.  # Fewer cycles for better time resolution (2 cycles)
        n_cycles_gamma = gamma_freqs / 4.  # More cycles for better frequency resolution (7.5 cycles)
        # Combine theta and gamma frequencies
        all_freqs = np.concatenate([theta_freqs, gamma_freqs])
        n_cycles = np.concatenate([n_cycles_theta, n_cycles_gamma])
        power = epoch.compute_tfr(method='multitaper', freqs=theta_freqs, n_cycles=n_cycles_theta, use_fft=True, return_itc=False, decim=3, n_jobs=jobs, average=False, verbose=verbose)
        avgpower = power.average()
        avgpower.plot(
                [0],
                baseline=(-.2, 0),
                mode='mean',
                show=True,
                colorbar=True)
        
        
        label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
        label = "Hippocampus-lh"
        power = tfr_array_morlet(label_tc[label], 
                                 sfreq=epoch.info['sfreq'],
                                 freqs=theta_freqs,
                                 n_cycles=n_cycles_theta,
                                 output="avg_power",
                                 zero_mean=False)
        # Put it into a TFR container for easy plotting
        tfr = AverageTFRArray(
            info=epoch.info, data=power, times=epoch.times, freqs=theta_freqs, nave=len(stcs)
        )
        tfr.plot(
            baseline=(-.2, 0),
            picks=[0],
            mode="mean",
            title="TFR calculated on a NumPy array",
)