import os
import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
from base import ensure_dir, ensured
from config import *
import gc
import sys
from autoreject import get_rejection_threshold

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

subjects = SUBJS15
mode_ICA = True
filtering = True
overwrite = True
verbose = True
        
# Set path
# data_path = RAW_DATA_DIR
data_path = op.join("/Volumes/Ultra_Touch/asrt/raws")
res_path = ensured(DATA_DIR / 'for_timeg' / 'noise_cov')
        
def process_subject(subject, mode_ICA, filtering, overwrite, jobs, verbose):        
        # Loop across sessions
        meg_session = '7_RESTING_2' if subject == 'sub01' else '7_RESTING_STATE_2'
                
        noise_cov_fname = res_path / f"{subject}-cov.fif"
        
        if not op.exists(noise_cov_fname) or overwrite:
                
                # Read the raw data
                raw_fname = op.join(data_path, subject, 'meg_data', meg_session, 'results', 'c,rfDC_EEG')
                hs_fname = op.join(data_path, subject, "meg_data", meg_session, "hs_file")
                config_fname = op.join(data_path, subject, "meg_data", meg_session, "config")
                raw = mne.io.read_raw_bti(raw_fname, preload=True, config_fname=config_fname, head_shape_fname=hs_fname, verbose=verbose)
                # Set some channel types and rename them
                raw.set_channel_types({'EEG 001': 'ecg',
                                'VEOG': 'eog',
                                'HEOG': 'eog',
                                'UTL 001': 'misc'})
                raw.rename_channels({'UTL 001': 'MISC 001',
                                'EEG 001': 'ECG 001'})
                to_drop = ['MISC 001', 'MEG 059', 'MEG 173', 'MEG 028']
                raw.drop_channels(to_drop)
                # Save a filtered version of the raw to run the ICA on
                filt_raw = raw.copy().filter(l_freq=1., h_freq=None, n_jobs=jobs)
                # Find rejection thresholds
                try:
                        epochs_for_thresh = mne.make_fixed_length_epochs(filt_raw, duration=2., preload=True, verbose=verbose)
                        reject = get_rejection_threshold(epochs_for_thresh, verbose=verbose)
                        reject = dict(mag=reject['mag'])
                        print(f"AutoReject thresholds: {reject}")
                        del epochs_for_thresh
                        gc.collect()
                except Exception as e:
                        print(f"AutoReject failed, falling back to default reject: {e}")
                        reject = dict(mag=4e-12)
                if mode_ICA:
                        # Initialize the ICA asking for 30 components
                        # ica = ICA(n_components=30, method='infomax', fit_params=dict(extended=True))
                        ica = ICA(n_components=30, method='picard', fit_params=dict(ortho=True), random_state=42, verbose=verbose)
                        # Fit the ica on the filtered raw
                        try:
                                ica.fit(filt_raw, reject=reject)
                        except RuntimeError as e:
                                print(f"ICA fitting failed for {subject} {meg_session}: {e}")
                                print("Retrying with higher reject threshold...")
                                ica.fit(filt_raw, reject=dict(mag=1e-11))
                        # Find the bad components based on the VEOG, HEOG and hearbeat
                        veog_indices, _ = ica.find_bads_eog(filt_raw, ch_name='VEOG')
                        heog_indices, _ = ica.find_bads_eog(filt_raw, ch_name='HEOG')
                        hbeat_indices, _ = ica.find_bads_ecg(filt_raw, ch_name='ECG 001')
                        # Exclude bad components and apply it to the unfiltered raw
                        ica.exclude = np.unique(np.concatenate([veog_indices, heog_indices, hbeat_indices]))
                        # Filter raw
                        ica.apply(raw)
                if filtering:
                        raw.filter(0.1, 30, n_jobs=jobs) # test by 25
                # Create epochs time locked on stimulus onset and button response, and baseline epochs
                picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False) # by default eog is True                
                noise_cov = mne.compute_raw_covariance(raw, 
                                                        tmin=0, tmax=None, 
                                                        reject=reject, 
                                                        picks=picks, 
                                                        rank='info', 
                                                        method='empirical', 
                                                        n_jobs=jobs, 
                                                        verbose=verbose)
                mne.write_cov(noise_cov_fname, noise_cov, overwrite=True, verbose=verbose)
                
                # Free memory
                del raw
                gc.collect()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 10
        process_subject(subject, mode_ICA, filtering, overwrite, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
        jobs = -1
        for subject in subjects:
                process_subject(subject, mode_ICA, filtering, overwrite, jobs, verbose)