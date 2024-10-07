import mne
import os.path as op
import numpy as np
import pandas as pd
from base import ensure_dir
from config import *
import matplotlib.pyplot as plt

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

#####
    epoch = epoch_filt.copy()
    events = epoch.events
    event_samples = events[:, 0]
    sample_intervals = np.diff(event_samples)

    sfreq = epoch.info['sfreq']
    time_intervals_ms = (sample_intervals / sfreq) * 1000
    # std_interval = np.std(time_intervals_ms)

    # Plot the time intervals
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals_ms, marker='o', linestyle='-', color='b')
    plt.title('Time Intervals Between Events')
    plt.xlabel('Event Index')
    plt.ylabel('Interval (s)')
    plt.grid(True)
    plt.show()
####
    data_path = Path('/Users/coum/Desktop/rawz/raws/')
    raw_fname = op.join(data_path, subject, 'meg_data', epo, 'results', 'c,rfDC_EEG')
    hs_fname = op.join(data_path, subject, "meg_data", epo, "hs_file")
    config_fname = op.join(data_path, subject, "meg_data", epo, "config")
    raw = mne.io.read_raw_bti(raw_fname, preload=True, config_fname=config_fname, head_shape_fname=hs_fname, verbose=verbose)

    all_events = mne.find_events(raw, verbose=verbose)
    sfreq = raw.info['sfreq']
    
    # events = [event for event in all_events if event[1] in [30, 32, 34]]
    events = np.array(events)

    mne.viz.plot_events(events, sfreq)
    
    event_samples = events[:, 0]
    sample_intervals = np.diff(event_samples)
    time_intervals_ms = (sample_intervals / sfreq) * 1000
    std_interval = np.std(time_intervals_ms)
    time_interval = [ti for ti in time_intervals_ms if ti < std_interval]

    # Plot the time intervals
    plt.figure(figsize=(10, 6))
    plt.plot(time_interval, marker='o')
    plt.title('Time Intervals Between Events')
    plt.xlabel('Event Index')
    plt.ylabel('Interval (ms)')
    plt.grid(True)
    plt.show()
    
    break


