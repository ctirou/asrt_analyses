#!/usr/bin/env python

# DO NOT USE TO CREATE EPOCHS FOR PRACTICE SESSION

# reject flat or noisy channels, search automatically or manually
# add rejection threshold in mne.Epochs

import os
import os.path as op
import numpy as np
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
from mne import Epochs, pick_types
from mne.epochs import concatenate_epochs
from mne.viz.utils import plt_show
from mne.preprocessing import ICA
from Levenshtein import editops
from pathlib import Path
from autoreject import AutoReject, Ransac
import sys
import pandas as pd
import warnings

subjects = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
            'sub10', 'sub12', 'sub13', 'sub14', 'sub15']
subjects = ['sub01']

# To run on cluster
# path = '/sps/crnl/Romain/ASRT_MEG/data/preprocessed'
# path_data = '/sps/crnl/Romain/ASRT_MEG/data/raws'
# subject_num = int(os.environ["SLURM_ARRAY_TASK_ID"])
# subject = subjects[subject_num]

def int_to_unicode(array):
        return ''.join([str(chr(int(ii))) for ii in array]) #permet de convertir int en unicode (pour editops)

mode_ICA = False

# Set path
path = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/preprocessed'
path_data = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/raws'
path_results = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results'


all_epochs_stim = list()
all_epochs_button = list()
all_behav_df = pd.DataFrame({'position': [], 'triplet': [], 'trialtype': [], 'RT': [], 'block':[]})

meg_sessions = ['2_PRACTICE']

# Create preprocessed sub-folders
folders = ['stim', 'button', 'behav', 'bsl']
for fold in folders:
        if not op.exists(op.join(path, fold)):
                os.mkdir(op.join(path, fold))      

# Loop across subjects
for subject in subjects:
        # Sort behav files
        path_to_behav_dir = f'{path_data}/{subject}/behav_data'
        behav_dir = os.listdir(path_to_behav_dir)
        behav_files_filter = [f for f in behav_dir if not f.startswith('.')]
        behav_sessions = sorted([f for f in behav_files_filter if '_eASRT_Practice' in f])
        # Loop across sessions
        for session_num, (meg_session, behav_session) in enumerate(zip(meg_sessions, behav_sessions)):
        # for session_num, meg_session, behav_session in zip([3], ['6_EPOCH_4'], ['1_eASRT_Epoch_4.txt']):
                # Read the raw data
                raw_fname = op.join(path_data, subject, 'meg_data', meg_session, 'results', 'c,rfDC_EEG')
                hs_fname = op.join(path_data, subject, "meg_data", meg_session, "hs_file")
                config_fname = op.join(path_data, subject, "meg_data", meg_session, "config")
                raw = mne.io.read_raw_bti(raw_fname, preload=True, config_fname=config_fname, head_shape_fname=hs_fname, verbose=True)
                # Set some channel types and rename them
                raw.set_channel_types({'EEG 001': 'ecg',
                                'VEOG': 'eog',
                                'HEOG': 'eog',
                                'UTL 001': 'misc'})
                raw.rename_channels({'UTL 001': 'MISC 001',
                                'EEG 001': 'ECG 001'})
                # Channels 059 and 173 are flat
                to_drop = ['MISC 001', 'MEG 059', 'MEG 173']
                raw.drop_channels(to_drop)
                if mode_ICA:
                        # Save a filtered version of the raw to run the ICA on
                        filt_raw = raw.copy().filter(l_freq=1., h_freq=None, n_jobs=10)
                        reject = dict(mag=5e-12)
                        # Initialize the ICA asking for 30 components
                        # ica = ICA(n_components=30, method='infomax', fit_params=dict(extended=True))
                        ica = ICA(n_components=30, method='fastica')
                        # Fit the ica on the filtered raw
                        ica.fit(filt_raw, reject=reject)
                        # Find the bad components based on the VEOG, HEOG and hearbeat
                        veog_indices, veog_scores = ica.find_bads_eog(filt_raw, ch_name='VEOG')
                        heog_indices, heog_scores = ica.find_bads_eog(filt_raw, ch_name='HEOG')
                        hbeat_indices, hbeat_scores = ica.find_bads_ecg(filt_raw, ch_name='ECG 001')
                        # Exclude bad components and apply it to the unfiltered raw
                        ica.exclude = np.unique(np.concatenate([veog_indices, heog_indices, hbeat_indices]))
                        ica.apply(raw)
                        # Filter raw
                raw.filter(0.1, 30, n_jobs=10)     
                # Select events of interest (only photodiode for good triplets and correct answers)
                if subject == 'sub06' and meg_session == '6_EPOCH_4':
                        events = mne.find_events(raw, shortest_event=1, verbose=True)
                elif subject == 'sub08' and meg_session == '4_EPOCH_2':
                        events = mne.find_events(raw, shortest_event=1, verbose=True)
                elif subject == 'sub14' and meg_session == '3_EPOCH_1':
                        events = mne.find_events(raw, shortest_event=1, verbose=True)
                elif subject == 'sub11': # sub11 has wrong triggers so we need to read a txt file with correct events
                        file_path = op.join(path_data, subject, 'meg_data', meg_session, 'events_info.txt')
                        ev = open(file_path, 'r')
                        lines = ev.readlines()
                        column_names = lines[0].split()
                        samples = list()
                        start = list()
                        end = list()
                        for line in lines[2:]:
                                samples.append(int(line.split()[column_names.index('sample')]))
                                start.append(0)
                                end.append(int(line.split()[column_names.index('value')]))
                        events = np.vstack([np.array(samples), np.array(start), np.array(end)]).T 
                else:
                        events = mne.find_events(raw, verbose=True)
                events_stim = list()
                events_button = list()
                triggs = [546]
                ranger = range(len(events))
                for ii in ranger:
                        if events[ii, 2] in triggs and events[ii+1, 2] in [12, 14, 16, 18]:
                                event_stim = events[ii]
                                event_button = events[ii+1]
                                # Replace photodiode values by triplet values (as in behavior)
                                if event_stim[2] == 546:
                                        event_stim[2] = 34
                                events_stim.append(event_stim)
                                events_button.append(event_button)
                events_stim = np.array(events_stim)
                events_button = np.array(events_button)
                # Read behav data
                fname_behav = op.join(path_data, subject, 'behav_data', behav_session)
                behav = open(fname_behav, 'r')
                lines = behav.readlines()
                column_names = lines[0].split()
                del column_names[7]
                del column_names[7]
                positions = list()
                triplets = list()
                trialtypes = list()
                RTs = list()
                corrects = list()
                expec_triggers = list()
                blocks = list()
                for line in lines[1:]:
                        corr = int(line.split()[column_names.index('talalat')])
                        if corr == 1:
                                pos = int(line.split()[column_names.index('position')])
                                positions.append(pos)
                                triplets.append(int(line.split()[column_names.index('triplet')]))
                                trialtypes.append(int(line.split()[column_names.index('trialtype')]))
                                RTs.append(float(line.split()[column_names.index('RT')]))
                                blocks.append(int(line.split()[column_names.index('block')]))
                                # add column of expected trigger in the epoch
                                if pos == 1:
                                        expec_trigg = 12
                                elif pos == 2:
                                        expec_trigg = 14
                                elif pos == 3:
                                        expec_trigg = 16
                                elif pos == 4:
                                        expec_trigg = 18
                                expec_triggers.append(expec_trigg)
                behav_dict = {'positions': np.array(positions), 'triplets': np.array(triplets),
                        'trialtypes': np.array(trialtypes), 'RTs': np.array(RTs),
                        'expec_triggers': np.array(expec_triggers), 'blocks': np.array(blocks)}
                behav_df = pd.DataFrame(behav_dict)
                # Indices of good triplets 
                good_triplets = np.where((behav_dict['triplets']==30) |
                                        (behav_dict['triplets']==32) |
                                        (behav_dict['triplets']==34))[0]
                behav_df = behav_df.reindex(index = good_triplets)
                stim_df = behav_df.copy()
                button_df = behav_df.copy()
                # Create epochs time locked on stimulus onset and button response, and baseline epochs
                reject = dict(mag=5e-12)
                picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False)
                epochs_stim = mne.Epochs(raw, events_stim, tmin=-0.2, tmax=0.6, baseline=(-.2, 0), preload=True, picks=picks, decim=10, reject=reject, verbose=True)                   
                epochs_button = mne.Epochs(raw, events_button, tmin=-0.2, tmax=0.6, baseline=None, preload=True, picks=picks, decim=10, reject=reject, verbose=True)                        
                # Free memory
                # del raw
                # Automatic rejection of bad epochs
                # ar = AutoReject(n_jobs=-1)
                # epochs_stim = ar.fit_transform(epochs_stim)
                # rsc = Ransac(n_jobs=-1)
                # epochs_stim = rsc.fit_transform(epochs_stim)
                # Check similarity between epochs and behavior
                for df, df_column, epochs in zip([stim_df, button_df], ['triplets', 'expec_triggers'], [epochs_stim, epochs_button]):
                        changes = editops(int_to_unicode(df[df_column]), int_to_unicode(epochs.events[:, 2]))
                        # Make modification
                        if len(changes) !=0:
                                del_from_epoch = list()
                                del_from_behav = list()
                                for change in changes:
                                        if change[0] == 'insert':
                                                del_from_epoch.append(change[2])
                                        elif change[0] == 'replace':
                                                del_from_epoch.append(change[2])
                                                del_from_behav.append(change[1])
                                        elif change[0] == 'delete':
                                                del_from_behav.append(change[1])
                                epochs_stim.drop(del_from_epoch)
                                epochs_button.drop(del_from_epoch)
                                df.drop(df.index[del_from_behav], inplace=True)
                # Equalize number of trials in epochs
                changes = editops(int_to_unicode(stim_df['positions']), int_to_unicode(button_df['positions']))
                if len(changes) !=0:
                        del_from_stimdf, del_from_buttondf, del_from_stimepo, del_from_buttonepo = [], [], [], []
                        for change in changes:
                                if change[0] == 'insert':
                                        del_from_buttondf.append(change[2])
                                        del_from_buttonepo.append(change[2])
                                elif change[0] == 'delete':
                                        del_from_stimdf.append(change[1])
                                        del_from_stimepo.append(change[1])
                        stim_df.drop(stim_df.index[del_from_stimdf], inplace=True)
                        button_df.drop(button_df.index[del_from_buttondf], inplace=True)
                        epochs_stim.drop(del_from_stimepo)
                        epochs_button.drop(del_from_buttonepo)
                # Last check if behav and epochs have same shapes
                changes = editops(int_to_unicode(stim_df['triplets']), int_to_unicode(epochs_stim.events[:, 2]))
                if len(changes) != 0:
                        warnings.warn("Behav file and stim epochs have different shapes.")
                changes = editops(int_to_unicode(stim_df['expec_triggers']), int_to_unicode(epochs_button.events[:, 2]))
                if len(changes) != 0:
                        warnings.warn("Behav file and button epochs have different shapes.")
                # Apply baseline from before the stimulus in the epochs_button
                epochs_baseline = epochs_stim.copy().crop(None, 0)         
                bsl_channels = mne.pick_types(epochs_button.info, meg=True)
                bsl_data = epochs_baseline.get_data()[:, bsl_channels, :]
                bsl_data = np.mean(bsl_data, axis=2)
                epochs_button._data[:, bsl_channels, :] -= bsl_data[:, :, np.newaxis]
                # Save epochs 
                epochs_stim.save(op.join(path, 'stim', f'{subject}_{session_num}_s-epo.fif'), overwrite=True)
                epochs_button.save(op.join(path, 'button', f'{subject}_{session_num}_b-epo.fif'), overwrite=True)
                epochs_baseline.save(op.join(path, 'bsl', f'{subject}_{session_num}_bl-epo.fif'), overwrite=True)
                # Save behavioral data
                behav_df = stim_df
                behav_df.to_pickle(op.join(path, 'behav', f'{subject}_{session_num}.pkl'))