import os
import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
from Levenshtein import editops
import pandas as pd
import warnings
from base import ensure_dir, ensured
from config import *
import gc
import sys
from autoreject import get_rejection_threshold

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

subjects = SUBJS15
mode_ICA = True
generalizing = True
filtering = True
overwrite = True
verbose = True
jobs = -1

def int_to_unicode(array):
        return ''.join([str(chr(int(ii))) for ii in array]) # permet de convertir int en unicode (pour editops)

def process_subject(subject, mode_ICA, generalizing, filtering, overwrite, jobs, verbose):      
        # Set path
        data_path = RAW_DATA_DIR
        if generalizing:
                res_path = ensured(DATA_DIR / 'for_timeg')
                tmin, tmax = -4, 4
        else:
                res_path = ensured(DATA_DIR / 'for_rsa')
                tmin, tmax = -0.2, 0.6

        meg_sessions = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

        # Create preprocessed sub-folders
        folders = ['epochs', 'behav']
        for fol in folders:
                ensure_dir(res_path / fol)
        
        # Sort behav files
        path_to_behav_dir = f'{data_path}/{subject}/behav_data'
        behav_dir = os.listdir(path_to_behav_dir)
        behav_files_filter = [f for f in behav_dir if not f.startswith('.')]
        behav_files = sorted([f for f in behav_files_filter if '_eASRT_Practice' in f or '_eASRT_Epoch' in f])
        behav_sessions = [behav_files[-1]] + behav_files[:-1]
        # Loop across sessions
        for session_num, meg_session, behav_session in zip([0, 1, 2, 3, 4], meg_sessions, behav_sessions):
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
                # Channels 059 and 173 are flat, 028 is noisy as fuck
                to_drop = ['MISC 001', 'MEG 059', 'MEG 173', 'MEG 028']
                raw.drop_channels(to_drop)
                # Save a filtered version of the raw to run the ICA on
                filt_raw = raw.copy().filter(l_freq=1., h_freq=None, n_jobs=jobs)
                # Find rejection thresholds
                try:
                        epochs_for_thresh = mne.make_fixed_length_epochs(filt_raw, duration=2., preload=True, verbose=verbose)
                        reject = get_rejection_threshold(epochs_for_thresh, verbose=verbose)
                        print(f"AutoReject thresholds: {reject}")
                        del epochs_for_thresh
                        gc.collect()
                except Exception as e:
                        print(f"AutoReject failed, falling back to default reject: {e}")
                        reject = dict(mag=4e-12)
                if mode_ICA:
                        # Initialize the ICA asking for 30 components
                        # ica = ICA(n_components=30, method='fastica', random_state=42, verbose=verbose)
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
                        raw.filter(0.1, 30, n_jobs=jobs)
                if subject == 'sub11' and session_num != 0: # sub11 has wrong triggers so we need to read a txt file with correct events
                        file_path = data_path / subject / 'meg_data' / meg_session / 'events_info.txt'
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
                        events = mne.find_events(raw, shortest_event=1, verbose=verbose) # shortest_event=1 for sub06, sub08, sub14 and possibly others
                if subject == 'sub05' and session_num == 0:
                        # offsets = np.array([ 9400, 11000, 12400, 12500, 14000, 15500, 17000, 17100, 18600]) + 97
                        # offset = 0
                        offset = 97
                        behav_fname = data_path / subject / 'behav_data' / behav_session
                        log = pd.read_csv(behav_fname, sep='\t')
                        log.columns = [col for col in log.columns if col not in ['isi_if_correct', 'isi_if_incorrect']] + [''] * len(['isi_if_correct', 'isi_if_incorrect'])
                        # Créer l'array d'événement
                        events_stim = np.column_stack((
                                log['stim_pres_time'].values.astype(int) + offset, # To re-synchronize with photodiode time-samples
                                np.zeros(len(log), dtype=int),
                                log['triplet'].values.astype(int)))
                        # # Use alternative log file to read events
                        # fname = op.join("/Users/coum/Desktop/asrt/raws/sub05/behav_data/-Explicit_ASRT_practice.log")
                        # log2 = open(fname, 'r', encoding='utf-8')
                        # lines = log2.readlines()
                        # lines = lines[3:]
                        # events = list()
                        # times = list()
                        # pos = list()
                        # column_names = lines[0].split(sep='\t')
                        # for i, line in enumerate(lines[2:]):
                        #         try:
                        #                 yep = str(line.split()[column_names.index('Code')])
                        #                 if any(direction in yep for direction in ['Left', 'Right', 'Up', 'Down']):
                        #                         events.append(str(line.split()[column_names.index('Event Type')])+ str(line.split()[column_names.index('Code')]))
                        #                         times.append(int(line.split()[column_names.index('Time')]) + offset)
                        #                         if 'Left' in yep:
                        #                                 pos.append(12)
                        #                         elif 'Up' in yep:
                        #                                 pos.append(14)
                        #                         elif 'Down' in yep:
                        #                                 pos.append(16)
                        #                         elif 'Right' in yep:
                        #                                 pos.append(18)
                        #         except IndexError:
                        #                 continue                        
                        # for i, event in enumerate(events):
                        #         if event == events[i+1]:
                        #                 del events[i+1]
                        #                 del times[i+1]
                        #                 del pos[i+1]
                        #                 if 'Pattern' in event:
                        #                         events[i] = 30
                        #                 elif 'Random' in event:
                        #                         events[i] = 32
                        #                 else:
                        #                         events[i] = 40
                        # times = [int(raw.info['sfreq'] * (t/1000)) for t in times]
                        # events_log = np.column_stack((np.array(times), np.array(pos), np.array(events)))
                else:
                        events_stim = list()
                        keys = [12, 14, 16, 18]
                        if subject == 'sub11':
                                triggs = [30, 32, 34, 36, 38, 40] # sub11 does not have 544 = photodiode for 32 (random high)
                                # triggs = [30, 32, 34] # sub11 does not have 544 = photodiode for 32 (random high)
                                for ii in range(len(events) - 2):
                                        cond = events[ii+1, 2] in keys if events[ii, 2] == 32 else events[ii+2, 2] in triggs
                                        if events[ii, 2] in triggs and events[ii+2, 2] in keys:
                                                event_stim = events[ii]
                                                # if event_stim[2] == 32:
                                                event_stim[0] = event_stim[0] + 97 # To re-synchronize with photodiode time-samples 
                                                event_stim[2] = {542: 30, 544: 32, 546: 34, 548: 36, 550: 38, 552: 40}.get(event_stim[2], event_stim[2])
                                                # event_stim[2] = {542: 30, 544: 32, 546: 34}.get(event_stim[2], event_stim[2])
                                                events_stim.append(event_stim)
                        else:
                                triggs = [542, 544, 546, 548, 550, 552]
                                # triggs = [542, 544, 546]
                                for ii in range(len(events)):
                                        if events[ii, 2] in triggs and events[ii+1, 2] in keys:
                                                event_stim = events[ii]
                                                event_stim[2] = {542: 30, 544: 32, 546: 34, 548: 36, 550: 38, 552: 40}.get(event_stim[2], event_stim[2])
                                                # event_stim[2] = {542: 30, 544: 32, 546: 34}.get(event_stim[2], event_stim[2])
                                                events_stim.append(event_stim)
                        events_stim = np.array(events_stim)                
                # Read behav data
                fname_behav = op.join(data_path, subject, 'behav_data', behav_session)
                behav = open(fname_behav, 'r')
                lines = behav.readlines()
                column_names = lines[0].split()
                if session_num == 0:
                        del column_names[7] # there are extra columns in the behav file that mess up the reading
                        del column_names[7] # there are extra columns in the behav file that mess up the reading
                positions = list()
                triplets = list()
                trialtypes = list()
                RTs = list()
                corrects = list()
                stim_pres_times = list()
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
                                stim_pres_times.append(int(line.split()[column_names.index('stim_pres_time')]))
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
                stim_pres_times = np.array(stim_pres_times)
                behav_df = pd.DataFrame(behav_dict)
                # Create epochs time locked on stimulus onset and baseline epochs
                picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False) # by default eog is True
                reject = dict(mag=reject['mag'])
                if generalizing:
                        epochs = mne.Epochs(raw, events_stim, tmin=tmin, tmax=tmax, baseline=None, preload=True, picks=picks, decim=20, reject=reject, verbose=verbose)
                else:
                        epochs = mne.Epochs(raw, events_stim, tmin=tmin, tmax=tmax, baseline=(-0.2, 0), preload=True, picks=picks, decim=20, reject=reject, verbose=verbose)
                # Free memory
                del raw
                gc.collect()
                changes = editops(int_to_unicode(behav_df['triplets']), int_to_unicode(epochs.events[:, 2]))
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
                        epochs.drop(del_from_epoch)
                        behav_df.drop(behav_df.index[del_from_behav], inplace=True)
                # Last check if behav and epochs have same shapes
                changes = editops(int_to_unicode(behav_df['triplets']), int_to_unicode(epochs.events[:, 2]))
                epochs.save(op.join(res_path, 'epochs', f'{subject}-{session_num}-epo.fif'), overwrite=overwrite)
                # Save behavioral data
                behav_df.to_pickle(op.join(res_path, 'behav', f'{subject}-{session_num}.pkl'))
                print("Final number of epochs: ", len(epochs), "out of", 255 if session_num == 0 else 425)
        
if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
        process_subject(subject, mode_ICA, generalizing, filtering, overwrite, jobs, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
        jobs = -1
        for subject in subjects:
                process_subject(subject, mode_ICA, generalizing, filtering, overwrite, jobs, verbose)
                
                
# diff_events = event_times - stim_pres_times

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(events_stim[:, 0], label='Event behav')
# ax.plot(events_log[:, 0], label='Event log')
# diff = events_stim[:, 0] - events_log[:, 0]
# ax.plot(diff, label='Difference')
# ax.legend()

# diff_behav, diff_log = np.diff(events_stim[:, 0]), np.diff(events_log[:, 0])

# fig, ax = plt.subplots()
# diff = events_stim[:, 0] - events_log[:, 0]
# ax.plot(diff, label='Difference')
# # ax.plot(diff_behav, label='Difference behav')
# # ax.plot(diff_log, label='Difference log')
# ax.legend()


# plt.plot(diff_events, label='Difference')
# plt.xlabel('Time (s)')

# # Calculate and plot the linear fit
# x = np.arange(len(diff_events))
# slope, intercept, _, _, _ = linregress(x, diff_events)
# linear_fit = slope * x + intercept
# plt.plot(linear_fit, label='Linear Fit', linestyle='--', color='red')

# plt.ylabel('Difference between event and stim pres time (s)')
# plt.legend()
# plt.ylabel('Difference between event and stim pres time (s)')

# fig, ax = plt.subplots()
# ax.plot(event_times, label='Event Times epochs')
# ax.plot(stim_pres_times, label='Stim Pres Times behav')
# ax.legend()

# # Calculate the correct slope and intercept
# scaling_factor = np.cov(x, diff_events)[0, 1] / np.var(x)
# offset = np.mean(diff_events) - scaling_factor * np.mean(x)

# # Apply the linear transformation
# transformed_stim_pres_times = stim_pres_times * scaling_factor + offset

# # Plot the transformed stim pres times against event times
# fig, ax = plt.subplots()
# ax.plot(event_times, label='Event Times epochs')
# ax.plot(stim_pres_times, label='Stim Pres Times behav')
# ax.plot(transformed_stim_pres_times, label='Transformed Stim Pres Times behav', linestyle='--')
# ax.legend()
# plt.show()

# print(f"Linear transformation: stim_pres_times * {scaling_factor:.4f} + {offset:.4f}")


# A = np.outer(event_times, stim_pres_times) / np.dot(stim_pres_times, stim_pres_times)
# event_times_approx = A @ stim_pres_times
# print("Transformation matrix A:\n", A)
# print("A @ v =", event_times_approx)

# fig, ax = plt.subplots()
# ax.plot(event_times, label='Event Times epochs', lw=4)
# ax.plot(stim_pres_times, label='Stim Pres Times behav', lw=4)
# ax.plot(event_times_approx, label='Transformed Event Times epochs', linestyle='--')
# ax.legend()

# stim_pres_times = np.array(stim_pres_times)
# event_times = epochs.events[:, 0]

# v = events_stim.copy()[:, 0]
# w = events_log.copy()[:, 0]

# v_prac = events_stim[:, 0]
# w_prac = events_log[:, 0]

# fig, ax = plt.subplots()
# ax.plot(v_prac, label='behav', lw=2, alpha=0.5)
# ax.plot(w_prac, label='epochs', lw=2, alpha=0.5)
# ax.legend()

# # Fit line: w = m*v + b
# # Using least squares
# A = np.vstack([v, np.ones_like(v)]).T
# m, b = np.linalg.lstsq(A, w, rcond=None)[0]

# print("Slope (m):", m)
# print("Intercept (b):", b)

# # Apply to a new v_i
# w_i = m * v + b
# print("Predicted w_i:", w_i)

# fig, ax = plt.subplots()
# ax.plot(v, label='behav', lw=2, alpha=0.5)
# ax.plot(w, label='epochs', lw=2, alpha=0.5)
# ax.plot(w_i, label="predicted epochs", ls='--', alpha=1)
# ax.legend()
# plt.show()