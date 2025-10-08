import os
import os.path as op
import numpy as np
import mne
from mne.preprocessing import ICA
from Levenshtein import editops
import pandas as pd
from base import ensure_dir, ensured
from config import *
import gc
import sys
from autoreject import get_rejection_threshold

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

subjects = SUBJS15
mode_ICA = True
generalizing = False
filtering = True
overwrite = False
verbose = True
jobs = -1

def int_to_unicode(array):
        return ''.join([str(chr(int(ii))) for ii in array]) # permet de convertir int en unicode (pour editops)

def process_subject(subject, mode_ICA, generalizing, filtering, overwrite, jobs, verbose):      
        # Set path
        data_path = RAW_DATA_DIR
        if generalizing:
                res_path = ensured(DATA_DIR / 'for_timeg_new')
                tmin, tmax = -4, 4
        else:
                res_path = ensured(DATA_DIR / 'for_rsa_new')
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
                
                if not op.exists(op.join(res_path, 'epochs', f'{subject}-{session_num}-epo.fif')) or overwrite:
                
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
                        # Channels 059 and 173 are flat, 028 is noisy
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
                                offset = 97
                                behav_fname = data_path / subject / 'behav_data' / behav_session
                                log = pd.read_csv(behav_fname, sep='\t')
                                log.columns = [col for col in log.columns if col not in ['isi_if_correct', 'isi_if_incorrect']] + [''] * len(['isi_if_correct', 'isi_if_incorrect'])
                                # Créer l'array d'événement
                                events_stim = np.column_stack((
                                        log['stim_pres_time'].values.astype(int) + offset, # To re-synchronize with photodiode time-samples
                                        np.zeros(len(log), dtype=int),
                                        log['triplet'].values.astype(int)))
                        else:
                                events_stim = list()
                                if subject == 'sub11':
                                        triggs = [30, 32, 34, 36, 38, 40] # sub11 does not have 544 = photodiode for 32 (random high)
                                        for ii in range(len(events) - 2):
                                                if events[ii, 2] in triggs:
                                                        event_stim = events[ii]
                                                        event_stim[0] = event_stim[0] + 97 # To re-synchronize with photodiode time-samples 
                                                        event_stim[2] = {542: 30, 544: 32, 546: 34, 548: 36, 550: 38, 552: 40}.get(event_stim[2], event_stim[2])
                                                        events_stim.append(event_stim)
                                else:
                                        triggs = [542, 544, 546, 548, 550, 552]
                                        for ii in range(len(events)):
                                                if events[ii, 2] in triggs:
                                                        event_stim = events[ii]
                                                        event_stim[2] = {542: 30, 544: 32, 546: 34, 548: 36, 550: 38, 552: 40}.get(event_stim[2], event_stim[2])
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
                        stim_pres_times = list()
                        blocks = list()
                        for line in lines[1:]:
                                pos = int(line.split()[column_names.index('position')])
                                positions.append(pos)
                                triplets.append(int(line.split()[column_names.index('triplet')]))
                                trialtypes.append(int(line.split()[column_names.index('trialtype')]))
                                RTs.append(float(line.split()[column_names.index('RT')]))
                                blocks.append(int(line.split()[column_names.index('block')]))
                                stim_pres_times.append(int(line.split()[column_names.index('stim_pres_time')]))
                        behav_dict = {'positions': np.array(positions), 'triplets': np.array(triplets),
                                        'trialtypes': np.array(trialtypes), 'RTs': np.array(RTs),
                                        'blocks': np.array(blocks)}
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
                else:
                        print(f"Epochs for {subject} {meg_session} already exist, skipping...")
        
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