import os
import sys
from config import *

# stim disp = 500 ms
# RSI = 750 ms in task
data_path = TIMEG_DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = True

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs):
    import os.path as op
    import pandas as pd
    import numpy as np
    import gc
    from base import ensure_dir
    from mne import read_epochs, concatenate_epochs
    from mne.decoding import cross_val_multiscore, GeneralizingEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    cv = StratifiedKFold(folds, shuffle=True)

    for trial_type in ['pattern', 'random']:
        all_behavs = list()
        all_epochs = list()
        for epoch_num in [0, 1, 2, 3, 4]:
            res_path = data_path / 'results' / 'sensors' / lock
            ensure_dir(res_path)
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch_gen = read_epochs(epoch_fname, verbose="error", preload=False)
            
            if not op.exists(res_path / f"{subject}-epoch0-{trial_type}-scores.npy") or overwrite:
                print(f"Processing {subject} - session {epoch_num} - {trial_type}...")
                # run time generalization decoding on unique epoch
                if trial_type == 'pattern':
                    pattern = behav.trialtypes == 1
                    X = epoch_gen.get_data()[pattern]
                    y = behav.positions[pattern]
                elif trial_type == 'random':
                    random = behav.trialtypes == 2
                    X = epoch_gen.get_data()[random]
                    y = behav.positions[random]
                else:
                    X = epoch_gen.get_data()
                    y = behav.positions    
                y = y.reset_index(drop=True)            
                assert X.shape[0] == y.shape[0]
                gc.collect()
                scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
                np.save(res_path / f"{subject}-epoch0-{trial_type}-scores.npy", scores.mean(0))
            
            if epoch_num != 0:
                all_epochs.append(epoch_gen)
                all_behavs.append(behav)
            
        for epoch in all_epochs:
            epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
        # concatenate epochs
        epochs = concatenate_epochs(all_epochs)
        behav_df = pd.concat(all_behavs)    
        meg_data = epochs.get_data()    
        behav_data = behav_df.reset_index(drop=True)
        if trial_type == 'pattern':
            pattern = behav_data.trialtypes == 1
            X = meg_data[pattern]
            y = behav_data.positions[pattern]
        elif trial_type == 'random':
            random = behav_data.trialtypes == 2
            X = meg_data[random]
            y = behav_data.positions[random]
        else:
            X = meg_data
            y = behav_data.positions    
        y = y.reset_index(drop=True)            
        assert X.shape[0] == y.shape[0]
        del all_epochs, all_behavs, behav, epoch_fname, epoch_gen, epochs, behav_df, meg_data, behav_data
        gc.collect()
        if not op.exists(res_path / f"{subject}-epochall-{trial_type}-scores.npy") or overwrite:
            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
            np.save(res_path / f"{subject}-epochall-{trial_type}-scores.npy", scores.mean(0))
            del X, y, scores
            gc.collect()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 10
        process_subject(subject, lock, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        jobs = -1
        process_subject(subject, lock, jobs)