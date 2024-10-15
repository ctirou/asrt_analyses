import mne
import os
import os.path as op
import numpy as np
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from base import ensure_dir
from config import *
import gc
# stim disp = 500 ms
# RSI = 750 ms in task
analysis = 'pat_bsl_filtered_3400_3200'
data_path = TIMEG_DATA_DIR / analysis
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
jobs = -1

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
cv = StratifiedKFold(folds, shuffle=True)

for subject in subjects:
    print(f"Processing {subject}...")
    for trial_type in ['pattern', 'random']:
        res_dir = data_path / 'results' / 'sensors' / lock
        ensure_dir(res_dir)
        all_epochs = list()
        all_behavs = list()
        print(subject)
        for epoch_num, epo in zip([1, 2, 3, 4], epochs_list[1:]):
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch_gen = mne.read_epochs(epoch_fname, verbose="error", preload=False)
            # # run time generalization decoding on unique epoch
            # if trial_type == 'pattern':
            #     pattern = behav.trialtypes == 1
            #     X = epoch_gen.get_data()[pattern]
            #     y = behav.positions[pattern]
            # elif trial_type == 'random':
            #     random = behav.trialtypes == 2
            #     X = epoch_gen.get_data()[random]
            #     y = behav.positions[random]
            # else:
            #     X = epoch_gen.get_data()
            #     y = behav.positions    
            # y = y.reset_index(drop=True)            
            # assert X.shape[0] == y.shape[0]
            # gc.collect()
            # scores = cross_val_multiscore(clf, X, y, cv=cv)
            # np.save(res_dir / f"{subject}-epoch{epoch_num}-{trial_type}-scores.npy", scores.mean(0))
            # append epochs
            all_epochs.append(epoch_gen)
            all_behavs.append(behav)
        for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
            epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
        # concatenate epochs
        epochs = mne.concatenate_epochs(all_epochs)
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
        if not op.exists(res_dir / f"{subject}-epochall-{trial_type}-scores.npy"):
            scores = cross_val_multiscore(clf, X, y, cv=cv)
            np.save(res_dir / f"{subject}-epochall-{trial_type}-scores.npy", scores.mean(0))
            del X, y, scores
            gc.collect()