import os
import sys
from config import *
from joblib import Parallel, delayed
import os.path as op
import pandas as pd
import numpy as np
import gc
from base import ensured
from mne import read_epochs
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc

data_path = TIMEG_DATA_DIR
subjects = ALL_SUBJS
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = True

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    kf = KFold(n_splits=4, shuffle=False)

    for trial_type in ['pattern', 'random']:
        
        all_Xtraining, all_Xtesting = [], []
        all_ytraining, all_ytesting = [], []

        for epoch_num in [0, 1, 2, 3, 4]:
            res_path = ensured(data_path / 'results' / 'sensors' / f"split_20s_{trial_type}")
            
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch = read_epochs(epoch_fname, verbose=verbose, preload=True)
            
            times = epoch.times
            idx = np.where(times >= -1.5)[0]
            data = epoch.get_data(picks='mag', copy=True)[:, :, idx]
            assert len(behav) == len(data)
            
            blocks = np.unique(behav["blocks"])
            
            Xtraining, Xtesting, ytraining, ytesting = [], [], [], []
            
            del epoch
            gc.collect()
            
            for block in blocks:
                block = int(block)
                print(f"Processing {subject} - session {epoch_num} - block {block} - {trial_type}")
                                
                this_block = behav.blocks == block
                X = data[this_block]
                y = behav[this_block].reset_index(drop=True)
                assert len(X) == len(y), "Data and behavior lengths do not match"

                filter = np.where(y.trialtypes == 1)[0] if trial_type == 'pattern' else np.where(y.trialtypes == 2)[0]
                
                for train_index, test_index in kf.split(X):
                    
                    trainxfilter = np.intersect1d(train_index, filter)
                    testxfilter = np.intersect1d(test_index, filter)
                    
                    X_train, X_test = X[trainxfilter], X[testxfilter]
                    y_train, y_test = y.iloc[trainxfilter].positions, y.iloc[testxfilter].positions
                    
                    Xtraining.append(X_train)
                    Xtesting.append(X_test)
                    ytraining.append(y_train)
                    ytesting.append(y_test)
                    
                    if epoch_num != 0:                
                        all_Xtraining.append(X_train)
                        all_Xtesting.append(X_test)
                        all_ytesting.append(y_test)
                        all_ytraining.append(y_train)
            
            Xtraining = np.concatenate(Xtraining)
            ytraining = np.concatenate(ytraining)
            assert len(Xtraining) == len(ytraining), "Xtraining and ytraining lengths do not match"
            
            clf.fit(Xtraining, ytraining)

            assert len(Xtesting) == len(ytesting), "Xtesting and ytesting lengths do not match"

            for i, _ in enumerate(Xtesting):
                if not op.exists(res_path / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    ypred = clf.predict(Xtesting[i])
                    print(f"Scoring quarter {i+1} for subject {subject} epoch {epoch_num} {trial_type}")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytesting[i], x), 0, ypred)
                    np.save(res_path / f"{subject}-{epoch_num}-{i+1}.npy", acc_matrix)
            
            del epoch, behav
            gc.collect()
            
        res_path = ensured(data_path / 'results' / 'sensors' / f"split_20s_all_{trial_type}")
        
        all_Xtraining = np.concatenate(all_Xtraining)
        all_ytraining = np.concatenate(all_ytraining)
        assert len(all_Xtraining) == len(all_ytraining), "all_Xtraining and all_ytraining lengths do not match"
        
        clf.fit(all_Xtraining, all_ytraining)
        
        assert len(all_Xtesting) == len(all_ytesting), "all_Xtesting and all_ytesting lengths do not match"
        for i, _ in enumerate(all_Xtesting):
            if not op.exists(res_path / f"{subject}_{i+1}.npy") or overwrite:
                ypred = clf.predict(all_Xtesting[i])
                print(f"Scoring quarter {i+1} for subject {subject} all {trial_type}")
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting[i], x), 0, ypred)
                np.save(res_path / f"{subject}-{i+1}.npy", acc_matrix)
        
if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 20
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)