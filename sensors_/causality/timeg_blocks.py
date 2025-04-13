import os
import sys
from config import *
from joblib import Parallel, delayed
import os.path as op
import pandas as pd
import numpy as np
import gc
from base import ensure_dir
from mne import read_epochs
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc

# data_path = TIMEG_DATA_DIR / 'gen44'
data_path = TIMEG_DATA_DIR
subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs):
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    # loo = LeaveOneOut()
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for trial_type in ['pattern', 'random']:
        
        all_Xtraining, all_Xtesting = [], []
        all_ytraining, all_ytesting = [], []

        for epoch_num in [0, 1, 2, 3, 4]:
            res_path = data_path / 'results' / 'sensors' / lock / f"split_all_{trial_type}"
            ensure_dir(res_path)
            
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch_gen = read_epochs(epoch_fname, verbose="error", preload=True)
            assert len(behav) == len(epoch_gen)
            
            times = epoch_gen.times
            idx = np.where(times >= -1.5)[0]
            blocks = np.unique(behav["blocks"])
            
            Xtraining, Xtesting, ytraining, ytesting = [], [], [], []
            for block in blocks:
                
                filter = (behav.trialtypes == 1) & (behav.blocks == block) if trial_type == 'pattern' \
                    else (behav.trialtypes == 2) & (behav.blocks == block)
                X = epoch_gen.get_data()[filter][:, :, idx]
                y = behav.positions[filter]
                y = y.reset_index(drop=True)
                assert len(X) == len(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                Xtraining.append(X_train)
                Xtesting.append(X_test)
                ytraining.append(y_train)
                ytesting.append(y_test)
                
                all_Xtraining.append(X_train)
                all_Xtesting.append(X_test)
                all_ytesting.append(y_test)
                all_ytraining.append(y_train)
                
        
            Xtraining = np.concatenate(Xtraining)
            ytraining = np.concatenate(ytraining)            
            # clf.fit(Xtraining, ytraining)
            
            # for i in range(len(blocks)):
            #     if not op.exists(res_path / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
            #         ypred = clf.predict(Xtesting[i])
            #         print("Scoring...")
            #         acc_matrix = np.apply_along_axis(lambda x: acc(ytesting[i], x), 0, ypred)
            #         np.save(res_path / f"{subject}-{epoch_num}-{i+1}.npy", acc_matrix)
            
            del epoch_gen, behav
            gc.collect()
            
        all_Xtraining = np.concatenate(all_Xtraining)
        all_ytraining = np.concatenate(all_ytraining)
        clf.fit(all_Xtraining, all_ytraining)
        
        for block in range(23):
            if not op.exists(res_path / f"{subject}-{block+1}.npy") or overwrite:
                ypred = clf.predict(all_Xtesting[block])
                print("Scoring...")
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting[block], x), 0, ypred)
                np.save(res_path / f"{subject}-{block+1}.npy", acc_matrix)

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = 20
        process_subject(subject, lock, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    jobs = 15
    Parallel(-1)(delayed(process_subject)(subject, lock, jobs) for subject in subjects)
        