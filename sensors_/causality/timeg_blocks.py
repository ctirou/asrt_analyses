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

    for trial_type in ['pattern', 'random']:
        
        all_Xtraining, all_Xtesting = [], []
        all_ytraining, all_ytesting = [], []

        for epoch_num in [0, 1, 2, 3, 4]:
            res_path = ensured(data_path / 'results' / 'sensors' / lock / f"split_{trial_type}")
            
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch_gen = read_epochs(epoch_fname, verbose="error", preload=True)
            assert len(behav) == len(epoch_gen)
            
            times = epoch_gen.times
            idx = np.where(times >= -1.5)[0]
            blocks = np.unique(behav["blocks"])
            filter = (behav.trialtypes == 1) if trial_type == 'pattern' else (behav.trialtypes == 2)
            
            Xtraining, Xtesting, ytraining, ytesting = [], [], [], []
            for block in blocks:
                
                this_block = behav.blocks == block
                out_blocks = behav.blocks != block
                
                X_train = epoch_gen.get_data(copy=False)[out_blocks & filter][:, :, idx]
                y_train = behav[out_blocks & filter].positions
                y_train = y_train.reset_index(drop=True)
                
                X_test = epoch_gen.get_data(copy=False)[this_block & filter][:, :, idx]
                y_test = behav[this_block & filter].positions
                y_test = y_test.reset_index(drop=True)
                                            
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
            clf.fit(Xtraining, ytraining)
            
            for i, _ in enumerate(Xtesting):
                if not op.exists(res_path / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    ypred = clf.predict(Xtesting[i])
                    print("Scoring...")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytesting[i], x), 0, ypred)
                    np.save(res_path / f"{subject}-{epoch_num}-{i+1}.npy", acc_matrix)
            
            del epoch_gen, behav
            gc.collect()
        
        res_path = ensured(data_path / 'results' / 'sensors' / lock / f"split_all_{trial_type}")
        
        all_Xtraining = np.concatenate(all_Xtraining)
        all_ytraining = np.concatenate(all_ytraining)
        clf.fit(all_Xtraining, all_ytraining)
        
        for i, _ in enumerate(all_Xtesting):
            if not op.exists(res_path / f"{subject}-{i+1}.npy") or overwrite:
                ypred = clf.predict(all_Xtesting[i])
                print("Scoring...")
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting[i], x), 0, ypred)
                np.save(res_path / f"{subject}-{i+1}.npy", acc_matrix)

if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    jobs = 15
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
        