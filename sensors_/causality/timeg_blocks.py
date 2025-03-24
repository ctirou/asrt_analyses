import os
import sys
from config import *
from joblib import Parallel, delayed

data_path = TIMEG_DATA_DIR / 'gen44'
subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, jobs):
    import os.path as op
    import pandas as pd
    import numpy as np
    import gc
    from base import ensure_dir
    from mne import read_epochs
    from mne.decoding import cross_val_multiscore, GeneralizingEstimator
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    loo = LeaveOneOut()

    for trial_type in ['pattern', 'random']:
        for epoch_num in [0, 1, 2, 3, 4]:
            res_path = data_path / 'results' / 'sensors' / lock / f"loocv_{trial_type}"
            ensure_dir(res_path)
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch_gen = read_epochs(epoch_fname, verbose="error", preload=True)
            
            times = epoch_gen.times
            idx = np.where(times >= -1.5)[0]
            blocks = np.unique(behav["blocks"])
                        
            for block in blocks:
                if not op.exists(res_path / f"{subject}-{epoch_num}-{block}.npy") or overwrite:
                    filter = (behav.trialtypes == 1) & (behav.blocks == block) if trial_type == 'pattern' \
                        else (behav.trialtypes == 2) & (behav.blocks == block)
                    X = epoch_gen.get_data()[filter][:, :, idx]
                    y = behav.positions[filter]
                    y = y.reset_index(drop=True)
                    assert len(X) == len(y)
                    scores = cross_val_multiscore(clf, X, y, cv=loo, verbose=verbose)
                    np.save(res_path / f"{subject}-{epoch_num}-{block}.npy", scores.mean(0))
              
            del epoch_gen, behav
            gc.collect()

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
    jobs = -1
    Parallel(-1)(delayed(process_subject)(subject, lock, jobs) for subject in subjects)
        