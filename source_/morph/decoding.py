import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.decoding import SlidingEstimator, cross_val_multiscore, UnsupervisedSpatialFilter
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys
from joblib import Parallel, delayed


# params
subjects = SUBJS
analysis = 'RSA'
lock = 'stim'
trial_type = 'all' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
parc = 'aparc'
hemi = 'both'

solver = 'lbfgs'
scoring = "accuracy"
folds = 10

verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None
networks = NETWORKS[:-2]

def process_subject(subject, lock, trial_type, network, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    pca = UnsupervisedSpatialFilter(PCA(1000), average=False)
    # define networks labels path
    label_path = RESULTS_DIR / 'networks_200_7' / 'fsaverage2'
    
    # for network in networks:
    all_behavs = list()
    all_stcs = list()
    # define results path
    res_dir = RESULTS_DIR / "RSA" / "source" / network / lock / "power_morphed_scores" / trial_type
    ensure_dir(res_dir)
    
    lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
    
    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # fname_stcs = RESULTS_DIR / 'morphed_power_stc' / lock / f"{subject}-morphed-stcs-{epoch_num}.npy"
        fname_stcs = RESULTS_DIR / 'power_stc' / lock / f"{subject}-{epoch_num}.npy"
        stcs_data = np.load(fname_stcs, allow_pickle=True)
        
        # Check if imaginary components are present
        max_imag = np.max([np.abs(stc.data.imag).max() for stc in stcs_data])
        threshold = 1e-10  # Define a small threshold

        if max_imag > threshold:
            raise ValueError(f"Processing, {subject}, {network}, {epoch_num}\nSignificant imaginary components detected (max {max_imag}). Check beamformer parameters.")
        # else:
            # print(f"Processing, {subject}, {network}, {epoch_num}\nMax imaginary component {max_imag} is below threshold {threshold}.")

        morphed_stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_data])

        # morphed_stcs_data = np.array([stc.in_label(lh_label + rh_label).data for stc in stcs_data])
        data = pca.fit_transform(morphed_stcs_data)

        all_stcs.extend(stcs_data)

        # del stcs_data
        # gc.collect()

        # print("Processing", subject, epoch_num, trial_type, network)
        
        # assert len(morphed_stcs_data) == len(behav)
        
        # if not os.path.exists(res_dir / f"{subject}-{epoch_num}-scores.npy") or overwrite:
        #     if trial_type == 'pattern':
        #         pattern = behav.trialtypes == 1
        #         X = data[pattern]
        #         y = behav.positions[pattern]
        #     elif trial_type == 'random':
        #         random = behav.trialtypes == 2
        #         X = data[random]
        #         y = behav.positions[random]
        #     else:
        #         X = data
        #         y = behav.positions    
        #     y = y.reset_index(drop=True)            
        #     assert X.shape[0] == y.shape[0]
        #     scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs)   
        #     np.save(op.join(res_dir, f"{subject}-{epoch_num}-scores.npy"), scores.mean(0))
            
        #     del X, y, scores
        #     gc.collect()
    
        # append epochs
        all_behavs.append(behav)
        
        # del data, behav
        # if trial_type == 'button':
        #     del epoch_bsl
        # gc.collect()

    behav_df = pd.concat(all_behavs)
    del all_behavs
    gc.collect()
    
    print("Processing", subject, 'all', trial_type, network)
    
    # morphed_stcs_data = np.array([stc.in_label(lh_label + rh_label).data for stc in all_stcs])
    morphed_stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
    data = pca.fit_transform(morphed_stcs_data)
    behav_data = behav_df.reset_index(drop=True)
    assert len(data) == len(behav_data)
    
    if not op.exists(res_dir / f"{subject}-all-scores.npy") or overwrite:
        if trial_type == 'pattern':
            pattern = behav_data.trialtypes == 1
            X = data[pattern]
            y = behav_data.positions[pattern]
        elif trial_type == 'random':
            random = behav_data.trialtypes == 2
            X = data[random]
            y = behav_data.positions[random]
        else:
            X = data
            y = behav_data.positions
        y = y.reset_index(drop=True)
        assert X.shape[0] == y.shape[0]
        del data, behav_data
        gc.collect()
        scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs)
        np.save(op.join(res_dir, f"{subject}-all-scores.npy"), scores.mean(0))
        del X, y, scores
        gc.collect()
    
    del behav_df, all_stcs
    gc.collect()
    
if is_cluster:
    lock = str(sys.argv[1])
    trial_type = str(sys.argv[2])
    jobs = 20
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock, trial_type, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    lock = 'stim'
    trial_type = 'pattern'
    for network in networks[1:]:
        for subject in subjects:
            for trial_type in ['pattern', 'random']:
                process_subject(subject, lock, trial_type, network, jobs=-1)
        # Parallel(-1)(delayed(process_subject)(subject, lock, trial_type, networks[0], jobs=-1) for trial_type in ['pattern', 'random'])
