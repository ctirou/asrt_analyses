import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import gc
import mne
from mne.decoding import GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from base import ensured, get_volume_estimate_tc
from config import *
from joblib import Parallel, delayed

data_path = TIMEG_DATA_DIR
subjects = SUBJS
subjects_dir = FREESURFER_DIR

lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = data_path / 'results' / 'source' / 'max-power'

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    kf = KFold(n_splits=4, shuffle=False)

    # read volume source space
    vol_src_fname =  data_path / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)

    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    
    del vol_src
    gc.collect()

    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
        
        res_path = ensured(data_path / 'results' / 'source' / 'max-power' / region / lock / "kf2_no_shfl" / subject)
                
        for epoch_num in [0, 1, 2, 3, 4]:
            
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl')).reset_index(drop=True)
            behav['trials'] = behav.index
            
            # read epoch
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            big_epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
                            
            filter = behav.trialtypes == 2
            noise_epoch = big_epoch[filter]
            noise_cov = mne.compute_covariance(noise_epoch, tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
            epoch = big_epoch.copy().crop(-1.5, 1.5)
            
            del big_epoch, noise_epoch
            gc.collect()
            
            # compute data covariance matrix
            data_cov = mne.compute_covariance(epoch, method="empirical", rank="info", verbose=verbose)
            # conpute rank
            rank = mne.compute_rank(data_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)

            # compute forward solution
            fwd_fname = data_path / "fwd" / f"{subject}-htc-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            
            # compute source estimates
            filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='max-power', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                    
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            
            # get data from volume source space
            label_tc, _ = get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir)
            
            # get data from region of interest
            labels = [label for label in label_tc.keys() if region in label]
            data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works

            del epoch, noise_cov, data_cov, fwd, filters, stcs
            gc.collect()
            
            blocks = np.unique(behav["blocks"])
            for block in blocks:
                block = int(block)

                pat = behav.trialtypes == 1
                this_block = behav.blocks == block
                pat_this_block = pat & this_block
                ypat = behav[pat_this_block]
                
                for i, (_, test_index) in enumerate(kf.split(ypat)):
                    
                    test_in_ypat = ypat.iloc[test_index].trials.values
                                    
                    test_idx = [i for i in behav.trials.values if i in test_in_ypat]
                    train_idx = [i for i in behav.trials.values if i not in test_in_ypat]
                    
                    ytrain = [behav.iloc[i].positions.as_int() for i in train_idx]
                    ytest = [behav.iloc[i].positions.as_int() for i in test_idx]
                    
                    Xtrain = data[train_idx]
                    Xtest = data[test_idx]
                    
                    assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
                    assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
                    
                    clf.fit(Xtrain, ytrain)
                    if not op.exists(res_path / f"pat-{epoch_num}-{block}-{i+1}.npy") or overwrite:
                        ypred = clf.predict(Xtest)
                        print(f"Scoring pattern split {i+1} for {subject} epoch {epoch_num}")
                        acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                        np.save(res_path / f"pat-{epoch_num}-{block}-{i+1}.npy", acc_matrix)
                    else:
                        print(f"Pattern split {i+1} for {subject} epoch {epoch_num} block {block} already exists")

                rand = behav.trialtypes == 2
                rand_this_block = rand & this_block
                yrand = behav[rand_this_block]
                
                for i, (_, test_index) in enumerate(kf.split(yrand)):
                    test_in_yrand = yrand.iloc[test_index].trials.values
                    
                    test_idx = [i for i in behav.trials.values if i in test_in_yrand]
                    train_idx = [i for i in behav.trials.values if i not in test_in_yrand]
                    
                    ytrain = [behav.iloc[i].positions for i in train_idx]
                    ytest = [behav.iloc[i].positions for i in test_idx]
                    
                    Xtrain = data[train_idx]
                    Xtest = data[test_idx]
                    
                    assert len(Xtrain) == len(ytrain), "Xtrain and ytrain lengths do not match"
                    assert len(Xtest) == len(ytest), "Xtest and ytest lengths do not match"
                    
                    clf.fit(Xtrain, ytrain)
                    if not op.exists(res_path / f"rand-{epoch_num}-{block}-{i+1}.npy") or overwrite:
                        ypred = clf.predict(Xtest)
                        print(f"Scoring random split {i+1} for {subject} epoch {epoch_num}")
                        acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                        np.save(res_path / f"rand-{epoch_num}-{block}-{i+1}.npy", acc_matrix)
                    else:
                        print(f"Random split {i+1} for {subject} epoch {epoch_num} block {block} already exists")
                    
            del data, behav
            gc.collect()
            
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