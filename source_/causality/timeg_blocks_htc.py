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
overwrite = True

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = data_path / 'results' / 'source' / 'max-power'

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)

    # read volume source space
    vol_src_fname =  data_path / 'src' / f"{subject}-htc-vol-src.fif"
    vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)

    offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src]) # need vol src here, fwd["src"] is mixed so does not work
    
    del vol_src
    gc.collect()

    for region in ['Hippocampus', 'Thalamus', 'Cerebellum-Cortex']:
                
        all_Xtraining_pat, all_Xtesting_pat = [], []
        all_ytraining_pat, all_ytesting_pat = [], []

        all_Xtraining_rand, all_Xtesting_rand = [], []
        all_ytraining_rand, all_ytesting_rand = [], []

        for epoch_num in [0, 1, 2, 3, 4]:
            
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
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
            stcs_data = np.concatenate([np.real(label_tc[label]) for label in labels], axis=1) # this works

            del epoch, noise_cov, data_cov, fwd, filters, stcs
            gc.collect()
            
            blocks = np.unique(behav["blocks"])
            
            Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat = [], [], [], []
            Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand = [], [], [], []
            
            for block in blocks:
                
                good = behav.blocks == block
                bad = behav.blocks != block
                
                pattern = behav.trialtypes == 1                
                X_train = stcs_data[good & pattern]
                y_train = behav[good & pattern].positions
                y_train = y_train.reset_index(drop=True)
                
                X_test = stcs_data[bad & pattern]
                y_test = behav[bad & pattern].positions
                y_test = y_test.reset_index(drop=True)
                                            
                Xtraining_pat.append(X_train)
                Xtesting_pat.append(X_test)
                ytraining_pat.append(y_train)
                ytesting_pat.append(y_test)
                
                if epoch_num != 0:
                    all_Xtraining_pat.append(X_train)
                    all_Xtesting_pat.append(X_test)
                    all_ytraining_pat.append(y_train)
                    all_ytesting_pat.append(y_test)
                
                random = behav.trialtypes == 2                
                X_train = stcs_data[good & random]
                y_train = behav[good & random].positions
                y_train = y_train.reset_index(drop=True)
                
                X_test = stcs_data[bad & random]
                y_test = behav[bad & random].positions
                y_test = y_test.reset_index(drop=True)
                                            
                Xtraining_rand.append(X_train)
                Xtesting_rand.append(X_test)
                ytraining_rand.append(y_train)
                ytesting_rand.append(y_test)
                
                if epoch_num != 0:
                    all_Xtraining_rand.append(X_train)
                    all_Xtesting_rand.append(X_test)
                    all_ytraining_rand.append(y_train)
                    all_ytesting_rand.append(y_test)
            
            del stcs_data, behav
            gc.collect()
            
            # Fit on epoch data and test on block - pattern
            print("Fitting on epoch data and testing on block - pattern")
            res_dir = ensured(res_path / region / "split_pattern")
            Xtraining_pat = np.concatenate(Xtraining_pat)
            ytraining_pat = np.concatenate(ytraining_pat)            
            clf.fit(Xtraining_pat, ytraining_pat)
            for i, _ in enumerate(Xtesting_pat):
                if not op.exists(res_dir / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    ypred = clf.predict(Xtesting_pat[i])
                    print("Scoring...")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytesting_pat[i], x), 0, ypred)
                    np.save(res_dir / f"{subject}-{epoch_num}-{i+1}.npy", acc_matrix)

            # Fit on epoch data and test on block - random
            print("Fitting on epoch data and testing on block - random")
            res_dir = ensured(res_path / region / "split_random")
            Xtraining_rand = np.concatenate(Xtraining_rand)
            ytraining_rand = np.concatenate(ytraining_rand)
            clf.fit(Xtraining_rand, ytraining_rand)
            for i, _ in enumerate(Xtesting_rand):
                if not op.exists(res_dir / f"{subject}-{epoch_num}-{i+1}.npy") or overwrite:
                    ypred = clf.predict(Xtesting_rand[i])
                    print("Scoring...")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytesting_rand[i], x), 0, ypred)
                    np.save(res_dir / f"{subject}-{epoch_num}-{i+1}.npy", acc_matrix)
        
        # Train on all data and test on block - pattern
        print("Fitting on all data and testing on block - pattern")
        res_dir = ensured(res_path / region / "split_all_pattern")
        all_Xtraining_pat = np.concatenate(all_Xtraining_pat)
        all_ytraining_pat = np.concatenate(all_ytraining_pat)
        assert len(all_Xtraining_pat) == len(all_ytraining_pat), "Length mismatch in pattern"
        clf.fit(all_Xtraining_pat, all_ytraining_pat)
        
        for i, _ in enumerate(all_Xtesting_pat):
            if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
                print("Scoring pattern on block", i, region)
                ypred = clf.predict(all_Xtesting_pat[i])
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting_pat[i], x), 0, ypred)
                np.save(res_dir / f"{subject}-{i+1}.npy", acc_matrix)
        
        # Train on all data and test on block - random
        print("Fitting on all data and testing on block - random")
        res_dir = ensured(res_path / region / "split_all_random")
        all_Xtraining_rand = np.concatenate(all_Xtraining_rand)
        all_ytraining_rand = np.concatenate(all_ytraining_rand)
        assert len(all_Xtraining_rand) == len(all_ytraining_rand), "Length mismatch in random"
        clf.fit(all_Xtraining_rand, all_ytraining_rand)
        
        for i, _ in enumerate(all_Xtesting_rand):
            if not op.exists(res_dir / f"{subject}-{i+1}.npy") or overwrite:
                print("Scoring random on block", i+1, region)
                ypred = clf.predict(all_Xtesting_rand[i])
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting_rand[i], x), 0, ypred)
                np.save(res_dir / f"{subject}-{i+1}.npy", acc_matrix)

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
    lock = 'stim'
    jobs = 15
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)
        