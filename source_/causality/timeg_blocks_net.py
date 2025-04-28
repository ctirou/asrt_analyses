import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import gc
import mne
from mne import read_epochs
from mne.decoding import cross_val_multiscore, GeneralizingEstimator
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc
from base import ensure_dir
from config import *
from joblib import Parallel, delayed

# data_path = TIMEG_DATA_DIR / 'gen44'
data_path = TIMEG_DATA_DIR
subjects = SUBJS
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

networks = NETWORKS[:-2]

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

res_path = data_path / 'results' / 'source' / 'max-power'

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject    

    for network in networks:
        
        # read labels
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        
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
            
            # read forward solution
            fwd_fname = TIMEG_DATA_DIR / "fwd" / f"{subject}-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
                    
            # compute source estimates
            filters = make_lcmv(epoch.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                pick_ori='max-power', weight_norm="unit-noise-gain",
                                rank=rank, reduce_rank=True, verbose=verbose)
                    
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

            del epoch, noise_cov, data_cov, fwd, filters
            gc.collect()

            stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs])
            assert len(stcs_data) == len(behav), "Length mismatch"
            
            del stcs
            gc.collect()
            
            blocks = np.unique(behav["blocks"])
            
            Xtraining_pat, Xtesting_pat, ytraining_pat, ytesting_pat = [], [], [], []
            Xtraining_rand, Xtesting_rand, ytraining_rand, ytesting_rand = [], [], [], []
                        
            for block in blocks:
                
                print("Spliting pattern on epoch", epoch_num, "block", block, network)
                pattern = (behav.trialtypes == 1) & (behav.blocks == block)                
                X = stcs_data[pattern]
                y = behav.positions[pattern]
                y = y.reset_index(drop=True)
                assert len(X) == len(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                Xtraining_pat.append(X_train)
                Xtesting_pat.append(X_test)
                ytraining_pat.append(y_train)
                ytesting_pat.append(y_test)
                
                if epoch_num != 0:                
                    all_Xtraining_pat.append(X_train)
                    all_Xtesting_pat.append(X_test)
                    all_ytesting_pat.append(y_test)
                    all_ytraining_pat.append(y_train)
                
                print("Spliting random on epoch", epoch_num, "block", block, network)
                random =  (behav.trialtypes == 2) & (behav.blocks == block)
                X = stcs_data[random]
                y = behav.positions[random]
                y = y.reset_index(drop=True)
                assert len(X) == len(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                Xtraining_rand.append(X_train)
                Xtesting_rand.append(X_test)
                ytraining_rand.append(y_train)
                ytesting_rand.append(y_test)
                
                if epoch_num != 0:                
                    all_Xtraining_rand.append(X_train)
                    all_Xtesting_rand.append(X_test)
                    all_ytesting_rand.append(y_test)
                    all_ytraining_rand.append(y_train)
            
                del stcs_data, behav
                gc.collect()
            
            # Fit on epoch data and test on block - pattern
            res_dir = res_path / network / "split_pattern"
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
            res_dir = res_path / network / "split_random"
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
        print("Fitting pattern...")    
        res_dir = res_path / network / "split_all_pattern"
        ensure_dir(res_dir)
        all_Xtraining_pat = np.concatenate(all_Xtraining_pat)
        all_ytraining_pat = np.concatenate(all_ytraining_pat)
        assert len(all_Xtraining_pat) == len(all_ytraining_pat), "Length mismatch in pattern"
        clf.fit(all_Xtraining_pat, all_ytraining_pat)
        for block in range(23):
            if not op.exists(res_dir / f"{subject}-{block+1}.npy") or overwrite:
                print("Scoring pattern on block", block, network)
                ypred = clf.predict(all_Xtesting_pat[block])
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting_pat[block], x), 0, ypred)
                np.save(res_dir / f"{subject}-{block+1}.npy", acc_matrix)
        
        # Train on all data and test on block - random
        print("Fitting random...")
        res_dir = res_path / network / "split_all_random"
        ensure_dir(res_dir)
        all_Xtraining_rand = np.concatenate(all_Xtraining_rand)
        all_ytraining_rand = np.concatenate(all_ytraining_rand)
        assert len(all_Xtraining_rand) == len(all_ytraining_rand), "Length mismatch in random"
        clf.fit(all_Xtraining_rand, all_ytraining_rand)
        for block in range(23):
            if not op.exists(res_dir / f"{subject}-{block+1}.npy") or overwrite:
                print("Scoring random on block", block+1, network)
                ypred = clf.predict(all_Xtesting_rand[block])
                acc_matrix = np.apply_along_axis(lambda x: acc(all_ytesting_rand[block], x), 0, ypred)
                np.save(res_dir / f"{subject}-{block+1}.npy", acc_matrix)

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
        