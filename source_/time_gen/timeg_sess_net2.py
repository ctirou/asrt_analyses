import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys
from sklearn.metrics import accuracy_score as acc


# params
subjects = SUBJS15
data_path = DATA_DIR / 'for_timeg'
subjects_dir = FREESURFER_DIR

solver = 'lbfgs'
scoring = "accuracy"
folds = 10
verbose = True
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

networks = NETWORKS[:-2]

def process_subject(subject, jobs):
    # define classifier'
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state=42)

    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject    
    
    all_behavs = list()
    all_epochs = list()
    all_stcs = list()
        
    for epoch_num in [0, 1, 2, 3, 4]:
        # read behav
        behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
        # read epoch
        epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
                        
        # read forward solution
        fwd_fname = RESULTS_DIR / "fwd" / 'for_timeg' / f"{subject}-{epoch_num}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
        
        pattern = behav[behav.trialtypes == 1].reset_index(drop=True)
        pattern_epochs = epoch[pattern.index]
        
        random = behav[behav.trialtypes == 2].reset_index(drop=True)
        random_epochs = epoch[random.index]
                
        for network in networks:
            # read labels
            lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
            res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / "scores_skf" / subject)
            
            # random trials
            acc_matrices = list()
            for i, (train_idx, test_idx) in enumerate(sss.split(random_epochs, random.positions)):

                print(f"Processing {subject} epoch {epoch_num} random {network} split {i+1}")
                
                # get training data
                noise_cov = mne.compute_covariance(random_epochs[train_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                data_cov = mne.compute_covariance(random_epochs[train_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=random_epochs[train_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(random_epochs[train_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori='max-power', weight_norm="unit-noise-gain",
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs_train = apply_lcmv_epochs(random_epochs[train_idx], filters=filters, verbose=verbose)
                Xtrain = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_train])
                ytrain = random.positions[train_idx].reset_index(drop=True)
                assert Xtrain.shape[0] == ytrain.shape[0], "Length mismatch"
                                
                # get testing data
                noise_cov = mne.compute_covariance(random_epochs[test_idx], tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
                data_cov = mne.compute_covariance(random_epochs[test_idx], method="empirical", rank="info", verbose=verbose)
                rank = mne.compute_rank(data_cov, info=random_epochs[test_idx].info, rank=None, tol_kind='relative', verbose=verbose)
                filters = make_lcmv(random_epochs[test_idx].info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                                    pick_ori='max-power', weight_norm="unit-noise-gain",
                                    rank=rank, reduce_rank=True, verbose=verbose)
                stcs_test = apply_lcmv_epochs(random_epochs[test_idx], filters=filters, verbose=verbose)
                Xtest = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_test])
                ytest = random.positions[test_idx].reset_index(drop=True)
                assert Xtest.shape[0] == ytest.shape[0], "Length mismatch"                
                
                if not os.path.exists(res_path / f"rand-{epoch_num}.npy") or overwrite:
                    clf.fit(Xtrain, ytrain)
                    ypred = clf.predict(Xtest)
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    acc_matrices.append(acc_matrix)

            acc_matrices = np.array(acc_matrices).mean(0)
            
            times = random_epochs.times
            import matplotlib.pyplot as plt
            plt.imshow(acc_matrices, cmap='viridis', aspect='auto', extent=[times[0], times[-1], times[-1], times[0]])
            plt.axhline(0, color='k', linestyle='-')
            plt.axvline(0, color='k', linestyle='-')
            plt.axvline(-0.2, color='k', linestyle='--')
            plt.show()
            
            plt.axhline(0.25, color='k', linestyle='-')
            plt.axvspan(0, 0.2, color='grey', alpha=0.1)
            plt.plot(times, np.diag(acc_matrices))
            plt.show()
    
    behav_df = pd.concat(all_behavs)
    del all_behavs
    gc.collect()
    
    for network in networks:
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in all_stcs])
        behav_data = behav_df.reset_index(drop=True)
        assert len(stcs_data) == len(behav_data), "Shape mismatch"
        
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / "scores_skf" / subject)

        if not op.exists(res_path / "pat-all.npy") or overwrite:
            print("Processing", subject, 'all', "pattern", network)
            pattern = behav_data.trialtypes == 1
            X = stcs_data[pattern]
            y = behav_data.positions[pattern]
            y = y.reset_index(drop=True)
            assert X.shape[0] == y.shape[0]
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=True)
            np.save(op.join(res_path, "pat-all.npy"), scores.mean(0))
            del X, y, scores
            gc.collect()
        else:
            print("Skipping", subject, 'all', "pattern", network)          

        if not op.exists(res_path / "rand-all.npy") or overwrite:
            print("Processing", subject, 'all', "random", network)
            random = behav_data.trialtypes == 2
            X = stcs_data[random]
            y = behav_data.positions[random]
            y = y.reset_index(drop=True)
            assert X.shape[0] == y.shape[0]
            cv = loo if any(np.unique(y, return_counts=True)[1] < 10) else skf
            scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=jobs, verbose=True)
            np.save(op.join(res_path, "rand-all.npy"), scores.mean(0))
            del X, y, scores
            gc.collect()
        else:
            print("Skipping", subject, 'all', "random", network)          

        del stcs_data, behav_data
        gc.collect()
        
    del all_stcs, behav_df
    gc.collect()
        
if is_cluster:
    jobs = 20
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    for subject in subjects:
        process_subject(subject, jobs)