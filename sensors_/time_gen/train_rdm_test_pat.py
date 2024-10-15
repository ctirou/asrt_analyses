import os.path as op
import numpy as np
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import *
from config import *
import pandas as pd
import gc

# stim disp = 500 ms
# RSI = 750 ms in task
analysis = 'pat_bsl_filtered'
data_path = TIMEG_DATA_DIR / analysis
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
hemi = 'both'
parc = 'aparc'
jobs = -1
verbose = True
res_path = data_path / 'results' / 'sensors' / 'train_rdm_test_pat'
ensure_dir(res_path)

# define classifier
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
cv = StratifiedKFold(folds, shuffle=True)

def reset_all():
    for name in dir():
        if not name.startswith('__'):
            del globals()[name]
    
# for lock in ['stim', 'button']: 
for lock in ['stim']: 
    
    ensure_dir(res_path / lock / 'cv_scores')
    ensure_dir(res_path / lock / 'scores')
    
    for subject in subjects:

        all_epochs, all_behavs = [], []

        # for epoch_num, epo, score_list, scoring_list in zip([1, 2, 3, 4], epochs_list[1:], [scores_1, scores_2, scores_3, scores_4], [scoring_1, scoring_2, scoring_3, scoring_4]):
        for epoch_num, epo in zip([1, 2, 3, 4], epochs_list[1:]):
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl'))
            all_behavs.append(behav)
            # read epoch
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
            all_epochs.append(epoch)
            
            pattern = behav.trialtypes == 1
            random = behav.trialtypes == 2
            
            Xrand = epoch.get_data()[random]
            yrand = np.array(behav.positions[random])
            assert Xrand.shape[0] == yrand.shape[0]
            
            Xpat = epoch.get_data()[pattern]
            ypat = np.array(behav.positions[pattern])
            assert Xpat.shape[0] == ypat.shape[0]
            
            del behav, epoch_fname, epoch, pattern, random
            gc.collect()
            
            # method 1
            clf.fit(Xrand, yrand)
            if not op.exists(res_path / lock / 'scores' / f"{subject}-{epoch_num}.npy"):
                scores = clf.score(Xpat, ypat)
                np.save(res_path / lock / 'scores' / f"{subject}-{epoch_num}.npy", scores)            
                del scores
                gc.collect()
            
            # method 2
            if not op.exists(res_path / lock / 'cv_scores' / f"{subject}-{epoch_num}.npy"):
                cv_scores = list()
                for train, test in cv.split(Xpat, ypat):
                    cv_scores.append(np.array(clf.score(Xpat[test], ypat[test])))
                cv_scores = np.array(cv_scores)
                np.save(res_path / lock / 'cv_scores' / f"{subject}-{epoch_num}.npy", cv_scores.mean(0))
                del cv_scores
                gc.collect()
            
            del Xrand, yrand, Xpat, ypat
            gc.collect()

        behavs = pd.concat(all_behavs)
        for epoch in all_epochs: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
            epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
        # concatenate epochs
        epochs = mne.concatenate_epochs(all_epochs)
        
        del all_behavs, all_epochs
        gc.collect()
        
        pattern = behavs.trialtypes == 1
        random = behavs.trialtypes == 2
        
        Xrand = epochs.get_data()[random]
        yrand = np.array(behavs.positions[random])
        assert Xrand.shape[0] == yrand.shape[0]
        
        Xpat = epochs.get_data()[pattern]
        ypat = np.array(behavs.positions[pattern])
        assert Xpat.shape[0] == ypat.shape[0]
        
        del behavs, epochs
        gc.collect()
        
        clf.fit(Xrand, yrand)
        
        # method 1
        if not op.exists(res_path / lock / 'scores' / f"{subject}-all.npy"):
            scores = clf.score(Xpat, ypat)
            np.save(res_path / lock / 'scores' / f"{subject}-all.npy", scores)
            del scores
            gc.collect()
        
        # method 2
        if not op.exists(res_path / lock / 'cv_scores' / f"{subject}-all.npy"):
            cv_scores = list()
            for train, test in cv.split(Xpat, ypat):
                cv_scores.append(np.array(clf.score(Xpat[test], ypat[test])))
            cv_scores = np.array(cv_scores)
            np.save(res_path / lock / 'cv_scores' / f"{subject}-all.npy", cv_scores.mean(0))
            del cv_scores
            gc.collect()
        
        del Xrand, yrand, Xpat, ypat
        gc.collect()