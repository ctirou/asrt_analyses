import os
import sys
import os.path as op
import pandas as pd
import numpy as np
import gc
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from base import ensured, get_train_test_blocks_net
from config import *
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg'
subjects = SUBJS15
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
overwrite = False

mne.use_log_level(verbose)

networks = NETWORKS[:-2]

# pick_ori = sys.argv[1]
pick_ori = 'vector'
# pick_ori = 'max-power'

# use_resting = sys.argv[2]
use_resting = False

analysis = 'scores_blocks' + '_maxp' if pick_ori == 'max-power' else 'scores_blocks' + '_vect'
analysis += '_rs' if use_resting == 'True' else '_0200'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject

    for network in networks:
        
        print(f"Processing subject {subject} in network {network} with pick_ori={pick_ori} and use_resting={use_resting}")
        
        # read labels
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)

        for epoch_num in [0, 1, 2, 3, 4]:
            
            # read behav
            behav = pd.read_pickle(op.join(data_path, 'behav', f'{subject}-{epoch_num}.pkl')).reset_index(drop=True)
            behav['trials'] = behav.index
            # read epoch
            epoch_fname = op.join(data_path, 'epochs', f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True).crop(-1.5, 1.5)
            
            # read forward solution
            fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-{epoch_num}-fwd.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
                                
            blocks = np.unique(behav["blocks"])        
            
            for block in blocks:
                block = int(block)
                
                if not op.exists(res_path / f"rand-{epoch_num}-{block}.npy") or overwrite:
                    Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(data_path, subject, epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                        'random', block, use_resting, verbose=verbose)
                    clf.fit(Xtrain, ytrain)
                    ypred = clf.predict(Xtest)
                    print(f"Scoring random for {subject} epoch {epoch_num} block {block}")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    np.save(res_path / f"rand-{epoch_num}-{block}.npy", acc_matrix)
                    del Xtrain, ytrain, Xtest, ytest
                    gc.collect()
                else:
                    print(f"Random for {subject} epoch {epoch_num} block {block} already exists")
                
                if not op.exists(res_path / f"pat-{epoch_num}-{block}.npy") or overwrite:
                    Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(data_path, subject, epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                        'pattern', block, use_resting, verbose=verbose)
                    clf.fit(Xtrain, ytrain)
                    ypred = clf.predict(Xtest)
                    print(f"Scoring pattern for {subject} epoch {epoch_num} block {block}")
                    acc_matrix = np.apply_along_axis(lambda x: acc(ytest, x), 0, ypred)
                    
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.imshow(acc_matrix, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=0.5)
                    plt.colorbar(label='Accuracy', extend='both')
                    plt.clim(0, 0.5)
                    plt.title(f'Accuracy Matrix for {subject} epoch {epoch_num} block {block}')
                    plt.show()
                    
                    np.save(res_path / f"pat-{epoch_num}-{block}.npy", acc_matrix)
                    del Xtrain, ytrain, Xtest, ytest
                    gc.collect()
                else:
                    print(f"Pattern for {subject} epoch {epoch_num} block {block} already exists")
                
            del epoch, fwd
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
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)