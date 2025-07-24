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
from base import *
from config import *
from joblib import Parallel, delayed

data_path = DATA_DIR / 'for_timeg_new'
subjects = SUBJS15
lock = 'stim'
solver = 'lbfgs'
scoring = "accuracy"
verbose = True
overwrite = False

mne.set_log_level(verbose)

networks = NETWORKS[:-2]

pick_ori = 'vector'
# pick_ori = 'max-power'

analysis = 'scores_lobo_vector' if pick_ori == 'vector' else 'scores_lobo_maxpower'
analysis += '_new'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    # define classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = GeneralizingEstimator(clf, scoring=scoring, n_jobs=jobs)
    # network and custom label_names
    label_path = RESULTS_DIR / 'networks_200_7' / subject
        
    all_epochs, all_behavs = [], []
    for epoch_num in range(5):
        # read behav        
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname).reset_index(drop=True)
        behav['sessions'] = epoch_num
        all_behavs.append(behav)
        # read epochs
        epoch_fname = op.join(data_path, "epochs", "%s-%s-epo.fif" % (subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose).crop(-1.5, 1.5)
        all_epochs.append(epoch)
    # concatenate all epochs and behavs
    behav = pd.concat(all_behavs, ignore_index=True)
    behav['trials'] = behav.index
    for epo in all_epochs:
        epo.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epochs)
    assert len(epoch) == len(behav)
    
    del all_epochs, all_behavs
    gc.collect()

    # read forward solution
    fwd_fname = RESULTS_DIR / "fwd" / "for_timeg" / f"{subject}-all-fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

    # rename blocks columns
    behav.loc[behav.sessions != 0, 'blocks'] += 3
    blocks = np.unique(behav["blocks"])        

    for network in networks:
        
        print(f"Processing subject {subject} in network {network} with pick_ori={pick_ori}")
        
        # read labels
        lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')
        res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject)
        
        for block in blocks:
            block = int(block)

            if not op.exists(res_path / f"rand-{block}.npy") or overwrite:
                Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                    'random', block, blocks, verbose=verbose)
                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_path / f"rand-{block}.npy", acc_matrix)
                del Xtrain, ytrain, Xtest, ytest
                gc.collect()
            else:
                print(f"Random for {subject} epoch {epoch_num} block {block} in {network} already exists")

            if not op.exists(res_path / f"pat-{block}.npy") or overwrite:
                Xtrain, ytrain, Xtest, ytest = get_train_test_blocks_net(epoch, fwd, behav, pick_ori, lh_label, rh_label, \
                    'pattern', block, blocks, verbose=verbose)
                clf.fit(Xtrain, ytrain)
                acc_matrix = clf.score(Xtest, ytest)
                np.save(res_path / f"pat-{block}.npy", acc_matrix)
                del Xtrain, ytrain, Xtest, ytest
                gc.collect()
            else:
                print(f"Pattern for {subject} epoch {epoch_num} block {block} in {network} already exists")
            
    del epoch, fwd
    gc.collect()
            
if is_cluster:
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = 1
    Parallel(-1)(delayed(process_subject)(subject, jobs) for subject in subjects)