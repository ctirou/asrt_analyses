import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys

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

verbose = 'error'
overwrite = True
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, lock, trialtype, jobs, rsync):
    
    # set-up the classifier and cv structure
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
    cv = StratifiedKFold(folds, shuffle=True)
    
    # network and custom label_names
    n_parcels = 200
    n_networks = 7
    # networks = (NEW_LABELS + schaefer_7) if n_networks == 7 else (NEW_LABELS + schaefer_17)
    networks = schaefer_7 if n_networks == 7 else schaefer_17
    
    label_path = RESULTS_DIR / f'networks_{n_parcels}_{n_networks}' / subject
        
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epo)

    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)

    if lock == 'button': 
        bsl_data = data_path / "bsl"
        epoch_bsl_fnames = [bsl_data / f"{f}" for f in sorted(os.listdir(bsl_data)) if ".fif" in f and subject in f]
        all_bsl = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epoch_bsl_fnames]
        for epo in all_bsl:
            epo.info['dev_head_t'] = all_epo[0].info['dev_head_t']
        epoch_bsl = mne.concatenate_epochs(all_bsl)

    # read forward solution    
    # fwd_fname = res_path / analysis / "fwd" / lock / f"{subject}-mixed-lh-fwd.fif"
    # fwd = mne.read_forward_solution(fwd_fname, ordered=False, verbose=verbose)
    # labels = mne.get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
    
    src_fname = RESULTS_DIR / "src" / f"{subject}-src.fif"
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    bem_fname = RESULTS_DIR / "bem" / f"{subject}-bem-sol.fif"    

    # compute data covariance matrix on evoked data
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
    # compute noise covariance
    if lock == 'button':
        noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
    else:
        noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
    info = epoch.info
    # conpute rank
    rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)
    trans_fname = os.path.join(RESULTS_DIR, "trans", lock, "%s-all-trans.fif" % (subject))

    # compute forward solution
    fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                src=src, bem=bem_fname,
                                meg=True, eeg=False,
                                mindist=5.0,
                                n_jobs=jobs,
                                verbose=verbose)

    # compute source estimates
    filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                    pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
    stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)

    for inetwork, network in enumerate(networks):
        
        res_path = RESULTS_DIR / "RSA" / 'source' / f'networks_{n_parcels}_{n_networks}' / network / lock / 'scores' / subject
        ensure_dir(res_path)
    
        print("Processing", subject, lock, trial_type, network)
        
        # parc = f'aparc.{network}' if network not in networks[-n_networks:] else f"Shaefer2018_{str(n_parcels)}_{str(n_networks)}.{network}"
        # parc = f"Shaefer2018_{str(n_parcels)}_{str(n_networks)}.{network}"
        hemi = 'lh' if 'left' in network else 'rh' if 'right' in network else 'both'
        # labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
        parc = f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks"
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, regexp=network, verbose=verbose, sort=True)        
        
        lh_label = mne.read_label(label_path / f'{network}-lh.label')
        rh_label = mne.read_label(label_path / f'{network}-rh.label')
        
        stcs_data = [stc.in_label(lh_label + rh_label).data for stc in stcs]
        stcs_data = np.array(stcs_data)
        
        # all_labels = []    
        # # loop across labels
        # for ilabel, label in enumerate(labels):
        #     print(f"{str(ilabel+1).zfill(2)}/{len(labels)}", subject, epoch_num, label.name)
            # res_path = RESULTS_DIR / analysis / 'source' / label.name / lock / 'rdm' / subject
        #     # get stcs in label
        #     stcs_data = [stc.in_label(label).data for stc in stcs]
        #     stcs_data = np.array(stcs_data)
        #     all_labels.append(stcs_data)
        #     # assert len(stcs_data) == len(behav)
        # all_labels = np.array(all_labels)

        if not op.exists(res_path / f"{trial_type}-scores.npy") or overwrite:
            if trial_type == 'pattern':
                pattern = behav.trialtypes == 1
                X = stcs_data[pattern]
                y = behav.positions[pattern]
            elif trial_type == 'random':
                random = behav.trialtypes == 2
                X = stcs_data[random]
                y = behav.positions[random]
            else:
                X = stcs_data
                y = behav.positions
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]

            del stcs_data
            gc.collect()

            scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=verbose)
            np.save(res_path / f"{trial_type}-scores.npy", scores.mean(0))
                
            del X, y, scores
            gc.collect()
    
    del labels, stcs
    gc.collect()
    
    if rsync:
        source = RESULTS_DIR / analysis / 'source'
        destination = "coum@crnl-dycog369.crnl.local:/Users/coum/Desktop/asrt/results/RSA/"
        options = "-av"
        rsync_files(source, destination, options)
    
if is_cluster:
    lock = str(sys.argv[1])
    trial_type = str(sys.argv[2])
    jobs = 10
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, lock, trial_type, jobs, rsync=True)
    
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, lock, trial_type, jobs=-1, rsync=False)