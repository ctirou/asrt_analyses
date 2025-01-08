import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import ensure_dir, get_volume_estimate_tc
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import gc

# params
subjects, epochs_list = SUBJS, EPOCHS

lock = "stim" # "stim", "button"
trial_type = 'all' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR

ensure_dir(res_path)
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
verbose = 'error'
jobs = -1
parc = 'aparc'
overwrite = False

# set-up the classifier and cv structure
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=True)
cv = StratifiedKFold(folds, shuffle=True)

vertices_df = dict()

for subject in subjects:
    # read behav file
    beh_dir = data_path / 'behav'
    beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
    all_beh = [pd.read_pickle(fname) for fname in beh_fnames]
    behav = pd.concat(all_beh)
    # read source space file
    src_fname = op.join(res_path, "src", "%s-src.fif" % (subject))
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    # path to bem file
    bem_fname = op.join(res_path, "bem", "%s-bem-sol.fif" % (subject))    
    # to store info on vertices
    sub_vertices_info = dict()

    # for hemi in ['lh', 'rh', 'others']:
    for hemi in ['lh', 'rh']:
        # create mixed source space
        vol_src_fname = op.join(res_path, "src", "%s-%s-vol-src.fif" % (subject, hemi))
        vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)
        mixed_src = src + vol_src
        all_stcs = []
        
        for epoch_num, epo in enumerate(epochs_list):
            # read epoch
            epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
            epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=False)
            if lock == 'button': 
                epoch_bsl_fname = data_path / 'bsl' / f'{subject}-{epoch_num}-epo.fif'
                epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose, preload=False)
                # compute noise covariance
                noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
            else:
                noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=epoch.info, rank=None, tol_kind='relative', verbose=verbose)
            # path to trans file
            trans_fname = os.path.join(res_path, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
            # compute forward solution
            fwd_fname = op.join(res_path, "fwd", lock, f"{subject}-{epoch_num}-fwd.fif")
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=mixed_src, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=verbose)
            # compute source estimates
            # filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
            #                 pick_ori='max-power', weight_norm='nai', rank=rank, reduce_rank=True, verbose=verbose)
            filters = make_lcmv(epoch.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)

            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
            all_stcs.extend(stcs)
            del rank, trans_fname, noise_cov, data_cov, filters, epoch, stcs
            gc.collect()
                        
        ##### see activation time course #####
        # labels_cx = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose) # issue with hemi here, when others
        # lab_tc = mne.extract_label_time_course(stcs, labels_cx, mixed_src, mode='mean')
        # lab_tc = np.array(lab_tc)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for i in range(34, 50):
        #     ax.plot(times, lab_tc[:, i, :].mean(0), label=all_labels[i])
        # ax.legend()
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.plot(times, lab_tc[:, 40, :].mean(0), label='thalamus')
        # ax.plot(times, lab_tc[:, 3, :].mean(0), label='cuneus')
        # ax.legend()
        # plt.show()
        ##### see activation time course #####   
        
        offsets = np.cumsum([0] + [len(s["vertno"]) for s in vol_src])
        label_tc, vertices_info = get_volume_estimate_tc(all_stcs, fwd, offsets, subject, subjects_dir)
        # subcortex labels
        # labels_subcx = list(label_tc.keys())
        labels_subcx = [label for label in label_tc.keys() if label in ['Hippocampus-lh', 'Hippocampus-rh', 'Thalamus-Proper-lh', 'Thalamus-Proper-rh']]
        sub_vertices_info.update(vertices_info)
        del vol_src, mixed_src, fwd, offsets
        gc.collect()
        
        if labels_subcx: 
            for ilabel, label in enumerate(labels_subcx):
                print(subject, lock, trial_type, hemi, f"{str(ilabel+1).zfill(2)}/{len(labels_subcx)}", label) 
                # results dir
                res_dir = res_path / 'source' / lock / trial_type / label
                ensure_dir(res_dir)
                if not os.path.exists(res_dir / f"{subject}-scores.npy") or overwrite:
                    if trial_type == 'pattern':    
                        pattern = behav.trialtypes == 1
                        X = label_tc[label][pattern]
                        y = behav.positions[pattern]
                    elif trial_type == 'random':
                        random = behav.trialtypes == 2
                        X = label_tc[label][random]
                        y = behav.positions[random]
                    else:
                        X = label_tc[label]
                        y = behav.positions
                    y = y.reset_index(drop=True)            
                    assert X.shape[0] == y.shape[0]
                    if X.shape[1] < 1:
                        continue            
                    scores = cross_val_multiscore(clf, X, y, cv=cv, verbose=True)
                    np.save(res_dir / f"{subject}-scores.npy", scores.mean(0))
                    del X, y, scores
                    gc.collect()
        
        del label_tc, labels_subcx
        gc.collect()
    
    vertices_df[subject] = dict(sorted(sub_vertices_info.items()))

info_df = pd.DataFrame() 
for subject, vertices_info in vertices_df.items():
    for key, value in vertices_info.items():
        # If the key is not in the DataFrame, add it with the current subject's value
        if key not in info_df.index:
            info_df.loc[key, subject] = value
        else:
            info_df.at[key, subject] = value
info_df.replace(0, np.nan, inplace=True)
# Export the DataFrame to a CSV file
info_df.to_csv(res_path / f'subcx_vertices_info_{trial_type}.csv', sep="\t")            