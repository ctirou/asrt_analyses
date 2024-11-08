import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from jr.gat import scorer_spearman
from base import *
from config import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict
import gc

# params
subjects = SUBJS

analysis = 'RSA'
lock = "stim"
trial_type = "pattern"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
res_path = RESULTS_DIR
parc = 'aparc'
hemi = 'both'
verbose = "error"
jobs = -1

# figures dir
res_dir = res_path / analysis / 'source' / lock
ensure_dir(res_dir)
# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times

# get label names
# best_regions = [6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 42, 43, 50, 51, 58, 59]
best = SURFACE_LABELS if lock == 'stim' else SURFACE_LABELS_RT
labels = mne.read_labels_from_annot(subject='sub01', parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels if label.name in best]

del epochs, labels
gc.collect()

combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']
for subject in subjects:
    
    # get labels
    all_labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    labels = [label for label in all_labels if label.name in best]
    # to store dissimilarity distances
    rsa_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    del all_labels
    gc.collect()
    
    src_fname = res_path / "src" / f"{subject}-src.fif"
    src = mne.read_source_spaces(src_fname, verbose=verbose)
    bem_fname = res_path / "bem" / f"{subject}-bem-sol.fif"
    
    # Read the behav file to get the sequence 
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
    sequence = get_sequence(behav_dir)
        
    # create lists of possible combinations between stimuli
    one_twos_pat = list()
    one_threes_pat = list()
    one_fours_pat = list() 
    two_threes_pat = list()
    two_fours_pat = list() 
    three_fours_pat = list()

    one_twos_rand = list()
    one_threes_rand = list()
    one_fours_rand = list() 
    two_threes_rand = list()
    two_fours_rand = list() 
    three_fours_rand = list()
    
    for epoch_num in [1, 2, 3, 4]:
        
        # read stim epoch
        epoch_fname = data_path / lock / f"{subject}-{epoch_num}-epo.fif"
        epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
        # read behav
        behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
        behav = pd.read_pickle(behav_fname).reset_index()

        if lock == 'button': 
            epoch_bsl_fname = data_path / "bsl" / f"{subject}-{epoch_num}-epo.fif"
            epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
        # read forward solution    
        # fwd_fname = res_path / "fwd" / lock / f"{subject}-{epoch_num}-fwd.fif"
        # fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
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
        trans_fname = os.path.join(res_path, "trans", lock, "%s-all-trans.fif" % (subject))
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
        
        del epoch, epoch_fname, behav_fname, fwd, data_cov, noise_cov, rank, info, filters
        gc.collect()

        # loop across labels
        for ilabel, label in enumerate(labels):
            print(f"{str(ilabel+1).zfill(2)}/{len(labels)}", subject, epoch_num, label.name)            
            # get stcs in label
            stcs_data = [stc.in_label(label).data for stc in stcs]
            stcs_data = np.array(stcs_data)
            assert len(stcs_data) == len(behav)
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
            assert X.shape[0] == y.shape[0]
            
            if not op.exists(res_path / f"pat-{epoch_num}.npy"):
                epoch_pat = epoch[np.where(behav["trialtypes"]==1)].get_data(copy=False).mean(axis=0)
                behav_pat = behav[behav["trialtypes"]==1]
                assert len(epoch_pat) == len(behav_pat)
                rdm_pat = get_rdm(epoch_pat, behav_pat)
                np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
            else:
                rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
            
            if not op.exists(res_path / f"rand-{epoch_num}.npy"):
                epoch_rand = epoch[np.where(behav["triplets"]==34)].get_data(copy=False).mean(axis=0)
                behav_rand = behav[behav["triplets"]==34]
                assert len(epoch_rand) == len(behav_rand)
                rdm_rand = get_rdm(epoch_rand, behav_rand)
                np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
            else:
                rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")

                one_two_pat = list()
                one_three_pat = list()
                one_four_pat = list() 
                two_three_pat = list()
                two_four_pat = list()
                three_four_pat = list()

                one_two_rand = list()
                one_three_rand = list()
                one_four_rand = list() 
                two_three_rand = list()
                two_four_rand = list()
                three_four_rand = list()

                for itime in range(rdm_pat.shape[2]):
                    one_two_pat.append(rdm_pat[0, 1, itime])
                    one_three_pat.append(rdm_pat[0, 2, itime])
                    one_four_pat.append(rdm_pat[0, 3, itime])
                    two_three_pat.append(rdm_pat[1, 2, itime])
                    two_four_pat.append(rdm_pat[1, 3, itime])
                    three_four_pat.append(rdm_pat[2, 3, itime])

                    one_two_rand.append(rdm_rand[0, 1, itime])
                    one_three_rand.append(rdm_rand[0, 2, itime])
                    one_four_rand.append(rdm_rand[0, 3, itime])
                    two_three_rand.append(rdm_rand[1, 2, itime])
                    two_four_rand.append(rdm_rand[1, 3, itime])
                    three_four_rand.append(rdm_rand[2, 3, itime])
                                
            similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                            two_three_similarity, two_four_similarity, three_four_similarity]
            
            for combi, similarity in zip(combinations, similarities):
                rsa_dict[label.name][session_id][combi].append(similarity)
                
            del X, y, data, model, rdm_times, rdmx, rdm, similarities
            del one_two_similarity, one_three_similarity, one_four_similarity, two_three_similarity, two_four_similarity, three_four_similarity
            gc.collect()
        
        del stcs, stcs_data
        gc.collect()
            
    time_points = range(len(times))
    
    # RSA dataframe
    index = pd.MultiIndex.from_product([label_names, range(len(sessions)), combinations], names=['label', 'session', 'similarities'])
    rsa_df = pd.DataFrame(index=index, columns=time_points)
    for label in labels:
        for session_id in range(len(sessions)):
            for isim, similarity in enumerate(combinations):
                rsa_list = rsa_dict[label.name][session_id][similarity]
                if rsa_list:
                    rsa_scores = np.mean(rsa_list, axis=0)
                    rsa_df.loc[(label.name, session_id, similarity), :] = rsa_scores.flatten()
    rsa_df.to_hdf(res_dir / f"{subject}_rsa.h5", key='rsa', mode="w")
    
    del labels, rsa_dict, index, rsa_df
    gc.collect()