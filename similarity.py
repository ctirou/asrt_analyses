import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis, euclidean, pdist, squareform
from scipy.stats import ttest_1samp
from itertools import cycle
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
import scipy.stats
import statsmodels.api as sm
from tqdm.auto import tqdm
from sklearn.covariance import LedoitWolf
from mne.beamformer import make_lcmv, apply_lcmv_epochs

method = 'lcmv'
lock = 'stim'
trial_type = 'pattern'

subjects_dir = op.join("./freesurfer/")

def decod_stats(X):
    from mne.stats import permutation_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='indices', n_permutations=2**12, n_jobs=-1,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)

raw_behavs ='./raws/behav'
data_path = './preprocessed'
res_path = './results'

if not op.exists(op.join(res_path, 'figures', lock, 'similarity')):
    os.makedirs(op.join(res_path, 'figures', lock, 'similarity'))

figures = './results/figures/stim/similarity'

subjects = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
            'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

epochs_list = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

do_pca = False
do_ols = False

# for lab in range(34):
for lab in range(1):
    
    all_in_seqs, all_out_seqs = [], []

    for subject in subjects:
        # Read the behav file to get the sequence 
        behav_dir = "./raws/%s/behav_data/" % (subject)
        behav_files = [f for f in os.listdir(behav_dir) if (not f.startswith('.') and ('_eASRT_Epoch_' in f))]
        behav = open(op.join(behav_dir, behav_files[0]), 'r')
        lines = behav.readlines()
        column_names = lines[0].split()
        sequence = list()
        for line in lines[1:]:
                trialtype = int(line.split()[column_names.index('trialtype')])
                if trialtype == 1:
                    sequence.append(int(line.split()[column_names.index('position')]))
                if len(sequence) == 4:
                    break
        # create lists of possible combinations between stimuli
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()
        # loop across sessions
        for epoch_num, epo in enumerate(epochs_list):
            # read epochs
            if lock == 'stim':
                epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
            else:
                epoch_fname = op.join(data_path, "%s/%s_%s_b-epo.fif" % (lock, subject, epoch_num))
            epoch = mne.read_epochs(epoch_fname)              
            # PCA
            if do_pca:
                n_component = 30    
                pca = UnsupervisedSpatialFilter(PCA(n_component), average=False)
                pca_data = pca.fit_transform(epochs.get_data())
                sampling_freq = epochs.info['sfreq']
                info = mne.create_info(n_component, ch_types='mag', sfreq=sampling_freq)
                epochs = mne.EpochsArray(pca_data, info = info, events=epochs.events, event_id=epochs.event_id)
                
            if epo == '2_PRACTICE':
                epo_fname = 'prac'
            else:
                epo_fname = 'sess-%s' % (str(epoch_num).zfill(2))
            stc_fname_1 = op.join("./results/stcs/%s/%s/%s_%s_1-lh.stc" % (method, lock, subject, epo_fname))
            stc_fname_2 = op.join("./results/stcs/%s/%s/%s_%s_2-lh.stc" % (method, lock, subject, epo_fname))
            stc_fname_3 = op.join("./results/stcs/%s/%s/%s_%s_3-lh.stc" % (method, lock, subject, epo_fname))
            stc_fname_4 = op.join("./results/stcs/%s/%s/%s_%s_4-lh.stc" % (method, lock, subject, epo_fname))
            one_pattern = mne.read_source_estimate(stc_fname_1, subject=subject)
            two_pattern = mne.read_source_estimate(stc_fname_2, subject=subject)
            three_pattern = mne.read_source_estimate(stc_fname_3, subject=subject)
            four_pattern = mne.read_source_estimate(stc_fname_4, subject=subject)
            
            # compute cov matrix and inverse for mahalanobis
            # cov = mne.compute_covariance(epoch, tmin=-0.2, tmax=0)
            # inv = np.linalg.inv(cov.data)
            
            # extract label time courses (activation of a particular region)
            labels = mne.read_labels_from_annot(subject=subject, parc='aparc', hemi='lh', subjects_dir=subjects_dir)
            label = labels[lab]
            times  = epoch.times
                    
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list() 
            three_four_similarity = list()
            
            # cov = np.cov((one_pattern.data + two_pattern.data + three_pattern.data + four_pattern.data)/4.)
            # inv = np.linalg.inv(cov)
                    
            # for label in labels[3:4]:
                
            ts1 = one_pattern.in_label(label)
            ts2 = two_pattern.in_label(label)
            ts3 = three_pattern.in_label(label)
            ts4 = four_pattern.in_label(label)
            
            for time_sample in range(epoch._data.shape[2]):
                # mahalanobis
                # one_two_similarity.append(mahalanobis(one_pattern.data[:, time_sample], two_pattern.data[:, time_sample], inv))
                # one_three_similarity.append(mahalanobis(one_pattern.data[:, time_sample], three_pattern.data[:, time_sample], inv))
                # one_four_similarity.append(mahalanobis(one_pattern.data[:, time_sample], four_pattern.data[:, time_sample], inv))
                # two_three_similarity.append(mahalanobis(two_pattern.data[:, time_sample], three_pattern.data[:, time_sample], inv))
                # two_four_similarity.append(mahalanobis(two_pattern.data[:, time_sample], four_pattern.data[:, time_sample], inv))
                # three_four_similarity.append(mahalanobis(three_pattern.data[:, time_sample], four_pattern.data[:, time_sample], inv))
                # euclidean
                one_two_similarity.append(euclidean(ts1.data[:, time_sample], ts2.data[:, time_sample]))
                one_three_similarity.append(euclidean(ts1.data[:, time_sample], ts3.data[:, time_sample]))
                one_four_similarity.append(euclidean(ts1.data[:, time_sample], ts4.data[:, time_sample]))
                two_three_similarity.append(euclidean(ts2.data[:, time_sample], ts3.data[:, time_sample]))
                two_four_similarity.append(euclidean(ts2.data[:, time_sample], ts4.data[:, time_sample]))
                three_four_similarity.append(euclidean(ts3.data[:, time_sample], ts4.data[:, time_sample]))
                
            one_two_similarity = np.array(one_two_similarity)
            one_three_similarity = np.array(one_three_similarity)
            one_four_similarity = np.array(one_four_similarity) 
            two_three_similarity = np.array(two_three_similarity)
            two_four_similarity = np.array(two_four_similarity) 
            three_four_similarity = np.array(three_four_similarity)

            one_two_similarities.append(one_two_similarity)
            one_three_similarities.append(one_three_similarity)
            one_four_similarities.append(one_four_similarity) 
            two_three_similarities.append(two_three_similarity)
            two_four_similarities.append(two_four_similarity) 
            three_four_similarities.append(three_four_similarity)
            
    one_two_similarities = np.array(one_two_similarities)
    one_three_similarities = np.array(one_three_similarities)  
    one_four_similarities = np.array(one_four_similarities)   
    two_three_similarities = np.array(two_three_similarities)  
    two_four_similarities = np.array(two_four_similarities)   
    three_four_similarities = np.array(three_four_similarities) 

    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))

    in_seq = list()
    out_seq = list()
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
    similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
                    two_three_similarities, two_four_similarities, three_four_similarities]
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    all_in_seqs.append(np.array(in_seq))
    all_out_seqs.append(np.array(out_seq))

    all_in_seqs = np.array(all_in_seqs)
    all_out_seqs = np.array(all_out_seqs)

    diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
    times = epoch.times

    # plot the difference in vs. out sequence averaging all epochs
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    # plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
    # plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
    # plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
    # plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
    plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
    # plt.plot(times, diff_inout[:, 0:, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
    diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
    # diff = diff_inout[:, 3, :] - diff_inout[:, 0, :]
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(diff)
    sig = p_values < 0.05
    # plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C2', alpha=0.2)
    plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C3', alpha=0.3)
    plt.legend()
    plt.show()
    # plt.savefig(op.join(figures, 'all_epochs_%s_%s.png' % (trial_type, label.name)))
    # plt.close()