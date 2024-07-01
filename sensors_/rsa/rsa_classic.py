import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import ttest_1samp
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from base import *
from config import *
from tqdm.auto import tqdm

lock = 'stim'
path_data = DATA_DIR
figures = op.join(RESULTS_DIR, 'figures', lock, 'similarity')

subjects = SUBJS

epochs_list = EPOCHS

do_pca = True

all_in_seqs = list()
all_out_seqs = list()
# for trial_type in ['all', 'pattern', 'random']:
for trial_type in ['pattern']:
    all_in_seqs = list()
    all_out_seqs = list()
    for subject in subjects:
        # Read the behav file to get the sequence 
        behav_dir = RAW_DATA_DIR / subject / "behav_data"
        sequence = get_sequence(behav_dir)
        
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()

        for epoch_num, epo in enumerate(epochs_list):
            behav = pd.read_pickle(op.join(path_data, 'behav', f'{subject}_{epoch_num}.pkl'))
            epoch_fname = op.join(path_data, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
            epochs = mne.read_epochs(epoch_fname)
            times = epochs.times
            if do_pca:
                n_component = 30    
                pca = UnsupervisedSpatialFilter(PCA(n_component), average=False)
                pca_data = pca.fit_transform(epochs.get_data())
                sampling_freq = epochs.info['sfreq']
                info = mne.create_info(n_component, ch_types='mag', sfreq=sampling_freq)
                epochs = mne.EpochsArray(pca_data, info = info, events=epochs.events, event_id=epochs.event_id)
            if trial_type == 'pattern':
                one_pattern = epochs[np.where((behav['positions']==1) & (behav['trialtypes']==1))[0]].get_data().mean(axis=0)
                two_pattern = epochs[np.where((behav['positions']==2) & (behav['trialtypes']==1))[0]].get_data().mean(axis=0)
                three_pattern = epochs[np.where((behav['positions']==3) & (behav['trialtypes']==1))[0]].get_data().mean(axis=0)
                four_pattern = epochs[np.where((behav['positions']==4) & (behav['trialtypes']==1))[0]].get_data().mean(axis=0)
            elif trial_type == 'random':
                one_pattern = epochs[np.where((behav['positions']==1) & (behav['trialtypes']==2))[0]].get_data().mean(axis=0)
                two_pattern = epochs[np.where((behav['positions']==2) & (behav['trialtypes']==2))[0]].get_data().mean(axis=0)
                three_pattern = epochs[np.where((behav['positions']==3) & (behav['trialtypes']==2))[0]].get_data().mean(axis=0)
                four_pattern = epochs[np.where((behav['positions']==4) & (behav['trialtypes']==2))[0]].get_data().mean(axis=0)
            elif trial_type == 'all':
                one_pattern = epochs[np.where(behav['positions']==1)[0]].get_data().mean(axis=0)
                two_pattern = epochs[np.where(behav['positions']==2)[0]].get_data().mean(axis=0)
                three_pattern = epochs[np.where(behav['positions']==3)[0]].get_data().mean(axis=0)
                four_pattern = epochs[np.where(behav['positions']==4)[0]].get_data().mean(axis=0)
            assert one_pattern.shape == two_pattern.shape == three_pattern.shape == four_pattern.shape
            
            cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0)
            inv = np.linalg.inv(cov.data)

            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list() 
            three_four_similarity = list()
            for time_sample in range(epochs.get_data().shape[2]):
                one_two_similarity.append(mahalanobis(one_pattern[:, time_sample], two_pattern[:, time_sample], inv))
                one_three_similarity.append(mahalanobis(one_pattern[:, time_sample], three_pattern[:, time_sample], inv))
                one_four_similarity.append(mahalanobis(one_pattern[:, time_sample], four_pattern[:, time_sample], inv))
                two_three_similarity.append(mahalanobis(two_pattern[:, time_sample], three_pattern[:, time_sample], inv))
                two_four_similarity.append(mahalanobis(two_pattern[:, time_sample], four_pattern[:, time_sample], inv))
                three_four_similarity.append(mahalanobis(three_pattern[:, time_sample], four_pattern[:, time_sample], inv))
            one_two_similarity = np.array(one_two_similarity)
            one_three_similarity = np.array(one_three_similarity)
            one_four_similarity = np.array(one_four_similarity) 
            two_three_similarity = np.array(two_three_similarity)
            two_four_similarity = np.array(two_four_similarity) 
            three_four_similarity = np.array(three_four_similarity)

            ensure_dir(op.join(RESULTS_DIR, 'sensor_sims', subject, epo))
            # for sim, mis in zip([one_two_similarity, one_three_similarity, one_four_similarity, 
            #               two_three_similarity, two_four_similarity, three_four_similarity], 
            #                  ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']):
                # fname = op.join(RESULTS_DIR, 'sensor_sims', subject, epo, "%s.npy" % (mis))
                # np.save(fname, sim)
            
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

    # plot the similarity evolution for inseq and outseq
    plt.plot(times, all_in_seqs[:, :, 0, :].mean(axis=(0, 1)), label='practice', color='C1', alpha=0.3)
    plt.plot(times, all_in_seqs[:, :, 1, :].mean(axis=(0, 1)), label='first epoch', color='C1', alpha=0.5)
    plt.plot(times, all_in_seqs[:, :, 2, :].mean(axis=(0, 1)), label='second epoch', color='C1', alpha=0.6)
    plt.plot(times, all_in_seqs[:, :, 3, :].mean(axis=(0, 1)), label='third epoch', color='C1', alpha=0.7)
    plt.plot(times, all_in_seqs[:, :, 4, :].mean(axis=(0, 1)), label='fourth epoch', color='C1', alpha=0.8)
    plt.legend()
    plt.savefig(op.join(figures, 'inseq_sim_%s.png' % trial_type))
    plt.close()

    plt.plot(times, all_out_seqs[:, :, 1, :].mean(axis=(0, 1)), label='first epoch', color='C7', alpha=0.5)
    plt.plot(times, all_out_seqs[:, :, 0, :].mean(axis=(0, 1)), label='practice', color='C7', alpha=0.3)
    plt.plot(times, all_out_seqs[:, :, 2, :].mean(axis=(0, 1)), label='second epoch', color='C7', alpha=0.6)
    plt.plot(times, all_out_seqs[:, :, 3, :].mean(axis=(0, 1)), label='third epoch', color='C7', alpha=0.7)
    plt.plot(times, all_out_seqs[:, :, 4, :].mean(axis=(0, 1)), label='fourth epoch', color='C7', alpha=0.8)
    plt.legend()
    plt.savefig(op.join(figures, 'outseq_sim_%s.png' % trial_type))
    plt.close()

    # plot the similarity evolution (baseline by practice)
    bsl = all_in_seqs[:, :, 0, :].mean(axis=(0, 1))
    plt.plot(times, all_in_seqs[:, :, 1, :].mean(axis=(0, 1)) - bsl, label='first epoch', color='C1', alpha=0.5)
    plt.plot(times, all_in_seqs[:, :, 2, :].mean(axis=(0, 1)) - bsl, label='second epoch', color='C1', alpha=0.6)
    plt.plot(times, all_in_seqs[:, :, 3, :].mean(axis=(0, 1)) - bsl, label='third epoch', color='C1', alpha=0.7)
    plt.plot(times, all_in_seqs[:, :, 4, :].mean(axis=(0, 1)) - bsl, label='fourth epoch', color='C1', alpha=0.8)
    plt.legend()
    plt.savefig(op.join(figures, 'inseq_sim_bsl_%s.png' % trial_type))
    plt.close()

    bsl = all_out_seqs[:, :, 0, :].mean(axis=(0, 1))
    plt.plot(times, all_out_seqs[:, :, 1, :].mean(axis=(0, 1)) - bsl, label='first epoch', color='C1', alpha=0.5)
    plt.plot(times, all_out_seqs[:, :, 2, :].mean(axis=(0, 1)) - bsl, label='second epoch', color='C1', alpha=0.6)
    plt.plot(times, all_out_seqs[:, :, 3, :].mean(axis=(0, 1)) - bsl, label='third epoch', color='C1', alpha=0.7)
    plt.plot(times, all_out_seqs[:, :, 4, :].mean(axis=(0, 1)) - bsl, label='fourth epoch', color='C1', alpha=0.8)
    plt.legend()
    plt.savefig(op.join(figures, 'outseq_sim_bsl_%s.png' % trial_type))
    plt.close()

    # plot the difference in vs. out sequence
    plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    plt.plot(times, diff_inout[:, 1, :].mean(0), label='first epoch', color='C1', alpha=0.6)
    plt.plot(times, diff_inout[:, 2, :].mean(0), label='second epoch', color='C1', alpha=0.7)
    plt.plot(times, diff_inout[:, 3, :].mean(0), label='third epoch', color='C1', alpha=0.8)
    plt.plot(times, diff_inout[:, 4, :].mean(0), label='fourth epoch', color='C1', alpha=0.9)
    plt.legend()
    plt.savefig(op.join(figures, 'diff_inoutseq_%s.png' % trial_type))
    plt.close()

    # plot the difference in vs. out sequence averaging all epochs
    plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
    diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values_unc < 0.05
    p_values = decod_stats(diff)
    sig = p_values < 0.05
    plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C1', alpha=0.2)
    plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C1', alpha=0.3)
    plt.legend()
    plt.savefig(op.join(figures, 'diff_inoutseq_all_epochs_%s.png' % trial_type))
    plt.close()

    # plot the difference in vs. out sequence for each epoch
    for i in range(1, 5):
        plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
        plt.plot(times, diff_inout[:, i, :].mean(0), label='learning', color='C1', alpha=0.6)
        diff = diff_inout[:, i, :] - diff_inout[:, 0, :]
        p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
        sig_unc = p_values < 0.05
        p_values = decod_stats(diff)
        sig = p_values < 0.05
        plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig_unc, color='C1', alpha=0.2)
        plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig, color='C1', alpha=0.4, hatch='+')
        plt.legend()
        plt.savefig(op.join(figures, 'diff_inoutseq_all_epochs_%s_%s.png' % (trial_type, str(i))))
        plt.close()