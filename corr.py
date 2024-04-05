import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_1samp, spearmanr
import scipy.stats
import statsmodels.api as sm
from tqdm.auto import tqdm
from sklearn.covariance import LedoitWolf
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from base import *
from config import *

method = 'lcmv'
lock = 'stim'
trial_type = 'pattern'

data_path = DATA_DIR
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
subjects, epochs_list = SUBJS, EPOCHS

figures_dir = res_path / 'figures' / lock / 'similarity'
ensure_dir(figures_dir)

for lab in tqdm(range(34)):
    
    all_in_seqs, all_out_seqs = [], []
    
    for subject in subjects:
        # Read the behav file to get the sequence 
        behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject))
        sequence = get_sequence(behav_dir) 
        # create lists of possible combinations between stimuli
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()
        
        bem_fname = os.path.join(res_path, "bem", "%s-bem.fif" % (subject))
        src_fname = op.join(res_path, "src", "%s-src.fif" % subject)
        src = mne.read_source_spaces(src_fname)

        # loop across sessions
        for epoch_num, epo in enumerate(epochs_list):
        
            if epo == '2_PRACTICE':
                epo_fname = 'prac'
            else:
                epo_fname = 'sess-%s' % (str(epoch_num).zfill(2))
        
            behav_fname = op.join(data_path, "behav/%s_%s.pkl" % (subject, epoch_num))
            behav = pd.read_pickle(behav_fname)
            # read epochs
            if lock == 'button': 
                epoch_bsl_fname = op.join(data_path, "bsl/%s_%s_bl-epo.fif" % (subject, epoch_num))
                epoch_bsl = mne.read_epochs(epoch_bsl_fname)
                epoch_fname = op.join(data_path, "%s/%s_%s_b-epo.fif" % (lock, subject, epoch_num))
            else:
                epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
            epoch = mne.read_epochs(epoch_fname)

            trans_fname = os.path.join(res_path, "trans", lock,  "%s-trans-%s.fif" % (subject, epoch_num))
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=src, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=1)
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info")
            # compute noise covariance
            if lock == 'button':
                noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info")
            else:
                noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info")
            info = epoch.info
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative')
            # compute source estimate
            filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True)
            stcs = apply_lcmv_epochs(epoch, filters=filters)

            labels = mne.read_labels_from_annot(subject=subject, parc='aparc', hemi='lh', subjects_dir=subjects_dir)
            label = labels[lab]
            stcs_data = list()
            for stc in stcs:
                stcs_data.append(stc.in_label(label).data)
            stcs_data = np.array(stcs_data)
            assert len(stcs_data) == len(behav)
            
            del fwd

            behav = behav.reset_index()
            behav_pat = behav[behav["trialtypes"]==1]
            pat_data = stcs_data[np.where(behav["trialtypes"]==1)[0]]
            assert len(pat_data) == len(behav_pat)
        
            stcs_data, behav = pat_data, behav_pat
            
            # Prepare the design matrix                        
            ntrials = len(stcs_data)
            nconditions = 4
            design_matrix = np.zeros((ntrials, nconditions))
            
            for icondi, condi in enumerate(behav["positions"]):            
                # assert isinstance(condi, np.int64) 
                design_matrix[icondi, condi-1] = 1

            # Run OLS
            # z-score the data
            meg_data_V = stcs_data
            _, nvertex, ntimes = meg_data_V.shape
            meg_data_V = scipy.stats.zscore(meg_data_V, axis=0)

            coefs = np.zeros((nconditions, nvertex, ntimes))
            resids = np.zeros_like(meg_data_V)
            for ivertex in tqdm(range(nvertex)):
                for itime in range(ntimes):
                    y = meg_data_V[:, ivertex, itime]
                    
                    model = sm.OLS(endog=y, exog=design_matrix, missing="raise")
                    results = model.fit()
                    
                    coefs[:, ivertex, itime] = results.params
                    resids[:, ivertex, itime] = results.resid
            
            # Calculate pairwise mahalanobis distance between regression coefficients        
            rdm_times = np.zeros((nconditions, nconditions, ntimes))
            for itime in tqdm(range(ntimes)):
                response = coefs[:, :, itime]
                residuals = resids[:, :, itime]
                
                # Estimate covariance from residuals
                lw_shrinkage = LedoitWolf(assume_centered=True)
                cov = lw_shrinkage.fit(residuals)
                
                # Compute pairwise mahalanobis distances
                VI = np.linalg.inv(cov.covariance_) # covariance matrix needed for mahalonobis
                rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
                assert ~np.isnan(rdm).all()
                rdm_times[:, :, itime] = rdm
            
            rdm_dir = op.join(RESULTS_DIR, "rdms", "source", subject, label.name)
            ensure_dir(rdm_dir)
            rdm_fname = "rdm_%s.npy" % str(epoch_num)
            np.save(op.join(rdm_dir, rdm_fname), rdm_times)
            
            coefs_dir = op.join(RESULTS_DIR, "coefs", "source", subject, label.name)
            ensure_dir(coefs_dir)
            coefs_fname = "coefs_%s.npy" % str(epoch_num)
            np.save(op.join(coefs_dir, coefs_fname), coefs)
            response_fname = "response_%s.npy" % str(epoch_num)
            np.save(op.join(coefs_dir, response_fname), response)
            
            resids_dir = op.join(RESULTS_DIR, "resids", "source", subject, label.name)
            ensure_dir(resids_dir)
            resids_fname = "resids_%s.npy" % str(epoch_num)
            np.save(op.join(resids_dir, resids_fname), resids)
            residuals_fname = "residuals_%s.npy" % str(epoch_num)
            np.save(op.join(resids_dir, residuals_fname), residuals)
            
            rdmx = rdm_times
            
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list()
            three_four_similarity = list()
                        
            for itime in range(rdmx.shape[2]):
                one_two_similarity.append(rdmx[1, 0, itime])
                one_three_similarity.append(rdmx[2, 0, itime])
                one_four_similarity.append(rdmx[3, 0, itime])
                two_three_similarity.append(rdmx[1, 2, itime])
                two_four_similarity.append(rdmx[1, 3, itime])
                three_four_similarity.append(rdmx[2, 3, itime])
                            
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

    # with sns.plotting_context("talk"): 
    # plot the difference in vs. out sequence averaging all epochs
    # plt.figure(figsize=(12.8, 7.2))
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    # plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
    # diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
    # p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    # sig_unc = p_values_unc < 0.05
    # p_values = decod_stats(diff)
    # sig = p_values < 0.05
    # plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C2', alpha=0.2)
    # plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C3', alpha=0.3)
    # plt.legend()
    # # plt.show()
    # plt.savefig(figures_dir / 'all_epochs_%s_%s.png' % (trial_type, label.name))
    # plt.close()

    # plot per sessions
    # plt.figure(figsize=(12.8, 7.2))
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    # plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
    # plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
    # plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
    # plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
    # plt.legend()
    # plt.savefig(figures_dir / 'per_block_%s_%s.png' % (trial_type, label.name))
    # plt.close()

    all_rhos = []
    for sub in tqdm(range(len(subjects))):
        rhos = []
        for t in range(len(times)):
            rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)

    # plot rhos per subject
    for isub, sub in enumerate(subjects):
        plt.subplots(1, 1, figsize=(16, 7))
        plt.plot(times, all_rhos[isub, :, 0], label="rho")
        sig = all_rhos[isub, :, 1] < 0.05 # not good for small sample size
        plt.fill_between(times, 0, all_rhos[isub, :, 0], where=sig, alpha=0.3)
        plt.legend()
        plt.ylim(-1.5, 1.5)
        plt.axhline(y = -1, color='k', ls="dashed")
        plt.axhline(y = 1, color='k', ls="dashed")
        plt.axhline(y = 0, color='k')
        plt.title(sub)
        plt.savefig(figures_dir / f'{sub}.png')
        plt.close()

    # plot average rho across subjects
    plt.subplots(1, 1, figsize=(16, 7))
    plt.plot(times, all_rhos.mean(0)[:, 0], label='rho')
    diff = all_rhos[:, :, 0]
    p_values = decod_stats(diff)
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig = p_values < 0.05
    sig_unc = p_values_unc < 0.05
    plt.fill_between(times, 0, all_rhos.mean(0)[:, 0], where=sig_unc, color='C2', alpha=1)
    plt.fill_between(times, 0, all_rhos.mean(0)[:, 0], where=sig, alpha=0.3)
    plt.ylim(-1, 1)
    plt.axhline(y = 0, color='k')
    plt.legend()
    plt.title('mean')
    plt.savefig(figures_dir / 'mean.png')
    plt.close()