import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from mne.decoding import UnsupervisedSpatialFilter
import scipy.stats
import statsmodels.api as sm
from tqdm.auto import tqdm
from sklearn.covariance import LedoitWolf
from base import *
from config import RAW_DATA_DIR, DATA_DIR, RESULTS_DIR, FREESURFER_DIR, SUBJS, EPOCHS

lock = 'stim'
trial_type = 'pattern'

data_path = DATA_DIR
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
subjects, epochs_list = SUBJS, EPOCHS

figures = op.join(res_path, 'figures', lock, 'similarity')
ensure_dir(figures)

all_in_seqs, all_out_seqs = [], []

for subject in subjects:
    
    # all_in_seqs, all_out_seqs = [], []
    
    # Read the behav file to get the sequence 
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
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
    # random.shuffle(sequence)
        
    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()
    
    # loop across sessions
    for epoch_num, epo in enumerate(epochs_list):
    # for epoch_num, epo in zip([2], epochs_list):
                    
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
        times = epoch.times              
        
        epoch_pat = epoch[np.where(behav["trialtypes"]==1)].get_data().mean(axis=0)
        behav_pat = behav[behav["trialtypes"]==1]
        assert len(epoch_pat) == len(behav_pat)
        
        epoch, behav = epoch_pat, behav_pat
        
        # Prepare the design matrix                        
        ntrials = len(epoch)
        nconditions = 4
        design_matrix = np.zeros((ntrials, nconditions))
        
        for icondi, condi in enumerate(behav["positions"]):            
            # assert isinstance(condi, np.int64) 
            design_matrix[icondi, condi-1] = 1
        assert np.sum(design_matrix.sum(axis=1) == 1) == len(epoch)        
       
        # Run OLS
        # z-score the data
        # meg_data_V = epoch.get_data()
        meg_data_V = epoch
        _, nchs, ntimes = meg_data_V.shape
        meg_data_V = scipy.stats.zscore(meg_data_V, axis=0)

        coefs = np.zeros((nconditions, nchs, ntimes))
        resids = np.zeros_like(meg_data_V)
        for ich in tqdm(range(nchs)):
            for itime in range(ntimes):
                y = meg_data_V[:, ich, itime]
                
                model = sm.OLS(endog=y, exog=design_matrix, missing="raise")
                results = model.fit()
                
                coefs[:, ich, itime] = results.params # (4, 248, 163)
                resids[:, ich, itime] = results.resid # (ntrials, 248, 163)
        
        # Calculate pairwise mahalanobis distance between regression coefficients        
        rdm_times = np.zeros((nconditions, nconditions, ntimes))
        for itime in tqdm(range(ntimes)):
            response = coefs[:, :, itime] # (4, 248)
            residuals = resids[:, :, itime] # (51, 248)
            
            # Estimate covariance from residuals
            lw_shrinkage = LedoitWolf(assume_centered=True)
            cov = lw_shrinkage.fit(residuals)
            
            # Compute pairwise mahalanobis distances
            VI = np.linalg.inv(cov.covariance_) # covariance matrix needed for mahalonobis
            rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
            assert ~np.isnan(rdm).all()
            rdm_times[:, :, itime] = rdm # rdm_times (4, 4, 163), rdm (4, 4)
        
        rdm_dir = op.join(RESULTS_DIR, "rdms", "sensors", "shuffled", subject)
        # ensure_dir(rdm_dir)
        # rdm_fname = "rdm_%s.npy" % str(epoch_num)
        # np.save(op.join(rdm_dir, rdm_fname), np.random.shuffle(rdm_times))
        
        # coefs_dir = op.join(RESULTS_DIR, "coefs", "sensors", subject)
        # ensure_dir(coefs_dir)
        # coefs_fname = "coefs_%s.npy" % str(epoch_num)
        # np.save(op.join(coefs_dir, coefs_fname), coefs)
        # response_fname = "response_%s.npy" % str(epoch_num)
        # np.save(op.join(coefs_dir, response_fname), response)
        
        # resids_dir = op.join(RESULTS_DIR, "resids", "sensors", subject)
        # ensure_dir(resids_dir)
        # resids_fname = "resids_%s.npy" % str(epoch_num)
        # np.save(op.join(resids_dir, resids_fname), resids)
        # residuals_fname = "residuals_%s.npy" % str(epoch_num)
        # np.save(op.join(resids_dir, residuals_fname), residuals)
                
        rdmx = rdm_times
        
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()
                    
        for itime in range(rdmx.shape[2]):
            one_two_similarity.append(rdmx[0, 1, itime])
            one_three_similarity.append(rdmx[0, 2, itime])
            one_four_similarity.append(rdmx[0, 3, itime])
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

    in_seq, out_seq = [], []
    similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
                    two_three_similarities, two_four_similarities, three_four_similarities]
        
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
                        
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