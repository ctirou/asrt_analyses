import os.path as op
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, zscore, spearmanr
import statsmodels.api as sm
from tqdm.auto import tqdm
from base import *
from config import DATA_DIR_SSD, RESULTS_DIR, NEW_FIG_DIR, SUBJS, EPOCHS
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

lock = 'stim'
trial_type = 'pattern'
metric = 'mahalanobis'

data_path = DATA_DIR_SSD
res_path = RESULTS_DIR
subjects, epochs_list = SUBJS, EPOCHS

combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

figures_dir = NEW_FIG_DIR / "RSA" / "sensors" / lock
ensure_dir(figures_dir)

all_in_seqs, all_out_seqs = [], []

for subject in subjects:
        
    # Read the behav file to get the sequence 
    behav_dir = DATA_DIR_SSD / "raw_behavs" / f"{subject}"
    sequence = get_sequence(behav_dir)
        
    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()
    
    # loop across sessions
    for epoch_num, epo in enumerate(epochs_list):
                    
        if epo == '2_PRACTICE':
            epo_fname = 'prac'
        else:
            epo_fname = 'sess-%s' % (str(epoch_num).zfill(2))
    
        behav_fname = op.join(data_path, "preprocessed/behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        if lock == 'button': 
            epoch_bsl_fname = op.join(data_path, "bsl/%s-%s_bl-epo.fif" % (subject, epoch_num))
            epoch_bsl = mne.read_epochs(epoch_bsl_fname)
            epoch_fname = op.join(data_path, "%s/%s-%s-b-epo.fif" % (lock, subject, epoch_num))
        else:
            epoch_fname = op.join(data_path, "preprocessed/%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
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
        meg_data_V = epoch.copy()
        _, nchs, ntimes = meg_data_V.shape
        meg_data_V = zscore(meg_data_V, axis=0)

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
                    
            # rdm = cv_distance(response, residuals, metric)
            # assert ~np.isnan(rdm).all()
            # rdm_times[:, :, itime] = rdm  # rdm_times (4, 4, 163)                

            # Estimate covariance from residuals
            lw_shrinkage = LedoitWolf(assume_centered=True)
            cov = lw_shrinkage.fit(residuals)
            
            # Compute pairwise mahalanobis distances
            VI = np.linalg.inv(cov.covariance_) # covariance matrix needed for mahalonobis
            # rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
            rdm = squareform(pdist(response, metric="cosine"))
            assert ~np.isnan(rdm).all()
            rdm_times[:, :, itime] = rdm # rdm_times (4, 4, 163), rdm (4, 4)                                
        
        rdmx = rdm_times.copy()
        
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

rhos = [[spearmanr([1, 2, 3, 4], diff_inout[sub, 1:, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
rhos = np.array(rhos)

# plot correlations
plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, rhos.mean(0))
p_values = decod_stats(rhos, -1)
sig = p_values < 0.05
plt.fill_between(times, rhos.mean(0), 0, where=sig, color='green', alpha=.7)
plt.axhline(0, color="black", linestyle="dashed")
plt.title(f'{metric} {trial_type} correlations', style='italic')
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.axhline(0, color='black', linestyle='dashed')
plt.savefig(op.join(figures_dir, '%s_correlations_%s.pdf' % (metric, trial_type)))
plt.close()

# plot the difference in vs. out sequence averaging all epochs
plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C1', alpha=0.2)
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C1', alpha=0.3)
plt.axhline(0, color='black', linestyle='dashed')
plt.legend()
plt.title(f'{metric} diff in/out {trial_type} average', style='italic')
plt.savefig(op.join(figures_dir, '%s_diff_inout_ave_%s.pdf' % (metric, trial_type)))
plt.close()

# plot the difference in vs. out sequence for each epoch
for i in range(1, 5):
    plt.subplots(1, 1, figsize=(14, 5))
    plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    plt.plot(times, diff_inout[:, i, :].mean(0), label='learning', color='C1', alpha=0.6)
    diff = diff_inout[:, i, :] - diff_inout[:, 0, :]
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values < 0.05
    p_values = decod_stats(diff, -1)
    sig = p_values < 0.05
    plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig_unc, alpha=0.2)
    plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig, alpha=0.4, hatch='+')
    plt.axhline(0, color='black', linestyle='dashed')
    plt.axvspan(0, 0.2, color='grey', alpha=.2)
    plt.legend()
    plt.title(f'{metric} diff in/out {trial_type} epoch {i}', style='italic')
    plt.savefig(op.join(figures_dir, '%s_diff_inout_%s_%s.pdf' % (metric, trial_type, str(i))))
    plt.close()