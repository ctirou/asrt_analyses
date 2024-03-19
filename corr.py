import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr
from tqdm.auto import tqdm
from base import *
from config import *

subjects = SUBJS
epochs_list = EPOCHS

lock = 'stim'
trial_type = 'pattern'
params = 'correlations'

figures_dir = op.join(RESULTS_DIR, 'figures', lock, 'similarity', params)
ensure_dir(figures_dir)
figsize = (16, 7)
# get times
epoch_fname = op.join(DATA_DIR, lock, 'sub01_0_s-epo.fif')
epochs = mne.read_epochs(epoch_fname)
times = epochs.times
del epochs

similarities_list = ['one_two_similarities', 'one_three_similarities', 'one_four_similarities',
                     'two_three_similarities', 'two_four_similarities', 'three_four_similarities']

d = {i:{j: list() for j in similarities_list} for i in epochs_list}

all_in_seqs, all_out_seqs = [], []
# create lists for epochs
for subject in subjects:
    # all_in_seqs, all_out_seqs = [], [] # uncomment if you want fig per subject
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
    
    for epoch_num, epoch in enumerate(epochs_list):
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()

        # load and read rdm file
        rdm_fname = op.join(RESULTS_DIR, 'rdms', 'sensors', subject, 'rdm_%s.npy' % (epoch_num)) # (4, 4, 263)
        rdm = np.load(rdm_fname)            

        for itime in range(rdm.shape[2]):
            one_two_similarity.append(rdm[0, 1, itime])
            one_three_similarity.append(rdm[0, 2, itime])
            one_four_similarity.append(rdm[0, 3, itime])
            two_three_similarity.append(rdm[1, 2, itime])
            two_four_similarity.append(rdm[1, 3, itime])
            three_four_similarity.append(rdm[2, 3, itime])
        
        sim_list = [one_two_similarity, one_three_similarity, one_four_similarity, 
            two_three_similarity, two_four_similarity, three_four_similarity]
        
        for i, j in zip(similarities_list, sim_list):
            d[epoch][i].append(j)
                
        one_two_similarity = np.array(one_two_similarity)
        one_three_similarity = np.array(one_three_similarity)
        one_four_similarity = np.array(one_four_similarity) 
        two_three_similarity = np.array(two_three_similarity)
        two_four_similarity = np.array(two_four_similarity) 
        three_four_similarity = np.array(three_four_similarity)
        
        sim_list_arr = [one_two_similarity, one_three_similarity, one_four_similarity, 
                        two_three_similarity, two_four_similarity, three_four_similarity]
        
        # calculate mean baseline value
        all_means = []
        for sims in sim_list_arr:
            mimi = []
            for itime, sim in zip(times, sims):
                if itime < 0:
                    mimi.append(sim)
            all_means.append(np.mean(mimi))             
        # baseline correction for each transitions
        for mean_val, sim in zip(all_means, sim_list_arr):
            sim -= mean_val
        one_two_similarities.append(one_two_similarity)
        one_three_similarities.append(one_three_similarity)
        one_four_similarities.append(one_four_similarity) 
        two_three_similarities.append(two_three_similarity)
        two_four_similarities.append(two_four_similarity) 
        three_four_similarities.append(three_four_similarity)
                    
        # plot in/out distance per epoch, per sub ---- done
        # all_in_seqs, all_out_seqs = [], [] # uncomment if you want fig per subject
        # similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
        #                 two_three_similarities, two_four_similarities, three_four_similarities]

        # in_seq, out_seq = get_inout_seq(sequence, similarities)

        # all_in_seqs.append(np.array(in_seq))
        # all_out_seqs.append(np.array(out_seq))
        # all_in_seqs = np.array(all_in_seqs)
        # all_out_seqs = np.array(all_out_seqs)
        # diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
        
        # ensure_dir(op.join(figures_dir, "inOut_dist", epoch))
        # plt.figure(figsize=(16, 7))
        # plt.ylim(-1.5, 2)
        # plt.plot(times, diff_inout[:, 0, :].mean(0), label='diff')
        # plt.legend()
        # plt.title("in/out_%s" % (subject))
        # plt.savefig(op.join(figures_dir, "inOut_dist", epoch, "%s.png" % (subject)))
        # plt.close()
        
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

all_rhos = []
for sub in tqdm(range(len(subjects))):
    rhos = []
    for t in range(len(times)):
        rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
    all_rhos.append(rhos)
all_rhos = np.array(all_rhos)

plt.figure(figsize=(16, 7))
plt.plot(times, all_rhos.mean(0)[:, 0], label='rho')
diff = all_rhos[:, :, 0]
p_values = decod_stats(diff)
sig = p_values < 0.05
sig = all_rhos[:, :, 1] < 0.05
plt.fill_between(times, 0, all_rhos.mean(0)[:, 0], where=sig, color='C3', alpha=0.3)
plt.legend()