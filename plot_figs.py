import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from config import RESULTS_DIR, SUBJS, RAW_DATA_DIR, DATA_DIR

subjects = SUBJS
epochs_list = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']
lock = 'stim'
trial_type = 'pattern'
loca = 'sensors'

figures_dir = op.join(RESULTS_DIR, 'figures', lock, 'similarity')
figsize = (15, 7)

# get times
epoch_fname = op.join(DATA_DIR, lock, 'sub01_0_s-epo.fif')
epoch = mne.read_epoch(epoch_fname)
times = epoch.times
del epoch

all_in_seqs, all_out_seqs = [], []
for subject in subjects:
    # all_in_seqs, all_out_seqs = [], [] # uncomment if you want fig per subject
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
    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()
    for epoch_num, epo in enumerate(epochs_list):
        # load and read rdm file
        rdm_fname = op.join(RESULTS_DIR, 'rdms', loca, 'rdm_%s.npy' % (epoch_num))
        rdm = np.load(rdm_fname)
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()
                    
        for itime in range(rdm.shape[2]):
            one_two_similarity.append(rdm[0, 1, itime])
            one_three_similarity.append(rdm[0, 2, itime])
            one_four_similarity.append(rdm[0, 3, itime])
            two_three_similarity.append(rdm[1, 2, itime])
            two_four_similarity.append(rdm[1, 3, itime])
            three_four_similarity.append(rdm[2, 3, itime])
                        
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
            # print(pair)
            # print(rev_pair)
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    all_in_seqs.append(np.array(in_seq))
    all_out_seqs.append(np.array(out_seq))
    
all_in_seqs = np.array(all_in_seqs)
all_out_seqs = np.array(all_out_seqs)
diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

# plot the difference in vs. out sequence across epochs
plt.figure(figsize=(12.8, 7.2))
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
plt.legend()
plt.savefig(op.join(figures_dir, 'blocks_ols_%s.png' % (trial_type)))
plt.close()