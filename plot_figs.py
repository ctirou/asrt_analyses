import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from config import RESULTS_DIR, SUBJS, RAW_DATA_DIR, DATA_DIR, EPOCHS

subjects = SUBJS
epochs_list = EPOCHS
lock = 'stim'
trial_type = 'pattern'
loca = 'sensors'

figures_dir = op.join(RESULTS_DIR, 'figures', lock, 'similarity')
figsize = (16, 7)

# get times
epoch_fname = op.join(DATA_DIR, lock, 'sub01_0_s-epo.fif')
epochs = mne.read_epochs(epoch_fname)
times = epochs.times
del epochs

all_in_seqs, all_out_seqs = [], []
# create lists for epochs
prac_0, epo_1, epo_2, epo_3, epo_4 = [], [], [], [], []
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
    for epoch_num, (epoch, epo) in enumerate(zip(epochs_list, [prac_0, epo_1, epo_2, epo_3, epo_4])):
        # load and read rdm file
        rdm_fname = op.join(RESULTS_DIR, 'rdms', loca, subject, 'rdm_%s.npy' % (epoch_num))
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

        epo.append(one_two_similarity)
        epo.append(one_three_similarity)
        epo.append(one_four_similarity) 
        epo.append(two_three_similarity)
        epo.append(two_four_similarity) 
        epo.append(three_four_similarity)
        
        # plot paired distances per epoch, per sub
        if not op.exists(op.join(figures_dir, "paired_dist_epo", subject)):
            os.makedirs(op.join(figures_dir, "paired_dist_epo", subject))
        ylims = (1, 2.4)
        plt.figure(figsize=(figsize))
        plt.ylim(ylims)
        plt.plot(times, np.array(one_two_similarities).mean(0), label="one_two")
        plt.plot(times, np.array(one_three_similarities).mean(0), label="one_three")
        plt.plot(times, np.array(one_four_similarities).mean(0), label="one_four")
        plt.plot(times, np.array(two_three_similarities).mean(0), label="two_three")
        plt.plot(times, np.array(two_four_similarities).mean(0), label="two_four")
        plt.plot(times, np.array(three_four_similarities).mean(0), label="three_four")
        plt.legend()
        plt.title("%s" % (sequence))
        plt.savefig(op.join(figures_dir, "paired_dist_epo", "%s.png" % (epoch)))
        plt.close()

    one_two_similarities = np.array(one_two_similarities)
    one_three_similarities = np.array(one_three_similarities)  
    one_four_similarities = np.array(one_four_similarities)   
    two_three_similarities = np.array(two_three_similarities)  
    two_four_similarities = np.array(two_four_similarities)   
    three_four_similarities = np.array(three_four_similarities)
    
    pracs = np.array(prac_0)
    epos_1 = np.array(epo_1)
    epos_2 = np.array(epo_2)
    epos_3 = np.array(epo_3)
    epos_4 = np.array(epo_4)
    
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
    
    # plot paired distances averaged across epochs, per sub
    if not op.exists(op.join(figures_dir, "paired_dist_ave")):
        os.makedirs(op.join(figures_dir, "paired_dist_ave"))
    ylims = (1, 2.4)
    plt.figure(figsize=(figsize))
    plt.ylim(ylims)
    plt.plot(times, one_two_similarities.mean(0), label="one_two")
    plt.plot(times, one_three_similarities.mean(0), label="one_three")
    plt.plot(times, one_four_similarities.mean(0), label="one_four")
    plt.plot(times, two_three_similarities.mean(0), label="two_three")
    plt.plot(times, two_four_similarities.mean(0), label="two_four")
    plt.plot(times, three_four_similarities.mean(0), label="three_four")
    plt.legend()
    plt.title("%s" % (sequence))
    plt.savefig(op.join(figures_dir, "paired_dist_ave", "%s.png" % (subject)))
    plt.close()
            
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    all_in_seqs.append(np.array(in_seq))
    all_out_seqs.append(np.array(out_seq))
    
    # all_in_seqs = np.array(all_in_seqs)
    # all_out_seqs = np.array(all_out_seqs)

    # diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
 
    # if not op.exists(op.join(figures_dir, "inOut_dist", epoch)):
    #     os.makedirs(op.join(figures_dir, "inOut_dist", epoch))
    # plt.figure(figsize=figsize)
    # plt.ylim(-0.5, 3)
    # plt.plot(times, all_out_seqs.mean((0, 1, 2)), label="out_seq")
    # plt.plot(times, all_in_seqs.mean((0, 1, 2)), label="in_seq")
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='diff')
    # plt.legend()
    # plt.title("in/out_%s" % (subject))
    # plt.savefig(op.join(figures_dir, "inOut_dist", epoch, "%s.png" % (subject)))
    # plt.close()

# plot paired distances averaged across epochs and subs
plt.figure(figsize=(figsize))
plt.plot(times, one_two_similarities.mean(0), label="one_two")
plt.plot(times, one_three_similarities.mean(0), label="one_three")
plt.plot(times, one_four_similarities.mean(0), label="one_four")
plt.plot(times, two_three_similarities.mean(0), label="two_three")
plt.plot(times, two_four_similarities.mean(0), label="two_four")
plt.plot(times, three_four_similarities.mean(0), label="three_four")
plt.legend()
plt.title("paired_dist_ave")
plt.savefig(op.join(figures_dir, "paired_dist_ave.png"))
plt.close()

# all_in_seqs = np.array(all_in_seqs)
# all_out_seqs = np.array(all_out_seqs)
# diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

# # plot the difference in vs. out sequence across epochs
# plt.figure(figsize=figsize)
# plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
# plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
# plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
# plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
# plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
# plt.legend()
# plt.savefig(op.join(figures_dir, 'ols_ave.png'))
# plt.close()