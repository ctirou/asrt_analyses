import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from config import RESULTS_DIR, SUBJS, RAW_DATA_DIR, DATA_DIR, EPOCHS

subjects = SUBJS
epochs_list = EPOCHS

lock = 'stim'
trial_type = 'pattern'
params = 'bslned_transitions'

ols = True

def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
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
    for epoch_num, epoch in enumerate(epochs_list):
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()
        if ols:
            # load and read rdm file
            rdm_fname = op.join(RESULTS_DIR, 'rdms', 'sensors', subject, 'rdm_%s.npy' % (epoch_num)) # (4, 4, 263)
            # coefs_fname = op.join(RESULTS_DIR, 'coefs', loca, subject, 'coefs_%s.npy' % (epoch_num)) # (4, 248, 163)
            # resp_fname = op.join(RESULTS_DIR, 'coefs', loca, subject, 'response_%s.npy' % (epoch_num)) # (4, 248)
            # resids_fname = op.join(RESULTS_DIR, 'resids', loca, subject, 'resids_%s.npy' % (epoch_num)) # (ntrials, 248, 163)
            # residuals_fname = op.join(RESULTS_DIR, 'resids', loca, subject, 'residuals_%s.npy' % (epoch_num)) # (ntrials, 248)
            
            # coefs = np.load(coefs_fname)
            # ensure_dir(op.join(figures_dir, "coefs", loca, subject))
            # plt.figure(figsize=(16, 7))
            # plt.plot(times, coefs.mean(1).T, label=[i for i in range(1, 5)])
            # plt.legend()
            # plt.savefig(op.join(figures_dir, "coefs", loca, subject, "coefs_%s" % str(epoch)))
            # plt.close()
            
            # resids = np.load(resids_fname)
            # ensure_dir(op.join(figures_dir, "resids", loca, subject))
            # plt.figure(figsize=(16, 7))
            # plt.plot(times, resids.mean((0, 1)))
            # plt.savefig(op.join(figures_dir, "resids", loca, subject, "resids_%s" % str(epoch)))
            # plt.close()
            
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
        else:
            sim_list = [one_two_similarities, one_three_similarities, one_four_similarities, 
                        two_three_similarities, two_four_similarities, three_four_similarities]
            for sim, mis, sss in zip(sim_list, ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four'], similarities_list):
                sim_fname = op.join(RESULTS_DIR, 'sensor_sims', subject, epoch, "%s.npy" % mis)
                sim.append(np.load(sim_fname))
                d[epoch][sss].append(np.load(sim_fname))
            
        # # plot paired distances per epoch, per sub ---- done
        # ensure_dir(op.join(figures_dir, "paired_dist_epo", subject))
        # ylims = (-1.5, 3)
        # plt.figure(figsize=(16, 7))
        # plt.ylim(ylims)
        # plt.plot(times, np.array(one_two_similarities).mean(0), label="one_two")
        # plt.plot(times, np.array(one_three_similarities).mean(0), label="one_three")
        # plt.plot(times, np.array(one_four_similarities).mean(0), label="one_four")
        # plt.plot(times, np.array(two_three_similarities).mean(0), label="two_three")
        # plt.plot(times, np.array(two_four_similarities).mean(0), label="two_four")
        # plt.plot(times, np.array(three_four_similarities).mean(0), label="three_four")
        # plt.legend()
        # plt.title("%s" % (sequence))
        # plt.savefig(op.join(figures_dir, "paired_dist_epo", subject, "%s.png" % (epoch)))
        # plt.close()
        
        # plot in/out distance per epoch, per sub ---- done
        # all_in_seqs, all_out_seqs = [], [] # uncomment if you want fig per subject
        # pairs_in_sequence = list()
        # pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
        # pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
        # pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
        # pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
        # in_seq, out_seq = [], []
        # similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
        #                 two_three_similarities, two_four_similarities, three_four_similarities]
        # pairs = ['12', '13', '14', '23', '24', '34']
        # rev_pairs = ['21', '31', '41', '32', '42', '43']
        # for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        #     if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
        #         in_seq.append(similarity)
        #     else: 
        #         out_seq.append(similarity)
        # all_in_seqs.append(np.array(in_seq))
        # all_out_seqs.append(np.array(out_seq))
        # all_in_seqs = np.array(all_in_seqs)
        # all_out_seqs = np.array(all_out_seqs)
        # diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
        
        # ensure_dir(op.join(figures_dir, "inOut_dist", epoch))
        # plt.figure(figsize=figsize)
        # plt.ylim(-1.5, 2)
        # plt.plot(times, all_out_seqs.mean((0, 1, 2)), label="out_seq")
        # plt.plot(times, all_in_seqs.mean((0, 1, 2)), label="in_seq")
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
    
    # # plot paired distances averaged across epochs, per sub ---- done
    # ensure_dir(op.join(figures_dir, "paired_dist_ave"))
    # ylims = (-0.5, 2.5)
    # plt.figure(figsize=(16, 7))
    # plt.ylim(ylims)
    # plt.plot(times, one_two_similarities.mean(0), label="one_two")
    # plt.plot(times, one_three_similarities.mean(0), label="one_three")
    # plt.plot(times, one_four_similarities.mean(0), label="one_four")
    # plt.plot(times, two_three_similarities.mean(0), label="two_three")
    # plt.plot(times, two_four_similarities.mean(0), label="two_four")
    # plt.plot(times, three_four_similarities.mean(0), label="three_four")
    # plt.legend()
    # plt.title("%s" % (sequence))
    # plt.savefig(op.join(figures_dir, "paired_dist_ave", "%s.png" % (subject)))
    # plt.close()
            
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    all_in_seqs.append(np.array(in_seq))
    all_out_seqs.append(np.array(out_seq))

    # # plot in/out and diff averaged across epochs and subs ---- done
    # all_in_seqs = np.array(all_in_seqs)
    # all_out_seqs = np.array(all_out_seqs)
    # diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
    # ensure_dir(op.join(figures_dir, "inOut_dist"))
    # plt.figure(figsize=figsize)
    # plt.ylim(-1, 7)
    # plt.plot(times, all_out_seqs.mean((0, 1, 2)), label="out_seq")
    # plt.plot(times, all_in_seqs.mean((0, 1, 2)), label="in_seq")
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='diff')
    # plt.legend()
    # plt.title("in/out_%s" % (subject))
    # plt.savefig(op.join(figures_dir, "inOut_dist", "%s.png" % (subject)))
    # plt.close()

# plot paired distances averaged across epochs and subs ---- done
ensure_dir(op.join(figures_dir, "paired_dist_ave"))
plt.figure(figsize=(16, 7))
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

all_in_seqs = np.array(all_in_seqs)
all_out_seqs = np.array(all_out_seqs)
diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

# plot the difference in vs. out sequence across epochs
plt.figure(figsize=(16, 7))
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
plt.legend()
plt.savefig(op.join(figures_dir, 'ols_ave.png'))
plt.close()

# plot paired distances per epoch averaged across subs ---- done
ensure_dir(op.join(figures_dir, "paired_dist_ave"))
for epoch, sim in zip(epochs_list, similarities_list):
    plt.figure(figsize=(16, 7))
    plt.ylim(0, 9)
    for i, label in zip(d[epoch][sim], similarities_list):
        plt.plot(times, i, label=label)
    plt.legend()
    plt.title("%s" % (epoch))
    plt.savefig(op.join(figures_dir, "paired_dist_ave", "%s.png" % (epoch)))
    plt.close()
    
# plot the difference in vs. out sequence averaging all epochs
plt.figure(figsize=(16, 7))
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1:5, :].mean((0, 1)), label='learning', color='C1', alpha=0.6)
diff = diff_inout[:, 1:5, :].mean((1)) - diff_inout[:, 0, :]
p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(diff)
sig = p_values < 0.05
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig_unc, color='C2', alpha=0.2)
plt.fill_between(times, 0, diff_inout[:, 1:5, :].mean((0, 1)), where=sig, color='C3', alpha=0.3)
plt.legend()
plt.savefig(op.join(figures_dir, 'megave_sensors_%s.png' % ('ols' if ols else 'basic')))
plt.close()