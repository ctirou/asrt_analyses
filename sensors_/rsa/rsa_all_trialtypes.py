import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, zscore, spearmanr
from tqdm.auto import tqdm
from base import *
from config import *

lock = 'stim'
trial_type = 'pattern'
analysis = 'usual'
overwrite = False

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
figures_dir = FIGURES_DIR / "RSA" / "sensors" / lock / analysis
ensure_dir(figures_dir)
metric = 'mahalanobis'

all_in_seqs, all_out_seqs = [], []

# get times
epoch_fname = op.join(data_path, lock, 'sub01-0-epo.fif')
epochs = mne.read_epochs(epoch_fname)
times = epochs.times
del epochs

for subject in tqdm(subjects):
    
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "rdm" / subject
    ensure_dir(res_path)
    
    # all_in_seqs, all_out_seqs = [], []
    
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
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
        
        if not op.exists(res_path / f"pat-{epoch_num}.npy") or not op.exists(res_path / f"rand-{epoch_num}.npy"):
            # read behav
            behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
            behav = pd.read_pickle(behav_fname)
            # read epochs
            epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
            epoch = mne.read_epochs(epoch_fname)
        
            epoch_pat = epoch[np.where(behav["trialtypes"]==1)].get_data(copy=False).mean(axis=0)
            behav_pat = behav[behav["trialtypes"]==1]
            assert len(epoch_pat) == len(behav_pat)
            rdm_pat = get_rdm(epoch_pat, behav_pat)
            np.save(res_path / f"pat-{epoch_num}.npy", rdm_pat)
        
            epoch_rand = epoch[np.where(behav["triplets"]==34)].get_data(copy=False).mean(axis=0)
            behav_rand = behav[behav["triplets"]==34]
            assert len(epoch_rand) == len(behav_rand)
            rdm_rand = get_rdm(epoch_rand, behav_rand)
            np.save(res_path / f"rand-{epoch_num}.npy", rdm_rand)
        else:
            rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")
            rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
        
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
                        
        one_two_pat = np.array(one_two_pat)
        one_three_pat = np.array(one_three_pat)
        one_four_pat = np.array(one_four_pat) 
        two_three_pat = np.array(two_three_pat)
        two_four_pat = np.array(two_four_pat) 
        three_four_pat = np.array(three_four_pat)

        one_two_rand = np.array(one_two_rand)
        one_three_rand = np.array(one_three_rand)
        one_four_rand = np.array(one_four_rand) 
        two_three_rand = np.array(two_three_rand)
        two_four_rand = np.array(two_four_rand) 
        three_four_rand = np.array(three_four_rand)

        one_twos_pat.append(one_two_pat)
        one_threes_pat.append(one_three_pat)
        one_fours_pat.append(one_four_pat) 
        two_threes_pat.append(two_three_pat)
        two_fours_pat.append(two_four_pat) 
        three_fours_pat.append(three_four_pat)

        one_twos_rand.append(one_two_rand)
        one_threes_rand.append(one_three_rand)
        one_fours_rand.append(one_four_rand) 
        two_threes_rand.append(two_three_rand)
        two_fours_rand.append(two_four_rand) 
        three_fours_rand.append(three_four_rand)
                            
    one_twos_pat = np.array(one_twos_pat)
    one_threes_pat = np.array(one_threes_pat)  
    one_fours_pat = np.array(one_fours_pat)   
    two_threes_pat = np.array(two_threes_pat)  
    two_fours_pat = np.array(two_fours_pat)
    three_fours_pat = np.array(three_fours_pat)

    one_twos_rand = np.array(one_twos_rand)
    one_threes_rand = np.array(one_threes_rand)  
    one_fours_rand = np.array(one_fours_rand)   
    two_threes_rand = np.array(two_threes_rand)  
    two_fours_rand = np.array(two_fours_rand)   
    three_fours_rand = np.array(three_fours_rand)
    
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))

    in_seq, out_seq = [], []
    similarities = [one_twos_pat, one_threes_pat, one_fours_pat,
                    two_threes_pat, two_fours_pat, three_fours_pat]
    random_lows = [one_twos_rand, one_threes_rand, one_fours_rand,
                    two_threes_rand, two_fours_rand, three_fours_rand]
        
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']                        
    for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(pat_sim)
        else:
            out_seq.append(pat_sim)
    # for l in random_lows:
    #     out_seq.append(l)
    
    all_in_seqs.append(np.array(in_seq))    
    all_out_seqs.append(np.array(out_seq))
    
all_in_seqs = np.array(all_in_seqs)
all_out_seqs = np.array(all_out_seqs)

diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

# plot in out and diff
plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, all_in_seqs.mean((0, 1, 2)), label='in', color='C0', alpha=0.6)
plt.plot(times, all_out_seqs.mean((0, 1, 2)), label='out', color='C1', alpha=0.6)
plt.plot(times, diff_inout.mean((0, 1)), label='in - out', color='C2', alpha=0.6)
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.legend()
plt.savefig(op.join(figures_dir, '%s_all_%s.pdf' % (metric, trial_type)), transparent=True)
plt.close()

# plot correlations
rhos = [[spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
rhos = np.array(rhos)
plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, rhos.mean(0))
p_values = decod_stats(rhos, -1)
sig = p_values < 0.05
plt.fill_between(times, rhos.mean(0), 0, where=sig, color='green', alpha=.7)
plt.axhline(0, color="black", linestyle="dashed")
plt.title(f'{metric} correlations', style='italic')
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.axhline(0, color='black', linestyle='dashed')
plt.savefig(op.join(figures_dir, '%s_high_low_correlations_%s.pdf' % (metric, trial_type)), transparent=True)
plt.close()

# learning_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t")

# # plot across subjects
# all_pvalues = []
# for t in range(len(times)):
#     all_pvalues.append(spearmanr(learning_index_df["4"], diff_inout[:, -1, t])[1])
# plt.subplots(1, 1, figsize=(14, 5))
# plt.plot(times, all_pvalues)
# sig = (np.asarray(all_pvalues) < 0.05)
# plt.fill_between(times, -0.1, -0.11, where=sig, color='red', alpha=1.0)  # Solid line at the bottom when sig is true
# plt.title(f'{metric} across subjects correlations', style='italic')
# plt.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.savefig(op.join(figures_dir, '%s_high_low_across_sub_correlations_%s.pdf' % (metric, trial_type)), transparent=True)
# plt.close()

# # plot within subjects
# all_rhos = []
# for sub in tqdm(range(len(subjects))):
#     rhos = []
#     for t in range(len(times)):
#         rhos.append(spearmanr(learning_index_df.iloc[sub, 1:], diff_inout[sub, :, t])[0])
#     all_rhos.append(rhos)
# all_rhos = np.array(all_rhos)
# plt.subplots(1, 1, figsize=(14, 5))
# plt.plot(times, all_rhos.mean(0))
# p_values = decod_stats(all_rhos, -1)
# sig = p_values < 0.05
# plt.fill_between(times, all_rhos.mean(0), 0, where=sig, color='green', alpha=.7)
# plt.axhline(0, color="black", linestyle="dashed")
# plt.title(f'{metric} within subjects correlations', style='italic')
# plt.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.axhline(0, color='black', linestyle='dashed')
# plt.savefig(op.join(figures_dir, '%s_high_low_within_sub_correlations_%s.pdf' % (metric, trial_type)), transparent=True)
# plt.close()

# plot the difference in vs. out sequence averaging all epochs
plt.subplots(1, 1, figsize=(14, 5))
diff = diff_inout[:, 1:, :].mean(1) - diff_inout[:, 0, :]
plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
plt.plot(times, diff_inout[:, 1:, :].mean(1).mean(0), label='high - low', color='C0', alpha=0.6)
plt.plot(times, diff.mean(0), label='high - low - practice', color='C1', alpha=0.6)
p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
sig_unc = p_values_unc < 0.05
p_values = decod_stats(diff, -1)
sig = p_values < 0.05
# plt.fill_between(times, 0, diff_inout.mean((0, 1)), where=sig_unc, color='C1', alpha=0.2)
# plt.fill_between(times, 0, diff.mean(0), where=sig, color='black', alpha=0.3)
plt.axhline(0, color='black', linestyle='dashed')
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.legend()
plt.title(f'{metric} high – low average', style='italic')
plt.savefig(op.join(figures_dir, '%s_high_low_ave_%s.pdf' % (metric, trial_type)))
plt.close()

# plot the difference in vs. out sequence for each epoch
for i in range(1, 5):
    plt.subplots(1, 1, figsize=(14, 5))
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    plt.plot(times, diff_inout[:, i, :].mean(0), label='learning', color='C1', alpha=0.6)
    diff = diff_inout[:, i, :] - diff_inout[:, 0, :]
    p_values_unc = ttest_1samp(diff, axis=0, popmean=0)[1]
    sig_unc = p_values < 0.05
    p_values = decod_stats(diff, -1)
    sig = p_values < 0.05
    # plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig_unc, alpha=0.2)
    # plt.fill_between(times, 0, diff_inout[:, i, :].mean(0), where=sig, alpha=0.4, color='black')
    # plt.axhline(0, color='black', linestyle='dashed')
    plt.axvspan(0, 0.2, color='grey', alpha=.2)
    plt.legend()
    # plt.gca().set_ylim(-0.04, 0.12)
    plt.title(f'{metric} high – low epoch {i}', style='italic')
    plt.savefig(op.join(figures_dir, '%s_high_low_%s_%s.pdf' % (metric, trial_type, str(i))))
    plt.close()