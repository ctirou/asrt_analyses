import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr
from base import *
from config import *

subjects = SUBJS
epochs_list = EPOCHS
subjects_dir = FREESURFER_DIR
lock = 'stim'
trial_type = 'pattern'

ols = True

figures_dir = RESULTS_DIR /'figures' / lock / 'similarity' / 'source'
figsize = (16, 7)

# get times
epoch_fname = op.join(DATA_DIR, lock, 'sub01_0_s-epo.fif')
epochs = mne.read_epochs(epoch_fname)
times = epochs.times
del epochs

similarities_list = ['one_two_similarities', 'one_three_similarities', 'one_four_similarities',
                     'two_three_similarities', 'two_four_similarities', 'three_four_similarities']
d = {i:{j: list() for j in similarities_list} for i in epochs_list}

for lab in range(34):
    
    all_in_seqs, all_out_seqs = [], []

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
            labels = mne.read_labels_from_annot(subject=subject, parc='aparc', hemi='lh', subjects_dir=subjects_dir)
            label = labels[lab]
            if ols:
                # load and read rdm file
                rdm_fname = op.join(RESULTS_DIR, 'rdms', 'source', subject, label.name, 'rdm_%s.npy' % (epoch_num)) # (4, 4, 263)
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
                
                one_two_similarities.append(one_two_similarity)
                one_three_similarities.append(one_three_similarity)
                one_four_similarities.append(one_four_similarity) 
                two_three_similarities.append(two_three_similarity)
                two_four_similarities.append(two_four_similarity) 
                three_four_similarities.append(three_four_similarity)
                
            else:
                sim_list = [one_two_similarities, one_three_similarities, one_four_similarities, 
                            two_three_similarities, two_four_similarities, three_four_similarities]
                # all false need this for sensors
                for sim, mis, sss in zip(sim_list, ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four'], similarities_list):
                    sim_fname = op.join(RESULTS_DIR, 'sensor_sims', subject, epoch, "%s.npy" % mis)
                    sim.append(np.load(sim_fname))
                    d[epoch][sss].append(np.load(sim_fname))
                
            # plot paired distances per epoch, per sub ---- done
            # ensure_dir(op.join(figures_dir, "paired_dist_epo", subject, label.name))
            # ylims = (0, 12)
            # plt.figure(figsize=(16, 7))
            # # plt.ylim(ylims)
            # plt.plot(times, np.array(one_two_similarities).mean(0), label="one_two")
            # plt.plot(times, np.array(one_three_similarities).mean(0), label="one_three")
            # plt.plot(times, np.array(one_four_similarities).mean(0), label="one_four")
            # plt.plot(times, np.array(two_three_similarities).mean(0), label="two_three")
            # plt.plot(times, np.array(two_four_similarities).mean(0), label="two_four")
            # plt.plot(times, np.array(three_four_similarities).mean(0), label="three_four")
            # plt.legend()
            # plt.title("%s" % (sequence))
            # plt.savefig(op.join(figures_dir, "paired_dist_epo", subject, label.name, "%s.png" % (epoch)))
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
            
            # ensure_dir(op.join(figures_dir, "inOut_dist", epoch, label.name))
            # plt.figure(figsize=figsize)
            # # plt.ylim(-1, 7)
            # plt.plot(times, all_out_seqs.mean((0, 1, 2)), label="out_seq")
            # plt.plot(times, all_in_seqs.mean((0, 1, 2)), label="in_seq")
            # plt.plot(times, diff_inout[:, 0, :].mean(0), label='diff')
            # plt.legend()
            # plt.title("in/out_%s_%s" % (subject, label.name))
            # plt.savefig(op.join(figures_dir, "inOut_dist", epoch, label.name, "%s.png" % (subject)))
            # plt.close()
            
        one_two_similarities = np.array(one_two_similarities)
        one_three_similarities = np.array(one_three_similarities)  
        one_four_similarities = np.array(one_four_similarities)   
        two_three_similarities = np.array(two_three_similarities)  
        two_four_similarities = np.array(two_four_similarities)   
        three_four_similarities = np.array(three_four_similarities)
        
        similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
                        two_three_similarities, two_four_similarities, three_four_similarities]
        
        in_seq, out_seq = get_inout_seq(sequence, similarities)
        
        
        # plot paired distances averaged across epochs, per sub ---- done
        # ensure_dir(op.join(figures_dir, "paired_dist_ave", label.name))
        # ylims = (1, 8)
        # plt.figure(figsize=(16, 7))
        # # plt.ylim(ylims)
        # plt.plot(times, one_two_similarities.mean(0), label="one_two")
        # plt.plot(times, one_three_similarities.mean(0), label="one_three")
        # plt.plot(times, one_four_similarities.mean(0), label="one_four")
        # plt.plot(times, two_three_similarities.mean(0), label="two_three")
        # plt.plot(times, two_four_similarities.mean(0), label="two_four")
        # plt.plot(times, three_four_similarities.mean(0), label="three_four")
        # plt.legend()
        # plt.title("%s" % (sequence))
        # plt.savefig(op.join(figures_dir, "paired_dist_ave", label.name, "%s.png" % (subject)))
        # plt.close()
                
        all_in_seqs.append(in_seq)
        all_out_seqs.append(out_seq)
        
        # all_in_seqs = np.array(all_in_seqs)
        # all_out_seqs = np.array(all_out_seqs)

        # diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)
    
        # plot in/out and diff averaged across epochs and subs ---- done
        # ensure_dir(op.join(figures_dir, "inOut_dist", label.name))
        # plt.figure(figsize=(16, 7))
        # plt.ylim(-1, 7)
        # plt.plot(times, all_out_seqs.mean((0, 1, 2)), label="out_seq")
        # plt.plot(times, all_in_seqs.mean((0, 1, 2)), label="in_seq")
        # plt.plot(times, diff_inout[:, 0, :].mean(0), label='diff')
        # plt.legend()
        # plt.title("in/out_%s" % (subject))
        # plt.savefig(op.join(figures_dir, "inOut_dist", label.name, "%s.png" % (subject)))
        # plt.close()
        
    # plot paired distances averaged across epochs and subs ---- done
    # plt.figure(figsize=(16, 7))
    # plt.plot(times, one_two_similarities.mean(0), label="one_two")
    # plt.plot(times, one_three_similarities.mean(0), label="one_three")
    # plt.plot(times, one_four_similarities.mean(0), label="one_four")
    # plt.plot(times, two_three_similarities.mean(0), label="two_three")
    # plt.plot(times, two_four_similarities.mean(0), label="two_four")
    # plt.plot(times, three_four_similarities.mean(0), label="three_four")
    # plt.legend()
    # plt.title("paired_dist_ave")
    # plt.savefig(op.join(figures_dir, "paired_dist_ave.png"))
    # plt.close()

    all_in_seqs = np.array(all_in_seqs)
    all_out_seqs = np.array(all_out_seqs)
    diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

    # plot the difference in vs. out sequence across epochs
    # ensure_dir(op.join(figures_dir, "ave"))
    # plt.figure(figsize=(16, 7))
    # plt.plot(times, diff_inout[:, 0, :].mean(0), label='practice', color='C7', alpha=0.6)
    # plt.plot(times, diff_inout[:, 1, :].mean(0), label='block_1', color='C1', alpha=0.6)
    # plt.plot(times, diff_inout[:, 2, :].mean(0), label='block_2', color='C2', alpha=0.6)
    # plt.plot(times, diff_inout[:, 3, :].mean(0), label='block_3', color='C3', alpha=0.6)
    # plt.plot(times, diff_inout[:, 4, :].mean(0), label='block_4', color='C4', alpha=0.6)
    # plt.legend()
    # plt.savefig(op.join(figures_dir, 'ave', '%s.png' % (label.name)))
    # plt.close()

    # plot paired distances per epoch averaged across subs ---- done
    # ensure_dir(op.join(figures_dir, "paired_dist_ave", label.name))
    # for epoch, sim in zip(epochs_list, similarities_list):
    #     plt.figure(figsize=(16, 7))
    #     # plt.ylim(0, 9)
    #     for i, lab in zip(d[epoch][sim], similarities_list):
    #         plt.plot(times, i, label=lab)
    #     plt.legend()
    #     plt.title("%s" % (epoch))
    #     plt.savefig(op.join(figures_dir, "paired_dist_ave", label.name, "%s.png" % (epoch)))
    #     plt.close()
    
    # plot the difference in vs. out sequence averaging all epochs
    # ensure_dir(op.join(figures_dir, 'source', 'ave', '%s' % ('ols' if ols else 'basic')))
    # fname = label.name + '.png'
    # if ols:
    #     loca = 'ols'
    # else:
    #     loca = 'basic'
    # plt.figure(figsize=(15, 7))
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
    # plt.savefig(op.join(figures_dir, 'source', 'ave', loca, fname))
    # plt.close()
    
    all_rhos = []
    for sub in range(len(subjects)):
        rhos = []
        for t in range(len(times)):
            rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
        all_rhos.append(rhos)
    all_rhos = np.array(all_rhos)

    # corr_dir = figures_dir / 'correlations' / label.name
    # ensure_dir(corr_dir)
    # # plot rhos per subject
    # for isub, sub in enumerate(subjects):
    #     plt.subplots(1, 1, figsize=(16, 7))
    #     plt.plot(times, all_rhos[isub, :, 0], label="rho")
    #     sig = all_rhos[isub, :, 1] < 0.05 # not good for small sample size
    #     plt.fill_between(times, 0, all_rhos[isub, :, 0], where=sig, alpha=0.3)
    #     plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
    #     plt.axvline(0, color='grey')
    #     plt.legend()
    #     plt.ylim(-1.5, 1.5)
    #     plt.axhline(y = -1, color='k', ls="dashed")
    #     plt.axhline(y = 1, color='k', ls="dashed")
    #     plt.axhline(y = 0, color='k')
    #     plt.title(sub)
    #     plt.savefig(corr_dir / f'{sub}.png')
    #     plt.close()
        
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
    plt.axvspan(.0, .2, color='gray', label='stimulus', alpha=.1)
    plt.axvline(0, color='grey')
    plt.legend()
    plt.title('mean')
    plt.savefig(figures_dir / 'correlations' / f"{label.name}.png")
    plt.close()