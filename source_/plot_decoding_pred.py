import os
import numpy as np
import pandas as pd
from base import *
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import ttest_1samp, spearmanr
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score


trial_type = "pattern"
subjects = SUBJS
lock = "stim"
params = "pred_decoding"
sessions = EPOCHS
subjects_dir = FREESURFER_DIR
verbose = "error"
hemi = 'both'
chance = .5

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

blocks = ['prac', 'b1', 'b2', 'b3', 'b4']
similarity_names = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']        

in_seqs, out_seqs = [], []
decod_in_lab = {}
decod_in_lab2 = {}
corr_in_lab = {}

# get label names
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for label in labels]

for ilabel, label in enumerate(label_names):
    
    if ilabel in [1, 19, 34, 47]:
        continue

    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)
    
    all_in_seqs, all_out_seqs = [], []
    
    for subject in subjects[:1]:
    
        res_dir = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type / subject
        ensure_dir(res_dir / "plots")
    
        one_two_similarities = list()
        one_three_similarities = list()
        one_four_similarities = list() 
        two_three_similarities = list()
        two_four_similarities = list() 
        three_four_similarities = list()
        
        behav_dir = RAW_DATA_DIR / subject / 'behav_data'
        sequence = get_sequence(behav_dir)
        
        sub_scores, sub_rsa, sub_cms = [], [], []
                            
        for session_id, session in enumerate(sessions):
            
            sub_scores.append(np.load(res_dir / f"{label}_{subject}_{session_id}-scores.npy"))
            sub_rsa.append(np.load(res_dir / f"{label}_{subject}_{session_id}-rsa.npy"))
            sub_cms.append(np.load(res_dir / f"{label}_{subject}_{session_id}-cm.npy") )
            
        sub_scores = np.array(sub_scores)
        sub_cms = np.array(sub_cms)
            
        sub_rsa = np.array(sub_rsa)
        one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
        for sim, sim_list in enumerate([one_four, one_three, one_two, three_four, two_four, two_three]):
            sim_list.append(sub_rsa[session_id, sim, :])
        for all_sims, sim_list in zip([one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities], 
                                        [one_two, one_three, one_four, two_three, two_four, three_four]):
                all_sims.append(np.array(sim_list))
        
        
        for all_sims in [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]:
            all_sims = np.array(all_sims)
            
        similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]
        in_seq, out_seq = get_inout_seq(sequence, similarities)
        all_in_seqs.append(np.array(in_seq))
        all_out_seqs.append(np.array(out_seq))
        
        fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7), sharey=False)
        fig.suptitle(f'{label} // {subject}')
        for i, (ax, session) in enumerate(zip(axs.flat[:5], sessions)):
            ax.plot(times, sub_scores[i])
            ax.axvspan(0, 0.2, color='grey', alpha=.2)
            ax.set_title(session)
            ax.axhline(chance, color='black', ls='dashed', alpha=.5)
            ax.set_ylim(0, 1)
        for i, ax in zip(range(5), axs.flat[5:]):        
            disp = ConfusionMatrixDisplay(sub_cms[i, :, :, 40:80].mean(-1), display_labels=[1, 2, 3, 4])
            disp.plot(ax=ax)
            disp.im_.set_clim(0, 1)  # Set colorbar limits
        plt.savefig(res_dir / 'plots' / f"cms_{label}_{subject}.png")
        plt.close()

    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    decod_in_lab[label] = diff_inout
    
    decod_in_lab2[label] = [np.squeeze(all_in_seq).mean(axis=1), np.squeeze(all_out_seq).mean(axis=1)]
    
    # all_rhos = []
    # for sub in range(len(subjects)):
    #     rhos = []
    #     for t in range(len(times)):
    #         rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
    #     all_rhos.append(rhos)
    # all_rhos = np.array(all_rhos)
    # corr_in_lab[label] = all_rhos
        
nrows, ncols = 7, 10

# plot diff in/out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    practice = decod_in_lab[label][:, 0, :].mean(0)
    learning = decod_in_lab[label][:, 1:, :].mean((0, 1))
    
    ax.plot(times, practice, label='practice')
    ax.plot(times, learning, label='learning')
    ax.set_title(label)
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    
    for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
        tick.label.set_fontsize(8)
plt.savefig(res_dir / 'plots' / f"diff_in_out.png")
plt.close()

# plot in vs out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='constrained')
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    ins = decod_in_lab2[label][0].mean(axis=(0, 1))
    outs = decod_in_lab2[label][1].mean(axis=(0, 1))
    ax.plot(times, ins, label='in_seq') 
    ax.plot(times, outs, label='out_seq')
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
    ax.axhline(chance, color='black', ls='dashed', alpha=.5)
    
    ax.set_title(label)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=8)  # Adjust legend size
    
    for tick in ax.xaxis.get_major_ticks():  # Adjust x-axis label size
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():  # Adjust y-axis label size
        tick.label.set_fontsize(8)
plt.savefig(res_dir / 'plots' / f"in_vs_out.png")
plt.close()

# correlations
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
# for ax, label in zip(axs.flat, label_names):
#     rho = corr_in_lab[label][:, :, 0].mean(0)
#     ax.plot(times, rho, label='rho')
#     ax.axvspan(0, 0.2, color='grey', alpha=.2)
#     ax.axhline(0, color='black', ls='dashed', alpha=.5)
#     ax.set_title(label)
# plt.show()