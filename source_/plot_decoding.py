import os.path as op
import numpy as np
from base import ensure_dir, decod_stats, get_sequence, get_inout_seq
from config import *
import matplotlib.pyplot as plt
from mne import read_labels_from_annot, read_epochs
from scipy.stats import ttest_1samp, spearmanr
import gc
from pathlib import Path

lock = "stim"
trial_type = "pattern"
subjects = SUBJS
analysis = "concatenated"
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
verbose = "error"
hemi = 'both'
chance = .5

summary = False

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']

decod_in_lab = {}
decod_in_lab2 = {}
corr_in_lab = {}
decoding = dict()

# get label names
best_regions = [6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 42, 43, 50, 51, 58, 59]
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
label_names = [label.name for ilabel, label in enumerate(labels) if ilabel in best_regions]
del labels

figures = res_path / "figures" / analysis / 'source' / lock / trial_type
ensure_dir(figures)
gc.collect()

for ilabel, label in enumerate(label_names):
    
    print(f"{str(ilabel+1).zfill(2)}/{len(label_names)}", label)
    all_in_seqs, all_out_seqs = [], []
    decoding[label] = list()
    
    for subject in subjects:
        
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
            
            # res_dir = res_path / analysis / 'source' / lock / trial_type / label / subject / session
            res_dir = res_path / analysis / 'source' / lock / trial_type / label / subject
            
            # other_dir = Path('/Users/coum/Desktop/decoding_cvm')
            
            sub_scores.append(np.load(res_dir / "scores.npy"))
            
            # sub_scores.append(np.load(other_dir / 'source' / lock / trial_type / label / subject / session / "scores.npy"))
            # sub_cms.append(np.load(other_dir / 'source' / lock / trial_type / label / subject / session / "cms.npy") )
            # sub_rsa.append(np.load(other_dir / 'source' / lock / trial_type / label / subject / session / "rsa.npy"))
            
        sub_scores = np.array(sub_scores)
        # sub_cms = np.array(sub_cms)

        # sub_rsa = np.array(sub_rsa)
        # one_two, one_three, one_four, two_three, two_four, three_four = [], [], [], [], [], []
        # for session_id, _ in enumerate(sessions):
        #     for sim, sim_list in enumerate([one_four, one_three, one_two, three_four, two_four, two_three]):
        #         sim_list.append(sub_rsa[session_id, sim, :])
        
        # for all_sims, sim_list in zip(
        #     [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities], 
        #     [one_two, one_three, one_four, two_three, two_four, three_four]):
        #         all_sims.append(np.array(sim_list))
                    
        # similarities = [one_two_similarities, one_three_similarities, one_four_similarities, two_three_similarities, two_four_similarities, three_four_similarities]
        # in_seq, out_seq = get_inout_seq(sequence, similarities)
        # all_in_seqs.append(np.array(in_seq))
        # all_out_seqs.append(np.array(out_seq))
        
        if summary:
            from sklearn.metrics import ConfusionMatrixDisplay
            ensure_dir(figures / "summary" / label)
            fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7), sharey=False)
            fig.suptitle(f'{subject} / ${label}$')
            times_win = np.where((times >= 0) & (times <= 0.2))[0]
            for i, (ax1, ax2, session) in enumerate(zip(axs.flat[:5], axs.flat[5:], sessions)):
                ax1.plot(times, sub_scores[i])
                ax1.axvspan(0, 0.2, color='grey', alpha=.2)
                ax1.set_title(session)
                ax1.axhline(chance, color='white', ls='dashed', alpha=.5)
                ax1.set_ylim(0.2, 0.8)
                ax1.grid(True, color='grey', alpha=0.3)
                max_score = np.argmax(sub_scores[i][times_win]) + np.where(times==0)[0][0]
                ax1.annotate(f'Max Score: {sub_scores[i][max_score]:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
                ax1.annotate('', xy=(times[max_score], sub_scores[i][max_score]), xytext=(times[max_score], sub_scores[i][max_score] + 0.1),
                            arrowprops=dict(arrowstyle='->', color='white'))
                disp = ConfusionMatrixDisplay(sub_cms[i, max_score, :, :], display_labels=[1, 2, 3, 4])
                disp.plot(ax=ax2)
                disp.im_.set_clim(0, 1)  # Set colorbar limits
            plt.savefig(figures / "summary" / label / f"{subject}.png")
            plt.close()
            
        decoding[label].append(sub_scores)
    decoding[label] = np.array(decoding[label])
        
    all_in_seq = np.array(all_in_seqs)
    all_out_seq = np.array(all_out_seqs)
    diff_inout = np.squeeze(all_in_seq.mean(axis=1) - all_out_seq.mean(axis=1))
    decod_in_lab[label] = diff_inout
    decod_in_lab2[label] = [np.squeeze(all_in_seq).mean(axis=1), np.squeeze(all_out_seq).mean(axis=1)]
    
    # if not op.exists(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy"):
    #     corr = []
    #     for sub in range(len(subjects)):
    #         rhos = []
    #         for t in range(len(times)):
    #             rhos.append(spearmanr([0, 1, 2, 3, 4], diff_inout[sub, :, t]))
    #         corr.append(rhos)
    #     corr = np.array(corr)
    #     np.save(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy", corr)
    # corr = np.load(res_path / analysis / 'source' / lock / trial_type / label / "corr.npy")
    # corr_in_lab[label] = corr
        
nrows, ncols = 7, 10

# decoding
chance = 0.25
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
fig.suptitle(f"${lock}$ / ${trial_type}$ / decoding")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    # score = decoding[label][:, 1:5, :]
    score = decoding[label][:, :, :]
    # ax.plot(times, score.mean((0, 1)), label='roc_auc')
    ax.plot(times, score.mean((0, 1)), label='acc')
    ax.axhline(chance, color='black', ls='dashed', alpha=.5)
    ax.set_title(f"${label}$", fontsize=8)    
    p_values = decod_stats(score.mean(1) - chance)
    p_values_unc = ttest_1samp(score, axis=0, popmean=0)[1]
    sig = p_values < 0.05
    ax.fill_between(times, chance, score.mean((0, 1)), where=sig, alpha=.4)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
plt.savefig(figures / f"decoding_{lock}_{trial_type}.pdf")
plt.close()

# plot diff in/out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
fig.suptitle(f"${lock}$ / ${trial_type}$ / diff_in_out")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    practice = decod_in_lab[label][:, 0, :].mean(0)
    learning = decod_in_lab[label][:, 1:, :].mean((0, 1))
    ax.plot(times, practice, label='practice')
    ax.plot(times, learning, label='learning')
    ax.set_title(f"${label}$", fontsize=8)
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)        
plt.savefig(figures / f"diff_in_out.pdf")
plt.close()

# plot in vs out
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
fig.suptitle(f"${lock}$ / ${trial_type}$ / in_vs_out_decoding")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    ins = decod_in_lab2[label][0].mean(axis=(0, 1))
    outs = decod_in_lab2[label][1].mean(axis=(0, 1))
    ax.plot(times, ins, label='in') 
    ax.plot(times, outs, label='out')
    ax.set_title(f"${label}$", fontsize=8)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
plt.savefig(figures / f"in_vs_out.pdf")
plt.close()

# correlations
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(40, 13))
fig.suptitle(f"${lock}$ / ${trial_type}$ / correlations")
for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
    rho = corr_in_lab[label][:, :, 0]
    ax.plot(times, rho.mean(0), label='rho')
    ax.axhline(0, color='black', ls='dashed', alpha=.5)
    ax.set_title(f"${label}$", fontsize=8)
    if i == 0:
        legend = ax.legend()
        plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
    p_values = decod_stats(rho)
    p_values_unc = ttest_1samp(rho, axis=0, popmean=0)[1]
    sig = p_values < 0.05
    sig_unc = p_values_unc < 0.05
    ax.fill_between(times, 0, rho.mean(0), where=sig_unc, color='C2', alpha=1)
    ax.fill_between(times, 0, rho.mean(0), where=sig, alpha=0.3)
    if lock == 'stim':
        ax.axvspan(0, 0.2, color='grey', alpha=.2)
plt.savefig(figures / "correlations.pdf")
plt.close()