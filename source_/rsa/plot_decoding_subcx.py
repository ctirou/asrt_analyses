import os.path as op
import mne
import matplotlib.pyplot as plt
from base import *
from config import *
import gc
from tqdm.auto import tqdm

lock = 'stim'
trial_type = 'pattern'
analysis = 'decoding'
jobs = -1

subjects = SUBJS
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR / analysis / 'source' / lock / trial_type
figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
ensure_dir(figures)

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=False)
times = epochs.times
del epochs, epoch_fname
gc.collect()

# get labels
# labels = [l for l in sorted(os.listdir(res_path / 'source' / lock / trial_type)) if not l.startswith(".")]
labels = sorted(SURFACE_LABELS + VOLUME_LABELS)

vol_labels_lh = [l for l in labels if l.endswith('lh')]
vol_labels_rh = [l for l in labels if l.endswith('rh')]
vol_labels_others = [l for l in labels if not l.endswith(('lh', 'rh'))]

lh_scores, rh_scores, other_scores = {}, {}, {}

# plot per subject
for lock in ['stim', 'button']:
    for trial_type in ['pattern', 'random']:
        print(lock, trial_type)
        for subject in tqdm(subjects):
            sub_dict = dict()
    
            for hemi, labels_list, label_dict in zip(['lh', 'rh', 'others'], [vol_labels_lh, vol_labels_rh, vol_labels_others], [lh_scores, rh_scores, other_scores]):
            # for ilabel, label in enumerate(labels):    
                for label in labels_list:
                    if label not in label_dict:
                        label_dict[label] = []
                    res_dir = res_path / label
                    try:        
                        score = np.load(res_dir / f"{subject}-scores.npy")
                        label_dict[label].append(score)
                        if label not in sub_dict:
                            sub_dict[label] = score
                    except:
                        continue
            all_labels = sorted(vol_labels_lh + vol_labels_rh + vol_labels_others)

            nrows, ncols = 9, 5
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(15, 13))
            for i, (ax, label) in enumerate(zip(axs.flat, all_labels)):
                try:
                    ax.plot(times, sub_dict[label])
                    ax.axhline(.25, color='black', ls='dashed', alpha=.5)
                    ax.set_title(f"${label}$")    
                    ax.axvspan(0, 0.2, color='grey', alpha=.2)
                except:
                    continue
            fig.savefig(figures / f"{subject}.pdf", transparent=True)
            plt.close()

# plot average and run stats
nrows, ncols = 10, 4
chance = 25
color1 = "#1f77b4"
color2 = "#F79256"
for lock in ['stim', 'button']:
    trial_type = 'pattern'
    # for trial_type in ['pattern', 'random']:
    print(lock, trial_type)
    figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
    score_dict = {}
    for label in tqdm(labels):
        if label not in score_dict:
            score_dict[label] = []
        res_dir = res_path / label
        for subject in subjects:
            score = np.load(RESULTS_DIR / analysis / 'source' / lock / trial_type / label / f"{subject}-scores.npy")
            score_dict[label].append(score)
        score_dict[label] = np.array(score_dict[label])
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(10, 15))
    for i, (ax, label) in enumerate(zip(axs.flat, labels)):
        if lock == 'stim':
            ax.axvspan(0, 0.2, color='grey', alpha=.2, label="Stimulus")
            color = color1
        else:
            ax.axvline(0, color='black', alpha=.2)
            color=color2
        score = score_dict[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score.mean(0) + np.array(sem))
        m2 = np.array(score.mean(0) - np.array(sem))
        ax.axhline(chance, color='black', ls='dashed', alpha=.5, label='Chance')
        ax.set_title(f"${label.capitalize()}$", fontsize=10)    
        # ax.text(-0.19, 40, f"{label.capitalize()[:-3]}",
        # weight='semibold', style='italic', ha='left',
        # bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
        p_values = decod_stats(score - chance, jobs)
        sig = p_values < 0.05
        ax.fill_between(times, m1, m2, color='0.6')
        ax.fill_between(times, m1, m2, color=color, where=sig, alpha=1)
        ax.fill_between(times, chance, m2, where=sig, color=color, alpha=0.7, label='Significant')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.text(0.6, 22.5, "$Chance$", fontsize=9, zorder=10, ha='center')
        if i % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=8)
        if i == 0:
            legend = ax.legend()
            plt.setp(legend.get_texts(), fontsize=6)  # Adjust legend size
        if i >= (nrows-1) * ncols:
            ax.set_xlabel("Time (s)", fontsize=8)
    # plt.savefig(FIGURE_PATH / analysis / 'source' / f'new-{lock}-{trial_type}-ave.pdf', transparent=True)
    plt.savefig(FIGURE_PATH / analysis / 'source' / f'new-{lock}-{trial_type}-ave.png', transparent=True)
    plt.close()

    # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 15))
    # fig.subplots_adjust(hspace=0.5)  # Adjust space between subplots

    # for i, label in enumerate(labels[::2]):  # Step by 2 to handle pairs of plots
    #     row, col = divmod(i, ncols)  # Compute the row and column index

    #     ax_left = axs[row, col]
    #     ax_right = axs[row, col+1] if col+1 < ncols else None  # Handle the case where there's an odd number of labels

    #     # Fill in the left plot (first label)
    #     label_left = labels[2 * i]
    #     ax_left.text(-0.19, 40, f"{label_left.capitalize()[:-3]}", weight='semibold', style='italic', ha='left',
    #                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    #     score_left = score_dict[label_left] * 100
    #     sem_left = np.std(score_left, axis=0) / np.sqrt(len(subjects))
    #     m1_left = np.array(score_left.mean(0) + np.array(sem_left))
    #     m2_left = np.array(score_left.mean(0) - np.array(sem_left))
    #     p_values_left = decod_stats(score_left - chance, jobs)
    #     sig_left = p_values_left < 0.05
    #     ax_left.fill_between(times, m1_left, m2_left, facecolor='0.6')
    #     ax_left.fill_between(times, m1_left, m2_left, facecolor=color1, where=sig_left, alpha=1)
    #     ax_left.fill_between(times, chance, m2_left, facecolor=color1, where=sig_left, alpha=0.7)
    #     ax_left.axhline(chance, color='black', ls='dashed', alpha=.7, zorder=-1)
    #     ax_left.spines["top"].set_visible(False)
    #     ax_left.spines["right"].set_visible(False)

    #     # Fill in the right plot (second label, if available)
    #     if ax_right:
    #         label_right = labels[2 * i + 1]
    #         ax_right.text(-0.19, 40, f"{label_right.capitalize()[:-3]}", weight='semibold', style='italic', ha='left',
    #                     bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    #         score_right = score_dict[label_right] * 100
    #         sem_right = np.std(score_right, axis=0) / np.sqrt(len(subjects))
    #         m1_right = np.array(score_right.mean(0) + np.array(sem_right))
    #         m2_right = np.array(score_right.mean(0) - np.array(sem_right))
    #         p_values_right = decod_stats(score_right - chance, jobs)
    #         sig_right = p_values_right < 0.05
    #         ax_right.fill_between(times, m1_right, m2_right, facecolor='0.6')
    #         ax_right.fill_between(times, m1_right, m2_right, facecolor=color2, where=sig_right, alpha=1)
    #         ax_right.fill_between(times, chance, m2_right, facecolor=color2, where=sig_right, alpha=0.7)
    #         ax_right.axhline(chance, color='black', ls='dashed', alpha=.7, zorder=-1)
    #         ax_right.spines["top"].set_visible(False)
    #         ax_right.spines["right"].set_visible(False)

    #     # Set labels and ticks
    #     if row == nrows - 1:  # Only set xlabel for the bottom plots
    #         ax_left.set_xlabel("Time (s)")
    #         if ax_right:
    #             ax_right.set_xlabel("Time (s)")
    #     if col == 0:  # Only set ylabel for the first column
    #         ax_left.set_ylabel("Accuracy (%)")

    # # Final layout adjustment
    # plt.tight_layout()
    # plt.show()

# plot thalamus vs cuneus
for lock in ['stim', 'button']:
    for trial_type in ['pattern', 'random']:
        res_path = RESULTS_DIR / analysis / 'source' / lock / trial_type
        figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
        ensure_dir(figures)
        score_dict = {}
        for label in tqdm(labels):
            if label not in score_dict:
                score_dict[label] = []
            res_dir = res_path / label
            for subject in subjects:
                score = np.load(res_dir / f"{subject}-scores.npy")
                score_dict[label].append(score)
            score_dict[label] = np.array(score_dict[label])

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, layout='tight', figsize=(14,5))
        # left
        ax1.axhline(.25, color='black', ls="dashed")
        ax1.plot(times, score_dict['Thalamus-Proper-lh'].mean(0), color=color1, label='Thalamus')
        ax1.plot(times, score_dict['cuneus-lh'].mean(0), color=color2, label='Cuneus')
        ax1.axvspan(0, 0.2, color='grey', alpha=.2)
        ax1.set_title("$Left$")
        ax1.axvline(times[np.argmax(score_dict['Thalamus-Proper-lh'].mean(0))], color='black', ls=':')
        ax1.legend()
        # right
        ax2.axhline(.25, color='black', ls="dashed")
        ax2.plot(times, score_dict['Thalamus-Proper-rh'].mean(0), color=color1, label='Thalamus')
        ax2.plot(times, score_dict['cuneus-rh'].mean(0), color=color2, label='Cuneus')
        ax2.axvspan(0, 0.2, color='grey', alpha=.2)
        ax2.set_title("$Right$")
        ax2.axvline(times[np.argmax(score_dict['Thalamus-Proper-rh'].mean(0))], color='black', ls=':')
        ax2.legend()
        plt.savefig(figures / "thal_vs_cun.pdf")
