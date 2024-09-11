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

# get label_names
# label_names = [l for l in sorted(os.listdir(res_path / 'source' / lock / trial_type)) if not l.startswith(".")]
label_names = sorted(SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else sorted(SURFACE_LABELS_RT + VOLUME_LABELS_RT)

vol_labels_lh = [l for l in label_names if l.endswith('lh')]
vol_labels_rh = [l for l in label_names if l.endswith('rh')]
vol_labels_others = [l for l in label_names if not l.endswith(('lh', 'rh'))]

lh_scores, rh_scores, other_scores = {}, {}, {}

# plot per subject
for lock in ['stim', 'button']:
    for trial_type in ['pattern', 'random']:
        print(lock, trial_type)
        for subject in tqdm(subjects):
            sub_dict = dict()
    
            for hemi, labels_list, label_dict in zip(['lh', 'rh', 'others'], [vol_labels_lh, vol_labels_rh, vol_labels_others], [lh_scores, rh_scores, other_scores]):
            # for ilabel, label in enumerate(label_names):    
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
# color1 = "#1f77b4"
# color2 = "#F79256"
color1 = "#1fb45c"
color2 = "#00B2CA"
for lock in ['stim', 'button']:
    label_names = sorted(SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else sorted(SURFACE_LABELS_RT + VOLUME_LABELS_RT)
    trial_type = 'pattern'
    # for trial_type in ['pattern', 'random']:
    print(lock, trial_type)
    figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
    decoding = {}
    for label in tqdm(label_names):
        if label not in decoding:
            decoding[label] = []
        res_dir = res_path / label
        for subject in subjects:
            score = np.load(RESULTS_DIR / analysis / 'source' / lock / trial_type / label / f"{subject}-scores.npy")
            decoding[label].append(score)
        decoding[label] = np.array(decoding[label])
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(10, 15))
    for i, (ax, label) in enumerate(zip(axs.flat, label_names)):
        if lock == 'stim':
            ax.axvspan(0, 0.2, color='grey', alpha=.2, label="Stimulus")
            color = color1
        else:
            ax.axvline(0, color='black', alpha=.2)
            color=color2
        score = decoding[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score.mean(0) + np.array(sem))
        m2 = np.array(score.mean(0) - np.array(sem))
        ax.axhline(chance, color='black', ls='dashed', alpha=.5, label='Chance')
        
        if label.endswith('lh'):
            title = 'L. ' + label[:-3].capitalize()
        elif label.endswith('rh'):
            title = 'R. ' + label[:-3].capitalize()
        else:
            title = label.capitalize()
        ax.set_title(f"${title}$", fontsize=10)    
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

color1 = "#1fb45c"
color2 = "#00B2CA"
chance = 25
# plot per label
label_names = sorted(SURFACE_LABELS + VOLUME_LABELS) if lock == 'stim' else sorted(SURFACE_LABELS_RT + VOLUME_LABELS_RT)
nrows = 10 if lock == 'stim' else 9
ncols = 4
for ilabel in tqdm(range(len(label_names))):
    if ilabel % 2 == 0:
        fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        fig.subplots_adjust(hspace=0)
        label = label_names[ilabel]
        if label.endswith('lh'):
            title = 'L. ' + label[:-3].capitalize()
        elif label.endswith('rh'):
            title = 'R. ' + label[:-3].capitalize()
        axs[0].set_title(f"${title}$", fontsize=10)    
        # axs[0].text(-0.19, 40, f"{title}",
        #             weight='semibold', style='italic', ha='left',
        #             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
        
        if ilabel % nrows == 0:
            if lock == 'stim':
                axs[0].text(0.1, 0.22, "$Stimulus$", fontsize=9, zorder=10, ha='center')
            else:
                axs[0].text(0.1, 0.22, "$Button press$", fontsize=9, zorder=10, ha='center')
        
        for i in range(2):
            axs[i].set_ylim(20, 45)
            yticks = axs[i].get_yticks()
            yticks = yticks[1:-1]  # Remove first and last element
            axs[i].set_yticks(yticks)  # Set new y-ticks
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].axhline(chance, color='black', ls='dashed', alpha=.7, zorder=-1)
            axs[i].text(0.6, 22.5, "$Chance$", fontsize=9, zorder=10, ha='center')
            if lock == 'stim':
                axs[i].axvspan(0, 0.2, color='grey', alpha=.2, label="Stimulus")
            else:
                axs[i].axvline(0, color='black', alpha=.5)
            if ilabel == 0:
                axs[i].set_ylabel("Accuracy (%)")
            if ilabel % ncols != 0:
                axs[i].set_yticklabels([])
                
        if ilabel in range(len(label_names))[-4:]:
            axs[1].get_xaxis().set_visible(True)
            axs[1].set_xlabel("Time (s)") 
        else:
            axs[1].set_xticklabels([])
                    
        score = decoding[label] * 100
        sem = np.std(score, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score.mean(0) + np.array(sem))
        m2 = np.array(score.mean(0) - np.array(sem))
        p_values = decod_stats(score - chance, jobs)
        sig = p_values < 0.05
        axs[0].fill_between(times, m1, m2, facecolor='0.6')
        axs[0].fill_between(times, m1, m2, facecolor=color1, where=sig, alpha=1)
        axs[0].fill_between(times, chance, m2, facecolor=color1, where=sig, alpha=0.7)
        axs[0].spines["bottom"].set_visible(False)
        axs[0].xaxis.set_ticks_position('none')  # New line: remove x-ticks of the upper plot
        axs[0].xaxis.set_tick_params(labelbottom=False)  # New line: remove x-tick labels of the upper plot
        label = label_names[ilabel+1]
        score2 = decoding[label] * 100
        sem = np.std(score2, axis=0) / np.sqrt(len(subjects))
        m1 = np.array(score2.mean(0) + np.array(sem))
        m2 = np.array(score2.mean(0) - np.array(sem))
        p_values = decod_stats(score2 - chance, jobs)
        sig = p_values < 0.05
        axs[1].fill_between(times, m1, m2, facecolor='0.6')
        axs[1].fill_between(times, m1, m2, facecolor=color2, where=sig, alpha=1)
        axs[1].fill_between(times, chance, m2, facecolor=color2, where=sig, alpha=0.7)
        # if ilabel == 0:
        axs[0].text(0.23, 40, "Left hemisphere", fontsize=10, color=color1, ha='left', weight='normal', style='italic')
        axs[1].text(0.23, 40, "Right hemisphere", fontsize=10, color=color2, ha='left', weight='normal', style='italic')
        
        plt.savefig(FIGURE_PATH / analysis / 'source' / lock / trial_type / f'{ilabel+1}_{label}.pdf', transparent=True)
        plt.close()
