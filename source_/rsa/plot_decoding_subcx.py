import os.path as op
import mne
import matplotlib.pyplot as plt
from base import *
from config import *
import gc
from tqdm.auto import tqdm

lock = 'button'
trial_type = 'random'
analysis = 'subcx_decoding'

subjects = SUBJS
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
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
labels = VOLUME_LABELS

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
                
                for label in labels_list:
                    if label not in label_dict:
                        label_dict[label] = []
                    res_dir = res_path / 'source' / lock / trial_type / label
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
nrows, ncols = 9, 5
chance = 25
color1 = "#1f77b4"
color2 = "#F79256"
for lock in ['stim', 'button']:
    for trial_type in ['pattern', 'random']:
        print(lock, trial_type)
        figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
        score_dict = {}
        for label in tqdm(labels):
            if label not in score_dict:
                score_dict[label] = []
            res_dir = res_path / 'source' / lock / trial_type / label
            for subject in subjects:
                score = np.load(res_dir / f"{subject}-scores.npy")
                score_dict[label].append(score)
            score_dict[label] = np.array(score_dict[label])
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(20,7))
        for i, (ax, label) in enumerate(zip(axs.flat, labels)):
            if lock == 'stim':
                ax.axvspan(0, 0.2, color='grey', alpha=.2)
                color = color1
            else:
                ax.axvline(0, color='black', alpha=.2)
                color=color2
            score = score_dict[label] * 100
            sem = np.std(score, axis=0) / np.sqrt(len(subjects))
            m1 = np.array(score.mean(0) + np.array(sem))
            m2 = np.array(score.mean(0) - np.array(sem))
            ax.axhline(chance, color='black', ls='dashed', alpha=.5)
            ax.set_title(f"${label}$", fontsize=8)    
            p_values = decod_stats(score - chance)
            sig = p_values < 0.05
            ax.fill_between(times, m1, m2, color='0.6')
            ax.fill_between(times, m1, m2, color=color, where=sig, alpha=1)
            ax.fill_between(times, chance, m2, where=sig, color=color, alpha=0.7, label='significant')
            if i == 0:
                legend = ax.legend()
                plt.setp(legend.get_texts(), fontsize=7)  # Adjust legend size
        plt.savefig(FIGURE_PATH / analysis / 'source' / f'{lock}-{trial_type}-ave.pdf', transparent=True)
        plt.close()
