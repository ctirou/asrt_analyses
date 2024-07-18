import os.path as op
import mne
import matplotlib.pyplot as plt
from base import *
from config import *
import gc
from tqdm.auto import tqdm

lock = 'stim'
trial_type = 'pattern'
verbose = True
analysis = 'subcx_decoding'

subjects = SUBJS
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
figures = FIGURE_PATH / analysis / 'source' / lock / trial_type
ensure_dir(figures)

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

# get labels
labels = [l for l in sorted(os.listdir(res_path / 'source' / lock / trial_type)) if not l.startswith(".")]

vol_labels_lh = [l for l in labels if l.endswith('lh')]
vol_labels_rh = [l for l in labels if l.endswith('rh')]
vol_labels_others = [l for l in labels if not l.endswith(('lh', 'rh'))]

lh_scores, rh_scores, other_scores = {}, {}, {}

for subject in subjects:
    sub_dict = dict()
    
    for hemi, labels_list, label_dict in zip(['lh', 'rh', 'others'], [vol_labels_lh, vol_labels_rh, vol_labels_others], [lh_scores, rh_scores, other_scores]):
        
        for label in tqdm(labels_list):
            if label not in label_dict:
                label_dict[label] = []
            res_dir = res_path / 'source' / lock / trial_type / label        
            score = np.load(res_dir / f"{subject}-scores.npy")
            label_dict[label].append(score)
            if label not in sub_dict:
                sub_dict[label] = score
        
        nrows, ncols = 4, 4
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(25, 13))
        for i, (ax, label) in enumerate(zip(axs.flat, labels_list)):
            ax.plot(times, sub_dict[label])
            ax.axhline(.25, color='black', ls='dashed', alpha=.5)
            ax.set_title(f"${label}$")    
            ax.axvspan(0, 0.2, color='grey', alpha=.2)
        fig.savefig(figures / f"{subject}-{hemi}.pdf", transparent=True)
        plt.close()

all_labels = vol_labels_lh + vol_labels_rh + vol_labels_others
all_labels = sorted(all_labels)

nrows, ncols = 9, 5
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, layout='tight', figsize=(15, 13))
for i, (ax, label) in enumerate(zip(axs.flat, all_labels)):
    ax.plot(times, sub_dict[label])
    ax.axhline(.25, color='black', ls='dashed', alpha=.5)
    ax.set_title(f"${label}$")    
    ax.axvspan(0, 0.2, color='grey', alpha=.2)
fig.savefig(figures / f"{subject}-all.pdf", transparent=True)
plt.close()
