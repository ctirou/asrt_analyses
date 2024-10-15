from base import ensure_dir, gat_stats, rsync_files
from config import *
import numpy as np
from mne import read_labels_from_annot
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import numba

analysis = "time_generalization"
data_path = TIMEG_DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'
jobs = -1

res_dir = data_path / 'results'

@numba.jit
def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    d = rank_x - rank_y
    d_squared = np.sum(d * d)
    rho = 1 - (6 * d_squared) / (n * (n * n - 1))
    return rho

# get labels
labels = read_labels_from_annot(subject='sub01', parc='aparc', hemi='both', subjects_dir=FREESURFER_DIR, verbose=False)

for ilabel, label in enumerate(labels):
    print(label.name)
    # load patterns and randoms time-generalization on all epochs
    patterns, randoms = [], []
    ensure_dir(res_dir / "pval" / label.name)
    for subject in tqdm(subjects):
        pattern = np.load(res_dir / 'source' / lock / label.name / 'pattern' / f"{subject}-all-scores.npy")
        patterns.append(pattern)
        random = np.load(res_dir / 'source' / lock / label.name / 'random' / f"{subject}-all-scores.npy")
        randoms.append(random)
    patterns = np.array(patterns)
    randoms = np.array(randoms)

    # contrast with significance
    contrasts = patterns - randoms
    pval = gat_stats(contrasts, jobs)
    fname = res_dir / 'pval' / label.name / 'constrast-pval.npy' 
    np.save(fname, pval)

    # look at the correlations
    all_patterns, all_randoms = [], []
    for subject in subjects:
        patterns, randoms = [], []
        for epoch_num in [1, 2, 3, 4]:
            pattern = np.load(res_dir / 'source' / lock / label.name / 'pattern' / f"{subject}-{epoch_num}-scores.npy")
            patterns.append(pattern)
            random = np.load(res_dir / 'source' / lock / label.name / 'random' / f"{subject}-{epoch_num}-scores.npy")
            randoms.append(random)
        patterns = np.array(patterns)
        randoms = np.array(randoms)
        all_patterns.append(patterns)
        all_randoms.append(randoms)
    all_patterns = np.array(all_patterns)
    all_randoms = np.array(all_randoms)

    for trial_type, time_gen in zip(['pattern', 'random'], [all_patterns, all_randoms]):
        # Initialize the output array
        rhos = np.zeros((11, 813, 813))
        # Compute Spearman correlation for each subject
        for subject in tqdm(range(11)):
            for i in tqdm(range(813)):
                for j in range(813):
                    values = time_gen[subject, :, i, j]
                    # old_rho, _ = spearmanr(time_gen[subject, :, i, j], range(len(values)))
                    rho = spearman_rank_correlation(time_gen[subject, :, i, j], range(len(values)))
                    rhos[subject, i, j] = rho
        pval = gat_stats(rhos, jobs)
        fname = res_dir / 'pval' / label.name / f'{trial_type}-rhos-pval.npy' 
        np.save(fname, pval)
    
    source = "/Users/coum/Desktop/pred_asrt/results/pval"
    destination = "/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results"
    rsync_files(source, destination, options="-av")

