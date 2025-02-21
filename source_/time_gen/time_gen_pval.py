import os
from base import ensure_dir, gat_stats, decod_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, spearmanr, zscore
import numba
import pandas as pd
from joblib import Parallel, delayed

data_path = TIMEG_DATA_DIR
subjects, subjects_dir = SUBJS, FREESURFER_DIR

analysis = 'tg_rdm_emp'

lock = 'stim'
# network and custom label_names
n_parcels = 200
n_networks = 7
networks = NETWORKS
res_dir = data_path / 'results' / 'source' / lock
res_dir = data_path / analysis / lock
figures_dir = FIGURES_DIR / "time_gen" / "source" / lock
ensure_dir(figures_dir)
overwrite = False

names = NETWORK_NAMES
times = np.linspace(-1.5, 1.5, 305)
chance = .25
threshold = .05

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
all_diags = {}
patterns, randoms = {}, {}
all_patterns, all_randoms = {}, {}
for network in networks:
    print(f"Processing {network}...")
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        all_patterns[network], all_randoms[network] = [], []
    all_pat, all_rand, all_diag = [], [], []
    patpat, randrand = [], []
    for i, subject in enumerate(subjects):
        pat, rand = [], []
        for j in [0, 1, 2, 3, 4]:
            pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-{j}-scores.npy"))
            rand.append(np.load(res_dir / network / 'random' / f"{subject}-{j}-scores.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_dir / network / 'pattern' / f"{subject}-all-scores.npy"))
        all_rand.append(np.load(res_dir / network / 'random' / f"{subject}-all-scores.npy"))
        
        diag = np.array(all_pat) - np.array(all_rand)
        all_diag.append(np.diag(diag[i]))
        
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    all_diags[network] = np.array(all_diag)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)
    
    # save time gen pvals
    ensure_dir(res_dir / network / "pval")
    if not op.exists(res_dir / network / "pval" / "all_pattern-pval.npy") or overwrite:
        pval = gat_stats(all_patterns[network] - chance, -1)
        np.save(res_dir / network / "pval" / "all_pattern-pval.npy", pval)
    if not op.exists(res_dir / network / "pval" / "all_random-pval.npy") or overwrite:
        pval = gat_stats(all_randoms[network] - chance, -1)
        np.save(res_dir / network / "pval" / "all_random-pval.npy", pval)
    if not op.exists(res_dir / network / "pval" / "all_contrast-pval.npy") or overwrite:
        pval = gat_stats(all_patterns[network] - all_randoms[network], -1)
        np.save(res_dir / network / "pval" / "all_contrast-pval.npy", pval)
    
    # save blocks x time gen correlation and pvals
    ensure_dir(res_dir / network / "corr")
    if not op.exists(res_dir / network / "corr" / "rhos_blocks.npy") or overwrite:
        contrasts = patterns[network] - randoms[network]
        contrasts = zscore(contrasts, axis=-1)
        all_rhos = []
        for sub in range(len(subjects)):
            rhos = np.empty((times.shape[0], times.shape[0]))
            vector = [0, 1, 2, 3, 4]  # vector to correlate with
            contrast = contrasts[sub]
            results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
            for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
                rhos[t, g] = results[idx]
            all_rhos.append(rhos)
        all_rhos = np.array(all_rhos)
        np.save(res_dir / network / "corr" / "rhos_blocks.npy", all_rhos)
        pval = gat_stats(all_rhos, -1)
        np.save(res_dir / network / "corr" / "pval_blocks-pval.npy", pval)
    
    # save learn df x time gen correlation and pvals
    if not op.exists(res_dir / network / "corr" / "rhos_learn.npy") or overwrite:
        contrasts = patterns[network] - randoms[network]
        contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
        all_rhos = []
        for sub in range(len(subjects)):
            rhos = np.empty((times.shape[0], times.shape[0]))
            vector = learn_index_df.iloc[sub]  # vector to correlate with
            contrast = contrasts[sub]
            results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
            for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
                rhos[t, g] = results[idx]
            all_rhos.append(rhos)
        all_rhos = np.array(all_rhos)
        np.save(res_dir / network / "corr" / "rhos_learn.npy", all_rhos)
        pval = gat_stats(all_rhos, -1)
        np.save(res_dir / network / "corr" / "pval_learn-pval.npy", pval)    
