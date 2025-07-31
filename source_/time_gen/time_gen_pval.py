from base import *
from config import *
import os.path as op
import numpy as np
from scipy.stats import spearmanr, zscore
import numba
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

networks = NETWORKS + ['Cerebellum-Cortex']
res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source')
overwrite = False

times = np.linspace(-1.5, 1.5, 307)
chance = .25

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)

data_type = 'scores_skf_maxpower'  # 'scores_skf_vect' or 'scores_skf_maxpower'
data_type = 'scores_skf_vect'  # 'scores_skf_vect' or 'scores_skf_maxpower'
data_type = 'scores_skf_vect_new'  # 'scores_skf_vect' or 'scores_skf_maxpower'

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
            pat.append(np.load(res_path / network / data_type / subject /  f"pat-{j}.npy"))
            rand.append(np.load(res_path / network / data_type / subject /  f"rand-{j}.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_path / network / data_type / subject /  "pat-all.npy"))
        all_rand.append(np.load(res_path / network / data_type / subject /  "rand-all.npy"))
        
    # all_patterns[network] = np.array(all_pat)
    # all_randoms[network] = np.array(all_rand)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)
    
    # # save time gen pvals
    # res_dir = ensured(res_path / network / data_type / "pval")
    # if not op.exists(res_dir / "all_pattern-pval.npy") or overwrite:
    #     pval = gat_stats(all_patterns[network] - chance, -1)
    #     np.save(res_dir / "all_pattern-pval.npy", pval)
    # if not op.exists(res_dir / "all_random-pval.npy") or overwrite:
    #     pval = gat_stats(all_randoms[network] - chance, -1)
    #     np.save(res_dir / "all_random-pval.npy", pval)
    # if not op.exists(res_dir / "all_contrast-pval.npy") or overwrite:
    #     pval = gat_stats(all_patterns[network] - all_randoms[network], -1)
    #     np.save(res_dir / "all_contrast-pval.npy", pval)
    
    # save learn df x time gen correlation and pvals
    res_dir = ensured(res_path / network / data_type / "corr")    
    if not op.exists(res_dir / "rhos_learn.npy") or overwrite:
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
        np.save(res_dir / "rhos_learn.npy", all_rhos)
        pval = gat_stats(all_rhos, -1)
        np.save(res_dir / "pval_learn-pval.npy", pval)

data_type = 'scores_lobotomized'
learn_index_blocks = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
cont_blocks = {}
patterns = {}
randoms = {}
contrasts = {}
for network in tqdm(networks):
    pats_blocks, rands_blocks = [], []
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        contrasts[network] = []
    for subject in subjects:
        res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / subject
        pattern, random = [], []
        for block in range(1, 24):
            if network in networks[:-3]:
                pfname = res_path / f"pat-{block}.npy" if block not in [1, 2, 3] else res_path / f"pat-0-{block}.npy"
                rfname = res_path / f"rand-{block}.npy" if block not in [1, 2, 3] else res_path / f"rand-0-{block}.npy"
            else:
                pfname = res_path / f"pat-4-{block}.npy" if block not in [1, 2, 3] else res_path / f"pat-0-{block}.npy"
                rfname = res_path / f"rand-4-{block}.npy" if block not in [1, 2, 3] else res_path / f"rand-0-{block}.npy"
            pattern.append(np.load(pfname))
            random.append(np.load(rfname))
        if subject == 'sub05':
            pat_bsl = np.load(res_path / "pat-4.npy") if network in networks[:-3] else np.load(res_path / "pat-4-4.npy")
            rand_bsl = np.load(res_path / "rand-4.npy") if network in networks[:-3] else np.load(res_path / "rand-4-4.npy")
            for i in range(3):
                pattern[i] = pat_bsl.copy()
                random[i] = rand_bsl.copy()
        pats_blocks.append(np.array(pattern))
        rands_blocks.append(np.array(random))
    pats_blocks, rands_blocks = np.array(pats_blocks), np.array(rands_blocks)
    patterns[network] = pats_blocks
    randoms[network] = rands_blocks
    contrasts[network] = pats_blocks - rands_blocks

res_path = ensured(RESULTS_DIR / 'TIMEG' / 'source')
for network in networks:    
    # save time gen pvals
    res_dir = ensured(res_path / network / data_type / "pval")
    if not op.exists(res_dir / "all_pattern-pval.npy") or overwrite:
        print(f"Processing pattern {network}...")
        pattern = patterns[network][:, 3:].mean(1)
        pval = gat_stats(pattern - chance, -1)
        np.save(res_dir / "all_pattern-pval.npy", pval)
    if not op.exists(res_dir / "all_random-pval.npy") or overwrite:
        print(f"Processing random {network}...")
        random = randoms[network][:, 3:].mean(1)
        pval = gat_stats(random - chance, -1)
        np.save(res_dir / "all_random-pval.npy", pval)
    if not op.exists(res_dir / "all_contrast-pval.npy") or overwrite:
        print(f"Processing contrast {network}...")
        pval = gat_stats(pattern - random, -1)
        np.save(res_dir / "all_contrast-pval.npy", pval)

    # save learn df x time gen correlation and pvals
    res_dir = ensured(res_path / network / data_type / "corr")    
    if not op.exists(res_dir / "rhos_learn.npy") or overwrite:
        contrasts = patterns[network] - randoms[network]
        # contrasts = zscore(contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
        all_rhos = []
        for sub in range(len(subjects)):
            print(f"Processing {network} subject {sub+1}/{len(subjects)}...")
            rhos = np.empty((times.shape[0], times.shape[0]))
            vector = learn_index_blocks.iloc[sub]  # vector to correlate with
            contrast = contrasts[sub]
            results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrast) for t in range(len(times)) for g in range(len(times)))
            for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
                rhos[t, g] = results[idx]
            all_rhos.append(rhos)
        all_rhos = np.array(all_rhos)
        all_rhos = fisher_z_transform_3d(all_rhos)
        np.save(res_dir / "rhos_learn.npy", all_rhos)
        pval = gat_stats(all_rhos, -1)
        np.save(res_dir / "pval_learn-pval.npy", pval)
