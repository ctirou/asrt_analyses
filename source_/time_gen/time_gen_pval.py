from base import ensured, gat_stats
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
threshold = .05

def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]

# Load data, compute, and save correlations and pvals 
learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices15.csv', sep="\t", index_col=0)

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
            pat.append(np.load(res_path / network / 'scores_skf' / subject /  f"pat-{j}.npy"))
            rand.append(np.load(res_path / network / 'scores_skf' / subject /  f"rand-{j}.npy"))
        patpat.append(np.array(pat))
        randrand.append(np.array(rand))
    
        all_pat.append(np.load(res_path / network / 'scores_skf' / subject /  "pat-all.npy"))
        all_rand.append(np.load(res_path / network / 'scores_skf' / subject /  "rand-all.npy"))
        
    all_patterns[network] = np.array(all_pat)
    all_randoms[network] = np.array(all_rand)
    
    patterns[network] = np.array(patpat)
    randoms[network] = np.array(randrand)
    
    # save time gen pvals
    res_dir = ensured(res_path / network / "pval-skf")
    if not op.exists(res_dir / "all_pattern-pval.npy") or overwrite:
        pval = gat_stats(all_patterns[network] - chance, -1)
        np.save(res_dir / "all_pattern-pval.npy", pval)
    if not op.exists(res_dir / "all_random-pval.npy") or overwrite:
        pval = gat_stats(all_randoms[network] - chance, -1)
        np.save(res_dir / "all_random-pval.npy", pval)
    if not op.exists(res_dir / "all_contrast-pval.npy") or overwrite:
        pval = gat_stats(all_patterns[network] - all_randoms[network], -1)
        np.save(res_dir / "all_contrast-pval.npy", pval)
    
    # save learn df x time gen correlation and pvals
    res_dir = ensured(res_path / network / "corr-skf")    
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


# --- time gen pvals for bins ---
times = np.linspace(-1.5, 1.5, 307)
res_path = RESULTS_DIR / 'TIMEG' / 'source'
chance = .25

# data
for network in networks:
    all_pats, all_rands = [], []
    patterns, randoms = [], []
    for subject in subjects:    
        pattern, random = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            pats, rands = [], []
            for block in blocks:
                p, r = [], []
                for fold in range(1, 3):
                    p.append(np.load(res_path / network / 'scores_40s' / subject / f"pat-{epoch_num}-{block}-{fold}.npy"))
                    r.append(np.load(res_path / network / 'scores_40s' / subject / f"rand-{epoch_num}-{block}-{fold}.npy"))
                pats.append(np.array(p))
                rands.append(np.array(r))
            # if epoch_num != 0:
            pattern.append(np.mean(pats, 0))
            random.append(np.mean(rands, 0))
        
        pattern = np.array(pattern)
        random = np.array(random)
        
        patterns.append(pattern.mean(1))
        randoms.append(random.mean(1))
        
        if epoch_num != 0:
            all_pats.append(pattern.mean(1))
            all_rands.append(random.mean(1))
    
    patterns = np.array(patterns)
    randoms = np.array(randoms)
    
    all_pats = np.array(all_pats).mean(1)
    all_rands = np.array(all_rands).mean(1)
    
    res_dir = ensured(res_path / network / "pval-40s")
    if not op.exists(res_dir / "all_pattern-pval.npy") or overwrite:
        print(f"Processing pattern in {network}...")
        pval = gat_stats(all_pats - chance, -1)
        np.save(res_dir / "all_pattern-pval.npy", pval)
    if not op.exists(res_dir / "all_random-pval.npy") or overwrite:
        print(f"Processing random in {network}...")
        pval = gat_stats(all_rands - chance, -1)
        np.save(res_dir / "all_random-pval.npy", pval)
    if not op.exists(res_dir / "all_contrast-pval.npy") or overwrite:
        print(f"Processing contrast in {network}...")
        pval = gat_stats(all_pats - all_rands, -1)
        np.save(res_dir / "all_contrast-pval.npy", pval)
    
    # save learn df x time gen correlation and pvals
    res_dir = ensured(res_path / network / "corr-40s")    
    if not op.exists(res_dir / "rhos_learn.npy") or overwrite:
        print(f"Correlating in {network}")
        contrasts = patterns - randoms
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
