import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import cross_val_multiscore, SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from jr.gat import scorer_spearman
from base import *
from config import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# params
# trial_types = ["all", "pattern", "random"]
trial_type = "pattern"
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 10
chance = 0.25
threshold = 0.05
scoring = "accuracy"
hemi = 'both'
params = "clustering"
verbose = "error"
jobs = -1
# figures dir
figures = RESULTS_DIR / 'figures' / lock / params / 'sensors' / trial_type
ensure_dir(figures)
# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

big_nmi, big_ari = list(), list()

for subject in subjects:
    print(subject)
    
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=False, verbose=verbose) for fname in epo_fnames]
    times = all_epo[0].times

    beh_dir = data_path / "behav"
    beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
    all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]

    for epoch in all_epo:
        epoch.info["dev_head_t"] = all_epo[0].info["dev_head_t"]
    
    all_nmi_scores, all_ari_scores = list(), list()

    est = KMeans(n_clusters=4)
    
    for session_id, session in enumerate(sessions):
        nmi_scores, ari_scores = list(), list()
        # get session behav and epoch
        if session_id == 0:
            session = 'prac'
        else:
            session = 'sess-%s' % (str(session_id).zfill(2))
        behav = all_beh[session_id]
        epoch = all_epo[session_id]
        if lock == 'button': 
            epoch_bsl_fname = data_path / "bsl" / f"{subject}_{session_id}_bl-epo.fif"
            epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)

        X = epoch.get_data(verbose=verbose)
        y = np.array(behav.positions)
    
        for itime in tqdm(range(len(times))):
            X_t = X[:, :, itime]
            est.fit(X_t)
            
            labels = est.labels_
            nmi_scores.append(normalized_mutual_info_score(y, labels))
            ari_scores.append(adjusted_rand_score(y, labels))
            
        all_nmi_scores.append(np.array(nmi_scores))
        all_ari_scores.append(np.array(ari_scores))
    
    big_nmi.append(np.array(all_nmi_scores))
    big_ari.append(np.array(all_ari_scores))
    
big_nmi = np.array(big_nmi)
big_ari = np.array(big_ari)

# # Assuming 'labels' are your cluster labels and 'y' are your classes
# df = pd.DataFrame({'Cluster': labels, 'Class': y})
# cross_tab = pd.crosstab(df['Cluster'], df['Class'])
# print(cross_tab)

nmi_mean = np.mean(big_nmi, axis=0)
ari_mean = np.mean(big_ari, axis=0)

# plt.subplots(1, 1, figsize=(14, 5))
# for i, j in zip(range(1, 5), sessions[1:]):
#     plt.plot(times, ari_mean[i].T, label=f"{j}")
# plt.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.legend()
# plt.title("adjusted_rand_score")

plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, ari_mean[0].T, label="practice")
plt.plot(times, ari_mean[1:5].mean(axis=0).T, label="learning")
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.legend()
plt.title("adjusted_rand_score")

# plt.subplots(1, 1, figsize=(14, 5))
# for i, j in zip(range(1, 5), sessions[1:]):
#     plt.plot(times, nmi_mean[i].T, label=f"{j}")
# plt.axvspan(0, 0.2, color='grey', alpha=.2)
# plt.legend()
# plt.title("normalized_mutual_info_score")

plt.subplots(1, 1, figsize=(14, 5))
plt.plot(times, nmi_mean[0].T, label="practice")
plt.plot(times, nmi_mean[1:5].mean(axis=0).T, label="learning")
plt.axvspan(0, 0.2, color='grey', alpha=.2)
plt.legend()
plt.title("normalized_mutual_info_score")
