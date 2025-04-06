import os
import numpy as np
import pandas as pd
import mne
from base import *
from config import *
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import sys
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter

subjects_dir = FREESURFER_DIR
fname_fs_src = RESULTS_DIR / "src" / "fsaverage2-src.fif"

data_path = DATA_DIR
# subject = 'sub01'
lock = 'stim'
verbose = True
overwrite = False

pca = UnsupervisedSpatialFilter(PCA(1000), average=False)

def rsa_subject(subject, lock, epoch_num, network):
            
    label_path = RESULTS_DIR / 'networks_200_7' / 'fsaverage2'
    lh_label, rh_label = mne.read_label(label_path / f'{network}-lh.label'), mne.read_label(label_path / f'{network}-rh.label')        
    
    # for epoch_num in range(5):
    # read behav
    behav_fname = data_path / "behav" / f"{subject}-{epoch_num}.pkl"
    behav = pd.read_pickle(behav_fname).reset_index()
            
    fname_stcs = RESULTS_DIR / 'power_stc' / lock / f"{subject}-morphed-stcs-{epoch_num}.npy"
    stcs_data = np.load(fname_stcs, allow_pickle=True)
    
    # Check if imaginary components are present
    max_imag = np.max([np.abs(stc.data.imag).max() for stc in stcs_data])
    threshold = 1e-10  # Define a small threshold

    if max_imag > threshold:
        raise ValueError(f"Processing, {subject}, {network}, {epoch_num}\nSignificant imaginary components detected (max {max_imag}). Check beamformer parameters.")
    else:
        print(f"Processing, {subject}, {network}, {epoch_num}\nMax imaginary component {max_imag} is below threshold {threshold}.")
    
    morphed_stcs_data = np.array([np.real(stc.in_label(lh_label + rh_label).data) for stc in stcs_data])
    
    data = pca.fit_transform(morphed_stcs_data)

    res_dir = RESULTS_DIR / 'RSA' / 'source' / network / lock / 'power_morphed_rdm' / subject
    ensure_dir(res_dir)
    
    if not op.exists(res_dir / f"pat-{epoch_num}.npy") or overwrite:
        pattern = behav.trialtypes == 1
        X_pat = data[pattern]
        y_pat = behav.positions[pattern].reset_index(drop=True)
        assert X_pat.shape[0] == y_pat.shape[0]
        rdm_pat = cv_mahalanobis(X_pat, y_pat)
        np.save(res_dir / f"pat-{epoch_num}.npy", rdm_pat)

    if not op.exists(res_dir / f"rand-{epoch_num}.npy") or overwrite:
        random = behav.trialtypes == 2
        X_rand = data[random]
        y_rand = behav.positions[random].reset_index(drop=True)
        assert X_rand.shape[0] == y_rand.shape[0]
        rdm_rand = cv_mahalanobis(X_rand, y_rand)
        np.save(res_dir / f"rand-{epoch_num}.npy", rdm_rand)

subjects = SUBJS
networks = NETWORKS[:-2]

for network in networks[1:]:
    for subject in subjects:
        # print("Processing", subject, network)        
        Parallel(-1)(delayed(rsa_subject)(subject, lock, epoch_num, network) for epoch_num in range(5))
        # source = RESULTS_DIR / 'RSA' / 'source' / network / lock / 'morphed_rdm'
        # destination = Path(f"/Users/coum/MEGAsync/RSA/source/{network}/{lock}/")
        # ensure_dir(destination)
        # rsync_files(source, destination, "-av")