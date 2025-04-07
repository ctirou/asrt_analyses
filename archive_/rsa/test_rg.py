import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from pyriemann.estimation import Shrinkage
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from base import ensure_dir, get_volume_estimate_time_course
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
import gc
import os
from tqdm.auto import tqdm

# params
subjects = SUBJS

analysis = "decoding"
lock = "button" # "stim", "button"
trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
ensure_dir(res_path)
sessions = ['Practice', 'Block_1', 'Block_2', 'Block_3', 'Block_4']
folds = 10
solver = 'lbfgs'
scoring = "accuracy"
parc='aparc'
# parc='aseg'
hemi = 'both'
verbose = 'error'
jobs = -1

# get times
epoch_fname = DATA_DIR / lock / 'sub01-0-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs, epoch_fname
gc.collect()

subject = 'sub01'
epo_dir = data_path / lock
epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
epochs = mne.read_epochs(epo_fnames[1], preload=True, verbose="error")

beh_dir = data_path / 'behav'
beh_fnames = [beh_dir / f'{f}' for f in sorted(os.listdir(beh_dir)) if '.pkl' in f and subject in f]
behav = pd.read_pickle(beh_fnames[1])

X = epochs.get_data()[behav.trialtypes == 1]
pattern = behav.trialtypes == 1
y = behav[behav.trialtypes == 1].positions.reset_index(drop=True)

conditions = np.unique(y)

# Parameters for sliding time window
sfreq = epochs.info['sfreq']  # Sampling frequency
window_size = 0.1  # Window size in seconds
step_size = 0.05  # Step size in seconds

window_samples = int(window_size * sfreq)
step_samples = int(step_size * sfreq)
n_channels = len(epochs.info['ch_names'])
n_windows = (epochs.times.size - window_samples) // step_samples + 1

# Initialize dictionary to store covariance matrices through time for each condition
cov_matrices_time = {cond: np.zeros((n_windows, n_channels, n_channels)) for cond in conditions}

# Sliding window covariance estimation for each condition
for condition in conditions:
    X = epochs.get_data()[pattern][y == condition]
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        # Extract data for the time window
        window_data = X[:, :, start:end]  # Shape: (n_epochs, n_channels, window_samples)
        # Compute mean covariance for the time window
        cov_matrices_time[condition][i] = np.mean(Shrinkage(shrinkage=0.1).fit_transform(window_data), axis=0)

# Compute Riemannian distances between conditions through time
distances_time = {f"{cond1} vs {cond2}": np.zeros(n_windows)
                  for i, cond1 in enumerate(conditions) for j, cond2 in enumerate(conditions) if i < j}

for i in range(n_windows):
    for cond1 in conditions:
        for cond2 in conditions:
            if cond1 != cond2 and f"{cond2} vs {cond1}" not in distances_time:
                # Compute Riemannian distance for the current time window
                dist = distance_riemann(cov_matrices_time[cond1][i], cov_matrices_time[cond2][i])
                distances_time[f"{cond1} vs {cond2}"][i] = dist

# Plot Riemannian distances through time
times = epochs.times[:n_windows * step_samples:step_samples]
