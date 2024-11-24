import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from scipy.stats import spearmanr as spear

lock = 'button'
overwrite = True
verbose = True

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, verbose):
    
    corr_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "spear_direction" / subject
    ensure_dir(corr_path)
    
    rev_path =  RESULTS_DIR / 'RSA' / 'sensors' / lock / "rev_rsa" / subject
    ensure_dir(rev_path)
    
    all_corr1, all_corr2 = [], []
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
                    
        behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        # read epochs
        epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname, verbose=verbose)
        data = epoch.get_data(picks='mag', copy=True)
        times = epoch.times
        
        epoch_pat = data[np.where(behav["trialtypes"]==1)]
        behav_pat = behav[behav["trialtypes"]==1]
        
        epoch_rand = data[np.where(behav["trialtypes"]==2)]
        behav_rand = behav[behav["trialtypes"]==2]
        
        if epoch_num == 0:
            pre_pat_epoch,  pre_pat_behav = epoch_pat.copy(), behav_pat.copy()
            pre_rand_epoch, pre_rand_behav = epoch_rand.copy(), behav_rand.copy()
        else:
            corr1, corr2 = [], []
            for t in range(len(times)):
                corr1.append(spear(epoch_pat[:, :, t].mean(0), pre_rand_epoch[:, :, t].mean(0)))
                corr2.append(spear(epoch_rand[:, :, t].mean(0), pre_pat_epoch[:, :, t].mean(0)))
            all_corr1.append(corr1)
            all_corr2.append(corr2)
    
    all_corr1 = np.array(all_corr1)
    all_corr2 = np.array(all_corr2)
    
    np.save(corr_path / "corr1.npy", all_corr1)
    np.save(corr_path / "corr2.npy", all_corr2)    

times = np.load(data_path / "times.npy")
all_diff = list()
for subject in subjects:
    corr_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "spear_direction" / subject
    corr1 = np.load(corr_path / "corr1.npy")
    corr2 = np.load(corr_path / "corr2.npy")
    all_diff.append(corr1[:, :, 0] - corr2[:, :, 0])
all_diff = np.array(all_diff)

figures_dir = FIGURES_DIR / "RSA" / "sensors" / lock / 'rsa_direction'
ensure_dir(figures_dir)
import matplotlib.pyplot as plt
color1 = "#008080"
color2 = "#FFA500"
for i in range(4):
    plt.subplots(1, 1, figsize=(10, 5))
    plt.plot(times, all_diff[:, i, :].mean(0), label=f"Pre vs. session {i+1}")
    p_values = decod_stats(all_diff[:, i, :], -1)
    sig = p_values < .05
    plt.fill_between(times, 0, all_diff[:, i, :].mean(0), where=sig, color=color2, alpha=.4)
    plt.title(f'session {i+1}')
    plt.axhline(0, color='black', linestyle='--')
    if lock == 'stim':
        plt.axvspan(0, 0.2, color='grey', alpha=.2)
    else:
        plt.axvline(0, color='black')
    plt.legend()
    plt.savefig(figures_dir / f"session_{i+1}.pdf")
    plt.close()

rhos = [[spear([0, 1, 2, 3], all_diff[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
rhos = np.array(rhos)
plt.subplots(1, 1, figsize=(10, 5))
plt.plot(times, rhos.mean(0))
p_values = decod_stats(rhos, -1)
sig = p_values < .05
plt.fill_between(times, 0, rhos.mean(0), where=sig, color=color2, alpha=.4)
plt.axhline(0, color='black', linestyle='--')
plt.title('Correlation between session and direction')
if lock == 'stim':
    plt.axvspan(0, 0.2, color='grey', alpha=.2)
else:
    plt.axvline(0, color='black')
plt.savefig(figures_dir / "correlation.pdf")
plt.close()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        process_subject(subject, verbose)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    for subject in subjects:
        process_subject(subject, verbose)