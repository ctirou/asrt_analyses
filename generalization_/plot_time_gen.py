from base import *
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs

trial_types = ['all', 'random', 'pattern']
lock = 'stim'
fold = 'K10'
subjects = SUBJS

epoch_fname = op.join('/Users/coum/Desktop/pred_asrt/stim/sub01-0-epo.fif')
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch


for trial_type in trial_types:
    
    figures = op.join(RESULTS_DIR, 'figures', lock, 'generalizing', 'big_gen', trial_type, fold)
    ensure_dir(figures)

    scores = []

    for subject in subjects:
        
        np_f = op.join(RESULTS_DIR, 'figures', lock, 'generalizing', 'big_gen', trial_type, fold, 'npy', '%s.npy' % subject)
        np_score = np.load(np_f)
        scores.append(np.load(np_f))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        im = ax.imshow(
            np_score,
            interpolation="lanczos",
            origin="lower",
            cmap="RdBu_r",
            extent=times[[0, -1, 0, -1]],
            aspect=0.5,
            vmin=.20,
            vmax=.30)
        
        ax.set_xlabel("Testing Time (s)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title("Temporal generalization")
        ax.axvline(0, color="k")
        ax.axhline(0, color="k")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("accuracy")
        fig.savefig(op.join(figures, "%s.png" % (subject)))
    
    scores = np.array(scores).mean(axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    im = ax.imshow(
        scores,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=times[[0, -1, 0, -1]],
        aspect=0.5,
        vmin=.20,
        vmax=.30)

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("accuracy")
    fig.savefig(op.join(figures, "mean.png"))