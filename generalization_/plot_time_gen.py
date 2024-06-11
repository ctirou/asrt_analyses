from base import ensure_dir, gat_stats
from config import *
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import read_epochs
from tqdm.auto import tqdm

analysis = "time_generalization"
data_path = PRED_PATH
res_path = RESULTS_DIR
subjects, epochs_list = SUBJS, EPOCHS
lock = 'stim'

summary = False

epoch_fname = data_path / lock / 'sub01-0-epo.fif'
epoch = read_epochs(epoch_fname, verbose=False)
times = epoch.times
del epoch

res_dir = res_path / analysis / 'sensors' / lock
    
figures = RESULTS_DIR / 'figures' / analysis / 'sensors' / lock
ensure_dir(figures)

pattern, random = [], []

for subject in tqdm(subjects):

    pattern_f = res_dir / 'pattern' / f"{subject}_scores.npy"
    pattern.append(np.load(pattern_f))

    random_f = res_dir / 'random' / f"{subject}_scores.npy"
    random.append(np.load(random_f))
    
    if summary:
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        im = ax.imshow(
            pattern,
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

pattern = np.array(pattern).mean(axis=0)
random = np.array(random).mean(axis=0)

# plot pattern
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    pattern,
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
fig.savefig(op.join(figures, "mean_pattern.png"))

# plot random
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    random,
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
fig.savefig(op.join(figures, "mean_random.png"))

# plot contrast
contrast = pattern - random
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    contrast,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5)

ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")

pval = gat_stats(contrast)
sig = np.array(pval < 0.05)
xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
plt.contour(xx, yy, sig, colors='Gray', levels=[0],
                    linestyles='solid', linewidths=1)

fig.savefig(op.join(figures, "mean_contrast.png"))

