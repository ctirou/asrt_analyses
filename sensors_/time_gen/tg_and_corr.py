import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, zscore, spearmanr
from tqdm.auto import tqdm
from base import *
from config import *
from joblib import Parallel, delayed


def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[0]


lock = 'stim'
trial_type = 'pattern'
subjects, epochs_list = SUBJS, EPOCHS
tg_scores_path = '/Users/romainquentin/Desktop/data/stat_learning_coum/scores'
results_path = '/Users/romainquentin/Desktop/data/stat_learning_coum/results'
results_figure_path = '/Users/romainquentin/Desktop/data/stat_learning_coum/figures'
learning_index_df = pd.read_csv('/Users/romainquentin/Desktop/data/stat_learning_coum/learning_indices.csv', sep="\t")

times = np.linspace(-4, 4, 813)  # not sure to be accurate here. You are taking it from the epochs, it is better.

# ------------------------------ within subjects ------------------------------
all_patterns = list()
all_randoms = list()
for subject in subjects:
    patterns = list()
    randoms = list()  
    for epoch_num in [0, 1, 2, 3, 4]:
        patterns.append(np.load(op.join(tg_scores_path, f"{subject}-epoch{epoch_num}-pattern-scores.npy")))
        randoms.append(np.load(op.join(tg_scores_path, f"{subject}-epoch{epoch_num}-random-scores.npy")))
    all_patterns.append(np.array(patterns))
    all_randoms.append(np.array(randoms))
all_patterns = np.array(all_patterns)
all_randoms = np.array(all_randoms)

# plot pattern (averaged accross participants and sessions)
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_patterns[:, 1:5, :, :].mean((0, 1)),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=0,
    vmax=.50)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("pattern", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(results_figure_path, "mean_pattern.pdf"))
# plot random (averaged accross participants and sessions)
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_randoms[:, 1:5, :, :].mean((0, 1)),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=0,
    vmax=.50)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("random", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(results_figure_path, "mean_random.pdf"))
# plot the contrast patterns vs. randoms (averaged accross participants and sessions)
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_patterns[:, 1:5, :, :].mean((0, 1)) - all_randoms[:, 1:5, :, :].mean((0, 1)),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=0,
    vmax=.1)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("pattern - random", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
fig.savefig(op.join(results_figure_path, "mean_contrast.pdf"))

# correlations between contrast and epoch_num (within subjects)
all_contrasts = all_patterns - all_randoms
all_contrasts = zscore(all_contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
all_rhos = []
for sub in tqdm(range(len(subjects))):
    rhos = np.empty((813, 813))
    vector = [0, 1, 2, 3, 4]  # vector to correlate with
    contrasts = all_contrasts[sub]
    results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrasts) for t in range(len(times)) for g in range(len(times)))
    for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
        rhos[t, g] = results[idx]
    all_rhos.append(rhos)
all_rhos = np.array(all_rhos)
np.save(op.join(results_path, 'all_rhos_epoch_num_ws.npy'), all_rhos)
# # plot the within-subject correlations between contrast and epoch_num
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_rhos.mean(0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=-0.2,
    vmax=.2)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("rhos", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
plt.savefig(op.join(results_figure_path, 'all_rhos_epoch_num_ws.pdf'), transparent=True)
plt.close()


# correlations between contrast and learning index (within subjects)
all_contrasts = all_patterns - all_randoms
all_contrasts = zscore(all_contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
all_rhos = []
for sub in tqdm(range(len(subjects))):
    rhos = np.empty((813, 813))
    vector = learning_index_df.iloc[sub, 1:]  # vector to correlate with
    contrasts = all_contrasts[sub]
    results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, contrasts) for t in range(len(times)) for g in range(len(times)))
    for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
        rhos[t, g] = results[idx]
    all_rhos.append(rhos)
all_rhos = np.array(all_rhos)
np.save(op.join(results_path, 'all_rhos_learning_index_ws.npy'), all_rhos)
# # plot the within-subject correlations between contrast and epoch_num
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_rhos.mean(0),
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=-0.2,
    vmax=.2)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("rhos", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
plt.savefig(op.join(results_figure_path, 'all_rhos_learning_index_ws.pdf'), transparent=True)
plt.close()

# ------------------------------ accross subjects ------------------------------
all_patterns = list()
all_randoms = list()
for subject in subjects:
    all_patterns.append(np.load(op.join(tg_scores_path, f"{subject}-epochall-pattern-scores.npy")))
    all_randoms.append(np.load(op.join(tg_scores_path, f"{subject}-epochall-random-scores.npy")))
all_patterns = np.array(all_patterns)
all_randoms = np.array(all_randoms)
# correlations between contrast and epoch_num (across subjects)
all_contrasts = all_patterns - all_randoms
all_contrasts = zscore(all_contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
all_rhos = []
vector = learning_index_df.iloc[:, -1]  # vector to correlate with (here the learning index for block 4)
results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, all_contrasts) for t in range(len(times)) for g in range(len(times)))
for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
    rhos[t, g] = results[idx]
all_rhos = np.array(rhos)
np.save(op.join(results_path, 'all_rhos_learning_index_as.npy'), all_rhos)
# # plot the within-subject correlations between contrast and epoch_num
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
im = ax.imshow(
    all_rhos,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=-0.2,
    vmax=.2)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("rhos", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
plt.savefig(op.join(results_figure_path, 'all_rhos_learning_index_as.pdf'), transparent=True)
plt.close()

# change the function to get the p_value (fast and dirty code)
def compute_spearman(t, g, vector, contrasts):
    return spearmanr(vector, contrasts[:, t, g])[1]


all_contrasts = all_patterns - all_randoms
all_contrasts = zscore(all_contrasts, axis=-1)  # je sais pas si zscore avant correlation pour la RSA mais c'est mieux je pense
all_rhos = []
vector = learning_index_df.iloc[:, -1]  # vector to correlate with (here the learning index for block 4)
results = Parallel(n_jobs=-1)(delayed(compute_spearman)(t, g, vector, all_contrasts) for t in range(len(times)) for g in range(len(times)))
for idx, (t, g) in enumerate([(t, g) for t in range(len(times)) for g in range(len(times))]):
    rhos[t, g] = results[idx]
all_rhos = np.array(rhos)
np.save(op.join(results_path, 'all_rhos_learning_index_as_pvalue.npy'), all_rhos)
# # plot the within-subject correlations between contrast and epoch_num
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
all_rhos = all_rhos < 0.05
im = ax.imshow(
    all_rhos,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=times[[0, -1, 0, -1]],
    aspect=0.5,
    vmin=0,
    vmax=1)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("rhos", style='italic')
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")
plt.savefig(op.join(results_figure_path, 'all_rhos_learning_index_as_sig.pdf'), transparent=True)
plt.close()

