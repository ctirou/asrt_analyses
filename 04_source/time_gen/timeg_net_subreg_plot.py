# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from mne import read_labels_from_annot
from tqdm.auto import tqdm


def natural_sort_key(name):
    return [int(n) for n in re.findall(r'\d+', name)]

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

analysis = 'scores_blocks_subreg'

networks = ['SomMot', 'DorsAttn']

times = np.linspace(-1, 1.5, 256)

parc = "Schaefer2018_200Parcels_7Networks"
hemi = 'both'

net_cmap = {
    'SomMot': '#DE8F05',
    'DorsAttn': '#029E73',
}

net_labels_map = {
    'SomMot': 'Sensorimotor',
    'DorsAttn': 'Dorsal Attention',
}

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

# get significant time points from GAMM csv
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_source_subreg.csv")
seg_df = seg_df[seg_df['metric'].str.contains('PA')]
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["label"], np.zeros(len(times), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["label"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_source_subreg.csv")
sig_df = sig_df[sig_df['metric'].str.contains('PA')]
for i, net in enumerate(sig_df['label'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['label'] == net]['signif_holm'][i] == 'ns':
            del sig_dict[net]

for network in networks:

    # Get naturally sorted label names for this network
    label_names = sorted([
        label.name for label in read_labels_from_annot(
            subject=subjects[0], parc=parc, hemi=hemi,
            subjects_dir=subjects_dir, regexp=network, verbose='error'
        )
    ], key=natural_sort_key)

    # Load timeg data per label: contrasts[lname] -> (n_subjects, n_blocks, n_times)
    contrasts = {lname: [] for lname in label_names}

    for subject in tqdm(subjects):
        res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / analysis / subject

        for lname in label_names:
            pattern_blocks, random_blocks = [], []

            for block in range(4, 24):
                pattern_blocks.append(np.load(res_path / lname / f'pat-{block}.npy'))
                random_blocks.append(np.load(res_path / lname / f'rand-{block}.npy'))

            pattern_blocks = np.array(pattern_blocks) * 100
            random_blocks = np.array(random_blocks) * 100

            contrasts[lname].append(pattern_blocks - random_blocks)

    for lname in label_names:
        contrasts[lname] = np.array(contrasts[lname])  # (n_subjects, n_blocks, n_times)

    color = net_cmap[network]
    chance = 0

    # Split into left (lh) and right (rh), each naturally sorted
    lh_labels = sorted([lbl for lbl in label_names if lbl.endswith('-lh')], key=natural_sort_key)
    rh_labels = sorted([lbl for lbl in label_names if lbl.endswith('-rh')], key=natural_sort_key)
    col_labels = [lh_labels, rh_labels]

    n_rows = max(len(lh_labels), len(rh_labels))
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 1.75),
                             sharey=True, sharex=True, layout="tight")
    axes = np.array(axes).reshape(n_rows, n_cols)

    for col, labels_col in enumerate(col_labels):
        hemi_title = 'Left' if col == 0 else 'Right'
        for row, lname in enumerate(labels_col):
            ax = axes[row, col]

            data = contrasts[lname].mean(1)  # (n_subjects, n_times)
            mean = data.mean(0)
            sem = np.std(data, axis=0) / np.sqrt(len(subjects))

            sig = sig_dict[lname] if lname in sig_dict else np.zeros(len(times), dtype=bool)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
            ax.axhline(chance, color='grey', alpha=.5)

            ax.plot(times, mean, alpha=1, zorder=10, color='C7')
            for start, end in contiguous_regions(sig):
                ax.plot(times[start:end], mean[start:end], alpha=1, zorder=10, color=color)

            sig_level_rows = sig_df[sig_df['label'] == lname]['signif_holm']
            if len(sig_level_rows) > 0 and sig_level_rows.values[0] != 'ns':
                ax.text(0.4, 0.1, sig_level_rows.values[0], fontsize=20,
                        ha='center', va='bottom', color=color, weight='bold')

            ax.fill_between(times, mean - sem, mean + sem, alpha=0.2, zorder=5, facecolor='C7')
            ax.fill_between(times, mean - sem, mean + sem, where=sig, alpha=0.5, zorder=5, color=color)
            ax.fill_between(times, mean - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=color)

            # Short region label: strip network prefix and hemisphere suffix
            parts = lname.split(f'{network}_')
            region_label = parts[-1].replace('-lh', '').replace('-rh', '') if len(parts) > 1 else lname
            ax.text(0.05, 0.95, region_label, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', color='k', fontstyle='italic')

            if row == 0:
                ax.set_title(hemi_title, fontsize=10, color='black')
            if col == 0:
                ax.set_ylabel('D.i.A. (a.u.)', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=9)
                ax.tick_params(labelbottom=True)

        # Hide unused rows in this column
        for row in range(len(labels_col), n_rows):
            axes[row, col].set_visible(False)

    fig.suptitle(net_labels_map[network], fontsize=13, color=color)
    
    fig.savefig(figures_dir / f"timeg-net-subreg-{network}.pdf", transparent=True)
    plt.close()

    # Save table per network with label column
    rows = []
    for lname in label_names:
        data = contrasts[lname].mean(1)  # (n_subjects, n_times)
        for j, subject in enumerate(subjects):
            for t, time in enumerate(times):
                rows.append({
                    "label": lname,
                    "subject": subject,
                    "time": t,
                    "value": data[j, t]
                })
    df = pd.DataFrame(rows)
    df.to_csv(FIGURES_DIR / "TM" / "data" / f'{network}_subreg_pa_tr.csv', index=False, sep=",")
