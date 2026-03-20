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

subjects, subjects_dir = SUBJS15, FREESURFER_DIR
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")
analysis = 'rdm_blocks_subreg_merged'
networks = ['SomMot', 'DorsAttn']
parc = "Schaefer2018_200Parcels_7Networks"
hemi = 'both'

times = np.linspace(-0.2, 0.6, 82)
step = 1
use_decode = False
saving = True

net_cmap = {
    'SomMot': '#DE8F05',
    'DorsAttn': '#029E73',
}

net_labels_map = {
    'SomMot': 'Sensorimotor',
    'DorsAttn': 'Dorsal Attention',
}

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

def natural_sort_key(name):
    return [int(n) for n in re.findall(r'\d+', name)]

def get_base_name(label_name):
    """Strip hemisphere prefix (LH_/RH_) and suffix (-lh/-rh) to get bilateral base name."""
    name = label_name.replace('_LH_', '_').replace('_RH_', '_')
    return name.replace('-lh', '').replace('-rh', '')

if not use_decode:
    # get significant time points from GAMM csv
    seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_rs_tr_source_subreg_merged.csv")
    # seg_df = seg_df[seg_df['metric'] == 'RS']
    seg_df = seg_df[seg_df['metric'].str.contains('RS')]
    sig_dict = {}
    for _, row in seg_df.iterrows():
        arr = sig_dict.get(row["label"], np.zeros(82, dtype=bool))
        arr[row["start"]:row["end"] + 1] = True
        sig_dict[row["label"]] = arr
    sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_rs_tr_source_subreg_merged.csv")
    sig_df = sig_df[sig_df['metric'].str.contains('RS')]
    for i, net in enumerate(sig_df['label'].unique()):
        if net in sig_dict:
            if sig_df[sig_df['label'] == net]['signif_holm'][i] == 'ns':
                del sig_dict[net]

for network in networks:

    # Get naturally sorted bilateral base names for this network
    raw_labels = read_labels_from_annot(
        subject=subjects[0], parc=parc, hemi=hemi,
        subjects_dir=subjects_dir, regexp=network, verbose='error'
    )
    # Merged lh+rh pairs use the base name; singletons (one hemi only) keep their original name.
    label_groups = {}
    for label in raw_labels:
        base = get_base_name(label.name)
        label_groups.setdefault(base, []).append(label.name)
    label_names = sorted(
        [base if len(names) == 2 else names[0] for base, names in label_groups.items()],
        key=natural_sort_key
    )

    # Load RSA data per label: diff_rp[lname] -> (n_subjects, n_blocks, n_times)
    diff_rp = {lname: [] for lname in label_names}

    for subject in tqdm(subjects):
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / analysis / subject
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)

        for lname in label_names:
            pattern_blocks, random_blocks = [], []

            for epoch_num in [1, 2, 3, 4]:
                blocks = [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]

                for block in blocks:
                    pattern_blocks.append(np.load(res_path / lname / f"pat-{epoch_num}-{block}.npy"))
                    random_blocks.append(np.load(res_path / lname / f"rand-{epoch_num}-{block}.npy"))

            pattern_blocks = np.array(pattern_blocks)
            random_blocks = np.array(random_blocks)

            high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False, step)

            pat = np.nanmean(high, 0)
            rand = np.nanmean(low, 0)

            diff_rp[lname].append(rand - pat)

    for lname in label_names:
        diff_rp[lname] = np.array(diff_rp[lname])  # (n_subjects, n_blocks, n_times)

    color = net_cmap[network]
    chance = 0

    n_rows = len(label_names)
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5, n_rows * 2.25),
                             sharey=True, sharex=True, layout="tight")
    axes = np.array(axes).reshape(n_rows, n_cols)

    for row, lname in enumerate(label_names):
        ax = axes[row, 0]

        data = diff_rp[lname].mean(1)  # (n_subjects, n_times)
        mean = data.mean(0)
        sem = np.std(data, axis=0) / np.sqrt(len(subjects))

        if not use_decode:
            sig = sig_dict[lname] if lname in sig_dict else np.zeros(len(times), dtype=bool)
            sig_level_rows = sig_df[sig_df['label'] == lname]['signif_holm']
            if len(times[sig]) > 0 and sig_level_rows.values[0] != 'ns':
                ax.text(0.45, 0.1, sig_level_rows.values[0], fontsize=20,
                        ha='center', va='bottom', color=color, weight='bold')
        else:
            pval = decod_stats(data, -1)
            sig = pval < 0.05

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
        ax.axhline(chance, color='grey', alpha=.5)

        ax.plot(times, mean, alpha=1, zorder=10, color='C7')
        for start, end in contiguous_regions(sig):
            ax.plot(times[start:end], mean[start:end], alpha=1, zorder=10, color=color)

        ax.fill_between(times, mean - sem, mean + sem, alpha=0.2, zorder=5, facecolor='C7')
        ax.fill_between(times, mean - sem, mean + sem, where=sig, alpha=0.5, zorder=5, color=color)
        ax.fill_between(times, mean - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=color)

        # Short region label: strip network prefix and hemisphere suffix
        parts = lname.split(f'{network}_')
        region_label = parts[-1].replace('-lh', '').replace('-rh', '') if len(parts) > 1 else lname
        ax.text(0.05, 0.95, region_label, transform=ax.transAxes,
                fontsize=9, va='top', ha='left', color='k', fontstyle='italic')

        ax.set_ylabel('SI (a.u.)', fontsize=9)
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.tick_params(labelbottom=True)

    fig.suptitle(net_labels_map[network], fontsize=13, color=color)
    
    if saving: 
        fig.savefig(figures_dir / f"rsa-net-subreg-{network}-merged.pdf")
        plt.close()

    # Save table per network with label column
    rows = []
    for lname in label_names:
        data = diff_rp[lname].mean(1)  # (n_subjects, n_times)
        for j, subject in enumerate(subjects):
            for t, time in enumerate(times):
                rows.append({
                    "label": lname,
                    "subject": subject,
                    "time": t,
                    "value": data[j, t]
                })
    df = pd.DataFrame(rows)
    if saving:
        df.to_csv(FIGURES_DIR / "TM" / "data" / f'{network}_subreg_merged_rs_tr.csv', index=False, sep=",")
