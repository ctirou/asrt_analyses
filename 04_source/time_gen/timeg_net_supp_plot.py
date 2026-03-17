# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

networks = ['SomMot-precentral', 'SomMot-postcentral', 'SomMot-paracentral', \
    'DorsAttn-superiorparietal', 'DorsAttn-caudalmiddlefrontal']
    # 'DorsAttn-superiorparietal', 'DorsAttn-caudalmiddlefrontal', \
    # 'Cont-rostralmiddlefrontal', 'Cont-superiorfrontal', 'Cont-parsopercularis', 'Cont-parstriangularis', 'Cont-supramarginal']
    # 'Cont-rostralmiddlefrontal', 'Cont-superiorfrontal', 'Cont-supramarginal']
# network_names = NETWORK_NAMES
times = np.linspace(-1.5, 1.5, 307)
threshold = .05
res_dir = RESULTS_DIR / 'TIMEG' / 'source'

data_type = 'scores_blocks'
patterns = {}
randoms = {}
contrasts = {}
win4 = np.where((times >= -1))[0]
for network in tqdm(networks):
    failed_subjects = []
    pats_blocks, rands_blocks = [], []
    if network not in patterns:
        patterns[network], randoms[network] = [], []
        contrasts[network] = []
    for subject in subjects:
        try:
            res_path = RESULTS_DIR / 'TIMEG' / 'source' / network / data_type / subject
            pattern, random = [], []
            for block in range(1, 24):
                pfname = res_path / f'pat-{block}.npy'
                rfname = res_path / f'rand-{block}.npy'
                pattern.append(np.load(pfname))
                random.append(np.load(rfname))
            if subject == 'sub05':
                pat_bsl = np.load(res_path / "pat-4.npy")
                rand_bsl = np.load(res_path / "rand-4.npy")
                for i in range(3):
                    pattern[i] = pat_bsl.copy()
                    random[i] = rand_bsl.copy()
            pats_blocks.append(np.array(pattern))
            rands_blocks.append(np.array(random))
        except Exception as e:
            failed_subjects.append(subject)
    pats_blocks, rands_blocks = np.array(pats_blocks), np.array(rands_blocks)
    patterns[network] = pats_blocks[:, :, win4] * 100
    randoms[network] = rands_blocks[:, :, win4] * 100
    contrasts[network] = patterns[network] - randoms[network]
    # print(f"failed for {len(failed_subjects)} in {network}")

net_cmap = {
    'SomMot': '#DE8F05',
    'DorsAttn': '#029E73',
    # 'Cont': '#CA9161',
}

# Group regions by parent network (before '-')
net_groups = defaultdict(list)
for network in networks:
    parent = network.split('-')[0]
    net_groups[parent].append(network)

net_labels = {
    'SomMot': 'Sensorimotor',
    'DorsAttn': 'Dorsal Attention',
    # 'Cont': 'Central Executive',
}

# net_order = ['SomMot', 'DorsAttn', 'Cont']
net_order = ['SomMot', 'DorsAttn']
n_cols = len(net_order)
n_rows = max(len(net_groups[n]) for n in net_order)
times_win = times[win4]

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

data_arr = contrasts.copy()
chance = 0

# get significant time points from GAMM csv
seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_cont_source_supp.csv")
seg_df = seg_df[seg_df['metric'] == 'PA']
# dictionary of boolean arrays
sig_dict = {}
for _, row in seg_df.iterrows():
    arr = sig_dict.get(row["network"], np.zeros(len(win4), dtype=bool))
    arr[row["start"]:row["end"] + 1] = True
    sig_dict[row["network"]] = arr
sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_cont_source_supp.csv")
sig_df = sig_df[sig_df['metric'] == 'PA']
for i, net in enumerate(sig_df['network'].unique()):
    if net in sig_dict:
        if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
            del sig_dict[net]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 1.75),
                         sharey=True, sharex=True, layout="tight")

for col, net_name in enumerate(net_order):
    regions = net_groups[net_name]
    color = net_cmap[net_name]
    for row, network in enumerate(regions):
        
        ax = axes[row, col]
        region_label = network.split('-')[1]
        data = data_arr[network][:, 3:].mean(1)  # (n_subjects, n_times)
        sem = np.std(data, axis=0) / np.sqrt(len(subjects))

        # plot_onset(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
    
        ax.axhline(chance, color='grey', alpha=.5)

        # Non-sig line
        ax.plot(times_win, data.mean(0), alpha=1, zorder=10, color='C7')
        
        # Sig regions in color
        sig = sig_dict[network] if network in sig_dict else np.zeros(data.shape[1], dtype=bool)
        for start, end in contiguous_regions(sig):
            ax.plot(times_win[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=color)
        sig_level = sig_df[sig_df['network'] == network]['signif_holm'].values[0]
        if sig_level != 'ns':
            ax.text(-0.5, 3, sig_level, fontsize=20, ha='center', va='center', color=color, weight='bold')

        ax.fill_between(times_win, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
        ax.fill_between(times_win, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=color)
        ax.fill_between(times_win, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=color)

        if row == 0:
            ax.set_title(f'{net_labels[net_name]}', fontsize=11,
                         color=color)
        
        ax.text(0.05, 0.15, region_label, transform=ax.transAxes,
                fontsize=10, va='top', ha='left', color="k", fontstyle='italic')

        if col == 0:
            ax.set_ylabel('Diff in acc. (a.u.)', fontsize=9)
            if row == 0:
                ax.legend(fontsize=8, frameon=False)
        if row == len(regions) - 1:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.tick_params(labelbottom=True)

    for row in range(len(regions), n_rows):
        axes[row, col].set_visible(False)

# fig.suptitle("Predictive activity: contrast in subregions of significant networks", fontsize=12, fontweight='bold')

fig.savefig(figures_dir / "timeg-net-supp-regions2.pdf", transparent=True)
plt.close()

# save time resolved diagonals
rows = list()
for i, network in enumerate(networks):
    # get table
    for j, subject in enumerate(subjects):
        for t, _ in enumerate(win4):
            rows.append({
                "network": networks[i],
                "subject": subject,
                "time": t,
                "value": contrasts[network].mean(1)[j, t]
            })
df = pd.DataFrame(rows)
fname = 'pa_source_tr_contrast_supp.csv'
df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")