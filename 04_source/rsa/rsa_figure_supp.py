# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

analysis = 'rdm_blocks'

networks = ['SomMot-precentral', 'SomMot-postcentral', 'SomMot-paracentral', \
    'DorsAttn-superiorparietal', 'DorsAttn-caudalmiddlefrontal', \
    # 'Cont-rostralmiddlefrontal', 'Cont-superiorfrontal', 'Cont-parsopercularis', 'Cont-parstriangularis', 'Cont-supramarginal']
    'Cont-rostralmiddlefrontal', 'Cont-superiorfrontal', 'Cont-supramarginal']

times = np.linspace(-0.2, 0.6, 82)
threshold = .05
step = 1
bsl_practice = False

# Load RSA data
random = {}
pattern = {}
diff_rp = {}
for network in tqdm(networks):
    if network not in diff_rp:
        diff_rp[network] = []
        random[network] = []
        pattern[network] = []
    for subject in subjects:
        res_path = RESULTS_DIR / 'RSA' / 'source' / network / analysis / subject
        behav_dir = op.join(HOME / 'raw_behavs' / subject)
        sequence = get_sequence(behav_dir)
        pattern_blocks, random_blocks = [], []
        for epoch_num in range(5):
            blocks = [i for i in range(1, 4)] if epoch_num == 0 else [i for i in range(5 * (epoch_num - 1) + 1, epoch_num * 5 + 1)]
            for block in blocks:
                pattern_blocks.append(np.load(res_path / f"pat-{epoch_num}-{block}.npy"))
                random_blocks.append(np.load(res_path / f"rand-{epoch_num}-{block}.npy"))
        if subject == 'sub05':
            pat_bsl = np.load(res_path / "pat-1-1.npy")
            rand_bsl = np.load(res_path / "rand-1-1.npy")
            for i in range(3):
                pattern_blocks[i] = pat_bsl.copy()
                random_blocks[i] = rand_bsl.copy()
        pattern_blocks = np.array(pattern_blocks)
        random_blocks = np.array(random_blocks)
        high, low = get_all_high_low(pattern_blocks, random_blocks, sequence, False, step)
        bsl_pat = np.nanmean(high[:, :3, :], (0, 1))
        bsl_rand = np.nanmean(low[:, :3, :], (0, 1))
        pat = np.nanmean(high, 0) - bsl_pat[np.newaxis, :] if bsl_practice else np.nanmean(high, 0)
        rand = np.nanmean(low, 0) - bsl_rand[np.newaxis, :] if bsl_practice else np.nanmean(low, 0)
        pattern[network].append(pat)
        random[network].append(rand)
        diff_rp[network].append(rand - pat)
    pattern[network] = np.array(pattern[network])
    random[network] = np.array(random[network])
    diff_rp[network] = np.array(diff_rp[network])

net_cmap = {
    'SomMot': '#DE8F05',
    'DorsAttn': '#029E73',
    'Cont': '#CA9161',
}

# Group regions by parent network (before '-')
net_groups = defaultdict(list)
for network in networks:
    parent = network.split('-')[0]
    net_groups[parent].append(network)

net_labels = {
    'SomMot': 'Sensorimotor',
    'DorsAttn': 'Dorsal\nAttention',
    'Cont': 'Central\nExecutive',
}

net_order = ['SomMot', 'DorsAttn', 'Cont']
n_cols = len(net_order)
n_rows = max(len(net_groups[n]) for n in net_order)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})

data_arr = diff_rp.copy()
chance = 0

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5),
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
        ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1, label='Stimulus onset')
        

        ax.axhline(chance, color='grey', alpha=.5)

        # pval = decod_stats(data, -1)
        pval = decod_stats(data - chance, -1)
        sig = pval < threshold

        # Non-sig line
        ax.plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
        # Sig regions in color
        for start, end in contiguous_regions(sig):
            ax.plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=color)

        ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
        ax.fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=color)
        ax.fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=color)

        if row == 0:
            ax.set_title(f'{net_labels[net_name]}', fontsize=11,
                         color=color, fontweight='bold')
        
        ax.text(0.05, 0.15, region_label, transform=ax.transAxes,
                fontsize=10, va='top', ha='left', color=color, fontstyle='italic')

        if col == 0:
            ax.set_ylabel('Diff in acc. (%)', fontsize=9)
            if row == 0:
                ax.legend()
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=9)

    for row in range(len(regions), n_rows):
        axes[row, col].set_visible(False)

fig.savefig(figures_dir / "rsa-net-supp-regions.pdf", transparent=True)
plt.close()
