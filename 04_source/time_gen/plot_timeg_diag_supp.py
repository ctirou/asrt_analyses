# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

from base import *
from config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, linregress
import pandas as pd
from tqdm.auto import tqdm
from mne.viz import Brain

subjects, subjects_dir = SUBJS15, FREESURFER_DIR

# network and custom label_names
figures_dir = ensured(FIGURES_DIR / "time_gen" / "source")

networks = ['SomMot', 'DorsAttn', 'Cont', 'Default']
network_names = ['Sensorimotor', 'Dorsal Attention', 'Central Executive', 'Default Mode']
times = np.linspace(-1.5, 1.5, 307)
chance = 25
threshold = .05
res_dir = RESULTS_DIR / 'TIMEG' / 'source'

data_type = 'decode_blocks_-3_1.5'
patterns = {}
randoms = {}
contrasts = {}
for network in tqdm(networks):
    pats_blocks, rands_blocks = [], []
    if not network in patterns:
        patterns[network], randoms[network] = [], []
        contrasts[network] = []
    for subject in subjects:
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
    pats_blocks, rands_blocks = np.array(pats_blocks), np.array(rands_blocks)
    patterns[network] = pats_blocks * 100
    randoms[network] = rands_blocks * 100
    contrasts[network] = patterns[network] - randoms[network]

times = np.linspace(-3, 1.5, 459)
win = np.where((times >= -1) & (times <= 0))[0]

cmap = ['#DE8F05','#029E73', '#CA9161','#FBAFE4']

# save table
chance = 25
for d, (data_name, data_dict) in enumerate(zip(['pattern', 'random', 'contrast'], [patterns, randoms, contrasts])):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True, sharex=True, layout="tight")
    rows = list()
    for i, network in enumerate(networks):
        axes[i].set_title(network_names[i], fontsize=13, fontstyle='italic')
        data = data_dict[network][:, 3:, win].mean(axis=0)
        subj_means = data_dict[network][:, 3:, win].mean(axis=1)  # (n_subjects, len_win)
        data_sem = np.std(subj_means, axis=0) / np.sqrt(len(subjects))
        if d == 2:
            axes[i].axhline(0, color='grey', alpha=.5)
        else:
            axes[i].axhline(chance, color='grey', alpha=.5)
        group_mean = subj_means.mean(0)
        axes[i].plot(times[win], group_mean, alpha=0.6, zorder=10, color=cmap[i], label=network, linewidth=1.5)
        axes[i].fill_between(times[win], group_mean - data_sem, group_mean + data_sem, alpha=0.2, color=cmap[i], zorder=5)
        # Linear fit across participants
        r_values = []
        for j in range(len(subjects)):
            _, _, r_j, _, _ = linregress(times[win], subj_means[j])
            r_values.append(r_j)
        mean_r = np.mean(r_values)
        _, p_val = ttest_1samp(r_values, 0)
        slope, intercept, _, _, _ = linregress(times[win], group_mean)
        fit_line = slope * times[win] + intercept
        axes[i].plot(times[win], fit_line, '--', color=cmap[i], linewidth=2, alpha=1, zorder=9)
        p_str = f'p = {p_val:.3f}' if p_val >= 0.05 else 'p < 0.05'
        axes[i].text(0.05, 0.2, f'R = {mean_r:.2f}\n{p_str}', transform=axes[i].transAxes,
                     fontsize=10, va='top', ha='left', color=cmap[i], fontweight='bold')
        # get table
        for j, subject in enumerate(subjects):
            # for t in range(data.shape[1]):
            for k, t in enumerate(win):
                rows.append({
                    "network": network_names[i],
                    "subject": subject,
                    "time": k,
                    "value": data[j, k] - chance if data_name != 'contrast' else data[j, k]
                })
    df = pd.DataFrame(rows)
    fname = f'decode_{data_name}_source_tr.csv'
    fig.suptitle(f"Linear fit in {data_name.capitalize()} trials", fontsize=14)
    plt.savefig(figures_dir / f"linear_fit-{data_name}.pdf", transparent=True)
    plt.close()
    df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")
    
# save time resolved diagonals
for data, data_fname in zip([cont_tr, pat_tr, rand_tr], ['contrast', 'pattern', 'random']):
    rows = list()
    for i, network in enumerate(networks):
        # get table
        for j, subject in enumerate(subjects):
            for t, idx in enumerate(idxt):
                rows.append({
                    "network": network_names[i],
                    "subject": subject,
                    "time": t,
                    "value": data[network][j, idx] - 0.25 if data_fname in ['pattern', 'random'] else data[network][j, idx]
                })
    df = pd.DataFrame(rows)
    fname = f'pa_source_tr_{data_fname}_all.csv' if data_type.endswith("new") else f'pa_source_tr_{data_fname}.csv'
    df.to_csv(FIGURES_DIR / "TM" / "data" / fname, index=False, sep=",")

# cmap = "viridis"
# cmap1 = "RdBu_r"
# cmap2 = "coolwarm"
# cmap3 = 'magma'

# def plot_onset(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)

# # Define subplot design layout
# design = [['A', 'B', 'C'], 
#           ['D', 'E', 'F'],
#           ['G', 'H', 'I'],
#           ['J', 'K', 'L']]


# plot_brains = False

# plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': 'Arial'})
# fig, axes = plt.subplot_mosaic(design, figsize=(13, 8), sharey=False, sharex=False, layout="tight",
#                                gridspec_kw={'height_ratios': [1, 1, 1, 1],
#                                             'width_ratios': [.5, .5, .5]})

# ### Pattern ###
# # Sig from GAMM
# # seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_decode_source.csv")
# # seg_df = seg_df[seg_df['metric'] == 'PATTERN']
# # # dictionary of boolean arrays
# # sig_dict = {}
# # for _, row in seg_df.iterrows():
# #     arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
# #     arr[row["start"]:row["end"] + 1] = True
# #     sig_dict[row["network"]] = arr
# # sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_decode_source.csv")
# # sig_df = sig_df[sig_df['metric'] == 'PATTERN']
# # for i, net in enumerate(sig_df['network'].unique()):
# #     if net in sig_dict:
# #         if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
# #             del sig_dict[net]

# for i, (network, pattern_idx) in enumerate(zip(networks, ['A', 'D', 'G', 'J'])):
#     # plot_onset(axes[pattern_idx])
#     data = patterns[network][:, 3:, win].mean(1)
#     # axes[pattern_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
#     axes[pattern_idx].axhline(chance, color='grey', alpha=.5)
#     # Get significant clusters
#     # sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
#     pval = decod_stats(data - chance, -1)
#     sig = pval < threshold
#     # Main plot
#     axes[pattern_idx].plot(times[win], data.mean(0), alpha=1, zorder=10, color='C7')
#     # Plot significant regions separately
#     for start, end in contiguous_regions(sig):
#         axes[pattern_idx].plot(times[win][start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
#     sem = np.std(data, axis=0) / np.sqrt(len(subjects))
#     axes[pattern_idx].fill_between(times[win], data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
#     # Highlight significant regions
#     axes[pattern_idx].fill_between(times[win], data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
#     axes[pattern_idx].fill_between(times[win], data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
#     axes[pattern_idx].set_ylabel('Acc. (%)', fontsize=11)
#     axes[pattern_idx].set_ylim(23, 35)
#     axes[pattern_idx].set_xticks(np.arange(-1, 0.25, 0.25))
#     axes[pattern_idx].set_yticks(np.arange(25, 36, 5))
#     axes[pattern_idx].set_yticklabels(np.arange(25, 36, 5))
#     if pattern_idx == 'A':
#         axes[pattern_idx].set_title('Pattern')
#     elif pattern_idx == 'AB':
#         axes[pattern_idx].set_xlabel('Time (s)', fontsize=11)
#     # sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
#     # if sig_level != 'ns':
#     #     axes[pattern_idx].text(0.5, 33, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')
    
# ### Random ###    
# # Sig from GAMM
# seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_pat_rand_source.csv")
# seg_df = seg_df[seg_df['metric'] == 'RANDOM']
# # dictionary of boolean arrays
# sig_dict = {}
# for _, row in seg_df.iterrows():
#     arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
#     arr[row["start"]:row["end"] + 1] = True
#     sig_dict[row["network"]] = arr
# sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_pat_rand_source.csv")
# sig_df = sig_df[sig_df['metric'] == 'RANDOM']
# for i, net in enumerate(sig_df['network'].unique()):
#     if net in sig_dict:
#         if sig_df[sig_df['network'] == net]['signif_holm'][i+10] == 'ns':
#             del sig_dict[net]

# for i, (network, random_idx) in enumerate(zip(networks, ['B', 'E', 'H', 'K'])):
#     plot_onset(axes[random_idx])
#     data = randoms[network][:, 3:].mean(1)
#     axes[random_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
#     axes[random_idx].axhline(chance, color='grey', alpha=.5)
#     # Get significant clusters
#     # sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
#     pval = decod_stats(data - chance, -1)
#     sig = pval < threshold
#     # Main plot
#     axes[random_idx].plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
#     # Plot significant regions separately
#     for start, end in contiguous_regions(sig):
#         axes[random_idx].plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
#     sem = np.std(data, axis=0) / np.sqrt(len(subjects))
#     axes[random_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
#     # Highlight significant regions
#     axes[random_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
#     axes[random_idx].fill_between(times, data.mean(0) - sem, chance, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
#     axes[random_idx].set_ylabel('Acc. (%)', fontsize=11)
#     axes[random_idx].set_ylim(23, 35)
#     # axes[random_idx].set_xticks(np.arange(-1, 2, 0.5))
#     axes[random_idx].set_xticks(np.arange(-3, 1.5, 0.5))
#     axes[random_idx].set_yticks(np.arange(25, 36, 5))
#     axes[random_idx].set_yticklabels(np.arange(25, 36, 5))
#     if random_idx == 'B':
#         axes[random_idx].set_title('Random')
#     if random_idx == 'AC':
#         axes[random_idx].set_xlabel('Time (s)', fontsize=11)
#     # sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
#     # if sig_level != 'ns':
#     #     axes[random_idx].text(0.5, 33, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')

# ### Contrast ###
# win = np.where((times >= -0.5) & (times < 0))[0]
# msig = []
# for network in networks:
#     s = []
#     for sub in range(len(subjects)):
#         cont = contrasts[network][sub, 3:, win].mean()
#         s.append(cont)
#     sig = ttest_1samp(s, 0, axis=0)[1] < threshold
#     if sig:
#         print(f"Significant contrast for {network} in the window {times[win][0]} to {times[win][-1]}")
#     msig.append(sig)
    
# # Sig from GAMM
# # get significant time points from GAMM csv --- contrast
# seg_df = pd.read_csv(FIGURES_DIR / "TM" / "em_segments_pa_tr_cont_source.csv")
# seg_df = seg_df[seg_df['metric'] == 'PA']
# # dictionary of boolean arrays
# sig_dict = {}
# for _, row in seg_df.iterrows():
#     arr = sig_dict.get(row["network"], np.zeros(len(times), dtype=bool))
#     arr[row["start"]:row["end"] + 1] = True
#     sig_dict[row["network"]] = arr
# sig_df = pd.read_csv(FIGURES_DIR / "TM" / "smooth_pa_tr_cont_source.csv")
# sig_df = sig_df[sig_df['metric'] == 'PA']
# for i, net in enumerate(sig_df['network'].unique()):
#     if net in sig_dict:
#         if sig_df[sig_df['network'] == net]['signif_holm'][i] == 'ns':
#             del sig_dict[net]

# for i, (network, contrast_idx) in enumerate(zip(networks, ['C', 'F', 'I', 'L'])):
#     axes[contrast_idx].set_ylim(-2, 10)
#     plot_onset(axes[contrast_idx])
#     data = contrasts[network][:, 3:].mean(1)
#     axes[contrast_idx].axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.1)
#     axes[contrast_idx].axhline(0, color='grey', alpha=.5)
#     # Get significant clusters
#     # sig = sig_dict[network_names[i]] if network_names[i] in sig_dict else np.zeros(data.shape[1], dtype=bool)
#     pval = decod_stats(data, -1)
#     sig = pval < threshold
#     # Main plot
#     axes[contrast_idx].plot(times, data.mean(0), alpha=1, zorder=10, color='C7')
#     # Plot significant regions separately
#     for start, end in contiguous_regions(sig):
#         axes[contrast_idx].plot(times[start:end], data.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
#     sem = np.std(data, axis=0) / np.sqrt(len(subjects))
#     axes[contrast_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, alpha=0.2, zorder=5, facecolor='C7')
#     # Highlight significant regions
#     axes[contrast_idx].fill_between(times, data.mean(0) - sem, data.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
#     axes[contrast_idx].fill_between(times, data.mean(0) - sem, 0, where=sig, alpha=0.3, zorder=5, facecolor=cmap[i])
#     axes[contrast_idx].axhline(0, color='grey', alpha=.5)
#     axes[contrast_idx].set_ylabel('Diff in acc. (%)', fontsize=11)
#     axes[contrast_idx].set_yticks(np.arange(0, 5, 2))
#     axes[contrast_idx].set_yticklabels(np.arange(0, 5, 2))
#     if contrast_idx == 'C':
#         axes[contrast_idx].set_title('Contrast\n(Pattern - Random)')
#     elif contrast_idx == 'AD':
#         axes[contrast_idx].set_xlabel('Time (s)', fontsize=11)
#     # sig_level = sig_df[sig_df['network'] == network_names[i]]['signif_holm'].values[0]
#     # if sig_level != 'ns':
#     #     axes[contrast_idx].text(-0.5, 4, sig_level, fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')

# fig.savefig(figures_dir / "timeg-diag.pdf", transparent=True)

# # Correlation with behavior
# from scipy.stats import spearmanr as spear
# learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices_blocks.csv', sep=",", index_col=0)
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': 'Arial'})
# fig, axes = plt.subplots(5, 2, figsize=(7, 9), sharey=True, sharex=True, layout="tight")
# for i, ax in enumerate(axes.flatten()):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.axhline(0, color='black', alpha=1)
#     ax.axvspan(0, 0.2, facecolor='grey', edgecolor=None, alpha=.2)
#     network = networks[i]
#     all_rhos = np.array([[spear(learn_index_df.iloc[sub, :], contrasts[network][sub, :, t])[0] for t in range(len(times))] for sub in range(len(subjects))])
#     # all_rhos, _, _ = fisher_z_and_ttest(all_rhos)
#     sem = np.std(all_rhos, axis=0) / np.sqrt(len(subjects))
#     p_values = decod_stats(all_rhos, -1)
#     sig = p_values < 0.05
#     # Main plot
#     ax.plot(times, all_rhos.mean(0), alpha=1, zorder=10, color='C7')
#         # Plot significant regions separately
#     for start, end in contiguous_regions(sig):
#         ax.plot(times[start:end], all_rhos.mean(0)[start:end], alpha=1, zorder=10, color=cmap[i])
#     ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, alpha=0.5, zorder=5, facecolor='C7')
#     # Highlight significant regions
#     ax.fill_between(times, all_rhos.mean(0) - sem, all_rhos.mean(0) + sem, where=sig, alpha=0.5, zorder=5, color=cmap[i])
#     ax.set_title(network_names[i], fontsize=13, fontstyle='italic')
#     wo = np.where((times >= -0.5) & (times < 0))[0]
#     mrho = all_rhos[:, wo].mean(1)
#     mrho_sig = ttest_1samp(mrho, 0)[1]
#     if mrho_sig < 0.05:
#         print(f"Significant correlation for {network} in the window {times[wo][0]} to {times[wo][-1]}")
#         ax.axvspan(times[win][0], times[win][-1], facecolor=cmap[i], edgecolor=None, alpha=0.3, zorder=5)
#         ax.text(0.4, 0.7, '*', fontsize=20, ha='center', va='center', color=cmap[i], weight='bold')
#     if ax in axes[:, 0]:
#         ax.set_ylabel("Spearman's rho", fontsize=11)
#     # Only set xlabel for axes in the bottom row
#     if i >= (axes.shape[0] - 1) * axes.shape[1]:
#         ax.set_xlabel("Time (s)", fontsize=11)
#     ax.set_xticks(np.arange(-1, 2, 0.5))

# fig.savefig(figures_dir / "timeg-corr.pdf", transparent=True)
