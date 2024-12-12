import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr as spear
from tqdm.auto import tqdm
from base import *
from config import *

lock = 'button'
data = 'new_k10'
analysis = 'usual'
analysis = 'pat_high_rdm_high'
# analysis = 'pat_high_rdm_low'
# analysis = 'rdm_high_rdm_low'
analyses = ['usual', 'pat_high_rdm_high', 'pat_high_rdm_low', 'rdm_high_rdm_low']

overwrite = False

data_path = DATA_DIR
subjects, epochs_list = SUBJS, EPOCHS
metric = 'mahalanobis'

# get times
times = np.load(data_path / "times.npy")

for lock in ['stim', 'button']:

    for analysis in analyses:

        figures_dir = FIGURES_DIR / "RSA" / "sensors" / lock / f'{data}_{analysis}'
        ensure_dir(figures_dir)
        
        all_highs, all_lows = [], []

        for subject in tqdm(subjects):
            
            res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / f"{data}_rdm" / subject
            ensure_dir(res_path)
                
            # Read the behav file to get the sequence 
            behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
            sequence = get_sequence(behav_dir)

            high, low = get_all_high_low(res_path, sequence, analysis, cv=True)
                
            all_highs.append(high)    
            all_lows.append(low)
            
        all_highs = np.array(all_highs)
        all_lows = np.array(all_lows)
        
        color1 = "#008080"
        color2 = "#FFA500"
        color3 = "#dd1c77"
        
        # directly using the reverse formula
        # plot reverse difference high vs. low sequence averaging all sessions
        # rev_high = all_highs[:, :, 1:, :].mean(axis=1) - all_highs[:, :, 0, :].mean(axis=1, keepdims=True)
        # rev_low = all_lows[:, :, 1:, :].mean(axis=1) - all_lows[:, :, 0, :].mean(axis=1, keepdims=True)
        # diff = rev_low - rev_high
        
        # plot reverse difference high vs. low sequence averaging all sessions
        rev_high = all_highs[:, :, 1:, :].mean((1, 2)) - all_highs[:, :, 0, :].mean(axis=1)
        rev_low = all_lows[:, :, 1:, :].mean((1, 2)) - all_lows[:, :, 0, :].mean(axis=1)
        rev_diff = rev_low - rev_high
        
        plt.subplots(1, 1, figsize=(16, 11))
        plt.plot(times, rev_diff.mean(0), label='(low_post - low_pre) - (high_post - high_pre)', color=color3, alpha=0.6)
        plt.plot(times, rev_high.mean(0), label='high_post - high_pre', alpha=0.6)
        plt.plot(times, rev_low.mean(0), label='low_post - low_pre', alpha=0.6)
        plt.plot(times, all_highs[:, :, 1:, :].mean((0, 1, 2)), label='high_post')
        plt.plot(times, all_highs[:, :, 0, :].mean((0, 1)), label='high_pre')
        plt.plot(times, all_lows[:, :, 1:, :].mean((0, 1, 2)), label='low_post')
        plt.plot(times, all_lows[:, :, 0, :].mean((0, 1)), label='low_pre')
        p_values_unc = ttest_1samp(rev_diff,  axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(rev_diff, -1)
        sig = p_values < 0.05
        plt.fill_between(times, 0, rev_diff.mean(0), where=sig_unc, color=color1, alpha=0.2, label='uncorrected')
        plt.fill_between(times, 0, rev_diff.mean(0), where=sig, color=color2, alpha=0.3, label='corrected')
        plt.axhline(0, color='black', linestyle='dashed')
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.legend()
        plt.title(f'{analysis} average', style='italic')
        plt.savefig(op.join(figures_dir, 'ave_low_high.pdf'))
        plt.close()
        
        # reverse difference high vs. low sequence across sessions
        for i in range(1, 5): 
            rev_high = all_highs[:, :, i, :].mean(1) - all_highs[:, :, 0, :].mean(axis=1)
            rev_low = all_lows[:, :, i, :].mean(1) - all_lows[:, :, 0, :].mean(axis=1)
            diff_corr = rev_low - rev_high
            
            plt.subplots(1, 1, figsize=(16, 11))
            plt.plot(times,  diff_corr.mean(0), label='(low_post - low_pre) - (high_post - high_pre)', color=color3, alpha=0.6)
            plt.plot(times, rev_high.mean(0), label='high_post - high_pre', alpha=0.6)
            plt.plot(times, rev_low.mean(0), label='low_post - low_pre', alpha=0.6)
            plt.plot(times, all_highs[:, :, i, :].mean((0, 1)), label='high_post')
            plt.plot(times, all_highs[:, :, 0, :].mean((0, 1)), label='high_pre')
            plt.plot(times, all_lows[:, :, i, :].mean((0, 1)), label='low_post')
            plt.plot(times, all_lows[:, :, 0, :].mean((0, 1)), label='low_pre')
            p_values_unc = ttest_1samp(diff_corr,  axis=0, popmean=0)[1]
            sig_unc = p_values_unc < 0.05
            p_values = decod_stats(diff_corr, -1)
            sig = p_values < 0.05
            plt.fill_between(times, 0, diff_corr.mean(0), where=sig_unc, color=color1, alpha=0.2, label='uncorrected')
            plt.fill_between(times, 0, diff_corr.mean(0), where=sig, color=color2, alpha=0.3, label='corrected')
            plt.axhline(0, color='black', linestyle='dashed')
            if lock == 'stim':
                plt.axvspan(0, 0.2, color='grey', alpha=.2)
            else:
                plt.axvline(0, color='black')
            plt.legend()
            plt.title(f'{analysis} session {i}', style='italic')
            plt.savefig(op.join(figures_dir, '%s_low_high.pdf' % (str(i))))
            plt.close()
                        
        rev_diff_sess = list()   
        for i in range(5):
            rev_low = all_lows[:, :, i, :].mean(1) - all_lows[:, :, 0, :].mean(axis=1)
            rev_high = all_highs[:, :, i, :].mean(1) - all_highs[:, :, 0, :].mean(axis=1)
            rev_diff_sess.append(rev_low - rev_high)
        rev_diff_sess = np.array(rev_diff_sess).swapaxes(0, 1)
        
        # plot reverse correlations
        rhos = [[spear([0, 1, 2, 3, 4], rev_diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
        rhos = np.array(rhos)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, rhos.mean(0), color=color3, label='rhos')
        p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(rhos, -1)
        sig = p_values < .05
        plt.fill_between(times, 0, rhos.mean(0), where=sig_unc, color=color1, alpha=.2, label='uncorrected')
        plt.fill_between(times, 0, rhos.mean(0), where=sig, color=color2, alpha=.3, label='corrected')
        plt.axhline(0, color='black', linestyle='dashed')
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.legend()
        plt.title(f'{analysis} corr', style='italic')
        plt.savefig(op.join(figures_dir, 'corr.pdf'), transparent=True)
        plt.close()

        learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
        # plot reverse across subjects
        all_pvalues, all_rhos = [], []
        for t in range(len(times)):
            rho, pval = spear(learn_index_df["4"], rev_diff_sess[:, -1, t])
            all_rhos.append(rho)
            all_pvalues.append(pval)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, all_rhos, color=color3, label='rho')
        sig = (np.asarray(all_pvalues) < 0.05)
        plt.fill_between(times, all_rhos, 0, where=sig, color=color2, alpha=.4, label='significant')  # Solid line at the bottom when sig is true
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.axhline(0, color='black', linestyle='dashed')
        plt.legend()
        plt.title(f'{analysis} as corr', style='italic')
        plt.savefig(op.join(figures_dir, 'as.pdf'), transparent=True)
        plt.close()

        # plot reverse within subjects
        all_rhos = []
        for sub in tqdm(range(len(subjects))):
            rhos = []
            for t in range(len(times)):
                rhos.append(spear(learn_index_df.iloc[sub, :], rev_diff_sess[sub, :, t])[0])
            all_rhos.append(rhos)
        all_rhos = np.array(all_rhos)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, all_rhos.mean(0), color=color3, label='rhos')
        p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(all_rhos, -1)
        sig = p_values < 0.05
        plt.fill_between(times, all_rhos.mean(0), 0, where=sig_unc, color=color1, alpha=.2, label='uncorrected')
        plt.fill_between(times, all_rhos.mean(0), 0, where=sig, color=color2, alpha=.4, label='corrected')
        plt.axhline(0, color="black", linestyle="dashed")
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.axhline(0, color='black', linestyle='dashed')
        plt.legend()
        plt.title(f'{analysis} ws corr', style='italic')
        plt.savefig(op.join(figures_dir, 'ws.pdf'), transparent=True)
        plt.close()

        # WITHOUT PRACTICE #
        diff = all_highs[:, :, 1:, :].mean((1, 2)) - all_lows[:, :, 1:, :].mean((1, 2))
        rev_diff = diff.copy()
        
        plt.subplots(1, 1, figsize=(16, 11))
        plt.plot(times, rev_diff.mean(0), label='low_post - high_post', color=color3, alpha=0.6)
        p_values_unc = ttest_1samp(rev_diff,  axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(rev_diff, -1)
        sig = p_values < 0.05
        plt.fill_between(times, 0, rev_diff.mean(0), where=sig_unc, color=color1, alpha=0.2, label='uncorrected')
        plt.fill_between(times, 0, rev_diff.mean(0), where=sig, color=color2, alpha=0.3, label='corrected')
        plt.axhline(0, color='black', linestyle='dashed')
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.legend()
        plt.title(f'{analysis} average - no practice', style='italic')
        plt.savefig(op.join(figures_dir, 'np_ave_low_high.pdf'))
        plt.close()
        
        # reverse difference high vs. low sequence across sessions
        for i in range(1, 5): 
            diff_sess = all_highs[:, :, i, :].mean(1) -  all_lows[:, :, i, :].mean(1)
            plt.subplots(1, 1, figsize=(16, 11))
            plt.plot(times,  diff_sess.mean(0), label='low_post - high_post', color=color3, alpha=0.6)
            plt.plot(times, all_highs[:, :, i, :].mean((0, 1)), label='high_post')
            plt.plot(times, all_lows[:, :, i, :].mean((0, 1)), label='low_post')
            p_values_unc = ttest_1samp(diff_sess,  axis=0, popmean=0)[1]
            sig_unc = p_values_unc < 0.05
            p_values = decod_stats(diff_sess, -1)
            sig = p_values < 0.05
            plt.fill_between(times, 0, diff_sess.mean(0), where=sig_unc, color=color1, alpha=0.2, label='uncorrected')
            plt.fill_between(times, 0, diff_sess.mean(0), where=sig, color=color2, alpha=0.3, label='corrected')
            plt.axhline(0, color='black', linestyle='dashed')
            if lock == 'stim':
                plt.axvspan(0, 0.2, color='grey', alpha=.2)
            else:
                plt.axvline(0, color='black')
            plt.legend()
            plt.title(f'{analysis} session {i} - no practice', style='italic')
            plt.savefig(op.join(figures_dir, 'np_%s_low_high.pdf' % (str(i))))
            plt.close()
                        
        diff_sess = list()   
        for i in range(5):
            rev_low = all_lows[:, :, i, :].mean(1)
            rev_high = all_highs[:, :, i, :].mean(1)
            diff_sess.append(rev_low - rev_high)
        diff_sess = np.array(diff_sess).swapaxes(0, 1)
        
        # plot reverse correlations
        rhos = [[spear([0, 1, 2, 3, 4], diff_sess[sub, :, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
        # rhos = [[spear([1, 2, 3, 4], diff_sess[sub, 1:, itime])[0] for itime in range(len(times))] for sub in range(len(subjects))]
        rhos = np.array(rhos)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, rhos.mean(0), color=color3, label='rhos')
        p_values_unc = ttest_1samp(rhos, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(rhos, -1)
        sig = p_values < .05
        plt.fill_between(times, 0, rhos.mean(0), where=sig_unc, color=color1, alpha=.2, label='uncorrected')
        plt.fill_between(times, 0, rhos.mean(0), where=sig, color=color2, alpha=.3, label='corrected')
        plt.axhline(0, color='black', linestyle='dashed')
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.legend()
        plt.title(f'{analysis} corr - no practice', style='italic')
        plt.savefig(op.join(figures_dir, 'np_corr.pdf'), transparent=True)
        plt.close()

        learn_index_df = pd.read_csv(FIGURES_DIR / 'behav' / 'learning_indices.csv', sep="\t", index_col=0)
        # plot reverse across subjects
        all_pvalues, all_rhos = [], []
        for t in range(len(times)):
            rho, pval = spear(learn_index_df["4"], diff_sess[:, -1, t])
            all_rhos.append(rho)
            all_pvalues.append(pval)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, all_rhos, color=color3, label='rho')
        sig = (np.asarray(all_pvalues) < 0.05)
        plt.fill_between(times, all_rhos, 0, where=sig, color=color2, alpha=.4, label='significant')  # Solid line at the bottom when sig is true
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.axhline(0, color='black', linestyle='dashed')
        plt.legend()
        plt.title(f'{analysis} as corr - no practice', style='italic')
        plt.savefig(op.join(figures_dir, 'np_as.pdf'), transparent=True)
        plt.close()

        # plot reverse within subjects
        all_rhos = []
        for sub in tqdm(range(len(subjects))):
            rhos = []
            for t in range(len(times)):
                rhos.append(spear(learn_index_df.iloc[sub, :], diff_sess[sub, :, t])[0])
            all_rhos.append(rhos)
        all_rhos = np.array(all_rhos)
        plt.subplots(1, 1, figsize=(14, 5))
        plt.plot(times, all_rhos.mean(0), color=color3, label='rhos')
        p_values_unc = ttest_1samp(all_rhos, axis=0, popmean=0)[1]
        sig_unc = p_values_unc < 0.05
        p_values = decod_stats(all_rhos, -1)
        sig = p_values < 0.05
        plt.fill_between(times, all_rhos.mean(0), 0, where=sig_unc, color=color1, alpha=.2, label='uncorrected')
        plt.fill_between(times, all_rhos.mean(0), 0, where=sig, color=color2, alpha=.4, label='corrected')
        plt.axhline(0, color="black", linestyle="dashed")
        if lock == 'stim':
            plt.axvspan(0, 0.2, color='grey', alpha=.2)
        else:
            plt.axvline(0, color='black')
        plt.axhline(0, color='black', linestyle='dashed')
        plt.legend()
        plt.title(f'{analysis} ws corr - no practice', style='italic')
        plt.savefig(op.join(figures_dir, 'np_ws.pdf'), transparent=True)
        plt.close()
