import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict
from scipy.stats import ttest_1samp, spearmanr
import matplotlib.pyplot as plt
import gc

# params
trial_types = ["all", "pattern", "random"]
trial_type = 'pattern'
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 2
chance = 0.5
threshold = 0.05
scoring = "accuracy"
parc='aparc'
hemi = 'both'
params = "pred_decoding"
verbose = True
jobs = -1

# figures dir
figures = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type
ensure_dir(figures)
# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
del epochs

ensure_dir(figures / 'corr')

combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

decod_in_lab = dict()

gc.collect()

for ilabel in range(68): # put in subjects and divide
        
    sims_in_label, decod_in_subs = [], []
    
    for subject in subjects:
        
        sims_in_sub, decod_in_sess = [], []
        
        # read epochs
        epo_dir = data_path / lock
        epo_fnames = [epo_dir / f"{f}" for f in sorted(os.listdir(epo_dir)) if ".fif" in f and subject in f]
        all_epo = [mne.read_epochs(fname, preload=True, verbose=verbose) for fname in epo_fnames]
        # read behav files
        beh_dir = data_path / "behav"
        beh_fnames = [beh_dir / f"{f}" for f in sorted(os.listdir(beh_dir)) if ".pkl" in f and subject in f]
        all_beh = [pd.read_pickle(fname).reset_index() for fname in beh_fnames]
        # get labels
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
        label = labels[ilabel]
        
        all_session_cms, all_session_scores = [], []
        
        #### get stimuli proportions
        print_proportions(subject, all_beh)
        
        for session_id, session in enumerate(sessions):
                        
            # get session behav and epoch
            if session_id == 0:
                session = 'prac'
            else:
                session = 'sess-%s' % (str(session_id).zfill(2))
            behav = all_beh[session_id]
            epoch = all_epo[session_id]
            if lock == 'button': 
                epoch_bsl_fname = data_path / "bsl" / f"{subject}_{session_id}_bl-epo.fif"
                epoch_bsl = mne.read_epochs(epoch_bsl_fname, verbose=verbose)
            # read forward solution    
            fwd_fname = res_path / "fwd" / lock / f"{subject}-fwd-{session_id}.fif"
            fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(epoch, tmin=0, tmax=.6, method="empirical", rank="info", verbose=verbose)
            # compute noise covariance
            if lock == 'button':
                noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info", verbose=verbose)
            else:
                noise_cov = mne.compute_covariance(epoch, tmin=-.2, tmax=0, method="empirical", rank="info", verbose=verbose)
            info = epoch.info
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative', verbose=verbose)
            # compute source estimate
            filters = make_lcmv(info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                            pick_ori=None, rank=rank, reduce_rank=True, verbose=verbose)
            stcs = apply_lcmv_epochs(epoch, filters=filters, verbose=verbose)
                        
            print(f"{ilabel+1}/{len(labels)}", subject, session, label.name)
            
            # get stcs in label
            stcs_data = [stc.in_label(label).data for stc in stcs]
            stcs_data = np.array(stcs_data)
            assert len(stcs_data) == len(behav)
        
            if trial_type == 'pattern':
                pattern = behav.trialtypes == 1
                X = stcs_data[pattern]
                y = behav.positions[pattern]
            elif trial_type == 'random':
                random = behav.trialtypes == 2
                X = stcs_data[random]
                y = behav.positions[random]
            else:
                X = stcs_data
                y = behav.positions
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]
            
            # set-up the classifier and cv structure
            clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=10000)) # use JAX maybe
            clf = SlidingEstimator(clf, n_jobs=jobs, scoring=scoring, verbose=verbose) # get time of one sample (slide), try with less jobs maybe ?
            cv = StratifiedKFold(folds, shuffle=True)   
            
            pred = np.zeros((len(y), X.shape[-1]))
            pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
            # there is only randoms in practice sessions
            for train, test in cv.split(X, y):
                clf.fit(X[train], y[train])
                pred[test] = np.array(clf.predict(X[test]))
                pred_rock[test] = np.array(clf.predict_proba(X[test]))
                
            cms, scores = list(), list()
            for t in range(X.shape[-1]):
                cms.append(confusion_matrix(y[test], pred[test, t], normalize='true', labels=clf.classes_))
                scores.append(roc_auc_score(y[test], pred_rock[test, t, :], multi_class='ovr'))
                
            scores = np.array(scores)
            all_session_scores.append(scores)
            # np.save(figures / f'{label.name}_{subject}_{session_id}-scores.npy', scores)

            c = np.array(cms).T # don't tramspose, take peak decoding performance
            all_session_cms.append(c)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), layout='tight')
            fig.suptitle(f'{label.name} // {subject} // {session}')
            
            ax1.plot(times, scores)
            ax1.set_ylabel('roc-auc')
            ax1.axvspan(0, 0.2, color='grey', alpha=.2)
            ax1.axhline(chance, color='black', ls='dashed', alpha=.5)
            ax1.set_ylim(0.2, 0.8)
            
            # ConfusionMatrixDisplay.from_estimator(clf, pred[test], y[test]).plot()
            disp = ConfusionMatrixDisplay(c.mean(-1), display_labels=set(y))
            disp.plot(ax=ax2)
            disp.im_.set_clim(0, 1)  # Set colorbar limits

            # cax = ax3.imshow(c.mean(-1), cmap='viridis')
            # ax3.set_xticks(np.arange(len(set(y))), labels=set(y))
            # ax3.set_yticks(np.arange(len(set(y))), labels=set(y))
            # cax.set_clim(0, 1)
            # for i in range(len(set(y))):
            #     for j in range(len(set(y))):
            #         text = ax3.text(j, i, round(c[i, j, :].mean(-1), 2),
            #                        ha='center', va='center', color='w')
            # ax3.set_ylabel("True label")
            # ax3.set_xlabel("Predicted label")

            # plt.show()
            fig.savefig(figures / f"{label.name}_{subject}_{session}.png")
            plt.close()
              
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list()
            three_four_similarity = list()
                        
            for itime in range(len(times)):
                one_two_similarity.append(c[0, 1, itime])
                one_three_similarity.append(c[0, 2, itime])
                one_four_similarity.append(c[0, 3, itime])
                two_three_similarity.append(c[1, 2, itime])
                two_four_similarity.append(c[1, 3, itime])
                three_four_similarity.append(c[2, 3, itime])
                                
            similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                            two_three_similarity, two_four_similarity, three_four_similarity]
            
            similarities = np.array(similarities)
            sims_in_sub.append(similarities)
            
            # np.save(figures / f'{label.name}_{subject}_{session_id}-rsa.npy', similarities)
                    
        all_session_cms = np.array(all_session_cms)
        all_session_scores = np.array(all_session_scores)
        
        sims_in_sub = np.array(sims_in_sub)
        sims_in_label.append(np.array(sims_in_sub))
        
        decod_in_sess = np.array(decod_in_sess)
        
        best_time = [i for i, j in enumerate(times) if 0 <= j <= 0.2]
        
        fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7))
        fig.suptitle(f'{label.name} // {subject}')
        for i, (ax, session) in enumerate(zip(axs.flat[:5], sessions)):
            ax.plot(times, all_session_scores[i])
            ax.axvspan(0, 0.2, color='grey', alpha=.2)
            ax.axhline(0, color='black', ls='dashed', alpha=.5)
            ax.set_title(session)
            ax.axvspan(0, 0.2, color='grey', alpha=.2)
            ax.axhline(chance, color='black', ls='dashed', alpha=.5)
            ax.set_ylim(0.2, 0.8)
            
            if i   != 0:
                ax.set_yticklabels([])

        
        for i, ax in zip(range(5), axs.flat[5:]):
            # cax = ax.imshow(all_session_cms[i].mean(-1), cmap='viridis')
            # ax.set_xticks(np.arange(len(set(y))), labels=set(y))
            # ax.set_yticks(np.arange(len(set(y))), labels=set(y))
            # cax.set_clim(0, 1)
            # for i in range(len(set(y))):
            #     for j in range(len(set(y))):
            #         text = ax.text(j, i, round(c[i, j, :].mean(-1), 2),
            #                        ha='center', va='center', color='w')
            # ax.set_ylabel("True label")
            # ax.set_xlabel("Predicted label")
            
            disp = ConfusionMatrixDisplay(all_session_cms[i, :, :, 40:80].mean(-1), display_labels=set(y))
            disp.plot(ax=ax)
            disp.im_.set_clim(0, 1)  # Set colorbar limits


    # sims_in_label = np.array(sims_in_label)
    # np.save(figures / f'{label.name}.npy', sims_in_label)
    
    # # need diff_inout for this, per sub
    # all_rhos = []
    # for sub in range(len(subjects)):
    #     rhos = []
    #     for t in range(len(times)):
    #         rhos.append(spearmanr([0, 1, 2, 3, 4], sims_in_label[sub, : t]))
    #     all_rhos.append(rhos)
    # all_rhos = np.array(all_rhos)
    # np.save(figures / 'corr' / f'{label}_rhos.npy', all_rhos)

    gc.collect() # only works if vars are deleted