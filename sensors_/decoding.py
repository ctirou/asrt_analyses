import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from base import ensure_dir
from config import *
import gc
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# params
subjects = SUBJS # done : none

trial_type = 'pattern' # "all", "pattern", or "random"
data_path = DATA_DIR
lock = "stim" 
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 5
scoring = "roc_auc"
parc='aparc'
hemi = 'both'
analysis = "pred_decoding"
verbose = True
jobs = -1

method = "logReg"

chance = 0.5

# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
    
del epochs, epoch_fname
gc.collect()

for subject in subjects[:4]:
        
    solvers = ['lbfgs', 'newton-cholesky']
    for solver in solvers:
        
        sub_scores, sub_cms, sub_rsa = [], [], []

        for session_id, session in enumerate(sessions):
            
            # results dir
            res_dir = res_path / analysis / 'source' / lock / trial_type / subject / session
            ensure_dir(res_dir)    
            # read stim epoch
            epoch_fname = data_path / lock / f"{subject}_{session_id}_s-epo.fif"
            epoch = mne.read_epochs(epoch_fname, preload=True, verbose=verbose)
            # read behav
            behav_fname = data_path / "behav" / f"{subject}_{session_id}.pkl"
            behav = pd.read_pickle(behav_fname).reset_index()
                
            X = epoch.get_data(copy=False)
            y = behav.positions
            y = y.reset_index(drop=True)            
            assert X.shape[0] == y.shape[0]
            
            del epoch, epoch_fname, behav_fname
            gc.collect()
            
            # set-up the classifier and cv structure
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, multi_class="ovr", max_iter=100000, solver=solver, class_weight="balanced", random_state=42))
            clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose)
            cv = StratifiedKFold(folds, shuffle=True)
            
            pred = np.zeros((len(y), X.shape[-1]))
            pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
            for train, test in cv.split(X, y):
                clf.fit(X[train], y[train])
                pred[test] = np.array(clf.predict(X[test]))
                pred_rock[test] = np.array(clf.predict_proba(X[test]))
                                        
            cms, scores = list(), list()
            for itime in range(len(times)):
                cms.append(confusion_matrix(y[:], pred[:, itime], normalize='true', labels=[1, 2, 3, 4]))
                scores.append(roc_auc_score(y[:], pred_rock[:, itime, :], multi_class='ovr', average='weighted'))
            
            cms_arr = np.array(cms)
            np.save(res_dir / "cms.npy", cms_arr)
            sub_cms.append(cms_arr)
            scores = np.array(scores)
            np.save(res_dir / "scores.npy", scores)
            sub_scores.append(scores)
            
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list()
            three_four_similarity = list()
            for itime in range(len(times)):
                one_two_similarity.append(cms_arr[itime, 0, 1])
                one_three_similarity.append(cms_arr[itime, 0, 2])
                one_four_similarity.append(cms_arr[itime, 0, 3])
                two_three_similarity.append(cms_arr[itime, 1, 2])
                two_four_similarity.append(cms_arr[itime, 1, 3])
                three_four_similarity.append(cms_arr[itime, 2, 3])
                                            
            similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                            two_three_similarity, two_four_similarity, three_four_similarity]
            similarities = np.array(similarities)
            np.save(res_dir / 'rsa.npy', similarities)
            sub_rsa.append(similarities)
            
            # del X, y, clf, cv, train, test, pred, pred_rock, cms, cms_arr, scores, similarities
            # del one_two_similarity, one_three_similarity, one_four_similarity, two_three_similarity, two_four_similarity, three_four_similarity
            # gc.collect()
            
        sub_cms = np.array(sub_cms)
        sub_scores = np.array(sub_scores)
        sub_rsa = np.array(sub_rsa)
        
        fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7), sharey=False)
        fig.suptitle(f'{subject} / ${solver}$')
        for i, (ax1, ax2, session) in enumerate(zip(axs.flat[:5], axs.flat[5:], sessions)):
            ax1.plot(times, sub_scores[i])
            ax1.axvspan(0, 0.2, color='grey', alpha=.2)
            ax1.set_title(session)
            ax1.axhline(chance, color='white', ls='dashed', alpha=.5)
            ax1.set_ylim(0, 1)
            ax1.grid(True, color='grey', alpha=0.3)
            times_win = np.where((times >= 0) & (times <= 0.2))[0]
            max_score = np.argmax(sub_scores[i][times_win]) + np.where(times==0)[0][0]
            ax1.annotate(f'Max Score: {sub_scores[i][max_score]:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')
            ax1.annotate('', xy=(times[max_score], sub_scores[i][max_score]), xytext=(times[max_score], sub_scores[i][max_score] + 0.1),
                        arrowprops=dict(arrowstyle='->', color='white'))
            disp = ConfusionMatrixDisplay(sub_cms[i, max_score, :, :], display_labels=[1, 2, 3, 4])
            disp.plot(ax=ax2)
            disp.im_.set_clim(0, 1)  # Set colorbar limits
        plt.savefig(res_dir / f"{solver}_{subject}.png")
        plt.close()