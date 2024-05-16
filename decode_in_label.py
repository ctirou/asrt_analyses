import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, classification_report
from base import *
from config import *
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from collections import defaultdict
from scipy.stats import ttest_1samp, spearmanr
import matplotlib.pyplot as plt
import gc
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from tensorflow.keras.utils import to_categorical, set_random_seed
# from jax import jit, grad, vmap, device_put, random
# import jax.numpy as jnp
# from jax.lib import xla_bridge
# from functools import partial
import time

from joblib import parallel_backend
from ray.util.joblib import register_ray
register_ray()

## Check if tensorflow finds GPU
# import tensorflow as tf
# tf.config.list_physical_devices('GPU') 

# params
trial_types = ["all", "pattern", "random"]
trial_type = 'pattern'
data_path = DATA_DIR
lock = "stim"
subjects = SUBJS
sessions = ['practice', 'b1', 'b2', 'b3', 'b4']
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
folds = 5
chance = 0.5
threshold = 0.05
# scoring = "accuracy"
scoring = "roc_auc"
parc='aparc'
hemi = 'both'
params = "pred_decoding"
verbose = True
jobs = -1

method = "classic" # "classic", "jax", or "nn"
decim = True

plt.style.use('dark_background')

# figures dir
figures = RESULTS_DIR / 'figures' / lock / params / 'source' / trial_type
ensure_dir(figures)
# get times
epoch_fname = DATA_DIR / lock / 'sub01_0_s-epo.fif'
epochs = mne.read_epochs(epoch_fname, verbose=verbose)
times = epochs.times
if decim:
    times = times[::3]
del epochs

ensure_dir(figures / 'corr')

combinations = ['one_two', 'one_three', 'one_four', 'two_three', 'two_four', 'three_four']

decod_in_lab = dict()

gc.collect()

# for ilabel in range(68): # put in subjects and divide
        
#     sims_in_label, decod_in_subs = [], []
    
for subject in subjects:
    
    sims_in_sub, decod_in_sess = [], []
    
    # get labels
    labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
    # label = labels[ilabel]
    
    all_session_cms, all_session_scores = [], []
            
    for session_id, session in enumerate(sessions):
        
        # read stim epoch
        epoch_fname = data_path / lock / f"{subject}_{session_id}_s-epo.fif"
        epoch = mne.read_epochs(epoch_fname, preload=True, verbose=True)
        
        # read behav
        behav_fname = data_path / "behav" / f"{subject}_{session_id}.pkl"
        behav = pd.read_pickle(behav_fname).reset_index()
                    
        # get session behav and epoch
        if session_id == 0:
            session = 'prac'
        else:
            session = 'sess-%s' % (str(session_id).zfill(2))

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
        
        del epoch, fwd, data_cov, noise_cov, rank, filters
        gc.collect()
        
        for ilabel, label in enumerate(labels): 
            print(f"{ilabel+1}/{len(labels)}", subject, session, label.name)
            
            sims_in_label, decod_in_subs = [], []            
            
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
            
            if decim:                 
                X = X[:, :, ::3]
            
            print("X shape :", X.shape)
            
            if method == "classic":
                with parallel_backend("ray"):
                    test, pred, pred_rock = make_predictions(X, y, folds, jobs, scoring, verbose) 
            
            elif method == "nn":
                y -= 1
                X = np.swapaxes(X, 1, 2)
                
                X_stand = np.zeros_like(X)
                for i in range(0, X.shape[0]):
                    X_stand[i,] = (X[i,] - X[i,].mean()) / X[i,].std()

                epochs = 25
                normalizer = keras.layers.Normalization(axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                normalizer.adapt(X_train)
                
                model = keras.models.Sequential(
                    [
                        keras.Input(shape=(X_train.shape[1:])),
                        # normalizer,
                        # A optimiser
                        layers.Flatten(),
                        layers.Dense(10, activation="relu"), 
                        layers.Dense(4, activation="softmax"),
                    ]
                )
                
                model.compile(optimizer=keras.optimizers.Adam(0.01),
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])
                
                history = model.fit(X_train, y_train, 
                                    epochs=epochs, 
                                    validation_split=0.1, 
                                    batch_size=X_train.shape[0])

                model.summary()

                import matplotlib.pyplot as plt
                plt.plot(history.history['val_loss'], label='val_loss')
                plt.plot(history.history['loss'], label='loss')
                plt.legend()
                plt.show()

                y_pred = model.predict(X_test)
                results = model.evaluate(X_test, y_test, verbose=1)

                # save best model over epochs: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

            elif method == "jax":
                y = y.to_numpy() - 1
                scaler = StandardScaler()
                X = X.reshape(X.shape[0], -1)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                
                print(xla_bridge.get_backend().platform) # Confirm GPU in use
                
                X_train_dp = device_put(X_train)
                y_train_dp = device_put(y_train.to_numpy())

                batches = [2**i for i in range(7, 14)] # Batch sizes are powers of two

                for x in batches:
                    lg_sgd_jax = JaxReg(learning_rate=1e-6, num_epochs = 20, size_batch = x)
                    lg_sgd_jax.fit(X_train_dp, y_train_dp)
            
            cms, scores, acc, bacc = list(), list(), list(), list()
            for t in range(X.shape[-1]):
                scores.append(roc_auc_score(y[:], pred_rock[:, t, :], multi_class='ovr'))
                cms.append(confusion_matrix(y[:], pred[:, t], normalize='true', labels=[1, 2, 3, 4]))
                acc.append(accuracy_score(y[:], pred[:, t]))
                bacc.append(balanced_accuracy_score(y[:], pred[:, t]))
                
            results = classification_report(y[:], pred[:, 0])
            print(results)
                
            scores = np.array(scores)
            acc = np.array(acc)
            bacc = np.array(bacc)
            
            all_session_scores.append(scores)
            # np.save(figures / f'{label.name}_{subject}_{session_id}-scores.npy', scores)

            c = np.array(cms) # don't tramspose, take peak decoding performance
            all_session_cms.append(c)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout='tight')
            fig.suptitle(f'{subject} / {session} / ${label.name}$')
            ax1.plot(times, scores, label='roc_auc')
            # ax1.plot(times, acc, label='acc')
            # ax1.plot(times, bacc, label='bacc')

            ax1.legend()
            
            ax1.set_ylabel('roc-auc')
            ax1.axvspan(0, 0.2, color='grey', alpha=.3)
            ax1.axhline(chance, color='white', ls='dashed', alpha=.5)
            max_score = max(scores)
            ax1.annotate(f'Max Score: {max_score:.2f}', xy=(0.1, 0.9), xycoords='axes fraction')                        
            ax1.set_ylim(0, 1)
            ax1.grid(True, color='grey', alpha=0.5)            
            # disp = ConfusionMatrixDisplay(c.mean(0), display_labels=set(y))
            disp = ConfusionMatrixDisplay(c[np.argmax(scores)], display_labels=set(y))
            disp.plot(ax=ax2)
            disp.im_.set_clim(0, 1)  # Set colorbar limits
            plt.show()
            # fig.savefig(figures / f"{label.name}_{subject}_{session}.png")
            # plt.close()
                
            one_two_similarity = list()
            one_three_similarity = list()
            one_four_similarity = list() 
            two_three_similarity = list()
            two_four_similarity = list()
            three_four_similarity = list()
                        
            for itime in range(len(times)):
                one_two_similarity.append(c[itime, 0, 1])
                one_three_similarity.append(c[itime, 0, 2])
                one_four_similarity.append(c[itime, 0, 3])
                two_three_similarity.append(c[itime, 1, 2])
                two_four_similarity.append(c[itime, 1, 3])
                three_four_similarity.append(c[itime, 2, 3])
                                
            similarities = [one_two_similarity, one_three_similarity, one_four_similarity, 
                            two_three_similarity, two_four_similarity, three_four_similarity]
            
            similarities = np.array(similarities)
            sims_in_sub.append(similarities)
            
            # np.save(figures / f'{label.name}_{subject}_{session_id}-rsa.npy', similarities)
            
            del stcs_data, X, y, cms, scores, c 
            del one_two_similarity, one_three_similarity, one_four_similarity, two_three_similarity, two_four_similarity, three_four_similarity
            del similarities
            gc.collect()
                
    all_session_cms = np.array(all_session_cms)
    all_session_scores = np.array(all_session_scores)
    
    sims_in_sub = np.array(sims_in_sub)
    sims_in_label.append(np.array(sims_in_sub))
    
    decod_in_sess = np.array(decod_in_sess)
    
    best_time = [i for i, j in enumerate(times) if 0 <= j <= 0.2]
    
    fig, axs = plt.subplots(2, 5, layout='tight', figsize=(23, 7))
    fig.suptitle(f'${label.name}$ // {subject}')
    for i, (ax, session) in enumerate(zip(axs.flat, sessions)):
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

    del labels
    gc.collect() # only works if vars are deleted