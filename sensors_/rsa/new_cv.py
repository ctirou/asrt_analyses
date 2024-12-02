import os.path as op
import os
import numpy as np
import mne
import pandas as pd
from base import *
from config import *
import sys
from sklearn.model_selection import KFold
from cv import ShuffleBin
from dissimilarity import *
import scipy
from sklearn.discriminant_analysis import _cov

overwrite = True
verbose = 'error'

data_path = DATA_DIR
subjects = SUBJS
lock = 'stim'
jobs = -1
subject = 'sub01'

res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "k10_rdm" / subject
ensure_dir(res_path)

# loop across sessions
# for epoch_num in [0, 1, 2, 3, 4]:
epoch_num = 1
    
print(f"Processing {subject} - {lock} - {epoch_num} - K10")
            
behav_fname = op.join(data_path, "behav/%s-%s.pkl" % (subject, epoch_num))
behav = pd.read_pickle(behav_fname)
# read epochs
epoch_fname = op.join(data_path, "%s/%s-%s-epo.fif" % (lock, subject, epoch_num))
epoch = mne.read_epochs(epoch_fname, verbose=verbose)
data = epoch.get_data(picks='mag', copy=True)

X_pat = data[np.where(behav["trialtypes"]==1)]
y_pat = behav[behav["trialtypes"]==1].reset_index(drop=True).positions

X_rand = data[np.where(behav["trialtypes"]==2)]
y_rand = behav[behav["trialtypes"]==2].reset_index(drop=True).positions

# cv = KFold(n_splits=5, shuffle=True, random_state=42)

ec = CvEuclidean2()
n_sensors = X_pat.shape[1]

n_perm = 10
n_pseudo = 5
n_conditions = 2
n_time = 82
s = epoch_num

# X, y = X_pat.copy(), y_pat.copy()
X, y = X_pat.copy(), y_pat.copy()

cv = ShuffleBin(y, n_iter=n_perm, n_pseudo=n_pseudo)

for f, (train_indices, test_indices) in enumerate(cv.split(X)):
    print('\tPermutation %g / %g' % (f + 1, n_perm))

    # 1. Compute pseudo-trials for training and test
    Xpseudo_train = np.full((len(train_indices), n_sensors, n_time), np.nan)
    Xpseudo_test = np.full((len(test_indices), n_sensors, n_time), np.nan)
    for i, ind in enumerate(train_indices):
        Xpseudo_train[i, :, :] = np.mean(X[ind, :, :], axis=0)
    for i, ind in enumerate(test_indices):
        Xpseudo_test[i, :, :] = np.mean(X[ind, :, :], axis=0)


    # 2. Whitening using the Epoch method
    sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
    sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
    for c in range(n_conditions):
        # compute sigma for each time point, then average across time
        sigma_[c] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, :, t], shrinkage='auto')
                                for t in range(n_time)], axis=0)
    sigma = sigma_.mean(axis=0)  # average across conditions
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
    Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
    Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

    for t in range(n_time):
        for c1 in range(n_conditions-1):
            for c2 in range(min(c1 + 1, n_conditions-1), n_conditions):
                    # 3. Apply distance measure to training data
                    data_train = Xpseudo_train[cv.ind_pseudo_train[c1, c2], :, t]
                    ec_cv.fit(data_train, cv.labels_pseudo_train[c1, c2])                            
                    ps_cv.fit(data_train, cv.labels_pseudo_train[c1, c2])

                    # 4. Validate distance measure on testing data
                    data_test = Xpseudo_test[cv.ind_pseudo_test[c1, c2], :, t]
                    result_cv['ec_cv'][s, f, c1, c2, t] = ec_cv.predict(data_test, y=cv.labels_pseudo_test[c1, c2])
                    result_cv['ps_cv'][s, f, c1, c2, t] = ps_cv.predict(data_test, y=cv.labels_pseudo_test[c1, c2])
