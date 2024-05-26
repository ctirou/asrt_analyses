import os
import os.path as op
import numpy as np
from mne.stats import permutation_cluster_1samp_test
import time
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def decod_stats(X):
    """Statistical test applied across subjects"""
    # check input
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=6,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)

def do_pca(epochs):
    import mne
    from mne.decoding import UnsupervisedSpatialFilter
    from sklearn.decomposition import PCA
    n_component = 30
    pca = UnsupervisedSpatialFilter(PCA(n_component), average=False)
    pca_data = pca.fit_transform(epochs.get_data())
    sampling_freq = epochs.info['sfreq']
    info = mne.create_info(n_component, ch_types='mag', sfreq=sampling_freq)
    all_epochs = mne.EpochsArray(pca_data, info = info, events=epochs.events, event_id=epochs.event_id)
    return all_epochs

def get_sequence(behav_dir):
    behav_files = [f for f in os.listdir(behav_dir) if (not f.startswith('.') and ('_eASRT_Epoch_' in f))]
    behav = open(op.join(behav_dir, behav_files[0]), 'r')
    lines = behav.readlines()
    column_names = lines[0].split()
    sequence = list()
    for line in lines[1:]:
            trialtype = int(line.split()[column_names.index('trialtype')])
            if trialtype == 1:
                sequence.append(int(line.split()[column_names.index('position')]))
            if len(sequence) == 4:
                break
    return sequence

def get_inout_seq(sequence, similarities):
    import numpy as np
    # create list of possible pairs
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
    in_seq, out_seq = [], []
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
    # look which are in, which are out
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    return np.array(in_seq), np.array(out_seq)


def get_best_pairs(sequence, similarities):
    # create list of possible pairs
    pairs_in_sequence, pairs_out_sequence = [], []
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
    
    pairs_out_sequence.append(str(sequence[0]) + str(sequence[2]))
    pairs_out_sequence.append(str(sequence[1]) + str(sequence[3]))

    in_seq, out_seq = [], []
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
    
    index = [0, 2]
    best_pairs = [val for idx, val in enumerate(pairs_in_sequence) if idx in index]
    
    # look which are in, which are out
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in best_pairs) or (rev_pair in best_pairs)):
            in_seq.append(similarity)
        elif ((pair in pairs_out_sequence) or (rev_pair in pairs_out_sequence)):
            out_seq.append(similarity)
    return np.array(in_seq), np.array(out_seq)
    
def get_inseq(sequence):
    # create list of possible pairs
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
    
    return pairs_in_sequence

def print_proportions(subject, all_beh):
    #### get stimuli proportions
    print(f"###############    {subject}")
    for i, sess in zip(range(5), ['prac', 'b1', 'b2', 'b3', 'b4']):
        print(f"{sess}    ----------------------")
        unique, values = np.unique(all_beh[i].positions, return_counts=True)
        for un, val in zip(unique, values):
            print(un, round((val/np.sum(values)*100), 2)) 

def make_predictions(X, y, folds, jobs, scoring, verbose):    
    # set-up the classifier and cv structure
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(multi_class="ovr", max_iter=100000, solver='saga', random_state=42)) # use JAX maybe
    # clf = make_pipeline(StandardScaler(), SGDRegressor(loss="squared_error", max_iter=100000, random_state=42)) # use JAX maybe
    clf = SlidingEstimator(clf, scoring=scoring, n_jobs=jobs, verbose=verbose) # get time of one sample (slide), try with less jobs maybe ?
    cv = StratifiedKFold(folds, shuffle=True)   

    pred = np.zeros((len(y), X.shape[-1]))
    pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
    # there is only randoms in practice sessions
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        pred[test] = np.array(clf.predict(X[test]))
        pred_rock[test] = np.array(clf.predict_proba(X[test]))

    return test, pred, pred_rock

def gat_stats(X):
    from mne.stats import spatio_temporal_cluster_1samp_test
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask',
        n_permutations=2**12, n_jobs=-1, verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T