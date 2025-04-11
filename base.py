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
        os.makedirs(path, exist_ok=True)
        
def decod_stats(X, jobs):
    """Statistical test applied across subjects"""
    # check input
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    X = X.astype(np.float64)
    
    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, out_type='mask', n_permutations=2**12, n_jobs=jobs,
        verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster] = pval

    return np.squeeze(p_values_)

def gat_stats(X, jobs):
    """Statistical test applied across subjects"""
    from mne.stats import spatio_temporal_cluster_1samp_test
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask',
        n_permutations=2**10, n_jobs=jobs, verbose=True)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T

def gat_t1samp(X):
    from scipy.stats import ttest_1samp
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    t_values = np.zeros_like(X[0])
    for itime in range(X.shape[1]):
        t_values[itime] = ttest_1samp(X[:, itime], 0)[0]
    return t_values

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

def get_random_low(behav_dir):
    behav_files = [f for f in os.listdir(behav_dir) if (not f.startswith('.') and ('_eASRT_Epoch_' in f))]
    behav = open(op.join(behav_dir, behav_files[0]), 'r')
    lines = behav.readlines()
    column_names = lines[0].split()
    rdm_low = list()
    for iline, line in enumerate(lines[1:]):
            triplet = int(line.split()[column_names.index('triplet')])
            if triplet == 34:
                first = line.split()[column_names.index('position')]
                second = lines[iline-2].split()[column_names.index('position')]
                pair = first + second
                if pair not in rdm_low:
                    # print(pair)
                    rdm_low.append(pair)
            # if len(rdm_low) == 3:
            #     break
    return rdm_low

def get_rdm(epoch, behav):
    from scipy.spatial.distance import pdist, squareform
    import scipy.stats
    import statsmodels.api as sm
    # from tqdm.auto import tqdm
    from sklearn.covariance import LedoitWolf
    import pandas as pd
    # Prepare the design matrix                        
    ntrials = len(epoch)
    nconditions = 4
    design_matrix = np.zeros((ntrials, nconditions))
    
    if type(behav) == pd.core.frame.DataFrame:
        y = behav["positions"]
    else:
        y = behav
    
    for icondi, condi in enumerate(y):            
        # assert isinstance(condi, np.int64) 
        design_matrix[icondi, condi-1] = 1
    assert np.sum(design_matrix.sum(axis=1) == 1) == len(epoch)  
    
    meg_data_V = epoch
    _, nchs, ntimes = meg_data_V.shape
    meg_data_V = scipy.stats.zscore(meg_data_V, axis=0)

    coefs = np.zeros((nconditions, nchs, ntimes))
    resids = np.zeros_like(meg_data_V)
    # for ich in tqdm(range(nchs)):
    for ich in range(nchs):
        for itime in range(ntimes):
            y = meg_data_V[:, ich, itime]
            
            model = sm.OLS(endog=y, exog=design_matrix, missing="raise")
            results = model.fit()
            
            coefs[:, ich, itime] = results.params # (4, 248, 163)
            resids[:, ich, itime] = results.resid # (ntrials, 248, 163)
    
    # Calculate pairwise mahalanobis distance between regression coefficients        
    rdm_times = np.zeros((nconditions, nconditions, ntimes))
    for itime in range(ntimes):
        response = coefs[:, :, itime] # (4, 248)
        residuals = resids[:, :, itime] # (51, 248)
        
        # Estimate covariance from residuals
        lw_shrinkage = LedoitWolf(assume_centered=True)
        cov = lw_shrinkage.fit(residuals)
        
        # Compute pairwise mahalanobis distances
        VI = np.linalg.inv(cov.covariance_) # inverse of covariance matrix needed for mahalonobis
        rdm = squareform(pdist(response, metric="mahalanobis", VI=VI))
        # rdm = squareform(pdist(response, metric="cosine"))
        assert ~np.isnan(rdm).any()
        rdm_times[:, :, itime] = rdm # rdm_times (4, 4, 163), rdm (4, 4)
    
    return rdm_times

    
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

def get_volume_estimate_time_course(stcs, fwd, subject, subjects_dir):
    """Extracts time courses for each label from volume source estimates.
    Args:
        stcs (list of mne.VolSourceEstimate): List of volume source estimates.
        fwd (dict): Forward solution.
        subject (str): Subject name.
        subjects_dir (str): Path to SUBJECTS_DIR.

    Returns:
        dict: A dictionary with label names as keys and arrays of shape
                (n_epochs, n_vertices_in_label, n_times) as values.    
    """
    import numpy as np
    from mne import get_volume_labels_from_src
    from tqdm.auto import tqdm
    
    labels = get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
    vertices_info = dict()
    for label in labels:
        vertices_info[label.name] = len(label.vertices)
    # Initialize a dictionary to hold time courses for each label
    label_time_courses = {}
    # Loop through each STC (source time course) for each epoch
    for stc in tqdm(stcs):
        # Extract data from the STC
        stc_data = stc.data  # shape: (n_vertices, n_times)
        # Loop through each label to extract the time course
        for ilabel, label in enumerate(labels):
            if ilabel >= len(stc.vertices):
                # If ilabel exceeds the number of vertex arrays, break the loop
                break
            # Get the vertices in the label
            label_vertices = np.intersect1d(stc.vertices[ilabel+2], label.vertices)
            if label_vertices.size == 0:
                continue
            # Get indices of these vertices in the STC data
            indices = np.searchsorted(stc.vertices[ilabel+2], label_vertices)
            # Extract the time courses for these vertices
            vertices_time_courses = stc_data[indices, :]
            # Store the time courses in the dictionary
            if label.name not in label_time_courses:
                label_time_courses[label.name] = []
            label_time_courses[label.name].append(vertices_time_courses)
    # Convert to numpy arrays
    for label in label_time_courses:
        label_time_courses[label] = np.array(label_time_courses[label])  # shape: (n_trials, n_vertices_in_label, n_times)
    return label_time_courses, vertices_info

def get_labels_from_vol_src(src, subject, subjects_dir):
    from mne import Label
    from mne import get_volume_labels_from_aseg
    """Return a list of Label of segmented volumes included in the src space.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space containing the volume regions.
    %(subject)s
    subjects_dir : str
        Freesurfer folder of the subjects.

    Returns
    -------
    labels_aseg : list of Label
        List of Label of segmented volumes included in src space.
    """
    # from ..label import Label

    # Read the aseg file
    aseg_fname = op.join(subjects_dir, subject, "mri", "aseg.mgz")
    all_labels_aseg = get_volume_labels_from_aseg(aseg_fname, return_colors=True)

    if any(np.any(s["type"] != "vol") for s in src):
        raise ValueError("source spaces have to be of vol type")

    labels_aseg = list()
    for nr in range(len(src)):
        vertices = src[nr]["vertno"]

        pos = src[nr]["rr"][src[nr]["vertno"], :]
        roi_str = src[nr]["seg_name"]
        try:
            ind = all_labels_aseg[0].index(roi_str)
            color = np.array(all_labels_aseg[1][ind]) / 255
        except ValueError:
            pass

        if "left" in roi_str.lower():
            hemi = "lh"
            roi_str = roi_str.replace("Left-", "") + "-lh"
        elif "right" in roi_str.lower():
            hemi = "rh"
            roi_str = roi_str.replace("Right-", "") + "-rh"
        else:
            hemi = "both"

        label = Label(
            vertices=vertices,
            pos=pos,
            hemi=hemi,
            name=roi_str,
            color=color,
            subject=subject,
        )
        labels_aseg.append(label)

    return labels_aseg

def get_volume_estimate_tc(stcs, fwd, offsets, subject, subjects_dir):
    import numpy as np
    from mne import get_volume_labels_from_src
    labels = get_volume_labels_from_src(fwd['src'], subject, subjects_dir)
    vertices_info = dict()
    for label in labels:
        vertices_info[label.name] = len(label.vertices)
    # Initialize a dictionary to hold time courses for each label
    label_time_courses = {}
    for stc in stcs:
        stc_data = stc.data  # shape: (n_vertices, n_times)
        for ilabel, label in enumerate(labels):
            tc = stc_data[offsets[ilabel]:offsets[ilabel+1]]
            if label.name not in label_time_courses:
                label_time_courses[label.name] = []
            label_time_courses[label.name].append(tc)
    # Convert to numpy arrays
    for label in label_time_courses:
        label_time_courses[label] = np.array(label_time_courses[label])  # shape: (n_trials, n_vertices_in_label, n_times)
    return label_time_courses, vertices_info

def rsync_files(source, destination, options=""):
    import subprocess
    try:
        # Construct the rsync command
        command = f"rsync {options} --progress --ignore-existing {source} {destination}"
        
        # Execute the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Decode and print the output and errors (if any)
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        print(stdout)
        if stderr:
            print(f"Errors during rsync: {stderr}")

        print("Rsync operation completed successfully.")
        return stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr.decode()}")
        return None
    
def get_in_out_seq(sequence, similarities, random_lows, analysis):
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
    if analysis == 'pat_high_rdm_high':
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                in_seq.append(pat_sim)
                out_seq.append(rand_sim)
    elif analysis == 'pat_high_rdm_low':
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                in_seq.append(pat_sim)
            else:
                out_seq.append(rand_sim)
    else:
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                in_seq.append(pat_sim)
            else:
                out_seq.append(pat_sim)
    return np.array(in_seq), np.array(out_seq)

def get_all_high_low(res_path, sequence, analysis, cv=False):
    import numpy as np
    # create lists of possible combinations between stimuli
    one_twos_pat = list()
    one_threes_pat = list()
    one_fours_pat = list() 
    two_threes_pat = list()
    two_fours_pat = list() 
    three_fours_pat = list()

    one_twos_rand = list()
    one_threes_rand = list()
    one_fours_rand = list()
    two_threes_rand = list()
    two_fours_rand = list() 
    three_fours_rand = list()
    
    # loop across sessions
    for epoch_num in [0, 1, 2, 3, 4]:
                
        rdm_rand = np.load(res_path / f"rand-{epoch_num}.npy")
        rdm_pat = np.load(res_path / f"pat-{epoch_num}.npy")
        
        one_two_pat = list()
        one_three_pat = list()
        one_four_pat = list() 
        two_three_pat = list()
        two_four_pat = list()
        three_four_pat = list()

        one_two_rand = list()
        one_three_rand = list()
        one_four_rand = list() 
        two_three_rand = list()
        two_four_rand = list()
        three_four_rand = list()

        if cv:
            for itime in range(rdm_pat.shape[0]):
                one_two_pat.append(rdm_pat[itime, 0, 1])
                one_three_pat.append(rdm_pat[itime, 0, 2])
                one_four_pat.append(rdm_pat[itime, 0, 3])
                two_three_pat.append(rdm_pat[itime, 1, 2])
                two_four_pat.append(rdm_pat[itime, 1, 3])
                three_four_pat.append(rdm_pat[itime, 2, 3])

                one_two_rand.append(rdm_rand[itime, 0, 1])
                one_three_rand.append(rdm_rand[itime, 0, 2])
                one_four_rand.append(rdm_rand[itime, 0, 3])
                two_three_rand.append(rdm_rand[itime, 1, 2])
                two_four_rand.append(rdm_rand[itime, 1, 3])
                three_four_rand.append(rdm_rand[itime, 2, 3])
        else:
            for itime in range(rdm_pat.shape[2]):
                one_two_pat.append(rdm_pat[0, 1, itime])
                one_three_pat.append(rdm_pat[0, 2, itime])
                one_four_pat.append(rdm_pat[0, 3, itime])
                two_three_pat.append(rdm_pat[1, 2, itime])
                two_four_pat.append(rdm_pat[1, 3, itime])
                three_four_pat.append(rdm_pat[2, 3, itime])

                one_two_rand.append(rdm_rand[0, 1, itime])
                one_three_rand.append(rdm_rand[0, 2, itime])
                one_four_rand.append(rdm_rand[0, 3, itime])
                two_three_rand.append(rdm_rand[1, 2, itime])
                two_four_rand.append(rdm_rand[1, 3, itime])
                three_four_rand.append(rdm_rand[2, 3, itime])

        one_two_pat = np.array(one_two_pat)
        one_three_pat = np.array(one_three_pat)
        one_four_pat = np.array(one_four_pat) 
        two_three_pat = np.array(two_three_pat)
        two_four_pat = np.array(two_four_pat) 
        three_four_pat = np.array(three_four_pat)

        one_two_rand = np.array(one_two_rand)
        one_three_rand = np.array(one_three_rand)
        one_four_rand = np.array(one_four_rand) 
        two_three_rand = np.array(two_three_rand)
        two_four_rand = np.array(two_four_rand) 
        three_four_rand = np.array(three_four_rand)

        one_twos_pat.append(one_two_pat)
        one_threes_pat.append(one_three_pat)
        one_fours_pat.append(one_four_pat) 
        two_threes_pat.append(two_three_pat)
        two_fours_pat.append(two_four_pat) 
        three_fours_pat.append(three_four_pat)

        one_twos_rand.append(one_two_rand)
        one_threes_rand.append(one_three_rand)
        one_fours_rand.append(one_four_rand) 
        two_threes_rand.append(two_three_rand)
        two_fours_rand.append(two_four_rand) 
        three_fours_rand.append(three_four_rand)
                            
    one_twos_pat = np.array(one_twos_pat)
    one_threes_pat = np.array(one_threes_pat)  
    one_fours_pat = np.array(one_fours_pat)   
    two_threes_pat = np.array(two_threes_pat)  
    two_fours_pat = np.array(two_fours_pat)
    three_fours_pat = np.array(three_fours_pat)

    one_twos_rand = np.array(one_twos_rand)
    one_threes_rand = np.array(one_threes_rand)  
    one_fours_rand = np.array(one_fours_rand)   
    two_threes_rand = np.array(two_threes_rand)  
    two_fours_rand = np.array(two_fours_rand)   
    three_fours_rand = np.array(three_fours_rand)
    
    similarities = [one_twos_pat, one_threes_pat, one_fours_pat,
                    two_threes_pat, two_fours_pat, three_fours_pat]
    random_lows = [one_twos_rand, one_threes_rand, one_fours_rand,
                    two_threes_rand, two_fours_rand, three_fours_rand]
        
    # create list of possible pairs
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
    high, low = [], []
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
    if analysis == 'pat_high_rdm_high':
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                high.append(pat_sim)
                low.append(rand_sim)
    elif analysis == 'pat_high_rdm_low':
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                high.append(pat_sim)
            else:
                low.append(rand_sim)
    elif analysis == 'rdm_high_rdm_low':
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                low.append(pat_sim)
            else:
                high.append(rand_sim)
    else:
        for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, similarities, random_lows):
            if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
                high.append(pat_sim)
            else:
                low.append(pat_sim)
    return np.array(high), np.array(low)

def get_all_high_low_blocks(res_path, sequence):
    import numpy as np
    # create list of possible pairs
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
    
    # create lists of possible combinations between stimuli
    one_twos_pat = list()
    one_threes_pat = list()
    one_fours_pat = list() 
    two_threes_pat = list()
    two_fours_pat = list() 
    three_fours_pat = list()

    one_twos_rand = list()
    one_threes_rand = list()
    one_fours_rand = list()
    two_threes_rand = list()
    two_fours_rand = list() 
    three_fours_rand = list()
    
    blocks = np.arange(23)
    
    rdm_pat = [np.load(res_path / f"pat-{epoch_num}.npy") for epoch_num in [0, 1, 2, 3, 4]]
    rdm_pat = np.vstack(rdm_pat)
    
    rdm_rand = [np.load(res_path / f"rand-{epoch_num}.npy") for epoch_num in [0, 1, 2, 3, 4]]
    rdm_rand = np.vstack(rdm_rand)
    
    for block in blocks:
                    
        one_two_pat = list()
        one_three_pat = list()
        one_four_pat = list() 
        two_three_pat = list()
        two_four_pat = list()
        three_four_pat = list()

        one_two_rand = list()
        one_three_rand = list()
        one_four_rand = list() 
        two_three_rand = list()
        two_four_rand = list()
        three_four_rand = list()

        for itime in range(rdm_pat.shape[1]):
            one_two_pat.append(rdm_pat[block, itime, 0, 1])
            one_three_pat.append(rdm_pat[block, itime, 0, 2])
            one_four_pat.append(rdm_pat[block, itime, 0, 3])
            two_three_pat.append(rdm_pat[block, itime, 1, 2])
            two_four_pat.append(rdm_pat[block, itime, 1, 3])
            three_four_pat.append(rdm_pat[block, itime, 2, 3])

            one_two_rand.append(rdm_rand[block, itime, 0, 1])
            one_three_rand.append(rdm_rand[block, itime, 0, 2])
            one_four_rand.append(rdm_rand[block, itime, 0, 3])
            two_three_rand.append(rdm_rand[block, itime, 1, 2])
            two_four_rand.append(rdm_rand[block, itime, 1, 3])
            three_four_rand.append(rdm_rand[block, itime, 2, 3])

        one_two_pat = np.array(one_two_pat)
        one_three_pat = np.array(one_three_pat)
        one_four_pat = np.array(one_four_pat) 
        two_three_pat = np.array(two_three_pat)
        two_four_pat = np.array(two_four_pat) 
        three_four_pat = np.array(three_four_pat)

        one_two_rand = np.array(one_two_rand)
        one_three_rand = np.array(one_three_rand)
        one_four_rand = np.array(one_four_rand) 
        two_three_rand = np.array(two_three_rand)
        two_four_rand = np.array(two_four_rand) 
        three_four_rand = np.array(three_four_rand)

        one_twos_pat.append(one_two_pat)
        one_threes_pat.append(one_three_pat)
        one_fours_pat.append(one_four_pat) 
        two_threes_pat.append(two_three_pat)
        two_fours_pat.append(two_four_pat) 
        three_fours_pat.append(three_four_pat)

        one_twos_rand.append(one_two_rand)
        one_threes_rand.append(one_three_rand)
        one_fours_rand.append(one_four_rand) 
        two_threes_rand.append(two_three_rand)
        two_fours_rand.append(two_four_rand) 
        three_fours_rand.append(three_four_rand)
                        
    one_twos_pat = np.array(one_twos_pat)
    one_threes_pat = np.array(one_threes_pat)  
    one_fours_pat = np.array(one_fours_pat)   
    two_threes_pat = np.array(two_threes_pat)  
    two_fours_pat = np.array(two_fours_pat)
    three_fours_pat = np.array(three_fours_pat)

    one_twos_rand = np.array(one_twos_rand)
    one_threes_rand = np.array(one_threes_rand)  
    one_fours_rand = np.array(one_fours_rand)   
    two_threes_rand = np.array(two_threes_rand)  
    two_fours_rand = np.array(two_fours_rand)   
    three_fours_rand = np.array(three_fours_rand)

    patterns = [one_twos_pat, one_threes_pat, one_fours_pat,
                    two_threes_pat, two_fours_pat, three_fours_pat]
    randoms = [one_twos_rand, one_threes_rand, one_fours_rand,
                    two_threes_rand, two_fours_rand, three_fours_rand]
    
    pat_blocks, rand_blocks = list(), list()
    for pair, rev_pair, pat_sim, rand_sim in zip(pairs, rev_pairs, patterns, randoms):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):                    
            pat_blocks.append(pat_sim)
            rand_blocks.append(rand_sim)            

    return np.array(pat_blocks), np.array(rand_blocks)


def get_cm(clf, cv, X, y, times):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    pred = np.zeros((len(y), X.shape[-1]))
    pred_rock = np.zeros((len(y), X.shape[-1], len(set(y))))
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        pred[test] = np.array(clf.predict(X[test]))
        pred_rock[test] = np.array(clf.predict_proba(X[test]))
                    
    cms, scores = list(), list()
    for itime in range(len(times)):
        cms.append(confusion_matrix(y[:], pred[:, itime], normalize='true', labels=[1, 2, 3, 4]))
        scores.append(roc_auc_score(y[:], pred_rock[:, itime, :], multi_class='ovr'))
    
    return np.array(cms), np.array(scores)


def cv_mahalanobis(X, y, n_splits=10):
    """
    Compute cross-validated Mahalanobis distances between conditions for each time point.

    Parameters:
        X: ndarray of shape (n_trials, n_channels, n_times)
        Multivariate time-series data.
        y: ndarray of shape (n_trials,)
        Condition labels for each trial.
        n_splits: int
        Number of cross-validation folds.

    Returns:
        distances: ndarray of shape (n_times, n_conditions, n_conditions)
        Cross-validated Mahalanobis distances between conditions at each time point.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import inv
    from scipy.spatial.distance import pdist, squareform
    from tqdm.auto import tqdm
    
    n_trials, n_channels, n_times = X.shape
    conditions = np.unique(y)
    n_conditions = len(conditions)

    # Initialize array to store distances
    distances = np.zeros((n_times, n_conditions, n_conditions))

    # Cross-validation with stratified splitting
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for time_idx in tqdm(range(n_times)):
        X_time = X[:, :, time_idx]  # Data at this time point
        cv_distances = np.zeros((n_conditions, n_conditions, n_splits))

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Split data into training and testing
            X_train, X_test = X_time[train_idx], X_time[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Compute the mean and covariance for each condition in training data
            means = {cond: X_train[y_train == cond].mean(axis=0) for cond in conditions}
            residuals = X_train - np.array([means[cond] for cond in y_train])

            # Regularized covariance estimation
            lw = LedoitWolf(assume_centered=True)
            cov = lw.fit(residuals)
            cov_inv = inv(cov.covariance_)

            # Compute Mahalanobis distances for each pair of conditions
            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions):
                    diff_mean = means[cond1] - means[cond2]
                    cv_distances[i, j, fold_idx] = np.sqrt(diff_mean.T @ cov_inv @ diff_mean)

        # Average distances across folds
        distances[time_idx] = cv_distances.mean(axis=2)

    return distances


def loocv_mahalanobis(X, y):
    """
    Compute LOOCV Mahalanobis distances between conditions for each time point.

    Parameters:
        X: ndarray of shape (n_trials, n_channels, n_times)
           Multivariate time-series data.
        y: ndarray of shape (n_trials,)
           Condition labels for each trial.

    Returns:
        distances: ndarray of shape (n_times, n_conditions, n_conditions)
           LOOCV Mahalanobis distances between conditions at each time point.
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import LeaveOneOut
    from scipy.linalg import inv
    from tqdm.auto import tqdm
    
    n_trials, n_channels, n_times = X.shape
    conditions = np.unique(y)
    n_conditions = len(conditions)

    # Initialize array to store distances
    distances = np.zeros((n_times, n_conditions, n_conditions))
    loo = LeaveOneOut()

    for time_idx in tqdm(range(n_times)):
        X_time = X[:, :, time_idx]  # Data at this time point
        cv_distances = np.zeros((n_conditions, n_conditions, n_trials))

        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X, y)):
            X_train, X_test = X_time[train_idx], X_time[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Compute the mean and covariance for each condition in training data
            means = {cond: X_train[y_train == cond].mean(axis=0) for cond in conditions}
            residuals = X_train - np.array([means[cond] for cond in y_train])

            # Regularized covariance estimation
            lw = LedoitWolf(assume_centered=True)
            cov = lw.fit(residuals)
            cov_inv = inv(cov.covariance_)

            # Compute Mahalanobis distances for each pair of conditions
            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions):
                    diff_mean = means[cond1] - means[cond2]
                    cv_distances[i, j, fold_idx] = np.sqrt(diff_mean.T @ cov_inv @ diff_mean)

        # Average distances across LOOCV iterations
        distances[time_idx] = cv_distances.mean(axis=2)

    return distances

def interpolate_rdm_nan(rdm):
    """Interpolate nan values in a RDM matrix by computing the mean of previous and subsequent block.

    Args:
        rdm: ndarray of shape (n_blocks, n_times, n_conditions, n_conditions)
        LOOCV Mahalanobis distances between conditions at each time point.
        
    Returns:
        rdm: ndarray of shape (n_blocks, n_times, n_conditions, n_conditions)
        LOOCV Mahalanobis distances between conditions at each time point.
        present_nan: bool
        True if nan values were present in the RDM, False otherwise
    """
    present_nan = False
    n_blocks, n_times, n_conditions, _ = rdm.shape
    for i in range(n_blocks):
        for j in range(n_times):
            if np.isnan(np.sum(rdm[i, j])):
                present_nan = True
                if i == 0:
                    rdm[i, j] = rdm[i + 1, j]
                elif i == n_blocks - 1:
                    rdm[i, j] = rdm[i - 1, j]
                else:
                    rdm[i, j] = (rdm[i - 1, j] + rdm[i + 1, j]) / 2
    return rdm, present_nan


def remove_common_vertex(base_label, target_label):
    """Remove common vertices between two labels"""
    import numpy as np
    from mne import Label
    
    base_vertices = base_label.vertices
    target_vertices = target_label.vertices
    
    intersection = set(base_vertices).intersection(target_vertices)
    
    corrected_vertices = np.setdiff1d(target_vertices, list(intersection))
    intersection_indices = [np.where(target_vertices == v)[0][0] for v in intersection]
    corrected_pos = np.delete(target_label.pos, intersection_indices, axis=0)
    corrected_label = Label(vertices=corrected_vertices, 
                  pos=corrected_pos, 
                  hemi=target_label.hemi, 
                  name=target_label.name, 
                  color=target_label.color, 
                  subject=target_label.subject)
    
    return corrected_label


def contiguous_regions(condition):
    import numpy as np
    """Find contiguous True regions in a boolean array."""
    d = np.diff(condition.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    if condition[0]:
        starts = np.r_[0, starts]
    if condition[-1]:
        ends = np.r_[ends, condition.size]

    return zip(starts, ends)

def svd(vector_data):
    print("Singular Value Decomposition in progress...")
    # Initialize an array for storing the dominant orientation time series
    dominant_data = np.zeros((vector_data.shape[0], vector_data.shape[1], vector_data.shape[-1]))  # (294, 8196, 82)
    for trial in range(vector_data.shape[0]):  # Loop over trials
        for source in range(vector_data.shape[1]):  # Loop over sources
            u, s, vh = np.linalg.svd(vector_data[trial, source, :, :], full_matrices=False)  # SVD over orientation axis (3)
            dominant_time_series = vh[0, :] * s[0]  # First right singular vector weighted by singular value
            dominant_data[trial, source, :] = dominant_time_series  # Store in new array
    return dominant_data

def check_stationarity(series):
    """
    Check if the series is stationary using the Augmented Dickey-Fuller test.
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    new_series = np.zeros((series.shape[0], series.shape[1] - 1))
    sig = []
    for sub in range(series.shape[0]):
        result = adfuller(series[sub])
        if result[1] < 0.05:
            sig.append(True)
            new_series[sub, :] = series[sub, 1:]
        else:
            sig.append(False)
            new_series[sub, :] = np.diff(series[sub, :])
    return sig, new_series

def ensure_stationarity(series, max_diff=2):
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    """
    Ensures that each subject's time series is stationary by applying up to max_diff levels of differencing.
    
    Args:
        series (numpy array): Shape (n_subjects, n_timepoints).
        max_diff (int): Maximum differencing order to apply (default = 2).
    
    Returns:
        stationarity_flags (list): List of booleans indicating if each subject's series is stationary.
        stationary_series (numpy array): Differenced series with max available timepoints.
        applied_diffs (list): List of how many differences were applied per subject.
    """
    n_subjects, n_timepoints = series.shape
    max_length = n_timepoints - 1  # Start with first-differencing max length
    stationary_series = []

    stationarity_flags = []
    applied_diffs = []

    for sub in range(n_subjects):
        diff_count = 0
        current_series = series[sub, :]

        # Check stationarity and apply differencing if needed
        while diff_count < max_diff:
            adf_p = adfuller(current_series)[1]
            if adf_p < 0.05:
                stationarity_flags.append(True)
                break  # Stop differencing if stationary
            else:
                current_series = np.diff(current_series)
                diff_count += 1

        # If still non-stationary after max_diff differencing
        if adfuller(current_series)[1] >= 0.05:
            stationarity_flags.append(False)
            print(f"⚠️ Warning: Subject {sub} still non-stationary after {max_diff} differences.")

        # Store the differenced data and track differencing steps
        applied_diffs.append(diff_count)
        max_length = min(max_length, len(current_series))  # Ensure equal timepoints
        stationary_series.append(current_series)

    # Convert list to array with equal-length time series
    stationary_series = np.array([s[:max_length] for s in stationary_series])

    return stationarity_flags, stationary_series, applied_diffs

def optimal_lag_mi(X_sub, Y_sub, max_lag=10):
    from sklearn.feature_selection import mutual_info_regression
    """
    Find optimal lag for Transfer Entropy using Mutual Information for one subject.
    
    Args:
        X_sub (array): Time series for X (shape: n_time_points).
        Y_sub (array): Time series for Y (shape: n_time_points).
        max_lag (int): Maximum lag to test.

    Returns:
        best_lag (int): Optimal lag with highest Mutual Information.
    """
    mi_scores = []
    for lag in range(1, max_lag + 1):
        mi = mutual_info_regression(X_sub[:-lag].reshape(-1, 1), Y_sub[lag:].reshape(-1, 1), random_state=42)
        # mi = mutual_info_regression(X_sub[:-lag].reshape(-1, 1), Y_sub[lag:].reshape(-1, 1))
        mi_scores.append(mi[0])  # Store MI score for this lag

    best_lag = np.argmax(mi_scores) + 1  # Best lag is the one with highest MI
    return best_lag, mi_scores

# ---- Run for all subjects ----
def find_optimal_lags(X, Y, max_lag=10):
    """
    Compute optimal lag for TE per subject.
    
    Args:
        X (array): Shape (n_subjects, n_time_points).
        Y (array): Shape (n_subjects, n_time_points).
        max_lag (int): Maximum lag to test.

    Returns:
        optimal_lags (list): Optimal lag per subject.
        global_lag (int): Median of all subject lags.
    """
    optimal_lags = []
    for sub in range(X.shape[0]):  # Iterate over subjects
        best_lag, _ = optimal_lag_mi(X[sub], Y[sub], max_lag)
        optimal_lags.append(best_lag)
    
    global_lag = int(np.median(optimal_lags))  # Take median for a global choice
    return optimal_lags, global_lag
