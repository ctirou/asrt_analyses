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