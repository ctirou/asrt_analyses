import os.path as op
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from mne.decoding import UnsupervisedSpatialFilter
import scipy.stats
import statsmodels.api as sm
from tqdm.auto import tqdm
from sklearn.covariance import LedoitWolf
from base import *
from config import RAW_DATA_DIR, DATA_DIR, RESULTS_DIR, FREESURFER_DIR, SUBJS, EPOCHS
from scipy.stats import pearsonr

lock = 'stim'
trial_type = 'pattern'
params = 'correlations'

data_path = DATA_DIR
res_path = RESULTS_DIR
subjects_dir = FREESURFER_DIR
subjects, epochs_list = SUBJS, EPOCHS

figures = res_path / 'figures' / lock / 'similarity' / params
ensure_dir(figures)

all_in_seqs, all_out_seqs = [], []

for subject in subjects:
    
    # all_in_seqs, all_out_seqs = [], []
    
    # Read the behav file to get the sequence 
    behav_dir = op.join(RAW_DATA_DIR, "%s/behav_data/" % (subject)) 
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
    # random.shuffle(sequence)
        
    # create lists of possible combinations between stimuli
    one_two_similarities = list()
    one_three_similarities = list()
    one_four_similarities = list() 
    two_three_similarities = list()
    two_four_similarities = list() 
    three_four_similarities = list()
    
    # loop across sessions
    for epoch_num, epo in enumerate(epochs_list):
        
        rdm_fname = res_path / 'rdms' / 'sensors' / subject / f'rdm_{epoch_num}.npy' # (4, 4, 263)                
        rdmx = np.load(rdm_fname)            
        
        one_two_similarity = list()
        one_three_similarity = list()
        one_four_similarity = list() 
        two_three_similarity = list()
        two_four_similarity = list()
        three_four_similarity = list()
                    
        for itime in range(rdmx.shape[2]):
            one_two_similarity.append(rdmx[0, 1, itime])
            one_three_similarity.append(rdmx[0, 2, itime])
            one_four_similarity.append(rdmx[0, 3, itime])
            two_three_similarity.append(rdmx[1, 2, itime])
            two_four_similarity.append(rdmx[1, 3, itime])
            three_four_similarity.append(rdmx[2, 3, itime])
                        
        one_two_similarity = np.array(one_two_similarity)
        one_three_similarity = np.array(one_three_similarity)
        one_four_similarity = np.array(one_four_similarity) 
        two_three_similarity = np.array(two_three_similarity)
        two_four_similarity = np.array(two_four_similarity) 
        three_four_similarity = np.array(three_four_similarity)

        one_two_similarities.append(one_two_similarity)
        one_three_similarities.append(one_three_similarity)
        one_four_similarities.append(one_four_similarity) 
        two_three_similarities.append(two_three_similarity)
        two_four_similarities.append(two_four_similarity) 
        three_four_similarities.append(three_four_similarity)
                            
    one_two_similarities = np.array(one_two_similarities)
    one_three_similarities = np.array(one_three_similarities)  
    one_four_similarities = np.array(one_four_similarities)   
    two_three_similarities = np.array(two_three_similarities)  
    two_four_similarities = np.array(two_four_similarities)   
    three_four_similarities = np.array(three_four_similarities)
    
    pairs_in_sequence = list()
    pairs_in_sequence.append(str(sequence[0]) + str(sequence[1]))
    pairs_in_sequence.append(str(sequence[1]) + str(sequence[2]))
    pairs_in_sequence.append(str(sequence[2]) + str(sequence[3]))
    pairs_in_sequence.append(str(sequence[3]) + str(sequence[0]))

    in_seq, out_seq = [], []
    similarities = [one_two_similarities, one_three_similarities, one_four_similarities,
                    two_three_similarities, two_four_similarities, three_four_similarities]
        
    pairs = ['12', '13', '14', '23', '24', '34']
    rev_pairs = ['21', '31', '41', '32', '42', '43']
                        
    for pair, rev_pair, similarity in zip(pairs, rev_pairs, similarities):
        if ((pair in pairs_in_sequence) or (rev_pair in pairs_in_sequence)):
            in_seq.append(similarity)
        else: 
            out_seq.append(similarity)
    all_in_seqs.append(np.array(in_seq))
    all_out_seqs.append(np.array(out_seq))

    # plot per subject
    
all_in_seqs = np.array(all_in_seqs)
all_out_seqs = np.array(all_out_seqs)

diff_inout = all_in_seqs.mean(axis=1) - all_out_seqs.mean(axis=1)

# diff_inout = np.reshape(diff_inout, (diff_inout.shape[1], diff_inout.shape[-1]))
data = diff_inout.copy().mean(0)

# correlations = {}
# for i in range(data.shape[0]):
#     for j in range(i+1, data.shape[0]):
#         corr, _ = pearsonr(data[i], data[j])
#         correlations[(i, j)] = corr
        
corr_matrix = np.corrcoef(data) # uses pearson's r

# Plot the correlation matrix
fig, ax = plt.subplots()
cax = ax.matshow(corr_matrix, cmap='coolwarm')
fig.colorbar(cax)

ax.set_xticks(np.arange(len(corr_matrix)))
ax.set_yticks(np.arange(len(corr_matrix)))

# Label the axes with the row numbers
ax.set_xticklabels(['P', 'B1', 'B2', 'B3', 'B4'])
ax.set_yticklabels(['P', 'B1', 'B2', 'B3', 'B4'])

# Add gridlines based on cell boundaries
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.set_xticks(np.arange(-.5, len(corr_matrix), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(corr_matrix), 1), minor=True)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                ha='center', va='center', color='w')
plt.title('mean')
plt.savefig(figures / 'mean.png')
# plt.show()
