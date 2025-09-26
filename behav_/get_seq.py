import os.path as op
import pandas as pd
from tqdm.auto import tqdm
from base import *
from config import *

subjects = SUBJS15

sequence_dict = {}
for subject in tqdm(subjects):
    sequence_dict[subject] = None
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    sequence_dict[subject] = sequence
    
sequence_df = pd.DataFrame.from_dict(sequence_dict, orient='index')
sequence_df.columns = [f'pos{i+1}' for i in range(sequence_df.shape[1])]
sequence_df.index.name = 'sub'
sequence_df.reset_index(inplace=True)
sequence_df.to_csv(FIGURES_DIR / 'behav' / 'sequences.csv', index=False)