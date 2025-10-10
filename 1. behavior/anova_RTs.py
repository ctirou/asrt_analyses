# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import pandas as pd
import pingouin as pg
from config import *

path_data = DATA_DIR / 'for_rsa'
figures = RESULTS_DIR
subjects = SUBJS15

runs = ['1','2','3','4']
blocks = [i for i in range(1, 21)]
columns = runs.copy()
columns.insert(0, 'sub')

all_RT = list()
pat_RT = list()
rand_RT = list()

all_RTb = list()
pat_RTb = list()
rand_RTb = list()

for subject in subjects:
    aRT = list()
    pRT = list()
    rRT = list()

    aRTb = list()
    pRTb = list()
    rRTb = list()

    for i in range(1, 5): 
        fname_behav = op.join(path_data, 'behav', f'{subject}-{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)

        aRT.append(behav_df['RTs'][np.where(behav_df['triplets'].isin([30, 32, 34]))[0]].mean())
        pRT.append(behav_df['RTs'][np.where(behav_df['triplets'] == 30)[0]].mean())
        rRT.append(behav_df['RTs'][np.where((behav_df['triplets'] == 32))[0]].mean())
        
        nblocks = np.unique(behav_df['blocks'])
        for block in nblocks:
            block_df = behav_df[behav_df['blocks'] == block].reset_index(drop=True)
            aRTb.append(block_df['RTs'][np.where(block_df['triplets'].isin([30, 32, 34]))[0]].mean())
            pRTb.append(block_df['RTs'][np.where(block_df['triplets'] == 30)[0]].mean())
            rRTb.append(block_df['RTs'][np.where((block_df['triplets'] == 32))[0]].mean())
    
    all_RTb.append(np.array(aRTb))
    pat_RTb.append(np.array(pRTb))
    rand_RTb.append(np.array(rRTb))
        
    all_RT.append(np.array(aRT))
    pat_RT.append(np.array(pRT))
    rand_RT.append(np.array(rRT))

all_RT = np.array(all_RT)
pat_RT = np.array(pat_RT)
rand_RT = np.array(rand_RT)

all_RTb = np.array(all_RTb)
pat_RTb = np.array(pat_RTb)
rand_RTb = np.array(rand_RTb)

# ------------------- PER RUN -------------------
all = pd.DataFrame(data=np.insert(all_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
all = pd.melt(all, id_vars=['sub'], value_vars=runs, var_name='run', value_name='RT')
all.insert(1, 'Triplet', np.zeros(len(all)))

pattern = pd.DataFrame(data=np.insert(pat_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
pattern = pd.melt(pattern, id_vars=['sub'], value_vars=runs, var_name='run', value_name='RT')
pattern.insert(1, 'Triplet', np.zeros(len(pattern)))

random = pd.DataFrame(data=np.insert(rand_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
random = pd.melt(random, id_vars=['sub'], value_vars=runs, var_name='run', value_name='RT')
random.insert(1, 'Triplet', np.ones(len(random)))

anova_stats_learning = pd.concat([pattern, random])
anova_stats_learning.to_csv(op.join(figures, 'anova_run.csv'))
aov_stats = pg.rm_anova(data=anova_stats_learning, dv='RT', within=['run', 'Triplet'], subject='sub', detailed=True)

# ------------------- PER BLOCK -------------------
all_block = pd.DataFrame(data=np.insert(all_RTb, 0, np.arange(len(subjects)), axis=1), columns=['sub'] + blocks)
all_block = pd.melt(all_block, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
all_block.insert(1, 'Triplet', np.zeros(len(all_block)))

pattern_block = pd.DataFrame(data=np.insert(pat_RTb, 0, np.arange(len(subjects)), axis=1), columns=['sub'] + blocks)
pattern_block = pd.melt(pattern_block, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
pattern_block.insert(1, 'Triplet', np.zeros(len(pattern_block)))

random_block = pd.DataFrame(data=np.insert(rand_RTb, 0, np.arange(len(subjects)), axis=1), columns=['sub'] + blocks)
random_block = pd.melt(random_block, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
random_block.insert(1, 'Triplet', np.ones(len(random_block)))

anova_stats_learning_block = pd.concat([pattern_block, random_block])
anova_stats_learning_block.to_csv(op.join(figures, 'anova_block.csv'))
aov_stats_block = pg.rm_anova(data=anova_stats_learning_block, dv='RT', within=['block', 'Triplet'], subject='sub', detailed=True)