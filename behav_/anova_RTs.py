import os
import os.path as op
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import pingouin as pg
from config import SUBJS, DATA_DIR, RESULTS_DIR
path_data = DATA_DIR
# path_data = '/Users/coum/Desktop/raws/behav'
figures = RESULTS_DIR

subjects = SUBJS
# subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

blocks = ['1','2','3','4']
columns = blocks.copy()
columns.insert(0, 'sub')

all_RT = list()
pat_RT = list()
rand_RT = list()
for subject in subjects:
    aRT = list()
    pRT = list()
    rRT = list()
    for i in range(1, 5):        
        fname_behav = op.join(path_data, 'behav', f'{subject}-{i}.pkl')
        behav_df = pd.read_pickle(fname_behav)
        behav_df.reset_index(inplace=True)      
        aRT.append(behav_df['RTs'].mean())
        pRT.append(behav_df['RTs'][np.where(behav_df['triplets'] == 30)[0]].mean())
        # rRT.append(behav_df['RTs'][np.where((behav_df['triplets'] == 32) | (behav_df['triplets'] == 34))[0]].mean())
        rRT.append(behav_df['RTs'][np.where((behav_df['triplets'] == 32))[0]].mean())
    all_RT.append(np.array(aRT))
    pat_RT.append(np.array(pRT))
    rand_RT.append(np.array(rRT))
all_RT = np.array(all_RT)
pat_RT = np.array(pat_RT)
rand_RT = np.array(rand_RT)

all = pd.DataFrame(data=np.insert(all_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
all = pd.melt(all, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
all.insert(1, 'Triplet', np.zeros(len(all)))

pattern = pd.DataFrame(data=np.insert(pat_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
pattern = pd.melt(pattern, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
pattern.insert(1, 'Triplet', np.zeros(len(pattern)))

random = pd.DataFrame(data=np.insert(rand_RT, 0, np.arange(len(subjects)), axis=1), columns=columns)
random = pd.melt(random, id_vars=['sub'], value_vars=blocks, var_name='block', value_name='RT')
random.insert(1, 'Triplet', np.ones(len(random)))

anova_stats_learning = pd.concat([pattern, random])

anova_stats_learning.to_csv(op.join(figures, 'anova.csv'))

aov_stats = pg.rm_anova(data=anova_stats_learning, dv='RT', within=['block', 'Triplet'], subject='sub', detailed=True)

print(pg.pairwise_tukey(data=anova_stats_learning, dv='RT', between='block', effsize='r').round(3))
# print(pg.pairwise_tukey(data=pattern, dv='RT', between='block', effsize='r'))
# print(pg.pairwise_tukey(data=random, dv='RT', between='block', effsize='r'))
print(pg.pairwise_tukey(data=anova_stats_learning, dv='RT', between='Triplet', effsize='r').round(3))
# print(pg.pairwise_tukey(data=anova_stats_learning, dv='RT', between=['Triplet', 'block'], effsize='r').round(3))
print(pg.pairwise_gameshowell(data=anova_stats_learning, dv='RT', between='block', effsize='r').round(3))
print(pg.pairwise_gameshowell(data=anova_stats_learning, dv='RT', between='Triplet', effsize='r').round(3))

print(pg.pairwise_ttests(data = anova_stats_learning, dv='RT', between='block', subject='sub',
                         parametric=False, padjust='bonf', effsize='r'))
print(pg.pairwise_ttests(data = anova_stats_learning, dv='RT', between='Triplet', subject='sub',
                         parametric=False, padjust='bonf', effsize='r'))

