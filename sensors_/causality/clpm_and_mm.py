import pandas as pd
from semopy import Model, Optimizer

import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'

times = np.linspace(-0.2, 0.6, 82)

# load RSA data
pattern, random = [], []
for subject in tqdm(subjects):
    res_path = RESULTS_DIR / 'RSA' / 'sensors' / lock / "loocv_rdm_blocks" / subject        
    behav_dir = op.join(HOME / 'raw_behavs' / subject)
    sequence = get_sequence(behav_dir)
    pat, rand = get_all_high_low_blocks(res_path, sequence)
    pattern.append(pat)
    random.append(rand)
pattern = np.array(pattern).mean(1)
random = np.array(random).mean(1)
win = np.where((times >= 0.28) & (times <= 0.51))[0]
m_pattern = pattern[:, :, win].mean(-1)
m_random = random[:, :, win].mean(-1)
prac_pat = m_pattern[:, :3].mean(-1)
prac_rand = m_random[:, :3].mean(-1)
sim_index = list()
for subject in range(len(subjects)):
    sub_sim = list()
    for i in range(23):
        diff = (m_random[subject, i] - prac_rand[subject]) - (m_pattern[subject, i] - prac_pat[subject])
        sub_sim.append(diff)
    sim_index.append(np.array(sub_sim))
sim_index = np.array(sim_index)

# load time gen data
timeg_data_path = TIMEG_DATA_DIR / 'results' / 'sensors' / lock
timesg = np.linspace(-1.5, 4, 559)
pattern, random = [], []
for subject in tqdm(subjects):
    pat, rand = [], []
    for block in range(1, 24):
        pat.append(np.load(timeg_data_path / 'split_all_pattern' /  f"{subject}-{block}.npy"))
        rand.append(np.load(timeg_data_path / 'split_all_random' / f"{subject}-{block}.npy"))
    pattern.append(np.array(pat))
    random.append(np.array(rand))
pattern, random = np.array(pattern), np.array(random)
contrast = pattern - random
idx_timeg = np.where((timesg >= -0.5) & (timesg < 0))[0]
# mean diag
mean_diag = []
for sub in range(len(subjects)):
    tg = []
    for block in range(23):
        data = np.diag(contrast[sub, block])
        tg.append(data[idx_timeg].mean())
    mean_diag.append(np.array(tg))
mean_diag = np.array(mean_diag)

X = mean_diag.copy() # Predictive coding
Y = sim_index.copy() # Representational change

# Create a pandas DataFrame with X and Y
data = {'X': X.flatten(), 'Y': Y.flatten()}
df_xy = pd.DataFrame(data)
# Load your data

df = pd.read_csv('your_data.csv')

# Create lagged variables per participant
df['A_lag'] = df.groupby('participant')['A'].shift(1)
df['B_lag'] = df.groupby('participant')['B'].shift(1)

# Remove missing lagged values (block 1 for each participant)
df = df.dropna()

# Define CLPM model
model_desc = """
A ~ A_lag + B_lag
B ~ B_lag + A_lag
"""

# Fit the model
model = Model(model_desc)
opt = Optimizer(model)
opt.optimize(df)

# Show results
print(model.inspect())