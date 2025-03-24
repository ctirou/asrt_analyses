import numpy as np
from base import *
from config import *
from tqdm.auto import tqdm

subjects = SUBJS + ['sub03', 'sub06']
lock = 'stim'

times = np.linspace(-0.2, 0.6, 82)
win = np.where((times >= 0.28) & (times <= 0.51))[0]

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

# Calculate peak for all participants
# Then mean
blocks = np.arange(23)
peaks = list()
for subject in range(len(subjects)):
    peaks.append(blocks[np.argmax(sim_index[subject])])
peaks = np.array(peaks)
m_peak = int(round(peaks.mean()))
# print('Peak:', m_peak)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.axhline(0, color='grey', linestyle='-', alpha=0.5)
for i in range(sim_index.shape[0]):
    ax.plot(blocks, sim_index[i], alpha=0.5)
# ax.set_xticks(range(1, 24))
ax.set_xticks(blocks)
ax.plot(blocks, sim_index.mean(0), lw=3, label='RSA')
ax.set_xlabel('Block')
ax.set_ylabel('Mean RSA effect')
ax.axvspan(m_peak-0.05, m_peak+0.05, color='red', alpha=0.5, label='peak')
ax.axvspan(0, 2, color='grey', alpha=0.1, label='practice')
ax.set_xticklabels(['01', '02', '03'] + [str(i) for i in range(1, 21)])
ax.legend()
plt.show()