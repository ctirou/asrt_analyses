import os.path as op
from pathlib import Path
import mne
import pandas as pd
import numpy as np
from base import *
from config import *

overwrite = False
verbose = 'error'

subjects = SUBJS15
results = []

# Loop over analyses
for analysis in ['for_rsa', 'for_rsa2', 'for_rsa3', 'for_rsa4']:
    
    data_path = Path(DATA_DIR) / analysis

    for subject in subjects:

        for epoch_num in range(5):

            epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
            
            if not op.exists(epoch_fname):
                print(f"Missing file: {epoch_fname}")
                continue

            try:
                epoch = mne.read_epochs(epoch_fname, verbose=verbose)
                data = epoch.get_data(picks='mag', copy=True)
                count = len(data)
            except Exception as e:
                print(f"Error reading {epoch_fname}: {e}")
                count = None

            results.append({
                'analysis': analysis,
                'subject': subject,
                'epoch_num': epoch_num,
                'count': count,
                # 'ratio': 100 - round(count * 100 / 255) if epoch_num == 0 else 100 - round(count * 100 / 425)
                'ratio': 100 - round(count * 100 / 255) if epoch_num == 0 else 100 - round(count * 100 / 425)
            })

# Convert results to DataFrame and save
df = pd.DataFrame(results)
df.to_csv('epoch_counts.csv', sep=',', index=False)

# mean drop
mean_rsa = df.loc[df['analysis'] == 'for_rsa', 'ratio'].mean()
print(f"Mean drop for for_rsa: {mean_rsa:.2f}%")
mean_rsa2 = df.loc[df['analysis'] == 'for_rsa2', 'ratio'].mean()
print(f"Mean drop for for_rsa2: {mean_rsa2:.2f}%")
mean_rsa3 = df.loc[df['analysis'] == 'for_rsa3', 'ratio'].mean()
print(f"Mean drop for for_rsa3: {mean_rsa3:.2f}%")
mean_rsa4 = df.loc[df['analysis'] == 'for_rsa4', 'ratio'].mean()
print(f"Mean drop for for_rsa4: {mean_rsa4:.2f}%")

# median drop
median_rsa = df.loc[df['analysis'] == 'for_rsa', 'ratio'].median()
print(f"Median drop for for_rsa: {median_rsa:.2f}%")
median_rsa2 = df.loc[df['analysis'] == 'for_rsa2', 'ratio'].median()
print(f"Median drop for for_rsa2: {median_rsa2:.2f}%")
median_rsa3 = df.loc[df['analysis'] == 'for_rsa3', 'ratio'].median()
print(f"Median drop for for_rsa3: {median_rsa3:.2f}%")
median_rsa4 = df.loc[df['analysis'] == 'for_rsa4', 'ratio'].median()
print(f"Median drop for for_rsa4: {median_rsa4:.2f}%")