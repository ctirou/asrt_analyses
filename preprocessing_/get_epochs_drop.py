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
for analysis in ['for_timeg', 'for_rsa']:
    
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
                'drop ratio (%)': 100 - round(count * 100 / 255) if epoch_num == 0 else 100 - round(count * 100 / 425)
            })

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(f'epoch_counts_{analysis}.csv', sep='\t', index=False)

    # mean drop
    mean = df.loc[df['analysis'] == analysis, 'drop ratio (%)'].mean()
    print(f"Mean drop for {analysis}: {mean:.2f}%\n---")
    # median drop
    median = df.loc[df['analysis'] == analysis, 'drop ratio (%)'].median()
    print(f"Median drop for {analysis}: {median:.2f}%\n---")
    