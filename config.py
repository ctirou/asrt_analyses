from pathlib import Path

# Path definitions
USER = 'coum'
# USER = 'romain'

SUBJS = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
        'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

if USER == 'coum':
    RAW_DATA_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/raws'
    DATA_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/preprocessed'
    RESULTS_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results'
    FREESURFER_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer'
else:
    RAW_DATA_DIR = None
    DATA_DIR = None
    RESULTS_DIR = None
    FREESURFER_DIR = None