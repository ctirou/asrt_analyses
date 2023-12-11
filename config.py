from pathlib import Path


# Path definitions
USER = 'coum'
# USER = 'romain'

if USER == 'Coum':
    RAW_DATA_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/raws'
    DATA_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/preprocessed'
    RESULTS_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results'
    FREESURFER_DIR = '/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer'
else:
    RAW_DATA_DIR = None
    DATA_DIR = None
    RESULTS_DIR = None
    FREESURFER_DIR = None