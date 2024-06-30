from pathlib import Path

SUBJS = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
        'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

EPOCHS = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']


RAW_DATA_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/raws')
DATA_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/preprocessed')
RESULTS_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results')
FREESURFER_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer')

PRED_PATH = Path('/Users/coum/Desktop/pred_asrt')
PRED_PATH_SSD = Path('/Volumes/Ultra_Touch/pred_asrt')
PRED_PATH_MB = Path('/Users/coum/Desktop/pred_asrt')

VOLUME_LABELS = ["Left-Cerebellum-Cortex", 
                 "Right-Cerebellum-Cortex", 
                 "Left-Thalamus-Proper", 
                 "Right-Thalamus-Proper", 
                 "Left-Hippocampus", 
                 "Right-Hippocampus",
                 "CSF",
                 "Brain-Stem"]