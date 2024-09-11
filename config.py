from pathlib import Path

SUBJS = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
        'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

EPOCHS = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

RAW_DATA_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/raws')
DATA_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/preprocessed')
RESULTS_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/results')
FREESURFER_DIR = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer')
FIGURE_PATH = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/figures')
HOME = Path('/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt')

PRED_PATH = Path('/Users/coum/Desktop/pred_asrt')
PRED_PATH_SSD = Path('/Volumes/Ultra_Touch/pred_asrt')
PRED_PATH_MB = Path('/Users/coum/Desktop/pred_asrt')

SURFACE_LABELS = ['cuneus-lh',
                'cuneus-rh',
                'fusiform-lh',
                'fusiform-rh',
                'inferiorparietal-lh',
                'inferiorparietal-rh',
                'isthmuscingulate-lh',
                'isthmuscingulate-rh',
                'lateraloccipital-lh',
                'lateraloccipital-rh',
                'lingual-lh',
                'lingual-rh',
                'pericalcarine-lh',
                'pericalcarine-rh',
                'precuneus-lh',
                'precuneus-rh',
                'superiorparietal-lh',
                'superiorparietal-rh']

SURFACE_LABELS_RT = ['lateraloccipital-lh',
                     'lateraloccipital-rh',
                     'lingual-lh',
                     'lingual-rh',
                     'paracentral-lh',
                     'paracentral-rh',
                     'pericalcarine-lh',
                     'pericalcarine-rh',
                     'postcentral-lh',
                     'postcentral-rh',
                     'precentral-lh',
                     'precentral-rh',
                     'superiorparietal-lh',
                     'superiorparietal-rh',
                     'supramarginal-lh',
                     'supramarginal-rh']

VOLUME_LABELS = ['Amygdala-lh',
                  'Amygdala-rh',
                  'Caudate-lh',
                  'Caudate-rh',
                  'Cerebellum-Cortex-lh',
                  'Cerebellum-Cortex-rh',
                  'Cerebellum-White-Matter-lh',
                  'Cerebellum-White-Matter-rh',
                  'Hippocampus-lh',
                  'Hippocampus-rh',
                  'Lateral-Ventricle-lh',
                  'Lateral-Ventricle-rh',
                  'Pallidum-lh',
                  'Pallidum-rh',
                  'Putamen-lh',
                  'Putamen-rh',
                  'Thalamus-Proper-lh',
                  'Thalamus-Proper-rh',
                  'VentralDC-lh',
                  'VentralDC-rh']

VOLUME_LABELS_RT = ['Caudate-lh',
                  'Caudate-rh',
                  'Cerebellum-Cortex-lh',
                  'Cerebellum-Cortex-rh',
                  'Cerebellum-White-Matter-lh',
                  'Cerebellum-White-Matter-rh',
                  'Hippocampus-lh',
                  'Hippocampus-rh',
                  'Lateral-Ventricle-lh',
                  'Lateral-Ventricle-rh',
                  'Putamen-lh',
                  'Putamen-rh',
                  'Thalamus-Proper-lh',
                  'Thalamus-Proper-rh',
                  'VentralDC-lh',
                  'VentralDC-rh']
