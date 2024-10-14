from pathlib import Path

SUBJS = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
        'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

EPOCHS = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

RAW_DATA_DIR = Path('/Users/coum/Desktop/asrt/raws')
RAW_DATA_DIR_SSD = Path('/Volumes/Ultra_Touch/asrt/raws')
DATA_DIR = Path('/Users/coum/Desktop/asrt/preprocessed')
RESULTS_DIR = Path('/Users/coum/Desktop/asrt/results')
FREESURFER_DIR = Path('/Users/coum/Desktop/asrt/freesurfer')

HOME = Path("/Users/coum/Desktop/asrt")
FIGURES_DIR = Path("/Users/coum/MEGAsync/figures")

PRED_PATH = Path('/Users/coum/Desktop/pred_asrt')
PRED_PATH_SSD = Path('/Volumes/Ultra_Touch/pred_asrt')
PRED_PATH_MB = Path('/Users/coum/Desktop/pred_asrt')

DATA_DIR_SSD = Path('/Volumes/Ultra_Touch/asrt')

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

BAD_VOLUME_LABELS = ['3rd-Ventricle',
                     '5th-Ventricle',
                     'Cerebral-Cortex-lh',
                     'Cerebral-Cortex-rh',
                     'Cerebral-White-Matter-lh',
                     'Cerebral-White-Matter-lh',
                     'Inf-Lat-Vent-lh', 
                     'Inf-Lat-Vent-rh',
                     'Optic-Chiasm',
                     'choroid-plexus-lh',
                     'choroid-plexus-rh',
                     'Unknown',
                     'vessel-lh',
                     'vessel-rh']

cortex_regions = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh', 'caudalanteriorcingulate-rh', 
                  'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh', 'cuneus-lh', 'cuneus-rh', 'entorhinal-lh',
                  'entorhinal-rh', 'frontalpole-lh', 'frontalpole-rh', 'fusiform-lh', 'fusiform-rh', 
                  'inferiorparietal-lh', 'inferiorparietal-rh', 'inferiortemporal-lh', 'inferiortemporal-rh',
                  'insula-lh', 'insula-rh', 'isthmuscingulate-lh', 'isthmuscingulate-rh', 'lateraloccipital-lh',
                  'lateraloccipital-rh', 'lateralorbitofrontal-lh', 'lateralorbitofrontal-rh', 'lingual-lh', 
                  'lingual-rh', 'medialorbitofrontal-lh', 'medialorbitofrontal-rh', 'middletemporal-lh', 
                  'middletemporal-rh', 'paracentral-lh', 'paracentral-rh', 'parahippocampal-lh', 'parahippocampal-rh',
                  'parsopercularis-lh', 'parsopercularis-rh', 'parsorbitalis-lh', 'parsorbitalis-rh', 'parstriangularis-lh',
                  'parstriangularis-rh', 'pericalcarine-lh', 'pericalcarine-rh', 'postcentral-lh', 'postcentral-rh',
                  'posteriorcingulate-lh', 'posteriorcingulate-rh', 'precentral-lh', 'precentral-rh', 'precuneus-lh', 
                  'precuneus-rh', 'rostralanteriorcingulate-lh', 'rostralanteriorcingulate-rh', 'rostralmiddlefrontal-lh', 
                  'rostralmiddlefrontal-rh', 'superiorfrontal-lh', 'superiorfrontal-rh', 'superiorparietal-lh', 
                  'superiorparietal-rh', 'superiortemporal-lh', 'superiortemporal-rh', 'supramarginal-lh', 
                  'supramarginal-rh', 'temporalpole-lh', 'temporalpole-rh', 'transversetemporal-lh', 
                  'transversetemporal-rh']

subcortex_regions = ['Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle', 
                     'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 
                     'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', '4th-Ventricle',
                     'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF', 'Left-Accumbens-area', 
                     'Left-VentralDC', 'Left-vessel', 'Left-choroid-plexus', 'Right-Cerebral-White-Matter',
                     'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 
                     'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 
                     'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
                     'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus',
                     'WM-hypointensities', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 
                     'CC_Anterior']

frontal_lobe = ['caudalmiddlefrontal-lh', 'frontalpole-lh', 'medialorbitofrontal-lh', 'rostralmiddlefrontal-lh',
                'superiorfrontal-lh', 'Left-Caudate', 'Left-Putamen', 'Left-Accumbens-area', 'Left-Thalamus-Proper',
                'caudalmiddlefrontal-rh', 'frontalpole-rh', 'medialorbitofrontal-rh', 'rostralmiddlefrontal-rh', 
                'superiorfrontal-rh', 'Right-Caudate', 'Right-Putamen', 'Right-Accumbens-area', 'Right-Thalamus-Proper']

parietal_lobe = ['inferiorparietal-lh', 'postcentral-lh', 'supramarginal-lh', 'paracentral-lh',
                 'inferiorparietal-rh', 'postcentral-rh', 'supramarginal-rh', 'paracentral-rh']

occipital_lobe = ['cuneus-lh', 'lateraloccipital-lh', 'lingual-lh', 'pericalcarine-lh',
                  'cuneus-rh', 'lateraloccipital-rh', 'lingual-rh', 'pericalcarine-rh']

temporal_lobe = ['entorhinal-lh', 'fusiform-lh', 'inferiortemporal-lh', 'middletemporal-lh',
                 'parahippocampal-lh', 'superiortemporal-lh', 'temporalpole-lh', 'transversetemporal-lh',
                 'Left-Hippocampus', 'Left-Amygdala', 
                 'entorhinal-rh', 'fusiform-rh', 'inferiortemporal-rh', 'middletemporal-rh',
                 'parahippocampal-rh', 'superiortemporal-rh', 'temporalpole-rh', 'transversetemporal-rh',
                 'Right-Hippocampus', 'Right-Amygdala']

cerebellum = ['Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 
              'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex']

ventricular_and_brainstem = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Right-Lateral-Ventricle', 
                             'Right-Inf-Lat-Vent', '4th-Ventricle', 'CSF', 'Left-choroid-plexus', 
                             'Right-choroid-plexus', 'Brain-Stem', 'Left-vessel', 'Right-vessel', 
                             'WM-hypointensities']

corpus_callosum = ['CC_Anterior', 'CC_Mid_Anterior', 'CC_Central', 'CC_Mid_Posterior', 'CC_Posterior']

left_hemisphere = ['bankssts-lh', 'caudalanteriorcingulate-lh', 'caudalmiddlefrontal-lh', 'cuneus-lh',
                   'entorhinal-lh', 'frontalpole-lh', 'fusiform-lh', 'inferiorparietal-lh', 'inferiortemporal-lh',
                   'insula-lh', 'isthmuscingulate-lh', 'lateraloccipital-lh', 'lateralorbitofrontal-lh', 
                   'lingual-lh', 'medialorbitofrontal-lh', 'middletemporal-lh', 'paracentral-lh', 'parahippocampal-lh',
                   'parsopercularis-lh', 'parsorbitalis-lh', 'parstriangularis-lh', 'pericalcarine-lh', 'postcentral-lh',
                   'posteriorcingulate-lh', 'precentral-lh', 'precuneus-lh', 'rostralanteriorcingulate-lh',
                   'rostralmiddlefrontal-lh', 'superiorfrontal-lh', 'superiorparietal-lh', 'superiortemporal-lh',
                   'supramarginal-lh', 'temporalpole-lh', 'transversetemporal-lh', 'Left-Cerebral-White-Matter',
                   'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                   'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                   'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
                   'Left-choroid-plexus']

right_hemisphere = ['bankssts-rh', 'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-rh', 'cuneus-rh',
                    'entorhinal-rh', 'frontalpole-rh', 'fusiform-rh', 'inferiorparietal-rh', 'inferiortemporal-rh',
                    'insula-rh', 'isthmuscingulate-rh', 'lateraloccipital-rh', 'lateralorbitofrontal-rh', 'lingual-rh',
                    'medialorbitofrontal-rh', 'middletemporal-rh', 'paracentral-rh', 'parahippocampal-rh',
                    'parsopercularis-rh', 'parsorbitalis-rh', 'parstriangularis-rh', 'pericalcarine-rh', 'postcentral-rh',
                    'posteriorcingulate-rh', 'precentral-rh', 'precuneus-rh', 'rostralanteriorcingulate-rh',
                    'rostralmiddlefrontal-rh', 'superiorfrontal-rh', 'superiorparietal-rh', 'superiortemporal-rh',
                    'supramarginal-rh', 'temporalpole-rh', 'transversetemporal-rh', 'Right-Cerebral-White-Matter',
                    'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter',
                    'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
                    'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
                    'Right-choroid-plexus']
