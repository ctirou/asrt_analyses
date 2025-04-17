import os
from pathlib import Path

SUBJS = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
        'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

EPOCHS = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

FIGURES_DIR = Path("/Users/coum/MEGAsync/figures")

if os.getenv("MB_ENV") is not None:
    HOME = Path("/Volumes/Ultra_Touch/asrt/")
    DATA_DIR = Path('/Volumes/Ultra_Touch/asrt/preprocessed')
    RAW_DATA_DIR = Path('/Volumes/Ultra_Touch/asrt/raws')
    RESULTS_DIR = Path('/Volumes/Ultra_Touch/asrt/results')
    FREESURFER_DIR = Path('/Volumes/Ultra_Touch/asrt/freesurfer')
    TIMEG_DATA_DIR = Path('/Volumes/Ultra_Touch/pred_asrt')
elif os.getenv("CLUSTER_ENV") is not None:
    RAW_DATA_DIR = Path('/sps/crnl/Romain/ASRT_MEG/data/raws')
    DATA_DIR = Path('/sps/crnl/Romain/ASRT_MEG/data/preprocessed')
    RESULTS_DIR = Path('/sps/crnl/Romain/ASRT_MEG/data/preprocessed/results')
    FREESURFER_DIR = Path('/sps/crnl/Romain/ASRT_MEG/data/freesurfer')
    TIMEG_DATA_DIR = Path('/sps/crnl/Romain/ASRT_MEG/data/pred_asrt')
else:
    RAW_DATA_DIR = Path('/Users/coum/Desktop/asrt/raws')
    DATA_DIR = Path('/Users/coum/Desktop/asrt/preprocessed')
    RESULTS_DIR = Path('/Users/coum/Desktop/asrt/results')
    FREESURFER_DIR = Path('/Users/coum/Desktop/asrt/freesurfer')
    TIMEG_DATA_DIR = Path('/Users/coum/Desktop/pred_asrt')
    HOME = Path("/Users/coum/Desktop/asrt")

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

subcortex_regions = ['Cerebral-White-Matter-lh', 'Cerebral-Cortex-lh', 'Lateral-Ventricle-lh', 
                     'Inf-Lat-Vent-lh', 'Cerebellum-White-Matter-lh', 'Cerebellum-Cortex-lh', 
                     'Thalamus-Proper-lh', 'Caudate-lh', 'Putamen-lh', 'Pallidum-lh', '4th-Ventricle',
                     'Brain-Stem', 'Hippocampus-lh', 'Amygdala-lh', 'CSF', 'Accumbens-area-lh', 
                     'VentralDC-lh', 'vessel-lh', 'choroid-plexus-lh', 'Cerebral-White-Matter-rh',
                     'Cerebral-Cortex-rh', 'Lateral-Ventricle-rh', 'Inf-Lat-Vent-rh', 
                     'Cerebellum-White-Matter-rh', 'Cerebellum-Cortex-rh', 'Thalamus-Proper-rh', 
                     'Caudate-rh', 'Putamen-rh', 'Pallidum-rh', 'Hippocampus-rh', 'Amygdala-rh',
                     'Accumbens-area-rh', 'VentralDC-rh', 'vessel-rh', 'choroid-plexus-rh',
                     'WM-hypointensities', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 
                     'CC_Anterior']

frontal_lobe = ['caudalmiddlefrontal-lh', 'frontalpole-lh', 'medialorbitofrontal-lh', 'rostralmiddlefrontal-lh',
                'superiorfrontal-lh', 'Caudate-lh', 'Putamen-lh', 'Accumbens-area-lh', 'Thalamus-Proper-lh',
                'caudalmiddlefrontal-rh', 'frontalpole-rh', 'medialorbitofrontal-rh', 'rostralmiddlefrontal-rh', 
                'superiorfrontal-rh', 'Caudate-rh', 'Putamen-rh', 'Accumbens-area-rh', 'Thalamus-Proper-rh']

parietal_lobe = ['inferiorparietal-lh', 'postcentral-lh', 'supramarginal-lh', 'paracentral-lh',
                 'inferiorparietal-rh', 'postcentral-rh', 'supramarginal-rh', 'paracentral-rh']

occipital_lobe = ['cuneus-lh', 'lateraloccipital-lh', 'lingual-lh', 'pericalcarine-lh',
                  'cuneus-rh', 'lateraloccipital-rh', 'lingual-rh', 'pericalcarine-rh']

temporal_lobe = ['entorhinal-lh', 'fusiform-lh', 'inferiortemporal-lh', 'middletemporal-lh',
                 'parahippocampal-lh', 'superiortemporal-lh', 'temporalpole-lh', 'transversetemporal-lh',
                 'Hippocampus-lh', 'Amygdala-lh', 
                 'entorhinal-rh', 'fusiform-rh', 'inferiortemporal-rh', 'middletemporal-rh',
                 'parahippocampal-rh', 'superiortemporal-rh', 'temporalpole-rh', 'transversetemporal-rh',
                 'Hippocampus-rh', 'Amygdala-rh']

cerebellum = ['Cerebellum-White-Matter-lh', 'Cerebellum-Cortex-lh', 
              'Cerebellum-White-Matter-rh', 'Cerebellum-Cortex-rh']

ventricular_and_brainstem = ['Lateral-Ventricle-lh', 'Inf-Lat-Vent-lh', 'Lateral-Ventricle-rh', 
                             'Inf-Lat-Vent-rh', '4th-Ventricle', 'CSF', 'choroid-plexus-lh', 
                             'choroid-plexus-rh', 'Brain-Stem', 'vessel-lh', 'vessel-rh', 
                             'WM-hypointensities']

corpus_callosum = ['CC_Anterior', 'CC_Mid_Anterior', 'CC_Central', 'CC_Mid_Posterior', 'CC_Posterior']

left_hemisphere = ['bankssts-lh', 'caudalanteriorcingulate-lh', 'caudalmiddlefrontal-lh', 'cuneus-lh',
                   'entorhinal-lh', 'frontalpole-lh', 'fusiform-lh', 'inferiorparietal-lh', 'inferiortemporal-lh',
                   'insula-lh', 'isthmuscingulate-lh', 'lateraloccipital-lh', 'lateralorbitofrontal-lh', 
                   'lingual-lh', 'medialorbitofrontal-lh', 'middletemporal-lh', 'paracentral-lh', 'parahippocampal-lh',
                   'parsopercularis-lh', 'parsorbitalis-lh', 'parstriangularis-lh', 'pericalcarine-lh', 'postcentral-lh',
                   'posteriorcingulate-lh', 'precentral-lh', 'precuneus-lh', 'rostralanteriorcingulate-lh',
                   'rostralmiddlefrontal-lh', 'superiorfrontal-lh', 'superiorparietal-lh', 'superiortemporal-lh',
                   'supramarginal-lh', 'temporalpole-lh', 'transversetemporal-lh', 'Cerebral-White-Matter-lh',
                   'Cerebral-Cortex-lh', 'Lateral-Ventricle-lh', 'Inf-Lat-Vent-lh', 'Cerebellum-White-Matter-lh',
                   'Cerebellum-Cortex-lh', 'Thalamus-Proper-lh', 'Caudate-lh', 'Putamen-lh', 'Pallidum-lh',
                   'Hippocampus-lh', 'Amygdala-lh', 'Accumbens-area-lh', 'VentralDC-lh', 'vessel-lh',
                   'choroid-plexus-lh']

right_hemisphere = ['bankssts-rh', 'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-rh', 'cuneus-rh',
                    'entorhinal-rh', 'frontalpole-rh', 'fusiform-rh', 'inferiorparietal-rh', 'inferiortemporal-rh',
                    'insula-rh', 'isthmuscingulate-rh', 'lateraloccipital-rh', 'lateralorbitofrontal-rh', 'lingual-rh',
                    'medialorbitofrontal-rh', 'middletemporal-rh', 'paracentral-rh', 'parahippocampal-rh',
                    'parsopercularis-rh', 'parsorbitalis-rh', 'parstriangularis-rh', 'pericalcarine-rh', 'postcentral-rh',
                    'posteriorcingulate-rh', 'precentral-rh', 'precuneus-rh', 'rostralanteriorcingulate-rh',
                    'rostralmiddlefrontal-rh', 'superiorfrontal-rh', 'superiorparietal-rh', 'superiortemporal-rh',
                    'supramarginal-rh', 'temporalpole-rh', 'transversetemporal-rh', 'Cerebral-White-Matter-rh',
                    'Cerebral-Cortex-rh', 'Lateral-Ventricle-rh', 'Inf-Lat-Vent-rh', 'Cerebellum-White-Matter-rh',
                    'Cerebellum-Cortex-rh', 'Thalamus-Proper-rh', 'Caudate-rh', 'Putamen-rh', 'Pallidum-rh',
                    'Hippocampus-rh', 'Amygdala-rh', 'Accumbens-area-rh', 'VentralDC-rh', 'vessel-rh',
                    'choroid-plexus-rh']

NEW_LABELS = ['corpus_callosum',
              'cortex_regions', 
              'frontal_lobe',
              'left_hemisphere',
              'occipital_lobe',
              'parietal_lobe',
              'right_hemisphere',
              'subcortex_regions',
              'temporal_lobe',
              'ventricular_and_brainstem']

NEW_LABELS2 = ['corpus_callosum',
              'cortex_regions', 
              'frontal_lobe',
              'left_hemisphere',
              'occipital_lobe',
              'parietal_lobe',
              'right_hemisphere',
              'subcortex_regions',
              'temporal_lobe',
              'ventricular_and_brainstem']

NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Hippocampus', 'Thalamus']
# NETWORK_NAMES = [' Visual', 'Somatomotor', 'Dorsal\nAttention', 'Salience /\nVentral\nAttention', 'Limbic', 'Control', 'Default', 'Hippocampus', 'Thalamus']
NETWORK_NAMES = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Control', 'Default', 'Hippocampus', 'Thalamus']

schaefer_17 = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB',
               'LimbicA', 'LimbicB', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar']

colors = {"BottleRocket1":["#A42820", "#5F5647", "#9B110E", "#3F5151", "#4E2A1E", "#550307", "#0C1707"],
          "BottleRocket2":["#FAD510", "#CB2314", "#273046", "#354823", "#1E1E1E"],
          "Rushmore1":["#E1BD6D", "#EABE94", "#0B775E", "#35274A","#F2300F"],
          "Royal1":["#899DA4", "#C93312", "#FAEFD1", "#DC863B"],
          "Royal2":["#9A8822", "#F5CDB4", "#F8AFA8", "#FDDDA0", "#74A089"],
          "Zissou1":["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"],
          "Darjeeling1":["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"],
          "Darjeeling2":["#ECCBAE", "#046C9A", "#D69C4E", "#ABDDDE", "#000000"],
          "Chevalier1":["#446455", "#FDD262", "#D3DDDC", "#C7B19C"],
          "FantasticFox1":["#DD8D29", "#E2D200", "#46ACC8", "#E58601", "#B40F20"],
          "Moonrise1":["#F3DF6C", "#CEAB07", "#D5D5D3", "#24281A"],
          "Moonrise2":["#798E87", "#C27D38", "#CCC591", "#29211F"],
          "Moonrise3":["#85D4E3", "#F4B5BD", "#9C964A", "#CDC08C", "#FAD77B"],
          "Cavalcanti1":["#D8B70A", "#02401B", "#A2A475", "#81A88D", "#972D15"],
          "GrandBudapest1":["#F1BB7B", "#FD6467", "#5B1A18", "#D67236"],
          "GrandBudapest2":["#E6A0C4", "#C6CDF7", "#D8A499", "#7294D4"],
          "IsleofDogs1":["#9986A5", "#79402E", "#CCBA72", "#0F0D0E", "#D9D0D3", "#8D8680"],
          "IsleofDogs2":["#EAD3BF", "#AA9486", "#B6854D", "#39312F", "#1C1718"]}

colorblind_cmap = ['#0173B2','#DE8F05','#029E73','#D55E00','#CC78BC','#CA9161','#FBAFE4','#ECE133','#56B4E9']

cud_colors = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00', 
    '#CC79A7', '#F0E442', '#56B4E9', '#D45F91'
]
