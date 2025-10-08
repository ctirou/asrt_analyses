import os
from pathlib import Path

SUBJS15 = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 
           'sub09', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14', 'sub15']

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

NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Hippocampus', 'Thalamus', 'Cerebellum-Cortex']
NETWORK_NAMES = ['Visual', 'Sensorimotor', 'Dorsal Attention', 'Salience', 'Limbic', 'Central Executive', 'Default Mode', 'Hippocampus', 'Thalamus', 'Cerebellum']

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
