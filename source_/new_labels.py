import os
import os.path as op
import mne
from base import ensure_dir
from config import *
from mne import get_volume_labels_from_aseg


overwrite = False
verbose = True
jobs = -1

subjects = SUBJS
epochs_list = EPOCHS
data_path = HOME
subjects_dir = FREESURFER_DIR
res_path = data_path / 'results' / "new_labels"
ensure_dir(res_path)

# Define your region groupings here
new_labels = [cortex_regions, subcortex_regions, 
              frontal_lobe, parietal_lobe, occipital_lobe, temporal_lobe, cerebellum, 
              ventricular_and_brainstem, 
              corpus_callosum, 
              left_hemisphere, right_hemisphere]

new_labels_names = ['cortex_regions', 'subcortex_regions', 
                    'frontal_lobe', 'parietal_lobe', 'occipital_lobe', 'temporal_lobe', 'cerebellum', 
                    'ventricular_and_brainstem', 
                    'corpus_callosum', 
                    'left_hemisphere', 'right_hemisphere']

for subject in subjects:
    ensure_dir(res_path / subject)
    
    # Get all cortical labels
    cx_labels = mne.read_labels_from_annot(subject=subject, parc="aparc", hemi="both", subjects_dir=subjects_dir, verbose=verbose)
    
    # Get all subcortical labels
    aseg_labels = list()
    for hemi in ['lh', 'rh', 'others']:
        # Create mixed source space
        vol_src_fname = op.join(RESULTS_DIR, "src", f"{subject}-{hemi}-vol-src.fif")
        vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)    
        aseg_labels.extend(mne.get_volume_labels_from_src(vol_src, subject, subjects_dir))

    # Combine cortical and subcortical labels 
    labels = cx_labels + aseg_labels
    
    # Iterate over region groups
    for i, (label_list, label_list_name) in enumerate(zip(new_labels, new_labels_names)):
        
        # Filter the labels based on the current list
        filtered_labels = [label for label in labels if label.name in label_list and label.name not in BAD_VOLUME_LABELS]
        
        if filtered_labels:
            # Start with the first label and sum the remaining to create a combined label
            big_label = filtered_labels[0]
            for label in filtered_labels[1:]:
                big_label += label
            
            # If it's a BiHemiLabel (both hemispheres), save left and right separately
            if isinstance(big_label, mne.BiHemiLabel):
                # Save left hemisphere
                if big_label.lh is not None:
                    big_label.lh.save(res_path / subject / f'{label_list_name}-lh.label')
                
                # Save right hemisphere
                if big_label.rh is not None:
                    big_label.rh.save(res_path / subject / f'{label_list_name}-rh.label')
            else:
                # If it's a single hemisphere label, save it directly
                big_label.save(res_path / subject / f'{label_list_name}.label')
                
subjects_dir = FREESURFER_DIR
subject = 'sub01'
parc = 'Schaefer2018_100Parcels_7Networks'
hemi = 'lh'
# Get all cortical labels    
cx_labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, sort=True, verbose=verbose)

aseg_fname = subjects_dir / subject / "mri" / f"{parc}.mgz"
aseg_fname = subjects_dir / subject / "mri" / f"BN_Atlas_subcotex_aseg.mgz"
labels_aseg = get_volume_labels_from_aseg(aseg_fname, return_colors=True)
