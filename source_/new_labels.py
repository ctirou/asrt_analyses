import os
import os.path as op
import mne
from base import ensure_dir
from config import *

overwrite = False
verbose = True
jobs = -1

subjects = SUBJS
epochs_list = EPOCHS
data_path = HOME
subjects_dir = FREESURFER_DIR
res_path = data_path / 'results' / "new_labels"
ensure_dir(res_path)

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
    
    # get all cortical labels
    cx_labels = mne.read_labels_from_annot(subject=subject, parc="aparc", hemi="both", subjects_dir=subjects_dir, verbose=verbose)
    # get all subcortical labels
    aseg_labels = list()
    for hemi in ['lh', 'rh', 'others']:
        # create mixed source space
        vol_src_fname = op.join(RESULTS_DIR, "src", "%s-%s-vol-src.fif" % (subject, hemi))
        vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)    
        aseg_labels.extend(mne.get_volume_labels_from_src(vol_src, subject, subjects_dir))

    labels = cx_labels + aseg_labels
    
    for i, (label_list, label_list_name) in enumerate(zip(new_labels, new_labels_names)):
        
        filtered_labels = [label for label in labels if label.name in label_list and label.name not in BAD_VOLUME_LABELS]
        
        if filtered_labels:
            big_label = filtered_labels[0]
            for label in filtered_labels[1:]:
                big_label += label
            
            # Handle if big_label is a BiHemiLabel
            if isinstance(big_label, mne.BiHemiLabel):
                # Save left hemisphere
                if big_label.lh is not None:
                    big_label.lh.save(res_path / subject / f'{label_list_name}-lh.label')
                
                # Save right hemisphere
                if big_label.rh is not None:
                    big_label.rh.save(res_path / subject / f'{label_list_name}-rh.label')
            else:
                # Save as a single label if it's not BiHemiLabel
                big_label.save(res_path / subject / f'{label_list_name}.label')