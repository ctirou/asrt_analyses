import os.path as op
import mne
from base import ensure_dir, remove_common_vertex
from config import *
import pandas as pd

overwrite = False
verbose = 'error'
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

save_label = True

for subject in subjects:
    ensure_dir(res_path / subject)
    
    # Get all cortical labels
    cx_labels = mne.read_labels_from_annot(subject=subject, parc="aparc", hemi="both", subjects_dir=subjects_dir, verbose=verbose)
    
    # Get all subcortical labels
    aseg_labels = list()
    for hemi in ['lh', 'rh', 'others']:
        # Create mixed source space
        vol_src_fname = op.join(RESULTS_DIR, "src", f"{subject}-htc-vol-src.fif")
        vol_src = mne.read_source_spaces(vol_src_fname, verbose=verbose)    
        aseg_labels.extend(mne.get_volume_labels_from_src(vol_src, subject, subjects_dir))

    # Combine cortical and subcortical labels 
    labels = cx_labels + aseg_labels
    
    # Iterate over region groups
    for i, (label_list, label_list_name) in enumerate(zip(new_labels, new_labels_names)):
        print(f"### Processing {subject} - {label_list_name} ###")
        
        # Filter the labels based on the current list
        filtered_labels = [label for label in labels if label.name in label_list and label.name not in BAD_VOLUME_LABELS]
        
        if filtered_labels:
            # Remove common vertices for each combination of filtered_labels
            corrected_labels = []
            for i, label1 in enumerate(filtered_labels):
                for label2 in filtered_labels[i+1:]:
                    # if label2.name[0].isupper(): # subcortical labels have their names starting with uppercase
                    label2 = remove_common_vertex(label1, label2)
                    print(f"Removed common vertices between {label1.name} and {label2.name}")
                corrected_labels.append(label1)
            
            # if save_label:
            #     # Start with the first label and sum the remaining to create a combined label
            #     big_label = corrected_labels[0]
            #     for label in corrected_labels[1:]:
            #         big_label += label
            #     # If it's a BiHemiLabel (both hemispheres), save left and right separately
            #     if isinstance(big_label, mne.BiHemiLabel):
            #         # Save left hemisphere
            #         if big_label.lh is not None:
            #             big_label.lh.save(res_path / subject / f'{label_list_name}-lh.label')
            #         # Save right hemisphere
            #         if big_label.rh is not None:
            #             big_label.rh.save(res_path / subject / f'{label_list_name}-rh.label')
            #     else:
            #         big_label.save(res_path / subject / f'{label_list_name}.label')
            #         # If it's a single hemisphere label, save it directly
            
            # label_list_name = 'frontal_lobe'
            parc = "aparc." + label_list_name
            hemi = 'lh' if 'left' in label_list_name else 'rh' if 'right' in label_list_name else 'both'
            mne.write_labels_to_annot(labels=corrected_labels, subject=subject, parc=parc, overwrite=True, subjects_dir=subjects_dir, hemi=hemi, sort=True, verbose=verbose)
            
            # read_labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, verbose=verbose)
            
            # Visualize on a brain plot
            brain = mne.viz.Brain(subject, hemi='both', subjects_dir=subjects_dir,
                                cortex='low_contrast', theme='dark', alpha=.5)

            for label in filtered_labels:
                brain.add_label(label, borders=True)
                
            # Add labels and store their colors for the legend
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            legend_elements = []
            for label in filtered_labels:
                brain.add_label(label, borders=False)
                legend_elements.append(Patch(facecolor=label.color, label=label.name))
            # Create a legend using matplotlib
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.legend(handles=legend_elements, loc='center')
            ax.axis('off')
            plt.show()
            
            # Load aseg.mgz for subcortical visualization
            aseg_path = subjects_dir / subject / 'mri' / "aseg.mgz"
            aseg_data = mne.read_source_spaces(aseg_path)

            # Visualize aseg volume
            brain = mne.viz.Brain(subject, hemi='both', subjects_dir=subjects_dir,
                                cortex='low_contrast', background='white', volume=aseg_path)
            

cuneus_lh = labels[6]
hpc_lh = labels[76]
# Check intersection
intersection = set(cuneus_lh.vertices).intersection(hpc_lh.vertices)
print(f"Intersection between cuneus_lh and hippocampus_lh: {intersection}")
# Remove intersection vertices from hpc_lh

new_hpc_lh = remove_common_vertex(cuneus_lh, hpc_lh)

for subject in subjects:    
    
    n_networks = "7"
    n_parcels = "200"
    parc = f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks"
    hemi = "both"
    # network_names = schaefer_7 if n_networks == "7" else schaefer_17    
    network_names = NETWORKS
    
    res_path = data_path / 'results' / f"networks_{n_parcels}_{n_networks}"
    ensure_dir(res_path / subject)
    
    res_path = FREESURFER_DIR / subject / 'label' / 'n7'
    ensure_dir(res_path)

    for i, network in enumerate(network_names):
        
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, regexp=network, verbose=verbose)        
        # print(f"Processing {subject} - {network} - {len(labels)}")
        
        # Start with the first label and sum the remaining to create a combined label
        big_label = labels[0]
        for label in labels[1:]:
            big_label += label
        # If it's a BiHemiLabel (both hemispheres), save left and right separately
        if isinstance(big_label, mne.BiHemiLabel):
            # Save left hemisphere
            if big_label.lh is not None:
                big_label.lh.save(res_path / f'lh.{network}.label')
            # Save right hemisphere
            if big_label.rh is not None:
                big_label.rh.save(res_path / f'rh.{network}.label')
        else:
            big_label.save(res_path / f'{network}.label')

        # parc_fname = f"Shaefer2018_{n_parcels}_{n_networks}.{network}"
        # mne.write_labels_to_annot(labels=labels, subject=subject, parc=parc_fname, subjects_dir=subjects_dir, hemi=hemi, sort=True, overwrite=True, verbose=verbose)
        
        
# Get aseg labels
subject = "sub01"
mgz_fname = op.join(FREESURFER_DIR, subject, 'mri', 'BN_Atlas.mgz')
labels = mne.get_volume_labels_from_aseg(mgz_fname=mgz_fname)