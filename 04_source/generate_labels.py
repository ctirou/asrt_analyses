# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import mne
from base import ensure_dir, ensured
from config import *

overwrite = False
verbose = 'error'
jobs = -1

subjects = SUBJS15
epochs_list = EPOCHS
data_path = HOME
subjects_dir = FREESURFER_DIR
res_path = data_path / 'results' / "new_labels"
ensure_dir(res_path)

for subject in subjects:    
    
    n_networks = "7"
    n_parcels = "200"
    parc = f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks"
    hemi = "both"
    networks = NETWORKS[:-3]
    
    res_path = data_path / 'results' / f"networks_{n_parcels}_{n_networks}"
    ensure_dir(res_path / subject)
    
    res_path = ensured(RESULTS_DIR / f"networks_{n_parcels}_{n_networks}" / subject)

    for i, network in enumerate(networks):        
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir, regexp=network, verbose=verbose)        
        # Start with the first label and sum the remaining to create a combined label
        big_label = labels[0]
        for label in labels[1:]:
            big_label += label
        # If it's a BiHemiLabel (both hemispheres), save left and right separately
        if isinstance(big_label, mne.BiHemiLabel):
            # Save left hemisphere
            if big_label.lh is not None:
                big_label.lh.save(res_path / f'{network}-lh.label')
            # Save right hemisphere
            if big_label.rh is not None:
                big_label.rh.save(res_path / f'{network}-rh.label')
        else:
            big_label.save(res_path / f'{network}.label')
            
    # supplementary work                
    networks = ['SomMot', 'DorsAttn', 'Cont']
    sensorimotor = ['precentral', 'postcentral', 'paracentral']
    dorsal_attention = ['superiorparietal', 'caudalmiddlefrontal']
    control = ['rostralmiddlefrontal', 'superiorfrontal', 'parsopercularis', 'parstriangularis', 'supramarginal']
    
    for i, (network_name, network) in enumerate(zip(networks, [sensorimotor, dorsal_attention, control])):
        for region in network:
            labels = mne.read_labels_from_annot(subject=subject, parc="aparc", hemi=hemi, subjects_dir=subjects_dir, regexp=region, verbose=verbose)
                    # Start with the first label and sum the remaining to create a combined label
            big_label = labels[0]
            for label in labels[1:]:
                big_label += label
            # If it's a BiHemiLabel (both hemispheres), save left and right separately
            if isinstance(big_label, mne.BiHemiLabel):
                # Save left hemisphere
                if big_label.lh is not None:
                    big_label.lh.save(res_path / f'{network_name}-{region}-lh.label')
                # Save right hemisphere
                if big_label.rh is not None:
                    big_label.rh.save(res_path / f'{network_name}-{region}-rh.label')
            else:
                big_label.save(res_path / f'{network_name}-{region}.label')

        