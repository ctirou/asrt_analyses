### PARCELLATION
cd /Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/mri_data
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR=/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

ls *.nii | parallel jobs 15 recon-all -s {.} -i {} -all -qcache

### SEGMENTATION -- BRAINNETOME ATLAS --
cd /Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/BN_Atlas_freesurfer
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR=/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

subjects=("sub01" "sub02" "sub04" "sub07" "sub08" "sub09" "sub10" "sub12" "sub13" "sub14" "sub15")
for Subject in "${subjects[@]}"; do
    echo "Processing $Subject..."
    mkdir -p ./$Subject
    ## mapping BN_atlas cortex to subjects
    mris_ca_label -l $SUBJECTS_DIR/$Subject/label/lh.cortex.label $Subject lh $SUBJECTS_DIR/$Subject/surf/lh.sphere.reg $SUBJECTS_DIR/lh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot
    mris_ca_label -l $SUBJECTS_DIR/$Subject/label/rh.cortex.label $Subject rh $SUBJECTS_DIR/$Subject/surf/rh.sphere.reg $SUBJECTS_DIR/rh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot
    ### Parcellation Stats
    mris_anatomical_stats -mgz -cortex $SUBJECTS_DIR/$Subject/label/lh.cortex.label -f $SUBJECTS_DIR/$Subject/stats/lh.BN_Atlas.stats -b -a $SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot -c $SUBJECTS_DIR/BN_Atlas_210_LUT.txt $Subject lh white 
    aparcstats2table -s $Subject --hemi lh --parc BN_Atlas --meas thickness --tablefile ./$Subject/lh.thickness.txt
    mris_anatomical_stats -mgz -cortex $SUBJECTS_DIR/$Subject/label/rh.cortex.label -f $SUBJECTS_DIR/$Subject/stats/rh.BN_Atlas.stats -b -a $SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot -c $SUBJECTS_DIR/BN_Atlas_210_LUT.txt $Subject rh white
    aparcstats2table -s $Subject --hemi rh --parc BN_Atlas --meas thickness --tablefile ./$Subject/rh.thickness.txt
    ### mapping BN_atlas subcortex to subjects 
    mri_ca_label $SUBJECTS_DIR/$Subject/mri/brain.mgz $SUBJECTS_DIR/$Subject/mri/transforms/talairach.m3z $SUBJECTS_DIR/BN_Atlas_subcortex.gca $SUBJECTS_DIR/$Subject/mri/BN_Atlas_subcortex_aseg.mgz
    ### Segmentation stats
    mri_segstats --seg $SUBJECTS_DIR/$Subject/mri/BN_Atlas_subcotex.mgz --ctab $SUBJECTS_DIR/BN_Atlas_246_LUT.txt --excludeid 0 --sum $SUBJECTS_DIR/$Subject/stats/BN_Atlas_subcortex_aseg.stats
    echo "$Subject processing completed."
done
echo "All subjects processed."