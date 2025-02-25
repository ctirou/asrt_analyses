# recon -all
cd /Users/coum/Desktop/asrt/mri_data
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR="/Users/coum/Desktop/asrt/freesurfer"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Single subject
# recon-all -s sub01 -i sub01.nii -all -qcache
# to be sure be on 
# mne watershed_bem -s fsaverage2 --overwrite
# mne make_scalp_surfaces -s fsaverage2 --force --overwrite

ls *.nii | parallel jobs 15 recon-all -s {.} -i {} -all -qcache

# Parcellation using Schaeffer2018 #
cd /Users/coum/Desktop/asrt/freesurfer
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR="/Users/coum/Desktop/asrt/freesurfer"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

n_parcels=200
n_networks=7

# subjects=("sub01" "sub02" "sub04" "sub07" "sub08" "sub09" "sub10" "sub12" "sub13" "sub14" "sub15")

subject="fsaverage2"

for subject in "${subjects[@]}"; do
    echo "### Processing $subject... ###"
  # In individual surface space
  # Left hemisphere
  mris_ca_label -l $SUBJECTS_DIR/$subject/label/lh.cortex.label \
    $subject lh $SUBJECTS_DIR/$subject/surf/lh.sphere.reg \
    $SUBJECTS_DIR/Schaefer2018/gcs/lh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.gcs \
    $SUBJECTS_DIR/$subject/label/lh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.annot
  # Right hemisphere
  mris_ca_label -l $SUBJECTS_DIR/$subject/label/rh.cortex.label \
    $subject rh $SUBJECTS_DIR/$subject/surf/rh.sphere.reg \
    $SUBJECTS_DIR/Schaefer2018/gcs/rh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.gcs \
    $SUBJECTS_DIR/$subject/label/rh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.annot

  # Visualize (optional)
  # freeview -f $SUBJECTS_DIR/$subject/surf/lh.inflated:annot=$SUBJECTS_DIR/$subject/label/lh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.annot

  # In individual volume space
  mri_aparc2aseg --s $subject --o $SUBJECTS_DIR/$subject/mri/Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.mgz --annot Schaefer2018_${n_parcels}Parcels_${n_networks}Networks
  # Visualize (optional)
  # freeview -v Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.mgz:colormap=lut:lut=$SUBJECTS_DIR/Schaefer2018/lut/Schaefer2018_${n_parcels}Parcels_${n_networks}Networks_order_LUT.txt

  mris_anatomical_stats \
    -f $SUBJECTS_DIR/$subject/stats/lh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.stats \
    -b -a $SUBJECTS_DIR/$subject/label/lh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.annot \
    $subject lh

  mris_anatomical_stats \
    -f $SUBJECTS_DIR/$subject/stats/rh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.stats \
    -b -a $SUBJECTS_DIR/$subject/label/rh.Schaefer2018_${n_parcels}Parcels_${n_networks}Networks.annot \
    $subject rh
    
    echo "### $subject processing completed. ###"

done
echo "################## All subjects processed. ##################"