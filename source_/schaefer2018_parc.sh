# recon -all
cd /Users/coum/Desktop/asrt/mri_data
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR="/Users/coum/Desktop/asrt/freesurfer"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

ls *.nii | parallel jobs 15 recon-all -s {.} -i {} -all -qcache

# Parcellation using Schaeffer2018 #
cd /Users/coum/Desktop/asrt/freesurfer
export FREESURFER_HOME="/Applications/freesurfer/7.3.2"
export SUBJECTS_DIR="/Users/coum/Desktop/asrt/freesurfer"
source $FREESURFER_HOME/SetUpFreeSurfer.sh

subject = 'sub01'

# In individual surface space
# Left hemisphere
mris_ca_label -l $SUBJECTS_DIR/$subject/label/lh.cortex.label \
  $subject lh $SUBJECTS_DIR/$subject/surf/lh.sphere.reg \
  <gcs_file_dir>/lh.Schaefer2018_<N>Parcels_<7/17>Networks.gcs \
  $SUBJECTS_DIR/$subject/label/lh.Schaefer2018_<N>Parcels_<7/17>Networks_order.annot
# Right hemisphere
mris_ca_label -l $SUBJECTS_DIR/$subject/label/rh.cortex.label \
  $subject rh $SUBJECTS_DIR/$subject/surf/rh.sphere.reg \
  <gcs_file_dir>/rh.Schaefer2018_<N>Parcels_<7/17>Networks.gcs \
  $SUBJECTS_DIR/$subject/label/rh.Schaefer2018_<N>Parcels_<7/17>Networks_order.annot
# Visualize (optional)
freeview -f $SUBJECTS_DIR/<subject_name>/surf/<?h>.inflated:annot=$SUBJECTS_DIR/<subject_name>/label/<?h>.Schaefer2018_<N>Parcels_<7/17>Networks_order.annot

# In individual volume space
mri_aparc2aseg --s $subject --o <output>.mgz --annot Schaefer2018_<N>Parcels_<7/17>Networks_order
# Visualize (optional)
freeview -v <output>.mgz:colormap=lut:lut=<lookup_table>

mris_anatomical_stats <subject_name> lh \
  -f $SUBJECTS_DIR/<subject_name>/stats/lh.Schaefer2018_<N>Parcels_<7/17>Networks.stats \
  -b -a $SUBJECTS_DIR/<subject_name>/label/lh.Schaefer2018_<N>Parcels_<7/17>Networks.annot

mris_anatomical_stats <subject_name> rh \
  -f $SUBJECTS_DIR/<subject_name>/stats/rh.Schaefer2018_<N>Parcels_<7/17>Networks.stats \
  -b -a $SUBJECTS_DIR/<subject_name>/label/rh.Schaefer2018_<N>Parcels_<7/17>Networks.annot