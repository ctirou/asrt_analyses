import os
import mne
from config import *
# -----------------------------
# Step 1: Setup paths and names
# -----------------------------
subject = 'sub03'
template_subject = 'fsaverage2'

# Use your freesurfer subjects directory
subjects_dir = FREESURFER_DIR

# -----------------------------
# Step 2: Copy fsaverage to make a substitute MRI for sub03
# -----------------------------
fsavg_path = os.path.join(subjects_dir, template_subject)
subj_path = os.path.join(subjects_dir, subject)

# Copy fsaverage if sub03 directory doesn't exist yet
if not os.path.exists(subj_path):
    print(f"Creating subject directory for {subject}...")
    os.system(f'cp -r {fsavg_path} {subj_path}')

# -----------------------------
# Step 3: Load MEG data and launch coregistration GUI
# -----------------------------
# Load MEG data
data_path = DATA_DIR
for epoch_num in range(5):
    epoch_fname = data_path / "stim" / f"{subject}-{epoch_num}-epo.fif"
    epoch = mne.read_epochs(epoch_fname, preload=True, verbose=True)

    # Launch coreg GUI to align head shape with fsaverage
    mne.gui.coregistration(subject=template_subject, subjects_dir=subjects_dir, inst=epoch_fname)

fiducials_fname = subjects_dir / subject / 'bem' / f'{subject}-fiducials.fif'
fiducials = mne.io.read_fiducials(fiducials_fname, verbose=True)