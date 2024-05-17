import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from base import *
from config import *

lock = 'button'
movie = False
overwrite = True

subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR

jobs = 10

# create results directory
folders = ["stcs", "bem", "src", "trans", "fwd"]
for f in folders:
    if f == "stcs":
        path = os.path.join(res_path, "stcs", lock)
        ensure_dir(path)
    elif f in folders[-2:]:
        path = op.join(res_path, f, lock)
        ensure_dir(path)
    else:
        ensure_dir(os.path.join(res_path, f))

for subject in subjects:
    # source space
    src_fname = op.join(res_path, "src", "%s-src.fif" % subject)
    if not op.exists(src_fname) or False:
        src = mne.setup_source_space(subject, spacing='oct6',
                                        subjects_dir=subjects_dir,
                                        add_dist=True,
                                        n_jobs=jobs)
        src.save(src_fname, overwrite=overwrite)
    src = mne.read_source_spaces(src_fname)
    # bem model
    bem_fname = os.path.join(res_path, "bem", "%s-bem.fif" % (subject))
    if not op.exists(bem_fname) or False:
        conductivity = (.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(bem_fname, bem, overwrite=overwrite)
    # loop across all epochs
    for epoch_num, epo in enumerate(epochs_list):
        # read behav and epoch files
        behav_fname = op.join(data_path, "behav/%s_%s.pkl" % (subject, epoch_num))
        behav = pd.read_pickle(behav_fname)
        if lock == 'stim':
            epoch_fname = op.join(data_path, "%s/%s_%s_s-epo.fif" % (lock, subject, epoch_num))
        else:
            epoch_fname = op.join(data_path, "%s/%s_%s_b-epo.fif" % (lock, subject, epoch_num))
        epoch = mne.read_epochs(epoch_fname)
        # open and read baseline epoch for button press epochs correction
        if lock == 'button': 
            epoch_bsl_fname = op.join(data_path, "bsl/%s_%s_bl-epo.fif" % (subject, epoch_num))
            epoch_bsl = mne.read_epochs(epoch_bsl_fname)                                                        
        # create trans file
        trans_fname = os.path.join(res_path, "trans", lock, "%s-%s-trans.fif" % (subject, epoch_num))
        if not op.exists(trans_fname) or overwrite:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=6, verbose=True)
            coreg.omit_head_shape_points(distance=5.0 / 1000)
            coreg.fit_icp(n_iterations=100, verbose=True)
            mne.write_trans(trans_fname, coreg.trans, overwrite=overwrite)
        fwd_fname = op.join(res_path, "fwd", lock, "%s-%s-fwd.fif" % (subject, epoch_num))
        if not op.exists(fwd_fname):
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=src, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)
            mne.write_forward_solution(fwd_fname, fwd, overwrite=True)