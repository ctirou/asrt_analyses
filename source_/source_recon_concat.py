import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc

lock = 'button'
overwrite = True
jobs = -1

subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR / "concatenated"
ensure_dir(res_path)

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
    # read epochs and concatenate    
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo: # see mne.preprocessing.maxwell_filter to realign the runs to a common head position. On raw data.
        epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epo)
    # create trans file
    trans_fname = os.path.join(res_path, "trans", lock, "%s-trans.fif" % (subject))
    if not op.exists(trans_fname) or overwrite:
        coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
        coreg.fit_fiducials(verbose=True)
        coreg.fit_icp(n_iterations=6, verbose=True)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=100, verbose=True)
        mne.write_trans(trans_fname, coreg.trans, overwrite=overwrite)
    fwd_fname = op.join(res_path, "fwd", lock, "%s-fwd.fif" % (subject))
    if not op.exists(fwd_fname) or overwrite:
        fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=src, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=True)
        mne.write_forward_solution(fwd_fname, fwd, overwrite=overwrite)
    
    del src_fname, src, bem_fname
    del epo_dir, epo_fnames, all_epo, epoch
    del trans_fname, fwd_fname, fwd
    gc.collect()