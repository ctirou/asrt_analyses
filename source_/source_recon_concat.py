import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc

lock = 'stim'
overwrite = True
verbose = True
jobs = -1

subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR / "concatenated"
ensure_dir(res_path)

volume_src = True

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
    
    # bem model
    bem_fname = os.path.join(res_path, "bem", "%s-bem.fif" % (subject))
    if not op.exists(bem_fname) or overwrite:
        conductivity = (.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(bem_fname, bem, overwrite=True, verbose=verbose)
    
    # cortex source space
    src_fname = op.join(res_path, "src", "%s-src.fif" % subject)
    if volume_src:
        src_fname = src_fname.replace("-src.fif", "+aseg-src.fif")    
    if not op.exists(src_fname) or overwrite:
        src = mne.setup_source_space(subject, spacing='oct6',
                                        subjects_dir=subjects_dir,
                                        add_dist=True,
                                        n_jobs=jobs,
                                        verbose=verbose)
                
        if volume_src:
            # volume source space
            aseg_fname = subjects_dir / subject / 'mri' / 'BN_Atlas_subcortex_aseg.mgz'
            aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname)
            
            # volume_label = ["Left-Hippocampus", "Right-Hippocampus"]
            aseg_src = mne.setup_volume_source_space(
                subject,
                bem=bem_fname,
                mri=aseg_fname,
                volume_label=aseg_labels,
                subjects_dir=subjects_dir,
                n_jobs=jobs,
                verbose=verbose)
            
            src += aseg_src
            
            # # for visualization
            # fig = mne.viz.plot_alignment(
            #     subject=subject,
            #     subjects_dir=subjects_dir,
            #     surfaces="white",
            #     coord_frame="mri",
            #     src=aseg_src)
    
            # mne.viz.set_3d_view(
            #     fig, azimuth=180, elevation=90, distance=0.30, focalpoint=(-0.03, -0.01, 0.03))
                        
        mne.write_source_spaces(src_fname, src, overwrite=True, verbose=verbose)
    
    src = mne.read_source_spaces(src_fname, verbose=verbose)
        
    # read epochs and concatenate    
    epo_dir = data_path / lock
    epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
    all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
    for epoch in all_epo:
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
    if volume_src:
        fwd_fname = fwd_fname.replace("-fwd.fif", "+aseg-fwd.fif")
    if not op.exists(fwd_fname) or overwrite:
        fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=src, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=True)        
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
    
    del src_fname, src, bem_fname
    del epo_dir, epo_fnames, all_epo, epoch
    del trans_fname, fwd_fname, fwd
    gc.collect()