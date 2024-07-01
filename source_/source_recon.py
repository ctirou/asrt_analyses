import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc
import sys

overwrite = True
verbose = True
jobs = -1

subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR
ensure_dir(res_path)
lock = "stim"
# lock = sys.argv[1]

# create results directory
folders = ["bem", "trans", "fwd"]
for f in folders:
    if f in folders[-2:]:
        path = op.join(res_path, f, lock)
        ensure_dir(path)
    else:
        ensure_dir(os.path.join(res_path, f))

for subject in subjects:
    # bem model
    bem_fname = op.join(res_path, "bem", "%s-bem-sol.fif" % (subject))
    model_fname = op.join(res_path, "bem", "%s-bem.fif" % (subject))
    if not op.exists(bem_fname) or False:
        conductivity = (.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir)
        mne.write_bem_surfaces(model_fname, model, overwrite=True, verbose=verbose)
        
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(bem_fname, bem, overwrite=True, verbose=verbose)
    
    # cortex source space
    src = mne.setup_source_space(subject, spacing='oct6',
                                    subjects_dir=subjects_dir,
                                    add_dist=True,
                                    n_jobs=jobs,
                                    verbose=verbose)
                                
    # volume source space
    ## Brainnetome atlas -- does not work for now
    # aseg_fname = subjects_dir / subject / 'mri' / 'BN_Atlas_subcotex_aseg.mgz'
    # lut_file = "/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer/BN_Atlas_246_LUT.txt"
    # lut_lab = mne.read_freesurfer_lut(lut_file)
    # aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname, atlas_ids=lut_lab[0])
    # volume_labels = ['rHipp_L', 'rHipp_R', 'cHipp_L', 'cHipp_R']
    
    ## Freesurfer default aseg atlas
    aseg_fname = subjects_dir / subject / 'mri' / 'aseg.mgz'
    aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname)
    
    vol_labels_lh = [l for l in aseg_labels if l.startswith('Left')]
    vol_labels_rh = [l for l in aseg_labels if l.startswith('Right')]
    vol_labels_others = [l for l in aseg_labels if not l.startswith(('Left', 'Right'))]
            
    vol_src_lh = mne.setup_volume_source_space(
        subject,
        bem=model_fname,
        mri="aseg.mgz",
        volume_label=vol_labels_lh,
        subjects_dir=subjects_dir,
        n_jobs=jobs,
        verbose=verbose)

    vol_src_rh = mne.setup_volume_source_space(
        subject,
        bem=model_fname,
        mri="aseg.mgz",
        volume_label=vol_labels_rh,
        subjects_dir=subjects_dir,
        n_jobs=jobs,
        verbose=verbose)

    vol_src_others = mne.setup_volume_source_space(
        subject,
        bem=model_fname,
        mri="aseg.mgz",
        volume_label=vol_labels_others,
        subjects_dir=subjects_dir,
        n_jobs=jobs,
        verbose=verbose)
    
    mixed_src_lh = src + vol_src_lh
    mixed_src_rh = src + vol_src_rh
    mixed_src_others = src + vol_src_others
    
    del src, vol_src_lh, vol_src_rh
    gc.collect()
    
    all_epochs = list()
    for epoch_num in range(5):
    
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, verbose="error", preload=False)
        all_epochs.append(epoch)
        
        # create trans file
        trans_fname = os.path.join(res_path, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
        # if not op.exists(trans_fname) or overwrite:
        if not op.exists(trans_fname) or False:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=6, verbose=True)
            coreg.omit_head_shape_points(distance=5.0 / 1000)
            coreg.fit_icp(n_iterations=100, verbose=True)
            mne.write_trans(trans_fname, coreg.trans, overwrite=True)
                
        mixed_fwd_fname_lh = op.join(res_path, "fwd", lock, "%s-%i-mixed-lh-fwd.fif" % (subject, epoch_num))
        if not op.exists(mixed_fwd_fname_lh) or overwrite:
            mixed_fwd_lh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=mixed_src_lh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(mixed_fwd_fname_lh, mixed_fwd_lh, overwrite=True, verbose=verbose)

        mixed_fwd_fname_rh = op.join(res_path, "fwd", lock, "%s-%i-mixed-rh-fwd.fif" % (subject, epoch_num))
        if not op.exists(mixed_fwd_fname_rh) or overwrite:
            mixed_fwd_rh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=mixed_src_rh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(mixed_fwd_fname_rh, mixed_fwd_rh, overwrite=True, verbose=verbose)

        mixed_fwd_fname_others = op.join(res_path, "fwd", lock, "%s-%i-mixed-others-fwd.fif" % (subject, epoch_num))
        if not op.exists(mixed_fwd_fname_rh) or overwrite:
            mixed_fwd_others = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=mixed_src_others, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(mixed_fwd_fname_others, mixed_fwd_others, overwrite=True, verbose=verbose)
        
        del epoch
        del mixed_fwd_lh, mixed_fwd_rh, mixed_fwd_others
        gc.collect()
        
    for epoch in all_epochs:
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epoch = mne.concatenate_epochs(all_epochs)
    
    del all_epochs
    gc.collect()
    
    # create trans file
    trans_fname = os.path.join(res_path, "trans", lock, "%s-all-trans.fif" % (subject))
    if not op.exists(trans_fname) or overwrite:
        coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
        coreg.fit_fiducials(verbose=True)
        coreg.fit_icp(n_iterations=6, verbose=True)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=100, verbose=True)
        mne.write_trans(trans_fname, coreg.trans, overwrite=True)
            
    mixed_fwd_fname_lh = op.join(res_path, "fwd", lock, "%s-all-mixed-lh-fwd.fif" % (subject))
    if not op.exists(mixed_fwd_fname_lh) or overwrite:
        mixed_fwd_lh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=mixed_src_lh, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=True)        
        mne.write_forward_solution(mixed_fwd_fname_lh, mixed_fwd_lh, overwrite=True, verbose=verbose)

    mixed_fwd_fname_rh = op.join(res_path, "fwd", lock, "%s-all-mixed-rh-fwd.fif" % (subject))
    if not op.exists(mixed_fwd_fname_rh) or overwrite:
        mixed_fwd_rh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=mixed_src_rh, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=True)        
        mne.write_forward_solution(mixed_fwd_fname_rh, mixed_fwd_rh, overwrite=True, verbose=verbose)

    mixed_fwd_fname_others = op.join(res_path, "fwd", lock, "%s-all-mixed-others-fwd.fif" % (subject))
    if not op.exists(mixed_fwd_fname_rh) or overwrite:
        mixed_fwd_others = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=mixed_src_others, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=True)        
        mne.write_forward_solution(mixed_fwd_fname_others, mixed_fwd_others, overwrite=True, verbose=verbose)

    del epoch
    del mixed_src_lh, mixed_src_rh, mixed_src_others
    del mixed_fwd_lh, mixed_fwd_rh, mixed_fwd_others
    gc.collect()
