import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc
import sys

subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
# data_path = TIMEG_DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR

lock = 'stim'
jobs = -1

overwrite = True
verbose = True

# create results directories
folders = ["src", "bem", "trans", "fwd"]
for lockf in ['stim', 'button']:
    for f in folders:
        if f in folders[-2:]:
            path = op.join(res_path, f, lockf)
            ensure_dir(path)
        else:
            ensure_dir(os.path.join(res_path, f))

# best_labels = [('Left-' + l.replace('-lh', '')) if '-lh' in l else ('Right-' + l.replace('-rh', '')) if '-rh' in l else l for l in VOLUME_LABELS]    

def process_subject(subject, lock, jobs):
    # bem model
    bem_fname = op.join(res_path, "bem", "%s-bem-sol.fif" % (subject))
    model_fname = op.join(res_path, "bem", "%s-bem.fif" % (subject))
    if not op.exists(bem_fname) or overwrite:
        conductivity = (.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir, verbose=verbose)
        mne.write_bem_surfaces(model_fname, model, overwrite=True, verbose=verbose)
        
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(bem_fname, bem, overwrite=True, verbose=verbose)
    
    # cortex source space
    src_fname = op.join(res_path, "src", "%s-src.fif" % (subject))
    if not op.exists(src_fname) or overwrite:
        src = mne.setup_source_space(subject, spacing='oct6',
                                        subjects_dir=subjects_dir,
                                        add_dist=True,
                                        n_jobs=jobs,
                                        verbose=verbose)
        mne.write_source_spaces(src_fname, src, overwrite=True, verbose=verbose)
    else:
        src = mne.read_source_spaces(src_fname, verbose=verbose)
                                
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
    
    # vol_labels_lh = [l for l in aseg_labels if l.startswith('Left')]
    # vol_labels_rh = [l for l in aseg_labels if l.startswith('Right')]
    # vol_labels_others = [l for l in aseg_labels if not l.startswith(('Left', 'Right'))]
    
    hipp_labels = [l for l in aseg_labels if 'Hipp' in l]
    thal_labels = [l for l in aseg_labels if 'Thal' in l]
    hipp_thal_labels = hipp_labels + thal_labels
    
    # vol_src_hipp_fname = op.join(res_path, "src", "%s-hipp-vol-src.fif" % (subject))
    # if not op.exists(vol_src_hipp_fname) or overwrite:
    #     vol_src_hipp = mne.setup_volume_source_space(
    #         subject,
    #         bem=model_fname,
    #         mri="aseg.mgz",
    #         volume_label=hipp_labels,
    #         subjects_dir=subjects_dir,
    #         n_jobs=jobs,
    #         verbose=verbose)
    #     mne.write_source_spaces(vol_src_hipp_fname, vol_src_hipp, overwrite=True)
    
    # vol_src_thal_fname = op.join(res_path, "src", "%s-thal-vol-src.fif" % (subject))
    # if not op.exists(vol_src_thal_fname) or overwrite:
    #     vol_src_thal = mne.setup_volume_source_space(
    #         subject,
    #         bem=model_fname,
    #         mri="aseg.mgz",
    #         volume_label=thal_labels,
    #         subjects_dir=subjects_dir,
    #         n_jobs=jobs,
    #         verbose=verbose)
    #     mne.write_source_spaces(vol_src_thal_fname, vol_src_thal, overwrite=True)
        
    vol_src_hipp_thal_fname = op.join(res_path, "src", "%s-hipp-thal-vol-src.fif" % (subject))
    if not op.exists(vol_src_hipp_thal_fname) or overwrite:
        vol_src_hipp_thal = mne.setup_volume_source_space(
            subject,
            bem=model_fname,
            mri="aseg.mgz",
            volume_label=hipp_thal_labels,
            subjects_dir=subjects_dir,
            n_jobs=jobs,
            verbose=verbose)
        mne.write_source_spaces(vol_src_hipp_thal_fname, vol_src_hipp_thal, overwrite=True, verbose=verbose)
    else:
        vol_src_hipp_thal = mne.read_source_spaces(vol_src_hipp_thal_fname, verbose=verbose)
    
    # vol_src_lh_fname = op.join(res_path, "src", "%s-lh-vol-src.fif" % (subject))
    # if not op.exists(vol_src_lh_fname) or overwrite:
    #     vol_src_lh = mne.setup_volume_source_space(
    #         subject,
    #         bem=model_fname,
    #         mri="aseg.mgz", # try with T1.mgz
    #         volume_label=vol_labels_lh,
    #         subjects_dir=subjects_dir,
    #         n_jobs=jobs,
    #         verbose=verbose)
    #     mne.write_source_spaces(vol_src_lh_fname, vol_src_lh, overwrite=True)

    # vol_src_rh_fname = op.join(res_path, "src", "%s-rh-vol-src.fif" % (subject))
    # if not op.exists(vol_src_rh_fname) or overwrite:
    #     vol_src_rh = mne.setup_volume_source_space(
    #         subject,
    #         bem=model_fname,
    #         mri="aseg.mgz",
    #         volume_label=vol_labels_rh,
    #         subjects_dir=subjects_dir,
    #         n_jobs=jobs,
    #         verbose=verbose)
    #     mne.write_source_spaces(vol_src_rh_fname, vol_src_rh, overwrite=True)

    # vol_src_others_fname = op.join(res_path, "src", "%s-others-vol-src.fif" % (subject))
    # if not op.exists(vol_src_others_fname) or overwrite:
    #     vol_src_others = mne.setup_volume_source_space(
    #         subject,
    #         bem=model_fname,
    #         mri="aseg.mgz",
    #         volume_label=vol_labels_others,
    #         subjects_dir=subjects_dir,
    #         n_jobs=jobs,
    #         verbose=verbose)
    #     mne.write_source_spaces(vol_src_others_fname, vol_src_others, overwrite=True)

    # vol_src_fname = op.join(res_path, "src", "%s-all-vol-src.fif" % (subject))
    # # if not op.exists(vol_src_others_fname) or overwrite:
    # vol_src = mne.setup_volume_source_space(
    #     subject,
    #     bem=model_fname,
    #     mri="aseg.mgz",
    #     volume_label=best_labels,
    #     subjects_dir=subjects_dir,
    #     n_jobs=jobs,
    #     verbose=verbose)
    # mne.write_source_spaces(vol_src_fname, vol_src, overwrite=True)
    
    # # Load the source space
    # src = mne.read_source_spaces(vol_src_fname)

    # # Check the structure of the source space
    # print(src)

    # # Plot the volume source spaces
    # mne.viz.plot_volume_source_space(src, src_color='red', show=True)
    
    # vol_src.plot(subjects_dir=subjects_dir)
    
    # labels = mne.get_volume_labels_from_src(src, subject, subjects_dir)
    mixed = src + vol_src_hipp_thal
    # # visualize source spaces
    # fig = mne.viz.plot_alignment(
    # subject=subject,
    # subjects_dir=subjects_dir,
    # surfaces="white",
    # coord_frame="mri",
    # src=mixed)
    # mne.viz.set_3d_view(fig, azimuth=180, elevation=90, distance=0.30, focalpoint=(-0.03, -0.01, 0.03))
    
    # del src, vol_src_lh, vol_src_rh, vol_src_others, vol_src
    # gc.collect()
    
    # all_epochs = list()
    
    for epoch_num in range(5):
        print(f"Processing {subject} {lock} {epoch_num}...")
        
        epoch_fname = op.join(data_path, lock, f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, preload=False, verbose=verbose)
        all_epochs.append(epoch)
        
        # create trans file
        trans_fname = os.path.join(res_path, "trans", lock, "%s-%i-trans.fif" % (subject, epoch_num))
        if not op.exists(trans_fname) or overwrite:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=verbose)
            coreg.fit_icp(n_iterations=6, verbose=verbose)
            coreg.omit_head_shape_points(distance=5.0/1000)
            coreg.fit_icp(n_iterations=100, verbose=verbose)
            mne.write_trans(trans_fname, coreg.trans, overwrite=True, verbose=verbose)
        # compute forward solution
        fwd_fname = op.join(res_path, "fwd", lock, f"{subject}-hipp-thal-{epoch_num}-fwd.fif")
        if not op.exists(fwd_fname) or overwrite:
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=mixed, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=verbose)
            mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=verbose)
            del fwd
        
        del epoch
        gc.collect()
    
    print(f"Processing {subject} {lock} all...")    
    for epoch in all_epochs:
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = mne.concatenate_epochs(all_epochs, verbose=verbose)
    
    del all_epochs
    gc.collect()
    
    # create trans file
    trans_fname = op.join(res_path, "trans", lock, "%s-all-trans.fif" % (subject))
    if not op.exists(trans_fname) or overwrite:
        coreg = mne.coreg.Coregistration(epochs.info, subject, subjects_dir)
        coreg.fit_fiducials(verbose=verbose)
        coreg.fit_icp(n_iterations=6, verbose=verbose)
        coreg.omit_head_shape_points(distance=5.0/1000)
        coreg.fit_icp(n_iterations=100, verbose=verbose)
        mne.write_trans(trans_fname, coreg.trans, overwrite=True, verbose=verbose)
    
    fwd_fname = op.join(res_path, "fwd", lock, f"{subject}-hipp-thal-all-fwd.fif")
    if not op.exists(fwd_fname) or overwrite:
        fwd = mne.make_forward_solution(epochs.info, trans=trans_fname,
                                    src=src, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=verbose)
        del fwd
            
    del epochs, src, vol_src_hipp_thal, mixed
    gc.collect()
        
for lock in ['stim', 'button']:
    for subject in subjects:
        process_subject(subject, lock, jobs=-1)