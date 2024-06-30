import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc

overwrite = True
verbose = True
jobs = -1

locks = ['stim', 'button']
subjects = SUBJS
epochs_list = EPOCHS
data_path = DATA_DIR
subjects_dir = FREESURFER_DIR
res_path = RESULTS_DIR / "concatenated"
ensure_dir(res_path)

for lock in locks:

    # create results directory
    folders = ["bem", "src", "trans", "fwd"]
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
        src_fname = op.join(res_path, "src", "%s-src.fif" % subject)
        if not op.exists(src_fname) or overwrite:
            src = mne.setup_source_space(subject, spacing='oct6',
                                            subjects_dir=subjects_dir,
                                            add_dist=True,
                                            n_jobs=jobs,
                                            verbose=verbose)
                                    
        # volume source space
        vol_src_fname = op.join(res_path, "src", "%s-vol-src.fif" % subject)
        if not op.exists(vol_src_fname) or overwrite:
            ## Brainnetome atlas -- does not work for now
            # aseg_fname = subjects_dir / subject / 'mri' / 'BN_Atlas_subcotex_aseg.mgz'
            # lut_file = "/Users/coum/Library/CloudStorage/OneDrive-etu.univ-lyon1.fr/asrt/freesurfer/BN_Atlas_246_LUT.txt"
            # lut_lab = mne.read_freesurfer_lut(lut_file)
            # aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname, atlas_ids=lut_lab[0])
            # volume_labels = ['rHipp_L', 'rHipp_R', 'cHipp_L', 'cHipp_R']
            
            ## Freesurfer default aseg atlas
            aseg_fname = subjects_dir / subject / 'mri' / 'aseg.mgz'
            aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname)
            
            vol_labels_lh = [l for l in aseg_labels if not l.startswith(('Right', 'Unknown'))]
            vol_labels_rh = [l for l in aseg_labels if l.startswith('Right')]
                    
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
            
        mixed_src_lh = src + vol_src_lh
        mixed_src_rh = src + vol_src_rh
        
        # read epochs and concatenate    
        epo_dir = data_path / lock
        epo_fnames = [epo_dir / f'{f}' for f in sorted(os.listdir(epo_dir)) if '.fif' in f and subject in f]
        all_epo = [mne.read_epochs(fname, preload=True, verbose="error") for fname in epo_fnames]
        for epoch in all_epo:
            epoch.info['dev_head_t'] = all_epo[0].info['dev_head_t']
        epoch = mne.concatenate_epochs(all_epo)
        
        # create trans file
        trans_fname = os.path.join(res_path, "trans", lock, "%s-trans.fif" % (subject))
        # if not op.exists(trans_fname) or overwrite:
        if not op.exists(trans_fname) or False:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=6, verbose=True)
            coreg.omit_head_shape_points(distance=5.0 / 1000)
            coreg.fit_icp(n_iterations=100, verbose=True)
            mne.write_trans(trans_fname, coreg.trans, overwrite=True)
        
        # compute forward solution
        # fwd_fname = op.join(res_path, "fwd", lock, "%s-fwd.fif" % (subject))
        # if not op.exists(fwd_fname) or overwrite:
        #     fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
        #                                     src=src, bem=bem_fname,
        #                                     meg=True, eeg=False,
        #                                     mindist=5.0,
        #                                     n_jobs=jobs,
        #                                     verbose=True)        
        #     mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
        
        vol_fwd_fname_lh = op.join(res_path, "fwd", lock, "%s-vol-lh-fwd.fif" % (subject))
        if not op.exists(vol_fwd_fname_lh) or overwrite:
            vol_fwd_lh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=vol_src_lh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(vol_fwd_fname_lh, vol_fwd_lh, overwrite=True, verbose=verbose)

        vol_fwd_fname_rh = op.join(res_path, "fwd", lock, "%s-vol-rh-fwd.fif" % (subject))
        if not op.exists(vol_fwd_fname_rh) or overwrite:
            vol_fwd_rh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=vol_src_rh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(vol_fwd_fname_rh, vol_fwd_rh, overwrite=True, verbose=verbose)

        mixed_fwd_fname_lh = op.join(res_path, "fwd", lock, "%s-mixed-lh-fwd.fif" % (subject))
        if not op.exists(mixed_fwd_fname_lh) or overwrite:
            mixed_fwd_lh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=mixed_src_lh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(mixed_fwd_fname_lh, mixed_fwd_lh, overwrite=True, verbose=verbose)

        mixed_fwd_fname_rh = op.join(res_path, "fwd", lock, "%s-mixed-rh-fwd.fif" % (subject))
        if not op.exists(mixed_fwd_fname_rh) or overwrite:
            mixed_fwd_rh = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                            src=mixed_src_rh, bem=bem_fname,
                                            meg=True, eeg=False,
                                            mindist=5.0,
                                            n_jobs=jobs,
                                            verbose=True)        
            mne.write_forward_solution(mixed_fwd_fname_rh, mixed_fwd_rh, overwrite=True, verbose=verbose)
            
        del src_fname, src, vol_src_lh, vol_src_rh, mixed_src_lh, mixed_src_rh, bem_fname
        del epo_dir, epo_fnames, all_epo, epoch
        del mixed_fwd_lh, mixed_fwd_rh, vol_fwd_lh, vol_fwd_rh
        gc.collect()