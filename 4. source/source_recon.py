import os
import os.path as op
import mne
from base import ensure_dir
from config import *
import gc
import sys

subjects = SUBJS15
epochs_list = EPOCHS
subjects_dir = FREESURFER_DIR
data_path, res_path = DATA_DIR, RESULTS_DIR

jobs = -1

overwrite = True
verbose = True

analyses = ['for_timeg', 'for_rsa']

is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

def process_subject(subject, jobs):
    # create results directories
    folders = ["src", "bem", "trans", "fwd"]
    for f in folders:
        if f == folders[-1]:
            ensure_dir(os.path.join(res_path, f, analysis))
        else:
            ensure_dir(os.path.join(res_path, f))
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
                                    
    ## Freesurfer default aseg atlas
    aseg_fname = subjects_dir / subject / 'mri' / 'aseg.mgz'
    aseg_labels = mne.get_volume_labels_from_aseg(aseg_fname)
    
    hipp_labels = [l for l in aseg_labels if 'Hipp' in l]
    thal_labels = [l for l in aseg_labels if 'Thal' in l]
    cerebellum_labels = [l for l in aseg_labels if 'Cerebellum-Cortex' in l]
    htc_labels = hipp_labels + thal_labels + cerebellum_labels
    
    vol_src_htc_fname = op.join(res_path, "src", "%s-htc-vol-src.fif" % (subject))
    if not op.exists(vol_src_htc_fname) or overwrite:
        surface = subjects_dir / subject / "bem" / "inner_skull.surf" # inner skull surface (to try)
        vol_src_htc = mne.setup_volume_source_space(
            subject,
            surface=surface,
            mri="aseg.mgz",
            volume_label=htc_labels,
            subjects_dir=subjects_dir,
            n_jobs=jobs,
            verbose=verbose)
        mne.write_source_spaces(vol_src_htc_fname, vol_src_htc, overwrite=True, verbose=verbose)
    else:
        vol_src_htc = mne.read_source_spaces(vol_src_htc_fname, verbose=verbose)
    
    mixed = src + vol_src_htc
    
    all_epochs = list()
    
    for epoch_num in range(5):
        print(f"Processing {subject} {epoch_num}...")
        
        epoch_fname = op.join(data_path, analysis, "epochs", f"{subject}-{epoch_num}-epo.fif")
        epoch = mne.read_epochs(epoch_fname, preload=False, verbose=verbose)
        if epoch_num != 0:
            all_epochs.append(epoch)
        
        # create trans file
        trans_fname = os.path.join(res_path, "trans", "%s-%i-trans.fif" % (subject, epoch_num))
        if not op.exists(trans_fname) or overwrite:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=verbose)
            coreg.fit_icp(n_iterations=6, verbose=verbose)
            coreg.omit_head_shape_points(distance=5.0/1000)
            coreg.fit_icp(n_iterations=100, verbose=verbose)
            mne.write_trans(trans_fname, coreg.trans, overwrite=True, verbose=verbose)
        
        # compute forward solution
        fwd_fname = res_path / "fwd" / analysis / f"{subject}-{epoch_num}-fwd.fif"
        if not op.exists(fwd_fname) or overwrite:
            fwd = mne.make_forward_solution(epoch.info, trans=trans_fname,
                                        src=src, bem=bem_fname,
                                        meg=True, eeg=False,
                                        mindist=5.0,
                                        n_jobs=jobs,
                                        verbose=verbose)
            mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=verbose)
            del fwd        
        
        fwd_fname = res_path / "fwd" / analysis / f"{subject}-htc-{epoch_num}-fwd.fif"
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
    
    for epoch in all_epochs:
        epoch.info['dev_head_t'] = all_epochs[0].info['dev_head_t']
    epochs = mne.concatenate_epochs(all_epochs, verbose=verbose)
    
    del all_epochs
    gc.collect()
    
    # create trans file
    trans_fname = op.join(res_path, "trans", "%s-all-trans.fif" % (subject))
    if not op.exists(trans_fname) or overwrite:
        coreg = mne.coreg.Coregistration(epochs.info, subject, subjects_dir)
        coreg.fit_fiducials(verbose=verbose)
        coreg.fit_icp(n_iterations=6, verbose=verbose)
        coreg.omit_head_shape_points(distance=5.0/1000)
        coreg.fit_icp(n_iterations=100, verbose=verbose)
        mne.write_trans(trans_fname, coreg.trans, overwrite=True, verbose=verbose)
    
    fwd_fname = res_path / "fwd" / analysis / f"{subject}-all-fwd.fif"
    if not op.exists(fwd_fname) or overwrite:
        fwd = mne.make_forward_solution(epochs.info, trans=trans_fname,
                                    src=src, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=verbose)
        del fwd
    
    fwd_fname = res_path / "fwd" / analysis / f"{subject}-htc-all-fwd.fif"
    if not op.exists(fwd_fname) or overwrite:
        fwd = mne.make_forward_solution(epochs.info, trans=trans_fname,
                                    src=mixed, bem=bem_fname,
                                    meg=True, eeg=False,
                                    mindist=5.0,
                                    n_jobs=jobs,
                                    verbose=verbose)
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=verbose)
        del fwd
            
    del epochs, src, mixed
    gc.collect()

if is_cluster:
    # Check that SLURM_ARRAY_TASK_ID is available and use it to get the subject
    try:
        subject_num = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        subject = subjects[subject_num]
        jobs = int(os.getenv("SLURM_CPUS_PER_TASK"))
        process_subject(subject, jobs)
    except (IndexError, ValueError) as e:
        print("Error: SLURM_ARRAY_TASK_ID is not set correctly or is out of bounds.")
        sys.exit(1)
else:
    jobs = -1
    for analysis in analyses:
        for subject in subjects:
            process_subject(subject, jobs=-1)