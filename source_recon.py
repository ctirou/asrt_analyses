import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.beamformer import make_lcmv, apply_lcmv
import autoreject

method = 'lcmv'
lock = 'stim'
# lock = 'button'
trial_type = 'pattern'
movie = False
overwrite = True

data_path = './preprocessed'
subjects_dir = op.join('./freesurfer')
res_path = op.join('./results')
if not op.exists(res_path):
    os.mkdir(res_path)

subjects = ['sub01', 'sub02', 'sub04', 'sub07', 'sub08', 'sub09',
            'sub10', 'sub12', 'sub13', 'sub14', 'sub15']

# create results directory
folders = ["stcs", "bem", "src", "trans", "fwd"]
for f in folders:
    if f == "stcs":
        path = os.path.join(res_path, "stcs", method, lock)
        if not op.exists(path):
            os.makedirs(path)
    elif f in folders[-2:]:
        path = op.join(res_path, f, lock)
        if not op.exists(path):
            os.makedirs(path)
    else:
        if not op.exists(op.join(res_path, f)):
            os.mkdir(op.join(res_path, f))

epochs_list = ['2_PRACTICE', '3_EPOCH_1', '4_EPOCH_2', '5_EPOCH_3', '6_EPOCH_4']

for subject in subjects: 
    # source space
    src_fname = op.join(res_path, "src", "%s-src.fif" % subject)
    if not op.exists(src_fname) or overwrite:
        src = mne.setup_source_space(subject, spacing='oct6',
                                        subjects_dir=subjects_dir,
                                        add_dist=True,
                                        n_jobs=-1)
        src.save(src_fname, overwrite=overwrite)
    # bem model
    bem_fname = os.path.join(res_path, "bem", "%s-bem.fif" % (subject))
    if not op.exists(bem_fname) or overwrite:
        conductivity = (.3,)
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        mne.bem.write_bem_solution(bem_fname, bem, overwrite=overwrite)
    # loop across all epochs
    for epoch_num, epo in enumerate(epochs_list): # if practice included
    # for epoch_num, epo in zip([1, 2, 3, 4], epochs_list): # if practice not included
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
        # get average of each stimuli if they are pattern
        if epoch_num == 0:
            one_pattern = epoch[np.where((behav['positions']==1))[0]]
            two_pattern = epoch[np.where((behav['positions']==2))[0]]
            three_pattern = epoch[np.where((behav['positions']==3))[0]]
            four_pattern = epoch[np.where((behav['positions']==4))[0]]
        else:
            one_pattern = epoch[np.where((behav['positions']==1) & (behav['trialtypes']==1))[0]]
            two_pattern = epoch[np.where((behav['positions']==2) & (behav['trialtypes']==1))[0]]
            three_pattern = epoch[np.where((behav['positions']==3) & (behav['trialtypes']==1))[0]]
            four_pattern = epoch[np.where((behav['positions']==4) & (behav['trialtypes']==1))[0]]
        check = 0
        for pat in [one_pattern, two_pattern, three_pattern, four_pattern]:
            check += pat.shape[0]
        assert check == epoch.shape[0]
        # create trans file
        trans_fname = os.path.join(res_path, "trans", lock,  "%s-trans-%s.fif" % (subject, epoch_num))
        if not op.exists(trans_fname) or overwrite:
            coreg = mne.coreg.Coregistration(epoch.info, subject, subjects_dir)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=6, verbose=True)
            coreg.omit_head_shape_points(distance=5.0 / 1000)
            coreg.fit_icp(n_iterations=100, verbose=True)
            mne.write_trans(trans_fname, coreg.trans, overwrite=overwrite)
        # compute source estimate with beamformer algorithm, on evoked data
        for pat, num in zip([one_pattern, two_pattern, three_pattern, four_pattern], [1, 2, 3, 4]):
            if epo == '2_PRACTICE':
                epo_fname = 'prac'
            else:
                epo_fname = 'sess-%s' % (str(epoch_num).zfill(2))
            # compute data covariance matrix on evoked data
            data_cov = mne.compute_covariance(pat, tmin=0, tmax=.6, method="empirical", rank="info")
            # compute noise covariance
            if lock == 'button':
                noise_cov = mne.compute_covariance(epoch_bsl, method="empirical", rank="info")
                info = pat.info
            else:
                noise_cov = mne.compute_covariance(pat, tmin=-.2, tmax=0, method="empirical", rank="info")
                info = pat.info
            # make forward solution
            fwd_fname = op.join(res_path, "fwd", lock, "%s-%s-fwd-%i.fif" % (subject, epo_fname, num))
            if not op.exists(fwd_fname) or overwrite:
                src = mne.read_source_spaces(src_fname)
                fwd = mne.make_forward_solution(pat.info, trans=trans_fname,
                                                src=src, bem=bem_fname,
                                                meg=True, eeg=False,
                                                mindist=5.0,
                                                n_jobs=1)
                mne.write_forward_solution(fwd_fname, fwd, overwrite=overwrite)
            else:
                fwd = mne.read_forward_solution(fwd_fname)        
            # conpute rank
            rank = mne.compute_rank(noise_cov, info=info, rank=None, tol_kind='relative')
            # create evoked
            evoked = pat.average()
            # compute source estimate
            filters = make_lcmv(pat.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                                pick_ori=None, rank=rank, reduce_rank=True)
            stc = apply_lcmv(evoked=evoked, filters=filters)
            # save source estimate
            stc_fname = op.join(res_path, "stcs", method, lock, "%s_%s_%s" % (subject, epo_fname, num))
            if not op.exists(stc_fname) or overwrite:
                stc.save(stc_fname, overwrite=True)
            # save movie per stim per session
            if movie:
                if not op.exists(op.join(res_path, 'movie')):
                    os.mkdir(op.join(res_path, 'movie'))
                fig = stc.plot(subjects_dir=subjects_dir)
                mov_fname = op.join(res_path, 'movie', '%s_%s_%s.mov' % (subject, epo_fname, num))
                # mov_fname = op.join(res_path, 'movie', 'epo-sub02-w-sub01.mov')
                fig.save_movie(mov_fname, time_dilation=12, tmin=pat.tmin, tmax=pat.tmax, framerate=24, interpolation='quadratic', codec='mpeg4')
                fig.close()
