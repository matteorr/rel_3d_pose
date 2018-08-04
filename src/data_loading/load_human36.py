import json
import torch
import numpy as np
import h5py

from os import listdir
from os.path import isfile, join, isdir
# from sklearn.model_selection import train_test_split

from src.data_loading.camera import project_point_radial, world_to_camera_frame, load_cameras

def load_h5_data( data_path,
                 subjects_list,
                 actions_list,
                 camera_coord_3d,
                 dtype = np.float32 ):
    # load camera parameters
    cameras_path = data_path + '/cameras.h5'
    cameras_dict = load_cameras(cameras_path, subjects_list)
    camera_info  = set([c[1] for c in cameras_dict.keys()])

    X_gts = [] # list for 2d ground_truth inputs
    X_dts = [] # list for 2d detections inputs
    Y     = [] # list for 3d ground_truth outputs
    A     = [] # list of action id
    C     = [] # list of camera id
    S     = [] # list of subject id
    F     = [] # list of file name and frame number (for visualization purposes)

    for s in subjects_list:

        subj_gts_path = join(data_path, 'S%d/MyPoses/3D_positions'%s)
        pose_files = [f for f in np.sort(listdir(subj_gts_path)) if isfile(join(subj_gts_path, f))]

        # there are 4 dt files for every h5 gt file, one per camera
        # the base name of the file is the same, but is followed by the camera id
        subj_dts_path  = join(data_path, 'S%d/StackedHourglassFineTuned240'%s)

        for pf in pose_files:

            # skip the annotation files of the actions not specified in the
            # actions_list variable, which can be a string or a list
            if type(actions_list) == str:
                # if the actions_list is a single string match the exact name
                # to select the file, i.e. "Phoning 1.h5"
                file_name = pf.split(".")[0]
                # add -1 to the action id list
                action_id = [-1]
                if file_name != actions_list:
                    continue

            elif type(actions_list) == list:
                # if the actions_list is a list string match the name
                file_name = pf.split(".")[0].split(" ")[0]
                if file_name not in actions_list:
                    continue
                # get the position of the action in the list
                action_id = [actions_list.index(file_name)]

            else:
                raise ValueError("Bad input type actions_list: [%s]"%type(actions_list))

            # if the code gets here the file should not be skipped and the annotations
            # should be extracted

            # pose of the subject for all frames in video in world coordinates
            abs_path = join(subj_gts_path, pf)
            h5_file = h5py.File(abs_path,'r')
            # w_poses is num_frames x 96;
            # where 96 is 3 x num_keypoints and human_36 has 32 keypoints
            w_poses    = h5_file['3D_positions'][:].T

            num_frames    = int(w_poses.shape[0])
            num_keypoints = int(w_poses.shape[1] / 3.)

            # this reshapes puts the x,y,z coordinates for every keypoint on a row
            # successive keypoints are described on successive rows
            # after num_keypoints rows the keypoints for the next frame in the video
            # are represented, thus there are a total of num_keypoints * num_frames rows
            w_coord_xyz = np.reshape(w_poses, [-1, 3])

            for ci in camera_info:
                # extract all the camera info from the camera ids dictionary
                # R -> rotation matrix
                # T -> translation
                # f -> focal length
                # c -> camera center
                # k -> radial distortion params
                # p -> tangential distortion params
                # name -> camera id
                R, T, f, c, k, p, name = cameras_dict[ (s, ci) ]

                # extend the lists containing the camera, subj and action identifiers
                C.extend([ci] * num_frames)
                # C.extend([int(name)] * num_frames)
                S.extend([int(s)] * num_frames)
                A.extend(action_id * num_frames)
                F.extend([(pf,nfi) for nfi in range(num_frames)])

                ################################################################
                ################################################################
                # 2D inputs
                # projects the 3d poses into the camera 2d space using extrinsic
                # and intrinsic camera parameters such as distortions
                i_coord_xy, D, radial, tan, r2 =  project_point_radial( w_coord_xyz, R, T, f, c, k, p )

                # reshape and append in the X_gts list
                i_coord_xy = np.reshape( i_coord_xy, [-1, num_keypoints * 2] )
                X_gts.append(i_coord_xy.astype(dtype))

                if name != '54138969' or s!=11 or pf != 'Directions.h5':
                    # store the detections in XX
                    file_name = join(subj_dts_path, pf.replace(' ','_').replace('.h5','.%s.h5'%name))
                    dt_poses = h5py.File(file_name,'r')['poses'][:]

                    # hard coded to 32 cause stacked hourglass detections
                    # have 32 coordinates
                    in_c_coord_xy = np.zeros((num_frames, 32))
                    in_c_coord_xy[:,0::2] = dt_poses[:,:,0]
                    in_c_coord_xy[:,1::2] = dt_poses[:,:,1]
                    X_dts.append(in_c_coord_xy.astype(dtype))
                else:
                    print(' - bad detection file: [%s][%d][%s]'%(name, s, pf))
                    # put nans in the values of the detections to make sure X_gts
                    # and X_dts keep the same number of frames
                    in_c_coord_xy = np.nan * np.ones((num_frames, 32))
                    X_dts.append(in_c_coord_xy.astype(dtype))

                ################################################################
                ################################################################
                # 3D outputs
                if camera_coord_3d:
                    # rotation matrix R is a 3x3 array, T is a 3x1 array
                    # and they get applied to all the rows of w_coord_xyz so they
                    # can represent each keypoint (x,y,z) from the world coordinates to specific camera
                    # camera_coord is the same shape as w_coord_xyz and is in camera coords
                    c_coord_xyz = world_to_camera_frame( w_coord_xyz, R, T)
                else:
                    c_coord_xyz = w_coord_xyz

                # reshape brings camera_coord to the original shape of num_frames x 96
                c_coord_xyz = np.reshape( c_coord_xyz, [-1, num_keypoints * 3] )
                # store the output (3d) poses
                Y.append(c_coord_xyz.astype(dtype))

    return cameras_dict, np.vstack(X_gts), np.vstack(X_dts), np.vstack(Y), np.array(A), np.array(C), np.array(S), np.array(F)

def prepare_data(misc, X_gts, X_dts, Y,
                 noise_type_2d, noise_amount_2d, subtract_2d_root,
                 only_depth, keep_root ):

    ###[1]######################################################################
    # extract only the keypoints of interest from the h5 annotations
    # note, this is not done for X_dts_train since they are already in the
    # format of shg_16k with 16 keypoints
    keypoints_indxs_2d, keypoints_indxs_3d = misc.get_keypoints()
    num_keypoints_2d = len(keypoints_indxs_2d)
    num_keypoints_3d = len(keypoints_indxs_3d)
    num_frames       = X_gts.shape[0]

    keep_idxs_2d = []
    keep_idxs_3d = []
    for i in keypoints_indxs_2d: keep_idxs_2d.extend([2*i,2*i+1])
    for i in keypoints_indxs_3d: keep_idxs_3d.extend([3*i,3*i+1,3*i+2])

    X_gts = X_gts[:,keep_idxs_2d]
    Y     = Y[:,keep_idxs_3d]

    ###[2]######################################################################
    ## apply noise to the 2d keypoints
    X = X_gts
    if noise_type_2d == 'uniform_linear':
        # uniform linear distance
        delta_x = np.random.randint(0, noise_amount_2d+1,(num_frames, num_keypoints_2d))
        delta_y = np.random.randint(0, noise_amount_2d+1,(num_frames, num_keypoints_2d))

    elif noise_type_2d == 'uniform_radial':
        # uniform radial distance
        random_ang_noise = np.random.randint(0,360,(num_frames, num_keypoints_2d))
        delta_x = noise_amount_2d * np.cos(np.deg2rad(random_noise))
        delta_y = noise_amount_2d * np.sin(np.deg2rad(random_noise))

    elif noise_type_2d == 'normal':
        # normal jitter
        mean = (0, 0)
        cov  = [[noise_amount_2d, 0], [0, noise_amount_2d]]
        delta_xy = np.random.multivariate_normal(mean, cov, (num_frames, num_keypoints_2d))
        delta_x = delta_xy[:,:,0]
        delta_y = delta_xy[:,:,1]

    elif noise_type_2d == 'detections':
        print(" - using detections as inputs")
        err_str = "If noise_type_2d is detections the dataset type must be shg_16k"
        assert misc.dataset_type == 'shg_16k', err_str
        delta_x = delta_y = 0
        X = X_dts

    else:
        delta_x = delta_y = 0
        assert noise_type_2d == 'none'

    X[:,0::2] += delta_x
    X[:,1::2] += delta_y

    print(' - num frames/keypoints [%d][%d].'%(X.shape[0],num_keypoints_3d))

    ###[3]######################################################################
    ## subtract the skeleton root from all coordinates so it is always in 0,0
    skeleton_root_indx_2d, skeleton_root_indx_3d = misc.get_skeleton_root_idx()
    root_coords_2d = [2 * skeleton_root_indx_2d,
                      2 * skeleton_root_indx_2d+1]
    root_coords_3d = [3 * skeleton_root_indx_3d,
                      3 * skeleton_root_indx_3d+1,
                      3 * skeleton_root_indx_3d+2]

    # subtract from 3d data (always done)
    Y_root = Y[:,root_coords_3d]
    Y -= np.tile(Y_root, num_keypoints_3d)

    # subtract from 2d data (only if specified in the opts)
    X_root = X[:,root_coords_2d]
    if subtract_2d_root:
        X -= np.tile(X_root, num_keypoints_2d)

    ###[4]######################################################################
    # remove the root from the output vector
    if not keep_root:
        # eliminate the root coordinates from the output matrix
        Y_train = np.delete(Y,root_coords_3d, axis=1)
        raise NotImplementedError("Our method always predicts the 3d root value.")

    ###[5]######################################################################
    # use only the z coordinates as outputs for the network
    if only_depth:
        # remove the x and y coordinates from the output matrix
        Y = Y[:,2::3]
        raise NotImplementedError("Our method always predicts all 3 coordinates (x,y,z).")

    return X, X_root, Y, Y_root

def compute_training_stats(misc, X_train, Y_train):
    # compute standard deviation and mean for standardizing input and output data
    # note that the standardization cannot be performed on the 3d data if the rel_loss
    # method is being used, as that would result in using some gt depth information.
    # the standardization is independent for each keypoint channel
    X_mean = X_train.mean(0);     X_std  = X_train.std(0)
    X_mean[np.isnan(X_mean)] = 0; X_std[np.isnan(X_std)]   = 0

    Y_mean = Y_train.mean(0);     Y_std  = Y_train.std(0)
    Y_mean[np.isnan(Y_mean)] = 0; Y_std[np.isnan(Y_std)]   = 0
    stat_2d = {'mean':X_mean, 'std':X_std}

    ############################################################################
    # compute stats about the actual length of limbs in the training set.
    # this corresponds to the 3d bone length of people in the data
    skeleton_pairs_2d, skeleton_pairs_3d = misc.get_skeleton_pairs()

    # average len of limbs in the training set
    avg_limb_lens      = []
    # compute if the first keypoint of each limb is in front or behind the second
    avg_limb_rel_label = []

    for limb in skeleton_pairs_3d:
        kpt_0_coord = [3 * limb[0], 3 * limb[0] + 1, 3 * limb[0] + 2]
        kpt_1_coord = [3 * limb[1], 3 * limb[1] + 1, 3 * limb[1] + 2]

        kpt_0 = Y_train[:,kpt_0_coord]
        kpt_1 = Y_train[:,kpt_1_coord]

        limb_lens = np.sqrt(np.power(kpt_0 - kpt_1, 2).sum(axis=1))
        avg_limb_lens.append(np.mean(limb_lens))
        avg_limb_rel_label.append(len(np.where(kpt_0[:,2] > kpt_1[:,2])[0])/float(len(Y_train)))

    avg_limb_lens = np.array(avg_limb_lens)[np.newaxis, ...]
    avg_limb_rel_label = np.array(avg_limb_rel_label)[np.newaxis, ...]
    print(" - average length of limbs: ")
    print(np.around(avg_limb_lens, 2))
    # print("Train prob of rel_limb location, ", avg_limb_rel_label)

    stat_3d = {'mean':Y_mean, 'std':Y_std, 'avg_limb_lens':avg_limb_lens}

    return stat_2d, stat_3d

def load_human36(misc, opt, actions_list):
    root_dir        = opt.data_dir
    # subjects_list   = opt.train_subjects + opt.test_subjects
    camera_coord_3d = opt.camera_coord_3d

    # load the training data from the h5 files
    print("Training set:")
    cameras_dict_train, X_gts_train, X_dts_train, Y_train, \
    A_train, C_train, S_train, F_train = \
        load_h5_data( data_path       = root_dir,
                      subjects_list   = opt.train_subjects,
                      actions_list    = actions_list,
                      camera_coord_3d = camera_coord_3d )

    # relevant parameters from opt
    subtract_2d_root = opt.subtract_2d_root
    only_depth       = opt.only_depth
    keep_root        = opt.keep_root
    noise_type_2d    = opt.train_noise_type_2d
    noise_amount_2d  = opt.train_noise_amount_2d

    # get rid of potentially bad data (containing np.nan) due to the bad
    # detection file
    if noise_type_2d == 'detections':
        bad_idx = np.unique(np.where(X_dts_train != X_dts_train)[0])
        X_gts_train = np.delete(X_gts_train, bad_idx, axis=0)
        X_dts_train = np.delete(X_dts_train, bad_idx, axis=0)
        Y_train     = np.delete(Y_train, bad_idx, axis=0)
        A_train     = np.delete(A_train, bad_idx, axis=0)
        C_train     = np.delete(C_train, bad_idx, axis=0)
        S_train     = np.delete(S_train, bad_idx, axis=0)
        F_train     = np.delete(F_train, bad_idx, axis=0)

    X_train, X_train_root, Y_train, Y_train_root \
                  = prepare_data(misc, X_gts_train, X_dts_train, Y_train,
                                 noise_type_2d, noise_amount_2d,
                                 subtract_2d_root, only_depth, keep_root)
    stat_2d, stat_3d = compute_training_stats(misc, X_train, Y_train)

    data_dict_train = {}
    data_dict_train['X'] = X_train
    data_dict_train['X_root'] = X_train_root
    data_dict_train['Y'] = Y_train
    data_dict_train['Y_root'] = Y_train_root
    data_dict_train['A'] = A_train
    data_dict_train['C'] = C_train
    data_dict_train['S'] = S_train
    data_dict_train['F'] = F_train

    # load the testing data from the h5 files
    print("\nTest set:")
    cameras_dict_test, X_gts_test, X_dts_test, Y_test, \
    A_test, C_test, S_test, F_test = \
        load_h5_data( data_path       = root_dir,
                      subjects_list   = opt.test_subjects,
                      actions_list    = actions_list,
                      camera_coord_3d = camera_coord_3d )

    # relevant parameters from opt
    subtract_2d_root = opt.subtract_2d_root
    only_depth       = opt.only_depth
    keep_root        = opt.keep_root
    noise_type_2d    = opt.test_noise_type_2d
    noise_amount_2d  = opt.test_noise_amount_2d

    # get rid of potentially bad data (containing np.nan) due to the bad
    # detection file
    if noise_type_2d == 'detections':
        bad_idx = np.unique(np.where(X_dts_test != X_dts_test)[0])
        X_gts_test = np.delete(X_gts_test, bad_idx, axis=0)
        X_dts_test = np.delete(X_dts_test, bad_idx, axis=0)
        Y_test     = np.delete(Y_test, bad_idx, axis=0)
        A_test     = np.delete(A_test, bad_idx, axis=0)
        C_test     = np.delete(C_test, bad_idx, axis=0)
        S_test     = np.delete(S_test, bad_idx, axis=0)
        F_test     = np.delete(F_test, bad_idx, axis=0)

    X_test, X_test_root, Y_test, Y_test_root = \
                       prepare_data(misc, X_gts_test, X_dts_test, Y_test,
                                    noise_type_2d, noise_amount_2d,
                                    subtract_2d_root, only_depth, keep_root)

    data_dict_test = {}
    data_dict_test['X'] = X_test
    data_dict_test['X_root'] = X_test_root
    data_dict_test['Y'] = Y_test
    data_dict_test['Y_root'] = Y_test_root
    data_dict_test['A'] = A_test
    data_dict_test['C'] = C_test
    data_dict_test['S'] = S_test
    data_dict_test['F'] = F_test

    return data_dict_train, cameras_dict_train, data_dict_test, cameras_dict_test, stat_2d, stat_3d
