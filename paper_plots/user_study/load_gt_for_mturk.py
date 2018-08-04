"""
This script loads the GT Mturk data
It also tries to account for camera rotation i.e. assumes that turkers measure
distance orthographically.

H3.6 has four cameras and they have a different coordinate system to the world,
for the cameras y points down, and z points into the scene.

     World coords
            ^  z
            |
        x   |
     <------/
           /
          /  y
         <

     Camera coords
              ^
        z    /
            /   x
           /----->
           |
        y  |
           <


"""

import h5py
import json
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import user_study_utils as utils

sys.path.append('../src/data_loading/')
from camera import project_point_radial, world_to_camera_frame, load_camera_params
sys.path.append('../src/data_formats/')
import misc


def load_cameras(bpath='cameras.h5', subjects=None, dtype=np.float64):
    if subjects is None:
        subjects = [1, 5, 6, 7, 8, 9, 11]
    rcams = {}

    with h5py.File(bpath, 'r') as hf:
        for s in subjects:
            for c in range(4):  # There are 4 cameras in human3.6m
                a = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s, c + 1), dtype)
                rcams[(s, a[-1])] = a
    return rcams


def load_h36_coords(h36_root, human_36m_file_path, hf_cameras_file, ims_of_interest, im_dir=''):

    # load data represented in our format
    with open(human_36m_file_path,'rb') as fp:
        data_us = json.load(fp)

    mturk_keypoints = data_us['pose'][0]['keypoints']
    if len(mturk_keypoints) == 14:
        SKELETON_2D  = misc.HUMAN_36M_14K_SKELETON_2D
        SKELETON_3D  = misc.HUMAN_36M_14K_SKELETON_3D
    elif len(mturk_keypoints) == 17:
        SKELETON_2D  = misc.HUMAN_36M_17K_SKELETON_2D
        SKELETON_3D  = misc.HUMAN_36M_17K_SKELETON_3D

    keypoints_of_int = [misc.HUMAN_36M_KEYPOINTS.index(kk) for kk in mturk_keypoints]
    SKELETON_2D_IDX = [[mturk_keypoints.index(k[0]),mturk_keypoints.index(k[1])] for k in SKELETON_2D]
    start_pt_2d = np.array( [k[0] for k in SKELETON_2D_IDX] )
    end_pt_2d = np.array( [k[1] for k in SKELETON_2D_IDX] )

    SKELETON_3D_IDX = [[mturk_keypoints.index(k[0]),mturk_keypoints.index(k[1])] for k in SKELETON_3D]
    start_pt_3d = np.array( [k[0] for k in SKELETON_3D_IDX] )
    end_pt_3d = np.array( [k[1] for k in SKELETON_3D_IDX] )


    imgs = [ii for ii in data_us['images'] if ii['id'] in ims_of_interest]
    subjects = list(set([img['s_id'] for img in imgs]))
    ip_2d_poses_vals = [ii['kpts_2d'] for ii in data_us['annotations'] if ii['id'] in ims_of_interest]
    ip_2d_poses_keys = [ii['id'] for ii in data_us['annotations'] if ii['id'] in ims_of_interest]
    ip_2d_poses = dict(zip(ip_2d_poses_keys, ip_2d_poses_vals))

    #
    # load camera parameters
    cameras_dict = load_cameras(hf_cameras_file, subjects, np.float32)

    #
    # load raw h3.6 data to get depth
    world_coords_3d = {}
    cam_coords_3d = {}
    cam_coords_3d_corr = {}
    im_coords_2d = {}
    im_coords_2d_ip = {}
    file_paths = {}
    for ii, img in enumerate(imgs):
        subj   = img['s_id']
        frame  = img['frame']
        name   = img['video'].split(".")[0]
        camera = img['video'].split(".")[1]
        #print img['id'], subj, frame, name, camera

        # inconsistency in file names
        if "WalkingDog" in name:
            name = name.replace("WalkingDog", "WalkDog")
        if "TakingPhoto" in name:
            name = name.replace("TakingPhoto", "Photo")

        # load raw data
        h5_file_path_w   = h36_root + '/S%d/MyPoses/3D_positions/%s.h5'%(subj,name)
        h5_file  = h5py.File(h5_file_path_w,'r')
        w_poses  = h5_file['3D_positions'][:]
        w_coord_xyz = w_poses[:,frame].reshape(-1,3)

        # project to 3D in camera coords
        R, T, f, c, k, p, _ = cameras_dict[(subj, camera)]
        i_coord_xy, D, radial, tan, r2 =  project_point_radial( w_coord_xyz, R, T, f, c, k, p )

        c_coord_xyz = world_to_camera_frame(w_coord_xyz, R, T)

        # correct_rotation of camera
        theta = utils.rotationMatrixToEulerAngles(R)
        # print ii, np.around(np.rad2deg(theta), 3), camera, subj

        # cancel rotation in X and Y
        theta[0] = 0  # front lean - set camera to be upright - world x
        #theta[1] = 0  # left right lean - world y
        # theta[2]  # world z, need to rotate by this to move camera

        R_corr = utils.eulerAnglesToRotationMatrix(theta)
        # print np.around(R,3), '\n\n', T, '\n\n', np.around(R_corr,3)

        c_coord_xyz_corr = R_corr.dot(w_coord_xyz.T - T).T
        # swap y and z, and negate y
        c_coord_xyz_corr = np.dot(c_coord_xyz_corr, np.asarray([[1,0,0], [0,0,1], [0,-1,0]]))


        world_coords_3d[img['id']] = w_coord_xyz[keypoints_of_int, :]
        cam_coords_3d[img['id']] = c_coord_xyz[keypoints_of_int, :]
        cam_coords_3d_corr[img['id']] = c_coord_xyz_corr[keypoints_of_int, :]
        im_coords_2d[img['id']] = i_coord_xy[keypoints_of_int, :]
        im_coords_2d_ip[img['id']] = np.array(ip_2d_poses[img['id']]).reshape(len(keypoints_of_int),2)
        file_paths[img['id']] = im_dir + img['filename']

    return cam_coords_3d, cam_coords_3d_corr, im_coords_2d_ip, file_paths, mturk_keypoints
