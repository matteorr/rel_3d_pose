#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os, json
from pprint import pprint

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from src.data_formats.misc import DatasetMisc
from src.model import LinearModel, weight_init

from src.utils.pose_plotter import PosePlotter
from src.data_formats.human36_17k_config import pose_config


def run_model(opt):

    # get misc file used for the specified data format
    misc = DatasetMisc(opt['dataset_type'])

    # class that takes care of plotting
    pose_plotter = PosePlotter(
                    pose_config['KEYPOINT_NAMES'],
                    pose_config['SKELETON_NAMES'],
                    pose_config['KEYPOINT_COLORS'],
                    pose_config['SKELETON_COLORS'])

    # load checkpoint file
    ckpt    = torch.load(opt['load'])
    stat_2d = ckpt['stat_2d']

    # load the pretrained model
    print("\n==================Model===================")
    print("Loading Pretrained Model:")
    print(" - Linear size: [{}]".format(opt['linear_size']))
    print(" - Num stages:  [{}]".format(opt['linear_size']))
    print("==========================================\n")

    pretrained_model = LinearModel(misc.NUM_KEYPOINTS_2D * 2,
                                   misc.NUM_KEYPOINTS_3D * 3,
                                   opt['linear_size'],
                                   opt['num_stage'],
                                   opt['dropout'],
                                   opt['predict_scale'],
                                   opt['scale_range'],
                                   opt['unnorm_op'],
                                   opt['unnorm_init'])

    pretrained_model = pretrained_model.cuda()
    pretrained_model.load_state_dict(ckpt['state_dict'])
    pretrained_model.eval()

    # load the data from a numpy file
    print("\n==================Data====================")
    print("Loading Data:")
    print(" - Data path:  [{}]".format(opt['data_dir']))
    print(" - Data type:  [{}]".format(opt['dataset_type']))

    with open(opt['data_dir'], 'r') as fp: data = np.load(fp)
    num_frames, num_coords = data.shape
    num_kpts               = int(num_coords/2)
    print(" - Num frames: [{}]".format(num_frames))
    print(" - Num kpts:   [{}]".format(num_kpts))
    print("==========================================\n")

    # subtract root if specified
    if opt['subtract_2d_root']:
        root_idx_2d, _ = misc.get_skeleton_root_idx()
        # subtract the 2d skeleton center from all coordinates so it is always in 0,0
        data_2d_root  = data[:, [2 * root_idx_2d, 2 * root_idx_2d + 1]]
        data         -= np.tile(data_2d_root, num_kpts)

    # normalize the inputs according to the stored mean and std
    data_mean = stat_2d['mean']
    data_std  = stat_2d['std']

    norm_data = (data - data_mean[np.newaxis, ...]) / data_std[np.newaxis, ...]
    norm_data[np.isnan(norm_data)] = 0
    norm_data = norm_data.astype(np.float32)

    seq_dataset = TensorDataset(torch.from_numpy(norm_data), torch.from_numpy(data))
    seq_loader  = DataLoader(dataset=seq_dataset,
                             batch_size=100, shuffle=False,
                             num_workers=4, drop_last=False)

    # predict 3d pose using the model
    in_2d_poses  = []
    out_3d_poses = []

    for indx, (norm_data, data) in enumerate(seq_loader):

        model_inps = Variable(norm_data.cuda())
        model_outs, model_scale = pretrained_model(model_inps)

        in_2d_poses.append(data.numpy())
        out_3d_poses.append(model_outs.data.cpu().numpy())

    in_2d_poses  = np.vstack(in_2d_poses)
    out_3d_poses = np.vstack(out_3d_poses)

    num_frames = out_3d_poses.shape[0]
    num_kpts   = int(out_3d_poses.shape[1] / 3)
    print("\n==================Outputs====================")
    print("Predicted Data:")
    print(" - Num frames:    [{}]".format(num_frames))
    print(" - Num keypoints: [{}]".format(num_kpts))


    f_no = np.random.randint(num_frames)

    ########################################################################
    ## load the 2d groundtruth keypoints in the frame
    kpts_2d_x = in_2d_poses[f_no, 0::2]
    kpts_2d_y = in_2d_poses[f_no, 1::2]

    ########################################################################
    ## get 3d predicted keypoints in the frame
    kpts_3d_x = out_3d_poses[f_no, 0::3]
    kpts_3d_y = out_3d_poses[f_no, 1::3]
    kpts_3d_z = out_3d_poses[f_no, 2::3]

    ########################################################################
    ## set the visibility flags (currently all keypoints are assumed visible)
    kpts_v = np.ones(np.shape(kpts_2d_x))

    pose_plotter.plot_2d(kpts_2d_x, kpts_2d_y, kpts_v)

    pose_plotter.plot_3d(kpts_3d_x, kpts_3d_y, kpts_3d_z, kpts_v)

    pose_plotter.plot_2d_3d(kpts_2d_x, kpts_2d_y, kpts_3d_x, kpts_3d_y, kpts_3d_z, kpts_v)


if __name__ == "__main__":
    """
    uses the function run_model to test a pretrained model on a numpy array
    """

    # NOTE: baseball.npy and running.npy contain poses with 17 keypoints
    # while random.npy contains poses with 14 keypoints
    DEMO_DATA = './demo_data/baseball.npy' # [baseball.npy, running.npy, random.npy]
    # NOTE: this model was trained for data with 17 keypoints so is compatible
    # with baseball.npy and running.npy, to run a model on random.npy you must
    # train a new model with 14 keypoints.
    LOAD_PATH = './checkpoint/default_human36_rel'

    opts_path  = LOAD_PATH + '/opt.json'
    model_path = LOAD_PATH + '/test_ckpt_last.pth.tar'

    with open(opts_path,'r') as fp: opt = json.load(fp)
    opt['data_dir']   = DEMO_DATA
    opt['load']       = model_path

    print("\n==================Options=================")
    pprint(opt, indent=4)
    print("==========================================\n")

    predicted_3d_poses = run_model(opt)
