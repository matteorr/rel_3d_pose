#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json


class LSP(Dataset):
    def __init__(self, misc, opt, is_train):

        merged_mturk_anns = opt.mturk_data
        lsp_root          = opt.data_dir
        subtract_2d_root  = opt.subtract_2d_root

        keypoints_2d_indxs, keypoints_3d_indxs       = misc.get_keypoints()
        skeleton_root_indx_2d, skeleton_root_indx_3d = misc.get_skeleton_root_idx()

        root_coords_2d = [ 2 * skeleton_root_indx_2d, 2 * skeleton_root_indx_2d + 1]

        # load merged mturk annotations
        with open(merged_mturk_anns) as da:
            lsp = json.load(da)

        # verify joint names
        assert(len(misc.KEYPOINTS_2D) == len(lsp['joint_names']))
        assert(len(misc.KEYPOINTS_2D) == np.sum([True for jj in lsp['joint_names'] if jj in misc.KEYPOINTS_2D]))
        joint_inds = np.array([misc.KEYPOINTS_2D.index(jj) for jj in lsp['joint_names']])

        # load keypoints
        data_all = []
        for ii, im in enumerate(lsp['images']):
            # load keypoints in the correct order
            kps_np = np.zeros(len(im['keypoints']))
            for kk, kp_id in enumerate(joint_inds):
                kps_np[kp_id*2] = im['keypoints'][kk*2]
                kps_np[kp_id*2 + 1] = im['keypoints'][kk*2 + 1]
            data_all.append(kps_np)
        data_all = np.vstack((data_all)).astype(np.float32)

        # subtract root
        data_root = data_all[:, root_coords_2d]
        if subtract_2d_root:
            data_all -= np.tile(data_root, len(keypoints_2d_indxs))

        train_ind = np.array([im['is_train'] for im in lsp['images']])
        inds_of_int = np.where(train_ind == is_train)[0]
        train_inds = np.where(train_ind == True)[0]

        self.input_data = data_all[inds_of_int, :]
        self.input_root = self.input_data[:, root_coords_2d]

        # load annotations
        self.annotations = []
        self.im_files = []
        im_ind = 0
        for ii, im in enumerate(lsp['images']):
            if im['is_train'] == is_train:
                self.im_files.append(lsp_root + im['im_file'])

                for ann in im['anns']:
                    rel = {'im_id':im_ind,
                           'label':ann['label']*2 -1,
                           'kp0':joint_inds[ann['kp0']],
                           'kp1':joint_inds[ann['kp1']]}
                    self.annotations.append(rel)
                im_ind += 1
        print(" - loaded data (images/annotations): [{}]/[{}]\n".format(len(self.im_files),len(self.annotations)))

        # standardize data using the training data
        data_train = data_all[train_inds, :]

        # normalize each dimension
        X_mean = data_train.mean(0)
        X_std  = data_train.std(0)
        X_mean[np.isnan(X_mean)] = 0
        X_std[np.isnan(X_std)]   = 0

        self.stat_2d = {}
        self.stat_2d['mean'] = X_mean
        self.stat_2d['std'] = X_std

        print(" - normalizing data\n")
        self.norm_inputs = (self.input_data - X_mean[np.newaxis, ...]) / X_std[np.newaxis, ...]
        self.norm_inputs[np.isnan(self.norm_inputs)] = 0

    def __getitem__(self, index):

        ann = self.annotations[index]
        inputs      = torch.from_numpy(self.input_data[ann['im_id'], :])
        inputs_root = torch.from_numpy(self.input_root[ann['im_id'], :])
        norm_inputs = torch.from_numpy(self.norm_inputs[ann['im_id'], :])

        # for LSP if kp0 < kp1 -> -1, otherwise 1
        rel_gt   = torch.from_numpy((np.ones(1)*ann['label']).astype(self.input_data.dtype))

        # get the z coordinate from the output 3d vector, multiplying the keypoint
        # location by 3 and then adding 2 (if you wanted the x it would just be *3,
        # for the y it is multiply by 3 and add 1.
        rel_inds = torch.from_numpy(np.array([ann['kp0']*3 + 2, ann['kp1']*3 + 2]).astype(np.int))

        return inputs, norm_inputs, inputs_root, rel_inds, rel_gt

    def __len__(self):
        return len(self.annotations)
