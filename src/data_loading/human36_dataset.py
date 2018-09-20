#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np

import time
from random import shuffle

class Human36M(Dataset):
    def __init__(self, misc, cameras_dict, data_dict, stat_2d, stat_3d,
                 tol_mm, num_pairs, amount_data, rel_labels_noise_prob,
                 in_dropout_p, is_train=True):

        self.cameras_dict = cameras_dict

        self.is_train = is_train

        skeleton_pairs_2d, skeleton_pairs_3d = misc.get_skeleton_pairs()
        self.skeleton_pairs = skeleton_pairs_2d

        self.tol_mm       = tol_mm
        self.amount_data  = amount_data
        self.in_dropout_p = in_dropout_p

        ########################################################################
        ## select subset of the data
        num_examples = data_dict['X'].shape[0]
        if self.is_train:
            # if is_train the amount of data represents the
            # percentage of data to select.
            if self.amount_data < 1.0:
                # here we select a subset of the data randomly
                amnt = int(num_examples * self.amount_data)
                inds = np.random.choice(np.arange(num_examples), amnt, replace=False)
            else:
                inds = np.arange(num_examples)

        else:
            # if not is_train the amount of data represents the
            # frequency at which the data should be subsampled,
            # i.e. 50 means that only a frame every 50 is selected
            if self.amount_data > 1.0:
                self.amount_data = int(self.amount_data)
                inds = []
                for data_idx in range(num_examples):
                    # here we sample every amt_data frames
                    frame_ind = int(data_dict['F'][data_idx][1])
                    if frame_ind % self.amount_data == 0:
                        inds.append(data_idx)
            else:
                inds = np.arange(num_examples)
        self.num_examples = len(inds)
        self.inds = np.array(inds)

        self.input_data  = data_dict['X'][inds, :]
        self.input_root  = data_dict['X_root'][inds, :]
        self.target_data = data_dict['Y'][inds, :]
        self.target_root = data_dict['Y_root'][inds, :]

        self.S = data_dict['S'][inds]
        self.C = data_dict['C'][inds]

        ########################################################################
        # normalize the targets and data
        print(" - normalizing data")
        self.norm_inputs = (self.input_data - stat_2d['mean'][np.newaxis, ...]) / stat_2d['std'][np.newaxis, ...]
        self.norm_inputs[np.isnan(self.norm_inputs)] = 0

        ########################################################################
        # apply dropout to the inputs with a certain probability
        if self.in_dropout_p > 0:
            print("\n - applying input data dropout with p=[{}]".format(self.in_dropout_p))
            assert self.in_dropout_p <= 1, "Cannot have dropout probability > 1."
            input_mask = np.random.choice([0, 1],
                            size=(self.input_data.shape[0],self.input_data.shape[1]/2),
                            p=[self.in_dropout_p, 1. - self.in_dropout_p])
            input_mask = np.repeat(input_mask, 2, axis=1)
            self.norm_inputs *= input_mask

        self.norm_targets = (self.target_data - stat_3d['mean'][np.newaxis, ...]) / stat_3d['std'][np.newaxis, ...]
        self.norm_targets[np.isnan(self.norm_targets)] = 0

        ########################################################################
        # sample the keypoint pairs for relative depth labels (possibly with errors)
        self.num_keypoints = int(self.input_data.shape[1] / 2.)
        all_pairs = [(i, j) for i in range(self.num_keypoints) for j in range(self.num_keypoints) if i < j]

        if num_pairs is None:
            self.num_pairs = 0
        elif num_pairs == 'all':
            self.num_pairs = len(all_pairs)
        else:
            self.num_pairs = num_pairs

        self.rel_labels_noise_prob = rel_labels_noise_prob
        if self.rel_labels_noise_prob != 'mturk':
            assert self.rel_labels_noise_prob >= 0 and self.rel_labels_noise_prob <= 1

        self.kpt_pairs = []
        self.pairs_err = []
        if self.num_pairs > 0:
            print("\n - sampling relative label pairs and errors (will take a while):")
            tic = time.time()
            for e in range(self.num_examples):
                ####################################################################
                # determine the pairs
                shuffle(all_pairs)
                selected_pairs = [all_pairs[i] for i in range(self.num_pairs)]
                self.kpt_pairs.append(selected_pairs)

                ####################################################################
                # determine the errors for every pair
                if self.rel_labels_noise_prob == 'mturk':
                    selected_pairs_coords_z0 = [3*z[0] + 2 for z in selected_pairs]
                    selected_pairs_coords_z1 = [3*z[1] + 2 for z in selected_pairs]

                    depth_diff = 0.1 * np.abs(self.target_data[e, selected_pairs_coords_z0] - self.target_data[e, selected_pairs_coords_z1])
                    err_prob   = 1 - np.minimum((1.25 * depth_diff + 50),100)/100.

                    err_label = np.sign(2 * (np.random.rand((self.num_pairs)) > err_prob) - 1.).astype(int)

                else:
                    err_label = np.sign(2 * (np.random.rand((self.num_pairs)) > self.rel_labels_noise_prob) - 1.).astype(int)
                self.pairs_err.append(err_label.tolist())

                if e % 100000 == 0:
                    print("   done %d: (%.3fs.)"%(e, time.time() -  tic))
                    tic = time.time()

    def __getitem__(self, index):
        inputs       = torch.from_numpy(self.input_data[index, :])
        inputs_root  = torch.from_numpy(self.input_root[index, :])
        norm_inputs  = torch.from_numpy(self.norm_inputs[index, :])

        outputs      = torch.from_numpy(self.target_data[index, :])
        outputs_root = torch.from_numpy(self.target_root[index, :])
        norm_outputs = torch.from_numpy(self.norm_targets[index, :])

        cam_id  = self.C[index]
        subj_id = self.S[index]

        # compute p_inv used in the reprojection loss if use_full_intrinsics==True
        R, T, f, c, k, p, name = self.cameras_dict[ (subj_id,cam_id) ]
        p_inv = np.zeros(p.shape).astype(self.target_data.dtype)
        p_inv[1,0] = p[0,0]
        p_inv[0,0] = p[1,0]
        camera_params = (R, T, f, c, k, p, p_inv)

        if self.num_pairs > 0:
            # one random pair of all the available ones is selected and the
            # ground truth relative location for that pair is provided as gt
            # this ensures that a possibly a different pair is used every
            # time this example is used in training
            rand_ind = np.random.randint(self.num_pairs)
            pts = self.kpt_pairs[index][rand_ind]

            # randomly flip the order of the keypoint pairs 50% of the time
            # to ensure that there is no bias in the ordering of keypoints
            if np.random.rand() > 0.5:
                pts = pts[::-1]

            rel_inds = torch.from_numpy(np.array([pts[0]*3 + 2, pts[1]*3 + 2]).astype(np.int))
            rel_gt   = torch.from_numpy(np.zeros(1).astype(self.target_data.dtype))

            depth_diff = self.target_data[index, pts[0]*3 + 2] - self.target_data[index, pts[1]*3 + 2]

            # NOTE: Tolerance is meaningful only if it is normalized in same way
            # as the data
            if depth_diff > self.tol_mm:
                rel_gt[0] = 1.0
            elif depth_diff < -self.tol_mm:
                rel_gt[0] = -1.0
            else:  # same depth within tolerance
                rel_gt[0] = 0.0

            # apply the noise - note that this  will not flip the label of 0
            # when using the tolerance parameter.
            err_str = "Cannot use tolerance and relative label noise at the same time."
            if self.rel_labels_noise_prob > 0.: assert self.tol_mm == 0., err_str
            rel_gt[0] *= self.pairs_err[index][rand_ind]

        else:
            rel_inds = torch.zeros(2)
            rel_gt = torch.from_numpy(np.zeros(1).astype(self.target_data.dtype))

        if self.is_train:
            # compute the length of the limbs of the target skeleton
            limb_lens_3d = torch.zeros((1,len(self.skeleton_pairs)))
            for l, limb in enumerate(self.skeleton_pairs):
                kpt_0_coord = [3 * limb[0], 3 * limb[0] + 1, 3 * limb[0] + 2]
                kpt_1_coord = [3 * limb[1], 3 * limb[1] + 1, 3 * limb[1] + 2]

                limb_lens_3d[:,l]= torch.sqrt(((outputs[kpt_0_coord] - outputs[kpt_1_coord]) ** 2).sum(dim=0))
        else:
            limb_lens_3d = torch.zeros((1,len(self.skeleton_pairs)))

        return inputs, norm_inputs, inputs_root, \
               outputs, norm_outputs, outputs_root, \
               rel_inds, rel_gt, \
               limb_lens_3d, camera_params

    def __len__(self):
        return len(self.input_data)
