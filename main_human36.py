#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import src.utils.log as log

from src.data_formats.misc import DatasetMisc
from src.data_formats.actions import define_actions

import src.data_loading.load_human36 as h36
from src.data_loading.human36_dataset import Human36M

from src.model import LinearModel, weight_init

from src.human_36.train_human36 import train_human
from src.human_36.test_human36 import test_human

import src.utils.viz as viz

def main_human(opt, save_op=True, return_proc=False):
    start_epoch   = 0
    err_test_best = 100000
    glob_step     = 0
    lr_now        = opt.lr

    # save options
    if save_op:
        log.save_options(opt, opt.ckpt)

    print("\n==================Actions=================")
    actions = define_actions(opt.action)
    print(">>> actions to use: {}".format(len(actions)))
    pprint(actions, indent=4)
    print("==========================================\n")

    print("\n==================Data=================")
    print(">>> loading data")

    # load structure for miscellaneous info
    misc = DatasetMisc(opt.dataset_type)

    # load the data from the h5 annotations
    data_dict_train, cameras_dict_train, data_dict_test, cameras_dict_test, \
        stat_2d, stat_3d  = h36.load_human36(misc, opt, actions)

    # relevant options for creating the train and test data loader
    tol_mm          = opt.tolerance_mm
    num_pairs       = opt.num_pairs
    amt_train_data  = opt.amt_train_data
    amt_test_data   = opt.amt_test_data
    train_rel_labels_noise_prob = opt.rel_labels_noise_prob
    test_rel_labels_noise_prob  = 0. #NOTE: hard coded to 0 for the test set
    in_dropout_p    = opt.in_dropout_p

    if opt.is_train:
        print("\n>>> creating Train dataset")
        # create dataset of type Human36M
        train_h36 = \
            Human36M( misc, cameras_dict_train, data_dict_train, stat_2d, stat_3d,
                      tol_mm, num_pairs, amt_train_data, train_rel_labels_noise_prob,
                      in_dropout_p, is_train=True )
        # create data loader
        train_loader = DataLoader(dataset=train_h36, batch_size=opt.train_batch,
                                  shuffle=True, num_workers=opt.job)
        print(" - number of batches: {}".format(len(train_loader)))

    if opt.is_test:
        print("\n>>> creating Test dataset")
        # create dataset of type Human36M
        test_h36 = \
            Human36M( misc, cameras_dict_test, data_dict_test, stat_2d, stat_3d,
                      tol_mm, num_pairs, amt_test_data, test_rel_labels_noise_prob,
                      in_dropout_p, is_train=False )
        # create data loader
        test_loader = DataLoader(dataset=test_h36, batch_size=opt.test_batch,
                                  shuffle=False, num_workers=0, drop_last=False)
        print(" - number of batches: {}".format(len(test_loader)))

    print("==========================================\n")
    print("\n==================Model=================")
    print(">>> creating model")

    num_2d_coords = misc.NUM_KEYPOINTS_2D * 2
    num_3d_coords = misc.NUM_KEYPOINTS_3D * 3

    model = LinearModel(num_2d_coords, num_3d_coords,
                    linear_size=opt.linear_size,
                    num_stage=opt.num_stage,
                    p_dropout=opt.dropout,
                    predict_scale=opt.predict_scale,
                    scale_range=opt.scale_range,
                    unnorm_op=opt.unnorm_op,
                    unnorm_init=opt.unnorm_init)

    model = model.cuda()
    model.apply(weight_init)
    print(" - total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("==========================================\n")

    ############################################################################
    # define losses and optimizers
    mse_loss = nn.MSELoss(size_average=True).cuda()
    # mse_loss = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    cudnn.benchmark = True

    ############################################################################
    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)

        stat_3d   = ckpt['stat_3d']
        stat_2d   = ckpt['stat_2d']

        err_best  = ckpt['err']
        lr_now    = ckpt['lr']
        glob_step = ckpt['step']

        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

        if not opt.resume:
            print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    if opt.resume:
        assert opt.load != ''
        start_epoch = ckpt['epoch']
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['it', 'lr', 'l_train', 'l_test', 'e_test', 'e_test_s', 'e_test_p'])

    ############################################################################
    ## TRAINING LOOP
    overall_train_losses = [[0],[0],[0],[0]]
    loss_lbls =  ['sup_loss','rel_loss', 'rep_loss', 'cam_loss']
    for epoch in range(start_epoch, opt.epochs):
        print('\n==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        ########################################################################
        ## TRAIN
        avg_loss_train = -1
        if opt.is_train:
            print('\n - Training')
            glob_step, lr_now, avg_loss_train, losses_train = \
                    train_human(
                        train_loader=train_loader, misc=misc,
                        stat_2d=stat_2d, stat_3d=stat_3d,
                        standardize_input_data=opt.standardize_input_data,
                        standardize_output_data=opt.standardize_output_data,
                        use_rel_loss=opt.use_rel_loss,
                        subtract_2d_root=opt.subtract_2d_root,
                        keep_root=opt.keep_root,
                        optimizer=optimizer, model=model, mse_loss=mse_loss,
                        reprojection=opt.reprojection,
                        use_full_intrinsics=opt.use_full_intrinsics,
                        predict_scale=opt.predict_scale, limb_type=opt.limb_type,
                        glob_step=glob_step, lr_init=opt.lr, lr_now=lr_now,
                        lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
                        max_norm=opt.max_norm,
                        distance_multiplier=opt.distance_multiplier,
                        loss_weights=opt.loss_weights)
            for li,l in enumerate(overall_train_losses): l.extend(losses_train[li])
            viz.plot_losses(overall_train_losses, loss_lbls,
                            opt.ckpt + '/train_losses.jpg', 'Training Set', 'iterations', 'losses')

        ########################################################################
        ## TEST
        loss_test = err_test = err_test_scale = err_test_proc = -1
        if opt.is_test and (glob_step) % opt.test_step == 0:
            print('\n - Testing')
            loss_test, target_poses, out_poses, proc_poses, scaled_poses = \
                    test_human(
                        test_loader=test_loader, misc=misc,
                        stat_2d=stat_2d, stat_3d=stat_3d,
                        standardize_input_data=opt.standardize_input_data,
                        standardize_output_data=opt.standardize_output_data,
                        subtract_2d_root=opt.subtract_2d_root, keep_root=opt.keep_root,
                        model=model, mse_loss=mse_loss, use_rel_loss=opt.use_rel_loss,
                        save_ims=opt.save_ims, epoch=epoch, op_dir=opt.ckpt_ims)

            target_poses = np.vstack(target_poses)
            out_poses    = np.vstack(out_poses)
            scaled_poses = np.vstack(scaled_poses)
            proc_poses   = np.vstack(proc_poses)

            ####################################################################
            ## compute error in mm for both protocols (with and without procrustes)
            sqerr        = (out_poses - target_poses) ** 2
            sqerr_proc   = (proc_poses - target_poses) ** 2
            sqerr_scaled = (scaled_poses - target_poses) ** 2

            all_err = np.sqrt(sqerr[:, 0::3] + sqerr[:,1::3] + sqerr[:,2::3])
            all_err_scaled = np.sqrt(sqerr_scaled[:, 0::3] + sqerr_scaled[:,1::3] + sqerr_scaled[:,2::3])
            all_err_proc = np.sqrt(sqerr_proc[:, 0::3] + sqerr_proc[:,1::3] + sqerr_proc[:,2::3])

            err_test       = np.mean(all_err)
            err_test_scale = np.mean(all_err_scaled)
            err_test_proc  = np.mean(all_err_proc)
            print ("> 3d error              {}".format(round(err_test, 3)))
            print ("> 3d error (scaled)     {}".format(round(err_test_scale, 3)))
            print ("> 3d error (procrustes) {}".format(round(err_test_proc, 3)))
            print("-"*25)

            # compute the errors per action
            a_test = data_dict_test['A'][test_h36.inds]
            vals, counts = np.unique(a_test, return_counts=True)
            assert a_test.shape[0] == all_err_proc.shape[0], "Bad shapes."

            err_test_actions_last = []
            for vv, cc in zip(vals, counts):
                action_inds = np.where(a_test == vv)[0]
                action_errs = all_err_proc[action_inds]
                err_test_actions_last.append(action_errs.mean())

            for aa, bb in zip(err_test_actions_last, actions):
                print("> {:<12} 3d error: {:.3f}".format(bb,aa))
            print("> Overall avg : {:.3f}".format(np.mean(err_test_actions_last)))
            print("-"*25)

            # change this line to decide what error to use for storing best
            err_test_last = err_test_proc

            is_best = err_test_last < err_test_best
            err_test_best = min(err_test_last, err_test_best)
            if save_op:
                log.save_ckpt({'epoch': epoch + 1,
                               'lr': lr_now,
                               'step': glob_step,
                               'err': err_test_last,
                               'stat_2d':stat_2d,
                               'stat_3d':stat_3d,
                               'opt':opt,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()},
                              ckpt_path=opt.ckpt,
                              data_split='test',
                              is_best=is_best)

        # update log file
        logs = [glob_step, lr_now, avg_loss_train, loss_test, err_test, err_test_scale, err_test_proc]
        logs_type = ['int', 'float', 'float', 'float', 'float', 'float', 'float']

        if save_op:
            logger.append(logs, logs_type)

    logger.close()
    if return_proc:
        return err_test_best, err_test_last, err_test_actions_last, all_err_proc, proc_poses, data_dict_test
    else:
        return err_test_best, err_test_last, err_test_actions_last

if __name__ == "__main__":

    # import the options
    from opts.default_human36 import opt

    print("\n==================Options=================")
    pprint(vars(opt), indent=4)
    if not os.path.isdir(opt.ckpt): os.makedirs(opt.ckpt)
    if not os.path.isdir(opt.ckpt_ims): os.makedirs(opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last = main_human(opt)
    print("Testing Errors:")
    print(" - Best:            [{}]".format(round(err_test_best), 3))
    print(" - Last:            [{}]".format(round(err_test_last), 3))
    print(" - Last Avg Action: [{}]".format(round(np.mean(err_test_actions_last)), 3))
