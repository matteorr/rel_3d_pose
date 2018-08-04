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

from src.data_loading.lsp_dataset import LSP

from src.model import LinearModel, weight_init

from src.lsp.train_lsp import train_lsp
from src.lsp.test_lsp import test_lsp

import src.utils.viz as viz

def main_lsp(opt, save_op=True, return_poses=False):
    start_epoch = 0
    err_test_best = 100000
    glob_step = 0
    lr_now = opt.lr

    # save options
    if save_op:
        log.save_options(opt, opt.ckpt)

    print("==================Data=================")

    # get info related to dataset type
    misc = DatasetMisc(opt.dataset_type)

    ############################################################################
    # create train data loader
    print(">>> creating Train dataset")
    train_dataset_lsp = LSP(misc, opt, is_train = True)
    train_loader = DataLoader(dataset=train_dataset_lsp, batch_size=opt.train_batch,
                              shuffle=True, num_workers=opt.job)
    stat_2d = {}
    stat_3d = {}
    stat_2d['lsp_mean'] = train_dataset_lsp.stat_2d['mean']
    stat_2d['lsp_std']  = train_dataset_lsp.stat_2d['std']
    print(" - number of batches: {}".format(len(train_loader)))

    ############################################################################
    # create test data loader
    print("\n>>> creating Test dataset")
    test_dataset_lsp = LSP(misc, opt, is_train = False)
    test_loader = DataLoader(dataset=test_dataset_lsp, batch_size=opt.test_batch,
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

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    cudnn.benchmark = True

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)

        loaded_stat_3d = ckpt['stat_3d']
        loaded_stat_2d = ckpt['stat_2d']
        # add the keys of the loaded stat dicts to the current stat dicts
        for k,v in loaded_stat_2d.items(): stat_2d[k] = v
        for k,v in loaded_stat_3d.items(): stat_3d[k] = v

        err_best = ckpt['err']
        lr_now = ckpt['lr']
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
        logger.set_names(['it', 'lr', 'l_train', 'e_train', 'e_test'])

    ############################################################################
    ## TRAINING LOOP
    overall_train_losses = [[0],[0],[0]]
    loss_lbls =  ['rel_loss', 'rep_loss', 'cam_loss']
    for epoch in range(start_epoch, opt.epochs):
        print('\n==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        ########################################################################
        ## TRAIN
        err_train = avg_loss_train = -1
        if opt.is_train:
            print('\n - Training')
            glob_step, lr_now, avg_loss_train, losses_train, all_err_train = \
                    train_lsp(train_loader=train_loader, misc=misc,
                              stat_2d=stat_2d, stat_3d=stat_3d, limb_type=opt.limb_type,
                              standardize_input_data=opt.standardize_input_data,
                              standardize_output_data=opt.standardize_output_data,
                              use_loaded_stats=opt.use_loaded_stats,
                              use_rel_loss=opt.use_rel_loss,
                              subtract_2d_root=opt.subtract_2d_root, keep_root=opt.keep_root,
                              optimizer=optimizer, model=model,
                              predict_scale=opt.predict_scale,
                              glob_step=glob_step, lr_init=opt.lr, lr_now=lr_now, lr_decay=opt.lr_decay,
                              gamma=opt.lr_gamma, max_norm=opt.max_norm, distance_multiplier=opt.distance_multiplier,
                              loss_weights=opt.loss_weights)
            for li,l in enumerate(overall_train_losses): l.extend(losses_train[li])
            viz.plot_losses(overall_train_losses, loss_lbls,
                            opt.ckpt + '/train_losses.jpg', 'Training Set', 'iterations', 'losses')

            err_train = np.mean(np.vstack(all_err_train))
            print ("> Rel labels error (train):   {}".format(round(err_train, 3)))

        ########################################################################
        ## TEST
        err_test_last = -1
        if opt.is_test and (glob_step) % opt.test_step == 0:
            print('\n - Testing')
            all_err_test, all_poses = test_lsp(
                                        test_loader=test_loader, misc=misc,
                                        stat_2d=stat_2d, stat_3d=stat_3d,
                                        standardize_input_data=opt.standardize_input_data,
                                        standardize_output_data=opt.standardize_output_data,
                                        use_loaded_stats=opt.use_loaded_stats,
                                        use_rel_loss=opt.use_rel_loss,
                                        subtract_2d_root=opt.subtract_2d_root,
                                        keep_root=opt.keep_root,
                                        model=model, save_ims=opt.save_ims,
                                        epoch=epoch, op_dir=opt.ckpt_ims)

            err_test_last = np.mean(np.vstack(all_err_test))
            print ("> Rel labels error (test):   {}".format(round(err_test_last, 3)))

            is_best = err_test_last < err_test_best
            err_test_best = min(err_test_last, err_test_best)
            if save_op:
                log.save_ckpt({'epoch': epoch + 1,
                               'lr': lr_now,
                               'step': glob_step,
                               'err': err_test_best,
                               'stat_2d':stat_2d,
                               'stat_3d':stat_3d,
                               'opt':opt,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()},
                              ckpt_path=opt.ckpt,
                              data_split='test',
                              is_best=is_best)

        # update log file
        logger.append([glob_step, lr_now, avg_loss_train, err_train, err_test_last],
                      ['int', 'float', 'float', 'float', 'float'])

    logger.close()
    if return_poses:
        return err_test_best, err_test_last, np.vstack([i[0] for i in all_poses])[0::5,:]
    else:
        return err_test_best, err_test_last

if __name__ == "__main__":

    # import the options
    from opts.default_lsp import opt

    print("\n==================Options=================")
    pprint(vars(opt), indent=4)
    if not os.path.isdir(opt.ckpt): os.makedirs(opt.ckpt)
    if not os.path.isdir(opt.ckpt_ims): os.makedirs(opt.ckpt_ims)
    print("==========================================\n")

    # if using lsp certain option values are forced
    err_str = "This parameter is not compatible with training/testing LSP data"
    assert opt.dataset_type == 'lsp_14k', err_str
    assert opt.only_depth   == False, err_str
    assert opt.keep_root    == True, err_str
    if opt.use_rel_loss:
        assert opt.reprojection == 'scaled_orthographic', err_str
    if opt.limb_type == 'avg_human36':
        assert opt.load != '', err_str
    assert opt.tolerance_mm == 0., err_str

    err_test_best, err_test_last = main_lsp(opt)
    print("Testing Errors:")
    print(" - Best: [{}]".format(err_test_best))
    print(" - Last: [{}]".format(err_test_last))
