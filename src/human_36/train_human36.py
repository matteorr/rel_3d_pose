import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.rel_losses as rel_losses
import src.utils.utils as utils

def train_human(train_loader, misc, stat_2d, stat_3d,
                standardize_input_data, standardize_output_data,
                use_rel_loss, subtract_2d_root, keep_root, optimizer, model,
                mse_loss, reprojection, use_full_intrinsics, predict_scale, limb_type,
                glob_step=None, lr_init=None, lr_now=None, lr_decay=None,
                gamma=None, max_norm=True, distance_multiplier=1., loss_weights=None):

    losses_sup = utils.AverageMeter()
    losses_rel = utils.AverageMeter()
    losses_rep = utils.AverageMeter()
    losses_cam = utils.AverageMeter()
    losses_tot = utils.AverageMeter()

    outputs_mean = Variable(torch.from_numpy(stat_3d['mean'][np.newaxis, ...]).cuda(),requires_grad=False)
    # outputs_mean = Variable(torch.from_numpy(stat_3d['mean'][np.newaxis, ...]),requires_grad=False)
    outputs_std  = Variable(torch.from_numpy(stat_3d['std'][np.newaxis, ...]).cuda(),requires_grad=False)
    # outputs_std  = Variable(torch.from_numpy(stat_3d['std'][np.newaxis, ...]),requires_grad=False)

    if limb_type == 'avg_person':
        # load this from the misc (NOTE that values of -1 in the misc will be ignored)
        avg_person_limb_lens = np.array(misc.SKELETON_3D_LENS_AVG_PERSON).astype(stat_3d['mean'].dtype)[np.newaxis,...]
        train_avg_limb_lens = Variable(torch.from_numpy(avg_person_limb_lens).cuda(),requires_grad=False)
        # train_avg_limb_lens = Variable(torch.from_numpy(avg_person_limb_lens),requires_grad=False)

    elif limb_type == 'avg_human36':
        train_avg_limb_lens = Variable(torch.from_numpy(stat_3d['avg_limb_lens']).cuda(),requires_grad=False)
        # train_avg_limb_lens = Variable(torch.from_numpy(stat_3d['avg_limb_lens']),requires_grad=False)

    else:
        if limb_type != 'gt': assert use_rel_loss == False

    # load mask of limbs to ignore in loss about body proportions
    mask = misc.SKELETON_3D_MASK

    model.train()

    tic = time.time()
    for i, train_data in enumerate(train_loader):
        ########################################################################
        # keep track of the global step and adjust the learning rate accordingly
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        ########################################################################
        # load data
        inps, norm_inps, inps_root, \
        tars, norm_tars, tars_root, \
        rel_inds, rel_gt, gt_limb_lens_3d, camera_params = train_data

        num_keypoints = int(inps.shape[1] / 2) # inps are the 2d coordinates
        batch_size    = inps.shape[0]

        inputs   = Variable(inps.cuda())
        # inputs   = Variable(inps)
        targets  = Variable(tars.cuda())
        # targets  = Variable(tars)
        rel_inds = Variable(rel_inds.cuda())
        # rel_inds = Variable(rel_inds)
        rel_gt   = Variable(rel_gt.cuda())
        # rel_gt   = Variable(rel_gt)

        ########################################################################
        # standardize data based on flags
        if standardize_input_data:
            # uses standardized 2d inputs
            model_inputs  = Variable(norm_inps.cuda())
            # model_inputs  = Variable(norm_inps)

        else:
            model_inputs  = inputs

        # if standardize_output_data:
        #     # uses standardized 3d outputs. NOTE: this is using 3d data
        #     model_targets = Variable(norm_tars.cuda())
        #
        #     # note that 3d outputs are not standardized based on the training data
        #     # for the relative loss since it cannot use any 3d information
        #     # (not even mean and std), relies on un-norm_op to unstandardize the data
        #     # assert use_rel_loss == False, "Cannot use 3d data for relative_loss!"
        #
        # else:
        #     model_targets = targets

        ########################################################################
        # pass through the network
        optimizer.zero_grad()
        model_outputs, model_scale = model(model_inputs)
        if np.isnan(model_outputs.mean().data[0]):
            print('nans in prediction')
            import ipdb;ipdb.set_trace()

        ########################################################################
        # un-standardize data based on flags
        if standardize_output_data:
            # can use the 3d info from training set to unstandardize
            outputs = outputs_mean + model_outputs * outputs_std
            assert use_rel_loss == False, "Cannot use 3d data for relative_loss!"

        else:
            # the network relies on the un-norm_op to unstandardize the data
            outputs = model_outputs
            assert use_rel_loss, "Supervised method must use output normalization."

        ########################################################################
        # add the root coordinates back if they were removed at data loading time
        if not keep_root: raise NotImplementedError

        if not use_rel_loss:
            # 3d supervised

            tars_root = Variable(tars_root.repeat(1, num_keypoints).cuda(),requires_grad=False)
            # tars_root = Variable(tars_root.repeat(1, num_keypoints),requires_grad=False)
            outputs = outputs + tars_root
            targets = targets + tars_root

            ####################################################################
            # SUPERVISED LOSS, requires the data to be normalized
            sup_loss = mse_loss(outputs, targets)
            # sup_loss = mse_loss(model_outputs, model_targets)
            losses_sup.update(sup_loss.data[0], batch_size)
            if np.isnan(sup_loss.mean().data[0]):
                print('nans in sup loss')
                import ipdb;ipdb.set_trace()

            losses_tot.update(-1)
            losses_cam.update(-1, batch_size)
            losses_rel.update(-1, batch_size)
            losses_rep.update(-1, batch_size)

            # take the backward step based on the supervised loss
            sup_loss.backward()

        else:
            # relative depth supervised

            ####################################################################
            # RELATIVE LOSS
            rel_loss = loss_weights['relative'] * \
                       rel_losses.relative_loss(outputs, rel_inds, rel_gt, distance_multiplier)
            losses_rel.update(rel_loss.data[0], batch_size)
            if np.isnan(rel_loss.mean().data[0]):
                print('nans in rel loss')
                import ipdb;ipdb.set_trace()

            ####################################################################
            # REPROJECTION LOSS
            if reprojection == 'scaled_orthographic':
                assert predict_scale, "Model must be predicting scale."
                # predicted model_scale gives the estimated values of f/z
                # NOTE: out 3d poses are assumed to be centered in 0,0,0 and in 2d poses in 0,0
                # since the tar_root or the inp_root are not added to outputs or inputs
                focal_length_over_dist = model_scale
                rep_loss = loss_weights['reproj'] * \
                           rel_losses.reproj_loss_scaled_orthographic(outputs, inputs, focal_length_over_dist)

            elif reprojection == 'weak_perspective':
                assert predict_scale, "Model must be predicting scale."
                # predict model_scale gives us the estimated z0 which we add to pred z
                # we use the ground truth focal length.
                # poses are still evaluated in 0,0,0 and 0,0
                # camera centers are unknown and true 3d location of pose is unknown
                # full_intrinsics not used
                f_batch  = Variable(camera_params[2].cuda(),requires_grad=False)
                # f_batch  = Variable(camera_params[2],requires_grad=False)
                # use inverted distance because it is less sensitive to saturation
                inverse_dist = 1. / model_scale
                rep_loss = loss_weights['reproj'] * \
                           rel_losses.reproj_loss_estimated_weak_perspective(outputs, inputs, inverse_dist, f_batch)

            # NOTE: this method is deprecated
            # elif reprojection == 'ground_truth_root_xyz':
            #     # model does not predict scale but we assume we have ground truth information
            #     # of where the person is located in the X,Y,Z space.
            #     # we do so by adding the target root location to the output root and
            #     # the 2d input root locations if previously subtracted.
            #     # this also allows us to use the full_intrinsics if they are known
            #     if subtract_2d_root:
            #         inps_root = Variable(inps_root.repeat(1, num_keypoints).cuda(), requires_grad=False)
            #         inputs = inputs + inps_root
            #
            #     # put the 3d prediction in the ground truth root coordinates
            #     tars_root = Variable(tars_root.repeat(1, num_keypoints).cuda(),requires_grad=False)
            #     outputs = outputs + tars_root
            #
            #     rep_loss = loss_weights['reproj'] * \
            #                rel_losses.reproj_loss_weak_perspective(outputs, inputs, camera_params, use_full_intrinsics)

            else:
                assert reprojection == 'none'

            losses_rep.update(rep_loss.data[0], batch_size)
            if np.isnan(rep_loss.mean().data[0]):
                print('nans in rep loss')
                import ipdb;ipdb.set_trace()

            ####################################################################
            # CAMERA LOSS
            # symmetry, length losses in the camera space need both norm and unnorm
            # weighting happens inside the loss

            # compute limb length for estimating losses
            if limb_type == 'gt':
                limb_lens_3d = Variable(gt_limb_lens_3d.squeeze(1).cuda(), requires_grad=False)
                # limb_lens_3d = Variable(gt_limb_lens_3d.squeeze(1), requires_grad=False)
            else:
                limb_lens_3d = train_avg_limb_lens
            if len(mask) > 0: limb_lens_3d[:,mask] = -1

            cam_loss = rel_losses.camera_coord_3d_loss(misc, model_outputs, outputs,
                                                       limb_lens_3d, loss_weights=loss_weights)
            losses_cam.update(cam_loss.data[0], batch_size)
            if np.isnan(cam_loss.mean().data[0]):
                print('nans in cam loss')
                import ipdb;ipdb.set_trace()

            ####################################################################
            # take the backward step based on the combination of all losses
            rel_loss_total = rel_loss + rep_loss + cam_loss
            rel_loss_total.backward()

            losses_tot.update(rel_loss_total.mean().data[0])
            losses_sup.update(-1, batch_size)

        ########################################################################
        # clip gradients and take a step with the optimizer
        if max_norm: nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        ########################################################################
        # update summary
        if (i + 1) % 1000 == 0:
            its_time = time.time() - tic

            # visualization code:
            # skeleton_pairs_2d, skeleton_pairs_3d = misc.get_skeleton_pairs()
            # pred_limb_lens = torch.zeros((batch_size,len(skeleton_pairs_3d)))
            # for l, limb in enumerate(skeleton_pairs_3d):
            #     kpt_0_coord = [3 * limb[0], 3 * limb[0] + 1, 3 * limb[0] + 2]
            #     kpt_1_coord = [3 * limb[1], 3 * limb[1] + 1, 3 * limb[1] + 2]
            #     pred_limb_lens[:,l] = torch.sqrt(((model_outputs[:,kpt_0_coord].data - model_outputs[:,kpt_1_coord].data) ** 2).sum(dim=1))
            # np.set_printoptions(precision=3)
            # print(model_scale.mean(dim=0).data[0])
            # print(pred_limb_lens.mean(dim=0).numpy())
            # print(stat_3d['train_avg_limb_lens'])

            # print_str  = ' ({global_step}) \t| sup loss {sup_loss:.4f}'
            print_str  = ' ({batch}/{size}) \t| sup loss {sup_loss:.4f}'
            print_str += ' | rel loss {rel_loss:.4f}'
            print_str += ' | rep loss {rep_loss:.4f}'
            print_str += ' | cam loss {cam_loss:.4f} | time {its_time:.3f}s'
            print(print_str.format(batch=i+1, size=len(train_loader), #global_step=glob_step,
                                   sup_loss=losses_sup.avg, rel_loss=losses_rel.avg,
                                   rep_loss=losses_rep.avg, cam_loss=losses_cam.avg,
                                   its_time=its_time))
            tic = time.time()

        ########################################################################
        # return the correct loss for validation
        losses_avg = losses_tot.avg if use_rel_loss else losses_sup.avg
        losses_list = [losses_sup.list, losses_rel.list, losses_rep.list, losses_cam.list]

    return glob_step, lr_now, losses_avg, losses_list
