import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.rel_losses as rel_losses
import src.utils.utils as utils

def train_lsp(train_loader, misc, stat_2d, stat_3d, limb_type,
              standardize_input_data, standardize_output_data, use_loaded_stats,
              use_rel_loss, subtract_2d_root, keep_root, optimizer, model,
              predict_scale, glob_step=None, lr_init=None, lr_now=None, lr_decay=None,
              gamma=None, max_norm=True, distance_multiplier=1., loss_weights=None):

    # list containing the errors done on every training set example
    all_err = []

    losses_rel = utils.AverageMeter()
    losses_rep = utils.AverageMeter()
    losses_cam = utils.AverageMeter()
    losses_tot = utils.AverageMeter()

    if standardize_output_data:
        # NOTE: in the case of the 3d supervised model the only the output is
        # un-standardized using the values from the data on which the model was
        # pre-trained.
        outputs_mean = Variable(torch.from_numpy(stat_3d['mean'][np.newaxis, ...]).cuda(),requires_grad=False)
        outputs_std  = Variable(torch.from_numpy(stat_3d['std'][np.newaxis, ...]).cuda(),requires_grad=False)

    if use_loaded_stats:
        # input mean and standard deviation loaded from the checkpoint file
        loaded_inputs_mean = torch.from_numpy(stat_2d['mean'][np.newaxis, ...])
        loaded_inputs_std  = torch.from_numpy(stat_2d['std'][np.newaxis, ...])

    if limb_type == 'avg_person':
        # load this from the misc
        # NOTE: values of -1 in the misc or joint in the misc.SKELETON MASK will be ignored
        limb_lens_3d = Variable(torch.from_numpy(np.array(misc.SKELETON_3D_LENS_AVG_PERSON).astype(
                    stat_2d['lsp_mean'].dtype)).unsqueeze(0).cuda(),requires_grad=False)

    elif limb_type == 'avg_human36':
        limb_lens_3d = Variable(torch.from_numpy(stat_3d['avg_limb_lens'].astype(
                                stat_2d['lsp_mean'].dtype)).cuda(),requires_grad=False)

    else:
        assert False, "Unkown value [%s] for parameter [limb_type]"%limb_type

    # load mask of limbs to ignore in loss about body proportions
    mask = misc.SKELETON_3D_MASK
    if len(mask) > 0: limb_lens_3d[:,mask] = -1

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
        inps, norm_inps, inps_root, rel_inds, rel_gt = train_data
        num_keypoints = int(inps.shape[1] / 2) # inps are the 2d coordinates
        batch_size    = inps.shape[0]

        inputs   = Variable(inps.cuda())
        rel_inds = Variable(rel_inds.cuda())
        rel_gt   = Variable(rel_gt.cuda())

        ########################################################################
        # normalize if specified by the flags
        if standardize_input_data:
            if not use_loaded_stats:
                # use the data that is already normalized
                model_inputs  = Variable(norm_inps.cuda())

            else:
                # normalize the data according to the stat_2d loaded from ckpt
                norm_data = (inps - loaded_inputs_mean) / loaded_inputs_std
                norm_data[np.isnan(norm_data)] = 0

                model_inputs  = Variable(norm_data.cuda())

        else:
            model_inputs  = inputs

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
        # add the root to the inputs if specified by the flags
        if not keep_root: raise NotImplementedError("Must add the root in prediction.")

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
        # predicted model_scale gives the estimated values of f/z
        # NOTE: out 3d poses are assumed to be centered in 0,0,0 and in 2d poses in 0,0
        # since the tar_root or the inp_root are not added to outputs or inputs
        focal_length_over_dist = model_scale
        rep_loss = loss_weights['reproj'] * \
                   rel_losses.reproj_loss_scaled_orthographic(outputs, inputs, focal_length_over_dist)
        losses_rep.update(rep_loss.data[0], batch_size)
        if np.isnan(rep_loss.mean().data[0]):
            print('nans in rep loss')
            import ipdb;ipdb.set_trace()

        ####################################################################
        # CAMERA LOSS
        # symmetry, length losses in the camera space need both norm and unnorm
        # weighting happens inside the loss

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

        ########################################################################
        # clip gradients and take a step with the optimizer
        if max_norm: nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        ########################################################################
        # compute the errors that were made in predicting the relative depth of
        # pairs of keypoints
        rel_error_outputs = outputs.data.cpu().numpy()
        rel_error_inds    = rel_inds.data.cpu().numpy()
        rel_error_gts     = rel_gt.data.view(-1).cpu().numpy()
        kpt_0_z = rel_error_outputs[range(batch_size), rel_error_inds[:, 0]]
        kpt_1_z = rel_error_outputs[range(batch_size), rel_error_inds[:, 1]]
        # kp0 < kp1 -> -1, otherwise 1
        rel_pred = -1 * np.sign((kpt_0_z < kpt_1_z) * 1 - .5)

        num_errs = (rel_pred != rel_error_gts).sum()
        all_err.append(num_errs / float(batch_size))

        ########################################################################
        # update summary
        if (i + 1) % 10 == 0:
            its_time = time.time() - tic

            # visualization code:
            # skeleton_pairs_2d, skeleton_pairs_3d = misc.get_skeleton_pairs()
            # pred_limb_lens = torch.zeros((batch_size,len(skeleton_pairs_3d)))
            # for l, limb in enumerate(skeleton_pairs_3d):
            #     kpt_0_coord = [3 * limb[0], 3 * limb[0] + 1, 3 * limb[0] + 2]
            #     kpt_1_coord = [3 * limb[1], 3 * limb[1] + 1, 3 * limb[1] + 2]
            #     pred_limb_lens[:,l] = torch.sqrt(((model_outputs[:,kpt_0_coord].data - model_outputs[:,kpt_1_coord].data) ** 2).sum(dim=1))
            #
            # np.set_printoptions(precision=3)
            # print(model_scale.mean(dim=0).data[0])
            # print(pred_limb_lens.mean(dim=0).numpy())

            # print_str  = ' ({global_step}) \t'
            print_str  = ' ({batch}/{size}) \t'
            print_str += ' | rel loss {rel_loss:.4f}'
            print_str += ' | rep loss {rep_loss:.4f}'
            print_str += ' | cam loss {cam_loss:.4f} | time {its_time:.3f}s'
            print(print_str.format(batch=i+1, size=len(train_loader), #global_step=glob_step,
                                   rel_loss=losses_rel.avg,
                                   rep_loss=losses_rep.avg, cam_loss=losses_cam.avg,
                                   its_time=its_time))
            tic = time.time()

        ########################################################################
        # return the correct loss for validation
        losses_avg = losses_tot.avg
        losses_list = [losses_rel.list, losses_rep.list, losses_cam.list]

    return glob_step, lr_now, losses_avg, losses_list, all_err
