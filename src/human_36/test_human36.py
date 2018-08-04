import numpy as np
import time

import torch
from torch.autograd import Variable

import src.utils.utils as utils
from src.utils.procrustes import get_transformation

import src.utils.viz as viz


def test_human(test_loader, misc, stat_2d, stat_3d,
               standardize_input_data, standardize_output_data,
               use_rel_loss, subtract_2d_root, keep_root,
               model, mse_loss, save_ims=False, epoch=None,
               op_dir=None):

    # list with all the predicted poses
    target_poses = []
    out_poses    = []
    scaled_poses = []
    proc_poses   = []

    # dictionary for storing a few images for visualization purposes
    output_for_viz = {'pts_2d':[], 'gt_3d':[], 'pred_3d':[], 'pred_3d_proc':[]}

    # at test time keep track only of the full 3d supervised mean squared error loss
    losses_sup = utils.AverageMeter()

    outputs_mean = Variable(torch.from_numpy(stat_3d['mean'][np.newaxis, ...]).cuda(),requires_grad=False)
    # outputs_mean = Variable(torch.from_numpy(stat_3d['mean'][np.newaxis, ...]),requires_grad=False)
    outputs_std  = Variable(torch.from_numpy(stat_3d['std'][np.newaxis, ...]).cuda(),requires_grad=False)
    # outputs_std  = Variable(torch.from_numpy(stat_3d['std'][np.newaxis, ...]),requires_grad=False)

    model.eval()

    tic = time.time()
    for i, test_data in enumerate(test_loader):
        ########################################################################
        # load data
        inps, norm_inps, inps_root, tars, norm_tars, tars_root, _, _, _, _ = test_data

        num_keypoints = int(inps.shape[1] / 2) # inps are the 2d coordinates
        batch_size    = inps.shape[0]

        inputs    = Variable(inps.cuda(), requires_grad=False)
        # inputs    = Variable(inps, requires_grad=False)
        targets   = Variable(tars.cuda(), requires_grad=False)
        # targets   = Variable(tars, requires_grad=False)
        tars_root = Variable(tars_root.repeat(1, num_keypoints).cuda(),requires_grad=False)
        # tars_root = Variable(tars_root.repeat(1, num_keypoints),requires_grad=False)

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

        model_outputs, _ = model(model_inputs)
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
            # the network relies on the un-norm_op to unstandardize the data so the
            # model outputs should already be un-standardized
            outputs = model_outputs

        # add the root back to the outputs and to the targets
        # outputs = outputs + tars_root
        # targets = targets + tars_root

        ########################################################################
        # supervised loss
        loss = mse_loss(outputs, targets)
        # loss = mse_loss(model_outputs, model_targets)
        losses_sup.update(loss.data[0], batch_size)

        ########################################################################
        # do plotting and compute errors using numpy
        # use unnormalized version of all the data
        targets   = targets.data.cpu().numpy()
        # targets = tars.numpy()
        outputs   = outputs.data.cpu().numpy()
        inputs    = inps.numpy()
        inps_root = inps_root.numpy()

        ########################################################################
        # add the root to the inputs if specified by the flags
        if subtract_2d_root:
            inputs += np.tile(inps_root, num_keypoints)

        ########################################################################
        # NOTE: MUST insert the root back to the targets if keep_root is False
        # otherwise the reconstruction fails
        if not keep_root: raise NotImplementedError("Must add the root in prediction.")

        # compute the error with procrustes alignment
        outputs_proc   = np.zeros(outputs.shape)
        outputs_scaled = np.zeros(outputs.shape)
        for ba in range(batch_size):
            gt  = targets[ba, :].reshape(-1, 3)
            out = outputs[ba, :].reshape(-1, 3)
            _, Z, T, b, c = get_transformation(gt, out, True)
            proc   = (b * out.dot(T)) + c
            scaled = b * out
            outputs_proc[ba, :]   = proc.reshape(1, num_keypoints * 3)
            outputs_scaled[ba, :] = scaled.reshape(1, num_keypoints * 3)

        target_poses.append(np.vstack(targets[np.newaxis,...]))
        out_poses.append(np.vstack(outputs[np.newaxis,...]))
        scaled_poses.append(np.vstack(outputs_scaled[np.newaxis,...]))
        proc_poses.append(np.vstack(outputs_proc[np.newaxis,...]))

        ########################################################################
        # save poses for visualization - select diverse data by spacing out selection
        if save_ims and (i + 1) % (len(test_loader)//15) == 0:
            output_for_viz['pts_2d'].append(inputs[0,:])
            output_for_viz['gt_3d'].append(targets[0,:])
            output_for_viz['pred_3d'].append(outputs[0,:])
            output_for_viz['pred_3d_proc'].append(outputs_proc[0,:])

        ########################################################################
        # update summary
        if (i + 1) % 1000 == 0:
            its_time = time.time() - tic
            print(' ({batch}/{size}) \t| sup loss {loss:.4f} | time {its_time:.3f}s' \
                  .format(batch=i+1, size=len(test_loader), loss=losses_sup.avg, its_time=its_time))
            tic = time.time()

    ############################################################################
    # save image
    if save_ims:
        op_file_name = op_dir + '/' + str(epoch+1).zfill(3) + '.jpg'
        viz.save_output_image(op_file_name, output_for_viz, misc)

    return losses_sup.avg, target_poses, out_poses, proc_poses, scaled_poses
