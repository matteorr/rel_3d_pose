import numpy as np
import time

import torch
from torch.autograd import Variable

import src.utils.utils as utils
from src.utils.procrustes import get_transformation

import src.utils.viz as viz


def test_lsp(test_loader, misc, stat_2d, stat_3d,
             standardize_input_data, standardize_output_data, use_loaded_stats,
             use_rel_loss, subtract_2d_root, keep_root, model,
             save_ims=False, epoch=None, op_dir=None):

    # list containing the errors done on every test set example
    all_err   = []
    # list with all the predicted poses
    all_poses = []
    # dictionary for storing a few images for visualization purposes
    output_for_viz = {'pts_2d':[], 'pred_3d':[]}

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

    model.eval()

    tic = time.time()
    for i, test_data in enumerate(test_loader):
        ########################################################################
        # load data
        inps, norm_inps, inps_root, rel_inds, rel_gt, = test_data

        num_keypoints = int(inps.shape[1] / 2) # inps are the 2d coordinates
        batch_size    = inps.shape[0]

        inputs   = Variable(inps.cuda(), requires_grad=False)
        rel_inds = rel_inds.numpy()
        rel_gt   = rel_gt.view(-1).numpy()

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
        # do plotting and compute errors using numpy
        # use unnormalized version of all the data
        outputs   = outputs.data.cpu().numpy()
        inputs    = inps.numpy()
        inps_root = inps_root.numpy()

        ########################################################################
        # add the root to the inputs if specified by the flags
        if subtract_2d_root:
            inputs += np.tile(inps_root, num_keypoints)
        # NOTE: this is not necessary for 3d outputs because the data loader always
        # subtracts the root from the 3d data, so it should not be added back.
        # NOTE: if there was a flag subtract_3d_root this would change...

        ########################################################################
        # NOTE: MUST insert the root back to the targets if keep_root is False otherwise the reconstruction fails
        if not keep_root: raise NotImplementedError("Must add the root in prediction.")

        # add poses to the list
        all_poses.append(outputs[np.newaxis,...])

        ########################################################################
        # compute the errors that were made in predicting the relative depth of
        # pairs of keypoints
        kpt_0_z = outputs[range(batch_size), rel_inds[:, 0]]
        kpt_1_z = outputs[range(batch_size), rel_inds[:, 1]]
        # kp0 < kp1 -> -1, otherwise 1
        rel_pred = -1 * np.sign((kpt_0_z < kpt_1_z) * 1 - .5)

        num_errs = (rel_pred != rel_gt).sum()
        all_err.append(num_errs / float(batch_size))

        ########################################################################
        # save poses for visualization - select diverse data by spacing out selection
        if save_ims and (i + 1) % (len(test_loader)//25) == 0:
            output_for_viz['pts_2d'].append(inputs[0,:])
            output_for_viz['pred_3d'].append(outputs[0,:])

        ########################################################################
        # update summary
        if (i + 1) % 10 == 0:
            its_time = time.time() - tic
            print(' ({batch}/{size}) \t| time {its_time:.3f}s' \
                  .format(batch=i+1, size=len(test_loader), its_time=its_time))
            tic = time.time()

    ############################################################################
    # save image
    if save_ims:
        op_file_name = op_dir + '/' + str(epoch+1).zfill(3) + '.jpg'
        viz.save_output_image_lsp(op_file_name, output_for_viz, misc)

        return all_err, all_poses
