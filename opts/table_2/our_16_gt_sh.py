import os

class Options(object):
    def __call__(self):
        return None

opt = Options()

# path of h5 annotations and camera params
opt.data_dir   = '../../h5_annotations'

# save path for experiments and experiment name
opt.ckpt = '../checkpoint/table_2'
opt.exp  = 'our_16_gt_sh'
opt.ckpt = os.path.join(opt.ckpt, opt.exp)
opt.ckpt_ims = opt.ckpt + '/ims'

# for loading and optionally resuming training
opt.load    = ''
opt.resume  = False

# do the training
opt.is_train = True
# do the testing
opt.is_test  = True

# running options
opt.epochs           = 25
opt.test_batch       = 64
opt.train_batch      = 64

opt.dropout          = 0.5
opt.lr               = 0.001
opt.lr_decay         = 100000
opt.lr_gamma         = 0.96
opt.test_step        = 1 # run test every test_step train calls
opt.save_ims         = True
opt.job              = 8

# actions to use
opt.action = "all"

# choose the type of dataset
opt.dataset_type = 'shg_16k'

# amount of data and splits
opt.amt_train_data = 1.  # percentage of data to keep.
opt.amt_test_data  = 1  # every how ever many frames to sample if larger than 1
opt.train_subjects = [1,5,6,7,8]
opt.test_subjects  = [9,11]

# decide the type and amount of 2d keypoints input noise
# here you can set all the options for training with GT / SH and testing with GT / SH
# options for noise are:
# 'none', 'uniform_linear', 'uniform_radial', 'normal', 'detections'
opt.train_noise_type_2d   = 'none'
opt.train_noise_amount_2d = 0
opt.test_noise_type_2d    = 'detections'
opt.test_noise_amount_2d  = 0

# probability of flipping a relative label (0 for noiseless)
# NOTE: this is hard coded to 0 for the TEST set
# use 'mturk' to simulate noise amount observed in turkers for training data
opt.rel_labels_noise_prob  = 0.

# input dropout probability
opt.in_dropout_p = 0.

# data formatting
opt.camera_coord_3d  = True
opt.keep_root        = True
opt.only_depth       = False
opt.subtract_2d_root = True

# model options
opt.max_norm    = True
opt.linear_size = 1024
opt.num_stage   = 1

# choose loss type (3d supervised or relative)
opt.use_rel_loss = True

# relative loss
opt.reprojection     = 'scaled_orthographic'

opt.standardize_input_data  = True

# unnorm_op is a layer in the model that does the unstandardize operation
# if the standardize_output_data flag is false
opt.standardize_output_data = False
opt.unnorm_op        = True
opt.unnorm_init      = 100.

# THESE OPTIONS DON'T MATTER FOR THE SUPERVISED LOSS

# if these values are > 0.0 the loss will be activated
# for supervised baseline it doesn't matter what they are, except subj should be 0
opt.loss_weights  = {}
opt.loss_weights['relative'] = 1.0
opt.loss_weights['reproj']   = 0.1
opt.loss_weights['root']     = 1.0
opt.loss_weights['symm']     = 1.0

# for symmetry loss computation
opt.limb_type = 'avg_human36' # other options are 'avg_human36', 'avg_person'

# for relative loss computation (don't matter for supervised loss)
opt.num_pairs    = 1 # 'all', None, or int ('all'=136 for 17 kpts)
opt.tolerance_mm = 0.
opt.distance_multiplier = 2.5 # 0.01

# reprojection loss parameters
if opt.reprojection == 'scaled_orthographic':
    opt.predict_scale       = True
    opt.scale_range         = 1.
    opt.use_full_intrinsics = False

elif opt.reprojection == 'weak_perspective':
    opt.predict_scale       = False
    opt.scale_range         = 0.001
    # opt.scale_range         = 10000.
    opt.use_full_intrinsics = False

else:
    assert opt.reprojection == 'none'
    opt.predict_scale       = False
    opt.scale_range         = -1
    opt.use_full_intrinsics = False
