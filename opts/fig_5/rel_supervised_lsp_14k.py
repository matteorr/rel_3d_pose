import os

class Options(object):
    def __call__(self):
        return None

opt = Options()

# if the model running on lsp is pretrained which stats should be used to
# normalized the data. If set to True then the stats of data the pretrained model
# was trained on will be used. If false the stats from the lsp data will be used.
opt.use_loaded_stats = False

# path of json annotations and camera params
opt.data_dir   = '/home/mronchi/Storage/Datasets/lsp/lsp_dataset/'
opt.mturk_data = './mturk_data/relative_depth_lsp_v1.json'

# save path for experiments and experiment name
opt.ckpt = '../checkpoint/fig_5'
opt.exp  = 'rel_supervised_lsp_14k'
opt.ckpt = os.path.join(opt.ckpt, opt.exp)
opt.ckpt_ims = opt.ckpt + '/ims'

# for loading and optionally resuming training
opt.load    = './checkpoint/fig_5/rel_supervised_human36_14k/test_ckpt_last.pth.tar'
opt.resume  = False

# do the training
opt.is_train = False
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

# choose the type of dataset
opt.dataset_type = 'lsp_14k'

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
opt.limb_type = 'avg_human36' # options are: None 'avg_human36', 'avg_person'

# for relative loss computation (don't matter for supervised loss)
opt.num_pairs    = 1 # 'all', None, or int ('all'=136 for 17 kpts)
opt.tolerance_mm = 0.
opt.distance_multiplier = 2.5 # 0.01

# reprojection loss parameters
opt.predict_scale       = True
opt.scale_range         = 1.
opt.use_full_intrinsics = False
