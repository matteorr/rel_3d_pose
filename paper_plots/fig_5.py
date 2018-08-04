#
# This script plots the predicted 3D skeletons for LSP.
#

import numpy as np
import sys
import json
import os

sys.path.append('../')
import src.utils.viz as viz
from src.data_formats.misc import DatasetMisc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


dataset_type = 'lsp_14k'
res_dir = '../checkpoint/fig_5/'
PATH_TO_LSP  = '../../../lsp_dataset/images/'
op_dir  = 'lsp_images/'

use_blender_rendering = False  # set this to True to render using blender
blender_path = 'PATH_TO_BLENDER' # if using blender this needs to point to blender 
blender_scene_path = './render_output/blank_scene_lsp.blend'
python_code_path = './render_output/render_single_frame_cmd.py'
op_dir_rend = op_dir + 'renders/'

if not os.path.isdir(op_dir_rend): os.makedirs(op_dir_rend)
if not os.path.isdir(op_dir): os.mkdir(op_dir)


with open('../mturk_data/relative_depth_lsp_v1.json','r') as fp: lsp = json.load(fp)
test_ids = [int(d['im_file'][9:13]) for d in lsp['images'] if d['is_train']==False]

# load data
pred_sup_pt = np.load(res_dir + 'fully_supervised_lsp_14k/all_poses.npy')
pred_rel_pt = np.load(res_dir + 'rel_supervised_lsp_14k/all_poses.npy')
pred_rel_ft = np.load(res_dir + 'rel_supervised_human36_lsp_14k/all_poses.npy')

pose_id = np.random.randint(pred_sup_pt.shape[0])
# [524, 808, 266, 52, 68, 166]

print 'pose', pose_id
im_id = test_ids[pose_id]
im = plt.imread(PATH_TO_LSP + 'im' + str(im_id).zfill(4) + '.jpg')
print 'image', im_id

misc = DatasetMisc(dataset_type)

# get 2D skeleton info
start_pt_2d = np.array( [k[0] for k in misc.SKELETON_2D_IDX] )
end_pt_2d = np.array( [k[1] for k in misc.SKELETON_2D_IDX] )
skeleton_color_2d = np.array(misc.SKELETON_2D_COLOR, dtype=bool)

# get 3D skeleton info
start_pt_3d = np.array( [k[0] for k in misc.SKELETON_3D_IDX] )
end_pt_3d = np.array( [k[1] for k in misc.SKELETON_3D_IDX] )
skeleton_color_3d = np.array(misc.SKELETON_3D_COLOR, dtype=bool)

plt.close('all')
fig = plt.figure(figsize=(12, 4))
gs1 = gridspec.GridSpec(1, 4)
gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
plt.axis('off')

ax = plt.subplot(gs1[0])
plt.imshow(im)
plt.axis('off')

# plot 3d predictions
ax3 = plt.subplot(gs1[1], projection='3d')
viz.show3Dpose(pred_sup_pt[pose_id, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
           lcolor="#3498db", rcolor="#e74c3c", radius=-1)
ax3.set_title('3D Supervised H36')

ax3 = plt.subplot(gs1[2], projection='3d')
viz.show3Dpose(pred_rel_pt[pose_id, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
           lcolor="#3498db", rcolor="#e74c3c", radius=-1)
ax3.set_title('Ours Relative H36')

ax3 = plt.subplot(gs1[3], projection='3d')
viz.show3Dpose(pred_rel_ft[pose_id, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
          lcolor="#3498db", rcolor="#e74c3c", radius=-1)
ax3.set_title('Ours Relative H36 + LSP FT')

plt.show()
# plt.savefig(op_dir + str(im_id) + '.pdf')


if use_blender_rendering:

    op_file = op_dir_rend + str(im_id).zfill(4) + '_sup.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(im_id) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in pred_sup_pt[pose_id, :]])
    os.system(cmd)

    op_file = op_dir_rend + str(im_id).zfill(4) + '_ours.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(im_id) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in pred_rel_pt[pose_id, :]])
    os.system(cmd)

    op_file = op_dir_rend + str(im_id).zfill(4) + '_ours_ft.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(im_id) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in pred_rel_ft[pose_id, :]])
    os.system(cmd)
