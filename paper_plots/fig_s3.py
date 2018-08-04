#
# This script plots the predicted 3D skeletons for Human3.6M.
# Selects a random frame and plots it.
# Need to set `videos_path` to the directory where the Human3.6M videos are.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import os

import sys
sys.path.append('../')
from src.misc import DatasetMisc
from src.utils.pose_plotter import PosePlotter
from src.data_formats.human36_config import pose_config_br

# loads gt and predicted poses
op_dir = 'h36/'
data_dir = '../checkpoint/fig_4ab/'

data = np.load(data_dir + 'rel_supervised/subject_info.npz')
data_sup = np.load(data_dir + 'fully_supervised/subject_info.npz')
err_str = 'The two data dictionaries are different. Mismatch in poses.'
for k in data.keys():
    assert data_rel[k] == data_sup[k], err_str

ours_pose = np.load(data_dir + 'rel_supervised/all_poses_proc.npy')
sup_pose = np.load(data_dir + 'fully_supervised/all_poses_proc.npy')
ours_err =  np.load(data_dir + 'rel_supervised/all_dist_proc.npy').mean(axis=1)
sup_err =  np.load(data_dir + 'fully_supervised/all_dist_proc.npy').mean(axis=1)

use_blender_rendering = False  # set this to True to render using blender
blender_path = 'PATH_TO_BLENDER'
blender_scene_path = './render_output/blank_scene.blend'
python_code_path = './render_output/render_single_frame_cmd.py'
op_dir_rend = op_dir + 'renders/'

if not os.path.isdir(op_dir): os.makedirs(op_dir)
if not os.path.isdir(op_dir_rend): os.makedirs(op_dir_rend)


NUM_EXAMPLES  = 5
IMAGES_WIDTH  = 512
IMAGES_HEIGHT = 512
ENLARGE       = .3
videos_path = 'PATH_TO_VIDEOS'
cam_list = ['54138969', '55011271', '58860488', '60457274']
dataset_type = 'human36_17k'

plt.close('all')

# get misc file used for the specified data format
misc = DatasetMisc(dataset_type)

# # class that takes care of plotting
pose_plotter = PosePlotter(
                pose_config_br['KEYPOINT_NAMES'],
                pose_config_br['SKELETON_NAMES'],
                pose_config_br['KEYPOINT_COLORS'],
                pose_config_br['SKELETON_COLORS'],
                plot_kps=False)

f_no = np.random.randint(data['X'].shape[0])

vid_name = 'S' + str(data['S'][f_no]) + '/Videos/' + data['F'][f_no][0][:-3] + \
           '.' + cam_list[data['C'][f_no]-1] + '.mp4'

print 'test frame num', f_no
print vid_name
print 'sup  error', round(sup_err[f_no], 2)
print 'ours error', round(ours_err[f_no], 2)

# get the keypoints
kpts_2d_x = data['X'][f_no, 0::2] + data['X_root'][f_no, 0]
kpts_2d_y = data['X'][f_no, 1::2] + data['X_root'][f_no, 1]
kpts_v = np.ones(np.shape(kpts_2d_x))

# gt 3d pose
kpts_3d_x = data['Y'][f_no, 0::3]
kpts_3d_y = data['Y'][f_no, 1::3]
kpts_3d_z = data['Y'][f_no, 2::3]

# pred pose sup
kpts_3d_x_s = sup_pose[f_no, 0::3]
kpts_3d_y_s = sup_pose[f_no, 1::3]
kpts_3d_z_s = sup_pose[f_no, 2::3]

# pred pose sup
kpts_3d_x_o = ours_pose[f_no, 0::3]
kpts_3d_y_o = ours_pose[f_no, 1::3]
kpts_3d_z_o = ours_pose[f_no, 2::3]


# load frame
cap = cv2.VideoCapture(videos_path + vid_name)
cap.set(cv2.CAP_PROP_POS_FRAMES,float(data['F'][f_no][1]))
ret, cap_frame = cap.read()
# convert the image
im = im = cap_frame[:, :, [2,1,0]]
width, height = im.shape[0], im.shape[1]



gt_pose_2d_x = kpts_2d_x
gt_pose_2d_y = kpts_2d_y

## make bounding box around 2d keypoints
w  = max(gt_pose_2d_x) - min(gt_pose_2d_x)
h  = max(gt_pose_2d_y) - min(gt_pose_2d_y)
cx = int(min(gt_pose_2d_x) + w/2.)
cy = int(min(gt_pose_2d_y) + h/2.)

bbox = [cx - (w*(1+ENLARGE))/2., cy - (h*1+ENLARGE)/2., w*(1+ENLARGE), h*(1+ENLARGE)]
slack = int(bbox[2]/2.) if w > h else int(bbox[3]/2.)
x_start = cx - slack
x_end   = cx + slack
y_start = cy - slack
y_end   = cy + slack

pad_left   = abs(x_start) if x_start < 0 else 0
#pad_right  = x_end - 2 * slack if x_end > 2 * slack else 0
pad_top    = abs(y_start) if y_start < 0 else 0
# pad_bottom = y_end - 2 * slack if y_end > 2 * slack else 0
padded_frame = np.pad(im,((0,0),(pad_left,0),(0,0)),'edge')
crop = padded_frame[y_start+pad_top:y_end+pad_top, x_start+pad_left:x_end+pad_left, :]
crop = np.array(Image.fromarray(crop).resize((IMAGES_WIDTH, IMAGES_HEIGHT)))

resize_ratio = [IMAGES_WIDTH / (2. * slack), IMAGES_HEIGHT / (2. * slack)]

gt_pose_2d_x_r = (gt_pose_2d_x - x_start) * resize_ratio[0]
gt_pose_2d_y_r = (gt_pose_2d_y - y_start) * resize_ratio[1]


plt.close('all')
fig = plt.figure(figsize=(16, 4))
gs1 = gridspec.GridSpec(1, 4)
gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
plt.axis('off')

ax = plt.subplot(gs1[0])
pose_plotter.plot_2d(gt_pose_2d_x_r, gt_pose_2d_y_r, kpts_v, ax=ax)
plt.imshow(crop)
ax.set_xlim(0, IMAGES_WIDTH)
ax.set_ylim(IMAGES_HEIGHT, 0)
ax.set_axis_off()

ax1 = plt.subplot(gs1[1], projection='3d')
pose_plotter.plot_3d(kpts_3d_x, kpts_3d_y, kpts_3d_z, kpts_v, ax=ax1)
plt.title('Ground Truth')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

ax2 = plt.subplot(gs1[2], projection='3d')
pose_plotter.plot_3d(kpts_3d_x_s, kpts_3d_y_s, kpts_3d_z_s, kpts_v, ax=ax2)
plt.title('3D Supervised - ' + str(round(sup_err[f_no], 1)) + ' mm')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

ax3 = plt.subplot(gs1[3], projection='3d')
pose_plotter.plot_3d(kpts_3d_x_o, kpts_3d_y_o, kpts_3d_z_o, kpts_v, ax=ax3)
plt.title('Ours Relative - ' + str(round(ours_err[f_no], 1)) + ' mm')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_zticklabels([])

plt.show()
plt.savefig(op_dir + str(f_no).zfill(8) + '.pdf')


#
# This will produces nicer plots, but they need to be manually composited.
# Needs blender to be installed.
#
if use_blender_rendering:

    op_file = op_dir_rend + str(f_no).zfill(8) + '_ours.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(f_no) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in ours_pose[f_no,:]])
    os.system(cmd)

    op_file = op_dir_rend + str(f_no).zfill(8) + '_sup.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(f_no) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in sup_pose[f_no,:]])
    os.system(cmd)

    op_file = op_dir_rend + str(f_no).zfill(8) + '_gt.png'
    cmd = blender_path + ' ' + blender_scene_path + ' --background --python ' + python_code_path + \
          ' -- frame_id ' + str(f_no) + ' op_file '+ op_file + ' pts_raw ' + \
          ''.join([str(round(pp,3)) + ' ' for pp in data['Y'][f_no, :]])
    os.system(cmd)
