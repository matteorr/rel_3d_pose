import numpy as np
import json
import sys

sys.path.append('../')

from src.utils.pose_plotter import PosePlotter
import matplotlib.pyplot as plt

lsp_path = "PATH TO LSP"

# for plotting purposes
pose_config = {

    'KEYPOINT_NAMES':
        ['nose','neck','left_shoulder',
         'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip',
         'right_hip', 'left_knee', 'right_knee',
         'left_ankle', 'right_ankle'],

   'KEYPOINT_COLORS':
       ['#4286f4', '#4286f4', '#9e1809',
        '#27a30b', '#9e1809', '#27a30b',
        '#9e1809', '#27a30b', '#9e1809',
        '#27a30b', '#9e1809', '#27a30b',
        '#9e1809', '#27a30b'],

    'SKELETON_NAMES':
        [['nose','neck'], ['neck','right_shoulder'], ['neck','left_shoulder'],
         ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
         ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
         ['right_shoulder','right_hip'], ['left_shoulder','left_hip'],
         ['right_hip','left_hip'],
         ['right_hip','right_knee'], ['right_knee','right_ankle'],
         ['left_hip','left_knee'], ['left_knee','left_ankle']],

    'SKELETON_COLORS':
        ['#a00899', '#ffe20c', '#ffe20c',
         '#0c99ff', '#0c99ff',
         '#0c99ff', '#0c99ff',
         '#ffe20c', '#ffe20c',
         '#ffe20c',
         '#a57e60', '#a57e60',
         '#a57e60', '#a57e60']
}

LABELS = ['nose','neck','l_shldr', 'r_shldr', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

pose_config['SKELETON_IDXS'] = [[pose_config['KEYPOINT_NAMES'].index(k[0]),pose_config['KEYPOINT_NAMES'].index(k[1])] for k in pose_config['SKELETON_NAMES']]

with open('../mturk_data/relative_depth_lsp_v1.json','r') as fp: data = json.load(fp)

pose_plotter = PosePlotter(pose_config['KEYPOINT_NAMES'],
                           pose_config['SKELETON_NAMES'],
                           pose_config['KEYPOINT_COLORS'],
                           pose_config['SKELETON_COLORS'])

occluded = [i for i in range(2000) if sum(data['images'][i]['occluded'])>0]

image_id = np.random.randint(2000)
kpts_v = 1-np.array(data['images'][image_id]['occluded'])

kpts_2d_xy = np.array(data['images'][image_id]['keypoints'])
kpts_2d_x  = kpts_2d_xy[0::2]
kpts_2d_y  = kpts_2d_xy[1::2]

cmap = plt.cm.get_cmap('jet')
num_colors = cmap.N
colors = cmap(np.arange(cmap.N))

skeleton_idxs = pose_config['SKELETON_IDXS']
start_pts_all = np.array( [k[0] for k in skeleton_idxs] )
end_pts_all   = np.array( [k[1] for k in skeleton_idxs] )

fig = plt.figure(figsize=(10,5))#plt.figaspect(0.667))
plt.clf()

ax_1 = fig.add_subplot(2, 3, 1)
plt.title('image: [%s]'%data['images'][image_id]['im_file'][-10:])
im = ax_1.imshow(np.random.random((10,10)), cmap='jet', vmin=0, vmax=1)

I = plt.imread("%s/%s"%(lsp_path,data['images'][image_id]['im_file']))
plt.imshow(I)
pose_plotter.plot_2d(kpts_2d_x, kpts_2d_y, np.ones(kpts_v.shape),ax=ax_1)
plt.scatter(kpts_2d_x[kpts_v==0],kpts_2d_y[kpts_v==0], c='k')
plt.xticks([])
plt.yticks([])
plt.xlabel("")
plt.ylabel("")

for pi,pair in enumerate(data['images'][image_id]['anns']):
    ax_i = fig.add_subplot(2, 3, pi+2)
    ax_i.invert_yaxis()
    ax_i.set_xlim(ax_1.get_xlim())
    ax_i.set_ylim(ax_1.get_ylim())
    plt.xticks([])
    plt.yticks([])

    for i,(start_pt,end_pt) in enumerate(zip(start_pts_all, end_pts_all)):
        if kpts_v[start_pt] * kpts_v[end_pt] == 0:
            continue

        x_limb = kpts_2d_x[start_pt], kpts_2d_x[end_pt]
        y_limb = kpts_2d_y[start_pt], kpts_2d_y[end_pt]

        plt.plot(x_limb, y_limb, lw=0.5, c='k')

    for i, kpt_n in enumerate(pose_config['KEYPOINT_NAMES']):
        if kpts_v[i] != 0:
            plt.scatter(kpts_2d_x[i], kpts_2d_y[i], color='gray')

        kp_0_x = kpts_2d_x[pair['kp0']]; kp_0_y = kpts_2d_y[pair['kp0']]
        kp_1_x = kpts_2d_x[pair['kp1']]; kp_1_y = kpts_2d_y[pair['kp1']]

        pair_x = kp_0_x, kp_1_x
        pair_y = kp_0_y, kp_1_y

        if pair['label'] == 0:
            index = int(num_colors*(1-pair['risk']))
            text  = (1-pair['risk'])*100
        else:
            index = int(num_colors*pair['prob'])
            text  = pair['prob']*100

        col = colors[index]
        plt.plot(pair_x, pair_y, c=col,lw=3)

        plt.text(0.7,0.1,'%.2f%%'%text,bbox={'facecolor':'white', 'alpha':0.5, 'pad':3},
                transform=ax_i.transAxes)

        if pair['label'] == 0:
            plt.scatter(kp_0_x,kp_0_y, c='limegreen')
            plt.scatter(kp_1_x,kp_1_y, c='r')
            str_l = "[%s] < [%s]"%(LABELS[pair['kp0']],LABELS[pair['kp1']])
        else:
            plt.scatter(kp_0_x,kp_0_y, c='r')
            plt.scatter(kp_1_x,kp_1_y, c='limegreen')
            str_l = "[%s] < [%s]"%(LABELS[pair['kp1']],LABELS[pair['kp0']])
        plt.title(str_l)

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
fig.colorbar(im, cax=cbar_ax)

plt.show()
# plt.savefig("./lsp/image_id_%d.png"%image_id, bbox_inches='tight')
