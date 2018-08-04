import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_mturk(data, images_to_exclude, random_flip=False):
    if 'human_subj_ids' in data[0].keys():
        im_ids_all = [dd['human_subj_ids'] for dd in data]
    else:
        im_ids_all = [dd['lsp_subj_ids'] for dd in data]
    im_ids_all = [item for sublist in im_ids_all for item in sublist]

    im_ids = []
    workers = []
    keypoints = []
    annotations = []
    for dd in data:
        for tt in dd['trials']:
            pairs = tt['depth']['keypoint_comparisons_res'].keys()
            res = tt['depth']['keypoint_comparisons_res'].values()

            if tt['img_id'] not in images_to_exclude:
                keypoints.extend(pairs)
                annotations.extend(res)
                im_ids.extend([tt['img_id']]*len(pairs))
                workers.extend([dd['worker_id']]*len(pairs))
                if tt['img_id'] not in im_ids_all:
                    print 'Note:', tt['img_id'], 'in trial but not in list'

    results = []
    for ii in range(len(im_ids)):
        kps = keypoints[ii].split(',')
        res = {'im_id':im_ids[ii], 'w_id':workers[ii], 'im_id_joint':str(im_ids[ii]) + '_' + keypoints[ii],
               'kp0':int(kps[0]), 'kp1':int(kps[1]), 'ann':annotations[ii]}
        results.append(res)

    return results


def bin_vals(distance, correct_pred, bin_size=10):
    binned_acc = []
    binned_inds = []
    intervals = range(0, int(distance.max())+bin_size, bin_size)
    cnt = []
    for ii in range(len(intervals)-1):
        inds = np.where((distance >= intervals[ii]) & (distance < intervals[ii+1]))[0]
        binned_inds.append(inds)
        if len(inds) > 0:
            binned_acc.append(correct_pred[inds].sum() / float(len(inds)))
            cnt.append(len(inds))
        else:
            binned_acc.append(np.nan)
            cnt.append(0)

    return np.array(intervals[:-1]), np.array(binned_acc), binned_inds


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                  [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                  [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                  ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                  ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                  [np.sin(theta[2]),    np.cos(theta[2]),     0],
                  [0,                     0,                      1]
                  ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def plot_2d_skel(fig, coords, start_pt_2d, end_pt_3d):
    ax = fig.add_subplot(111)
    for i in np.arange( len(start_pt_2d) ):
        x, y = [np.array( [coords[start_pt_2d[i], j], coords[end_pt_3d[i], j]] ) for j in range(2)]
        ax.plot(x, y, lw=2, c='b')

    plt.plot(coords[0,0], coords[0,1], 'ro')
    plt.plot(coords[6,0], coords[6,1], 'go')

    ax.set_aspect('equal')
    ax.invert_yaxis()


def plot_3d_skel(fig, coords, start_pt_3d, end_pt_3d):
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')

    for i in np.arange( len(start_pt_3d) ):
        x, y, z = [np.array( [coords[start_pt_3d[i], j], coords[end_pt_3d[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c='b')

    plt.plot([coords[0,0]], [coords[0,1]], [coords[0,2]], 'r.')
    plt.plot([coords[6,0]], [coords[6,1]], [coords[6,2]], 'g.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    radius = 1000
    xroot, yroot, zroot = coords.mean(0)
    ax.set_xlim3d([-radius+xroot, radius+xroot])
    ax.set_ylim3d([-radius+yroot, radius+yroot])
    ax.set_zlim3d([-radius+zroot, radius+zroot])
    ax.set_aspect('equal')
