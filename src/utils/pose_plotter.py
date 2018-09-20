import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PosePlotter():
    def __init__(self, KEYPOINT_NAMES,  SKELETON_NAMES,
                       KEYPOINT_COLORS, SKELETON_COLORS, plot_kps=True):
        self.keypoint_names  = KEYPOINT_NAMES
        self.keypoint_clrs   = KEYPOINT_COLORS

        self.skeleton_names  = SKELETON_NAMES
        self.skeleton_colors = SKELETON_COLORS
        self.skeleton_idxs   = \
                [[self.keypoint_names.index(k[0]),
                  self.keypoint_names.index(k[1])] for k in self.skeleton_names]

        self.start_pts = np.array( [k[0] for k in self.skeleton_idxs] )
        self.end_pts   = np.array( [k[1] for k in self.skeleton_idxs] )

        self.plot_kps = plot_kps

        self.ax_2d = None
        self.ax_2d_lims = False
        self.x_start_2d = None
        self.x_end_2d   = None
        self.y_start_2d = None
        self.y_end_2d   = None

        self.ax_3d = None
        self.ax_3d_lims = False
        self.x_start_3d = None
        self.x_end_3d   = None
        self.y_start_3d = None
        self.y_end_3d   = None
        self.z_start_3d = None
        self.z_end_3d   = None

    def set_2d_axis(self, x_start_2d, x_end_2d, y_start_2d, y_end_2d):
        self.ax_2d_lims = True
        self.x_start_2d = x_start_2d
        self.x_end_2d   = x_end_2d
        self.y_start_2d = y_start_2d
        self.y_end_2d   = y_end_2d

    def plot_2d(self, pose_2d_x, pose_2d_y, kpts_v, BLOCK=True, ax=None):
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(1.))
            plt.clf()
            self.ax = fig.add_subplot(1, 1, 1)
        else:
            self.ax = ax
        self._plot_skeleton(kpts_v, pose_2d_x, pose_2d_y)

        if self.ax_2d_lims:
            self.ax.set_xlim(self.x_start_2d, self.x_end_2d)
            self.ax.set_ylim(self.y_start_2d, self.y_end_2d)
        else:
            # uses the keypoint visibility flags to select the max and min
            # across the x and y dimensions for setting the plot axis limits
            max_x = np.max(pose_2d_x[kpts_v.astype(np.bool)])
            min_x = np.min(pose_2d_x[kpts_v.astype(np.bool)])
            max_y = np.max(pose_2d_y[kpts_v.astype(np.bool)])
            min_y = np.min(pose_2d_y[kpts_v.astype(np.bool)])

            w  = max_x - min_x
            h  = max_y - min_y

            cx = int(min_x + w/2.)
            cy = int(min_y + h/2.)

            ENLARGE = 0.
            bbox = [cx - (w*(1+ENLARGE))/2.,
                    cy - (h*(1+ENLARGE))/2., w*(1+ENLARGE), h*(1+ENLARGE)]
            slack = int(bbox[2]/2.) if w > h else int(bbox[3]/2.)

            x_start = cx - slack
            x_end   = cx + slack
            y_start = cy - slack
            y_end   = cy + slack
            self.ax.set_xlim(x_start, x_end)
            self.ax.set_ylim(y_start, y_end)

        self.ax.invert_yaxis()
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("y")

        if ax is None:
            if BLOCK:
                plt.show()
                #plt.close()
            else:
                plt.draw()
                plt.pause(0.01)

    def set_3d_axis(self, x_start_3d, x_end_3d,
                          y_start_3d, y_end_3d, z_start_3d, z_end_3d):
        self.ax_3d_lims = True
        self.x_start_3d = x_start_3d
        self.x_end_3d   = x_end_3d
        self.y_start_3d = y_start_3d
        self.y_end_3d   = y_end_3d
        self.z_start_3d = z_start_3d
        self.z_end_3d   = z_end_3d

    def plot_3d(self, pose_3d_x, pose_3d_y, pose_3d_z, kpts_v, BLOCK=True, ax=None):
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(1.))
            plt.clf()
            self.ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            self.ax = ax

        self._plot_skeleton(kpts_v, pose_3d_x, pose_3d_y, pose_3d_z)

        if self.ax_3d_lims:
            self.ax.set_xlim(self.x_start_3d, self.x_end_3d)
            self.ax.set_ylim(self.z_start_3d, self.z_end_3d)
            self.ax.set_zlim(self.y_start_3d, self.y_end_3d)
        else:
            max_range = np.array([pose_3d_x.max()-pose_3d_x.min(),
                          pose_3d_y.max()-pose_3d_y.min(),
                          pose_3d_z.max()-pose_3d_z.min()]).max() / 2.0
            mid_x = (pose_3d_x.max()+pose_3d_x.min()) * 0.5
            mid_y = (pose_3d_y.max()+pose_3d_y.min()) * 0.5
            mid_z = (pose_3d_z.max()+pose_3d_z.min()) * 0.5

            x_start = mid_x - max_range
            x_end   = mid_x + max_range
            y_start = mid_y - max_range
            y_end   = mid_y + max_range
            z_start = mid_z - max_range
            z_end   = mid_z + max_range
            self.ax.set_xlim(x_start, x_end)
            self.ax.set_ylim(z_start, z_end)
            self.ax.set_zlim(y_start, y_end)

        self.ax.invert_zaxis()
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("z")
        # self.ax.set_zlabel("y")

        if ax is None:
            if BLOCK:
                plt.show()
                #plt.close()
            else:
                plt.draw()
                plt.pause(0.01)

    def plot_2d_3d(self, pose_2d_x, pose_2d_y,
                         pose_3d_x, pose_3d_y, pose_3d_z, kpts_v, BLOCK=True):

        fig = plt.figure(figsize=plt.figaspect(.5))
        plt.clf()
        ax_2d = fig.add_subplot(1, 2, 1)
        self.plot_2d(pose_2d_x, pose_2d_y, kpts_v, BLOCK, ax_2d)

        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        self.plot_3d(pose_3d_x, pose_3d_y, pose_3d_z, kpts_v, BLOCK, ax_3d)

        if BLOCK:
            plt.show()
            #plt.close()
        else:
            plt.draw()
            plt.pause(0.01)


    def _plot_skeleton(self, kpts_v, pose_x, pose_y, pose_z=None):

        for i,(start_pt,end_pt) in enumerate(zip(self.start_pts, self.end_pts)):
            if kpts_v[start_pt] * kpts_v[end_pt] == 0:
                continue

            x_limb = pose_x[start_pt], pose_x[end_pt]
            y_limb = pose_y[start_pt], pose_y[end_pt]

            if pose_z is None:
                self.ax.plot(x_limb, y_limb, lw=2, c=self.skeleton_colors[i])
            else:
                z_limb = pose_z[start_pt], pose_z[end_pt]
                self.ax.plot(x_limb,
                            z_limb, y_limb, lw=2, c=self.skeleton_colors[i])

        if self.plot_kps:
            for i, kpt_n in enumerate(self.keypoint_names):
                if kpts_v[i] != 0:
                    if pose_z is None:
                        self.ax.scatter(pose_x[i],
                                        pose_y[i], color=self.keypoint_clrs[i])
                    else:
                        self.ax.scatter(pose_x[i],
                                        pose_z[i],
                                        pose_y[i], color=self.keypoint_clrs[i])
