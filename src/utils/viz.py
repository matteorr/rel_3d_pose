"""Functions to visualize human poses, adapted from:
   https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/viz.py
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def plot_losses(loss_vals, loss_names, filename, title, xlabel, ylabel, spacing=0):
    """
    Given a list of errors, plot the objectives of the training and show
    """
    plt.close('all')
    for li, lvals in enumerate(loss_vals):
        iterations = range(len(lvals))
        # lvals.insert(0, 0)
        if spacing == 0:
            plt.loglog(iterations, lvals, '-',label=loss_names[li])
            # plt.semilogx(iterations, lvals, 'x-')
        else:
            xvals = [ii*spacing for ii in iterations]
            plt.loglog( xvals, lvals, '-',label=loss_names[li])

    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close('all')

def save_output_image_lsp(op_file_name, output_for_viz, misc):

    pts_2d = np.vstack(output_for_viz['pts_2d'])
    pred_3d = np.vstack(output_for_viz['pred_3d'])

    # get 2D skeleton info
    start_pt_2d = np.array( [k[0] for k in misc.SKELETON_2D_IDX] )
    end_pt_2d = np.array( [k[1] for k in misc.SKELETON_2D_IDX] )
    skeleton_color_2d = np.array(misc.SKELETON_2D_COLOR, dtype=bool)

    # get 3D skeleton info
    start_pt_3d = np.array( [k[0] for k in misc.SKELETON_3D_IDX] )
    end_pt_3d = np.array( [k[1] for k in misc.SKELETON_3D_IDX] )
    skeleton_color_3d = np.array(misc.SKELETON_3D_COLOR, dtype=bool)

    plt.close('all')
    fig = plt.figure(figsize=(21.3, 10.8))
    gs1 = gridspec.GridSpec(5, 10) # 5 rows, 10 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 0, 0
    nsamples = 25
    for i in np.arange( nsamples ):

        # plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx])
        show2Dpose(pts_2d[exidx, :], ax1, start_pt_2d, end_pt_2d, skeleton_color_2d, radius=100, root_idx=12)
        ax1.invert_yaxis()

        # plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        show3Dpose(pred_3d[exidx, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
                   lcolor="#1a5277", rcolor="#7f271e", radius=-1)

        exidx = exidx + 1
        subplot_idx = subplot_idx + 2

    # plt.show()
    plt.savefig(op_file_name, bbox_inches='tight')

def save_output_image(op_file_name, output_for_viz, misc):

    pts_2d = np.vstack(output_for_viz['pts_2d'])
    gt_3d = np.vstack(output_for_viz['gt_3d'])
    pred_3d = np.vstack(output_for_viz['pred_3d'])
    pred_3d_proc = np.vstack(output_for_viz['pred_3d_proc'])

    # get 2D skeleton info
    start_pt_2d = np.array( [k[0] for k in misc.SKELETON_2D_IDX] )
    end_pt_2d = np.array( [k[1] for k in misc.SKELETON_2D_IDX] )
    skeleton_color_2d = np.array(misc.SKELETON_2D_COLOR, dtype=bool)

    # get 3D skeleton info
    start_pt_3d = np.array( [k[0] for k in misc.SKELETON_3D_IDX] )
    end_pt_3d = np.array( [k[1] for k in misc.SKELETON_3D_IDX] )
    skeleton_color_3d = np.array(misc.SKELETON_3D_COLOR, dtype=bool)

    plt.close('all')
    fig = plt.figure(figsize=(25.6, 10.8))
    gs1 = gridspec.GridSpec(5, 12) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    subplot_idx, exidx = 0, 0
    nsamples = 15
    for i in np.arange( nsamples ):

        # plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx])
        show2Dpose(pts_2d[exidx, :], ax1, start_pt_2d, end_pt_2d, skeleton_color_2d, root_idx=2)
        # show2Dpose(pts_2d[exidx, :], ax1, start_pt_2d, end_pt_2d, skeleton_color_2d,root_idx=12)
        ax1.invert_yaxis()

        # plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        show3Dpose(gt_3d[exidx, :], ax2, start_pt_3d, end_pt_3d, skeleton_color_3d,
                   lcolor="#3498db", rcolor="#e74c3c")

        # plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+2], projection='3d')
        show3Dpose(pred_3d_proc[exidx, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
                   lcolor="#1a5277", rcolor="#7f271e")

        # plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+3], projection='3d')
        show3Dpose(pred_3d[exidx, :], ax3, start_pt_3d, end_pt_3d, skeleton_color_3d,
                   lcolor="#1a5277", rcolor="#7f271e")

        exidx = exidx + 1
        subplot_idx = subplot_idx + 4

    plt.savefig(op_file_name, bbox_inches='tight')


def show3Dpose(channels, ax, start_pt, end_pt, skeleton_color, lcolor, rcolor,
               add_labels=False, radius=750):
  """
  Visualize 3d skeleton
  """

  vals = channels.reshape((int(channels.shape[0] / 3.), 3))

  # rotate the 3D pts so they are orientated correctly
  rot = np.asarray([[1,0,0], [0,0,-1], [0,1,0]])
  vals = np.dot(vals, rot)

  # Make connection matrix
  for i in np.arange( len(start_pt) ):
    x, y, z = [np.array( [vals[start_pt[i], j], vals[end_pt[i], j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=lcolor if skeleton_color[i] else rcolor)

  # set space around the subject
  if radius >= 0:
      xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
      ax.set_xlim3d([-radius+xroot, radius+xroot])
      ax.set_ylim3d([-radius+yroot, radius+yroot])
      ax.set_zlim3d([-radius+zroot, radius+zroot])
  else:
    X = vals[:,0]; Y = vals[:,1]; Z = vals[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def show2Dpose(channels, ax, start_pt, end_pt, skeleton_color, lcolor="#3498db",
               rcolor="#e74c3c", add_labels=False, radius=350, root_idx=0):
  """
  Visualize 2d skeleton
  """

  vals = channels.reshape((int(channels.shape[0] / 2.), 2))

  # Make connection matrix
  for i in np.arange( len(start_pt) ):
    x, y = [np.array( [vals[start_pt[i], j], vals[end_pt[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if skeleton_color[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  # set space around the subject
  xroot, yroot = vals[root_idx,0], vals[root_idx,1]
  ax.set_xlim([-radius+xroot, radius+xroot])
  ax.set_ylim([-radius+yroot, radius+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')
