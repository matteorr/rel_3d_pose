"""
Takes an image as input and lets to click on each joint location and then save
as numpy array.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

import time
import warnings
warnings.filterwarnings("ignore")

im_path = './input_image.jpg'
im = plt.imread(im_path)

from src.data_formats.misc import DatasetMisc
misc = DatasetMisc('human36_17k')
keypoints = misc.KEYPOINTS_2D

print("please click on:")
#plt.close('all')
fig = plt.figure(0)
plt.imshow(im)
plt.show(block=False)

pts = np.zeros((len(keypoints), 2))
for ii in range(len(keypoints)):
    print(ii, keypoints[ii])
    pts[ii, :] = plt.ginput(1)[0]
    #print(round(pts[ii, 0], 2), round(pts[ii, 1], 1))
    plt.plot(pts[ii, 0], pts[ii, 1], 'rx')
    fig.canvas.draw()
    time.sleep(0.01)

np.save(im_path[:-4], pts.astype(np.float32).ravel()[np.newaxis, ...])
