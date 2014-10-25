#XXXXXXXX

assert False

import matplotlib        # Must be first!
matplotlib.use('TkAgg')  # Required to allow the 3-D graphs to freely rotate

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d_indices(size_y, size_x):
    xindices = np.repeat(np.arange(size_x)-size_x//2,
                         size_y)
    yindices = np.tile(np.arange(size_y)-size_y//2,
                       size_x)
    xindices = xindices.reshape((size_y, size_x))
    yindices = yindices.reshape((size_y, size_x))

    return xindices, yindices

    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(xindices, yindices, scaled_psf,
                    rstride=1, cstride=1, color='red', alpha=0.3)
    ax.plot_surface(xindices, yindices, orig_subimage,
                    rstride=1, cstride=1, color='blue', alpha=0.3)
    plt.title('PSF (RED) and Orig Image-Gradient')

    ax = fig.add_subplot(223, projection='3d')
    ax.plot_surface(xindices, yindices, scaled_psf-subimage, rstride=1, cstride=1, alpha=0.3)
    plt.title('Masked Residuals')

    ax = fig.add_subplot(224, projection='3d')
    ax.plot_surface(xindices, yindices, scaled_psf-orig_subimage, rstride=1, cstride=1, alpha=0.3)
    plt.title('Orig Residuals')

    fig.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=0.05, hspace=0.05)
    plt.show()

