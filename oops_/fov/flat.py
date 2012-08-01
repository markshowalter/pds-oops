################################################################################
# oops_/fov/flat.py: Flat subclass of class FOV
#
# 2/1/12 Modified (MRS) - copy() added to as_pair() calls.
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
# 2/23/12 MRS - Gave each method the option to return partial derivatives.
################################################################################

import numpy as np

from oops_.fov.fov_ import FOV
from oops_.array.all import *

class Flat(FOV):
    """Flat is a subclass of FOV that describes a field of view that is free of
    distortion, implementing an exact pinhole camera model.
    """

    def __init__(self, uv_scale, uv_shape, uv_los=None, uv_area=None):
        """Constructor for a FlatFOV. The U-axis is assumed to align with X and
        the V-axis aligns with Y. A FlatFOV has no dependence on the optional
        extra indices that can be associated with time, wavelength band, etc.

        Input:
            uv_scale    a single value, tuple or Pair defining the ratios dx/du
                        and dy/dv. For example, if (u,v) are in units of
                        arcseconds, then
                            uv_scale = Pair((pi/180/3600.,pi/180/3600.))
                        Use the sign of the second element to define the
                        direction of increasing V: negative for up, positive for
                        down.

            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal field of view
                        of a pixel. If not provided, the area is calculated
                        based on the area of the central pixel.
        """

        self.uv_scale = Pair.as_float(uv_scale, copy=True)
        self.uv_shape = Pair.as_pair(uv_shape).copy()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_float(uv_los, copy=True)

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        scale = Pair.as_pair(uv_scale)

        self.dxy_duv = MatrixN([[  scale.vals[0], 0.], [0.,   scale.vals[1]]])
        self.duv_dxy = MatrixN([[1/scale.vals[0], 0.], [0., 1/scale.vals[1]]])

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        uv = Pair.as_pair(xy_pair)/self.uv_scale + self.uv_los

        if derivs:
            uv.insert_subfield("d_dxy", self.duv_dxy)

        return uv

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        xy = (Pair.as_pair(uv_pair) - self.uv_los) * self.uv_scale

        if derivs:
            xy.insert_subfield("d_duv", self.dxy_duv)

        return xy

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Flat(unittest.TestCase):

    def runTest(self):

        test = Flat((1/2048.,-1/2048.), 101, (50,75))

        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)

        xy = test.xy_from_uv(buffer)
        (x,y) = xy.as_scalars()

        self.assertEqual(xy[  0,  0], (-50./2048., 75./2048.))
        self.assertEqual(xy[100,  0], ( 50./2048., 75./2048.))
        self.assertEqual(xy[  0,100], (-50./2048.,-25./2048.))
        self.assertEqual(xy[100,100], ( 50./2048.,-25./2048.))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, Pair(buffer))

        self.assertEqual(test.area_factor(buffer), 1.)

        test2 = Flat((1/2048.,-1/2048.), 101, (50,75), uv_area = test.uv_area*2)
        self.assertEqual(test2.area_factor(buffer), 0.5)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
