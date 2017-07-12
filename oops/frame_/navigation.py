################################################################################
# oops_/frame/navigation.py: Fittable subclass Navigation of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.transform import Transform
from oops.fittable  import Fittable

class Navigation(Frame, Fittable):
    """Navigation is a Frame subclass describing a fittable, fixed offset from
    another frame, defined by two or three rotation angles.
    """

    def __init__(self, angles, reference, id=None, matrix=None):
        """Constructor for a Navigation Frame.

        Input:
            angles      the angles of rotation in radians. The order of the
                        rotations is about the y, x and z axes.
            reference   the frame or frame_id relative to which this rotation is
                        defined.
            id          the ID to use; None to use a temporary ID.
            matrix      an optional parameter, used internally, to speed up the
                        copying of Navigation objects. If not None, it must
                        contain the Matrix3 object that performs the defined
                        rotation.
        """

        self.angles = np.array(angles)
        assert self.angles.shape in ((2,),(3,))

        self.cache = {}
        self.param_name = "angles"
        self.nparams = self.angles.shape[0]

        if matrix is None:
            matrix = Navigation._rotmat(self.angles[0],1)
            matrix = Navigation._rotmat(self.angles[1],0) * matrix

            if self.nparams > 2 and self.angles[2] != 0.:
                matrix = Navigation._rotmat(self.angles[2], 2) * matrix

        self.frame_id  = id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.shape     = self.reference.shape
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        self.transform = Transform(matrix, Vector3.ZERO,
                                   self, self.reference, self.origin)

    ########################################

    @staticmethod
    def _rotmat(angle, axis):
        """Internal function to return a matrix that performs a rotation about
        a single specified axis."""

        axis2 = axis
        axis0 = (axis2 + 1) % 3
        axis1 = (axis2 + 2) % 3

        mat = np.zeros((3,3))
        mat[axis2, axis2] = 1.
        mat[axis0, axis0] = np.cos(angle)
        mat[axis0, axis1] = np.sin(angle)
        mat[axis1, axis1] =  mat[axis0, axis0]
        mat[axis1, axis0] = -mat[axis0, axis1]

        return Matrix3(mat)

    ########################################

    def transform_at_time(self, time, quick=False):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        return self.transform

    ########################################
    # Fittable interface
    ########################################

    def set_params_new(self, params):
        """Redefines the Fittable object, using this set of parameters. Unlike
        method set_params(), this method does not check the cache first.
        Override this method if the subclass should use a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        params = np.array(params).copy()
        assert self.angles.shape == params.shape

        self.angles = params

        matrix = Navigation._rotmat(self.angles[0],1)
        matrix = Navigation._rotmat(self.angles[1],0) * matrix

        if self.nparams > 2 and self.angles[2] != 0.:
            matrix = Navigation._rotmat(self.angles[2],2) * matrix

        self.transform = Transform(matrix, Vector3.ZERO, self,
                                   self.reference, self.origin)

    def copy(self):
        """Returns a deep copy of the given object. The copy can be safely
        modified without affecting the original."""

        return Navigation(self.angles.copy(), self.reference,
                          matrix=self.transform.matrix.copy())

        return result

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Navigation(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
