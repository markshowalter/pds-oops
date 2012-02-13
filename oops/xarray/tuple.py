################################################################################
# Tuple
#
# Created 1/12/12 (MRS)
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
################################################################################

import numpy as np
import numpy.ma as ma

from baseclass  import Array
from scalar     import Scalar
from pair       import Pair
from oops.units import Units

import utils as utils

################################################################################
# Tuple
################################################################################

class Tuple(Array):
    """An arbitrary Array of tuples, all of the same length."""

    def __init__(self, arg, mask=False, units=None):

        if mask is not False: mask = np.asarray(mask)

        if isinstance(arg, Array) and arg.rank == 1:
            mask = mask | arg.mask
            if units is None:
                units = arg.units
                arg = arg.vals
            elif arg.units is not None:
                arg = arg.units.convert(arg.vals, units)
            else:
                arg = arg.vals

        elif isinstance(arg, Array):
            raise ValueError("class " + type(arg).__name__ +
                             " cannot be converted to class " +
                             type(self).__name__)

        elif isinstance(arg, ma.MaskedArray):
            if arg.mask != ma.nomask: mask = mask | np.any(arg.mask, axis=-1)
            arg = arg.data

        self.vals = np.asarray(arg)
        ashape = list(self.vals.shape)

        self.rank  = 1
        self.item  = ashape[-1:]
        self.shape = ashape[:-1]
        self.mask  = mask

        if (self.mask is not False) and (list(self.mask.shape) != self.shape):
            raise ValueError("mask array is incompatible with Tuple shape")

        self.units = Units.as_units(units)

        return

    @staticmethod
    def as_tuple(arg):
        if isinstance(arg, Tuple): return arg
        return Tuple(arg)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Tuple): arg = Tuple(arg)
        return arg.convert_units(None)

    def as_scalar(self, axis):
        """Returns a Scalar containing one selected item from each tuple."""

        return Scalar(self.vals[...,axis], self.mask)

    def as_scalars(self):
        """Returns this object as a list of Scalars."""

        list = []
        for i in range(self.item[0]):
            list.append(Scalar(self.vals[...,i]), self.mask)

        return list

    def as_pair(self, axis=0):
        """Returns a Pair containing two selected items from each Tuple,
        beginning with the selected axis."""

        return Pair(self.vals[...,axis:axis+2], self.mask)

    @staticmethod
    def from_scalars(*args):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given as arguments.
        """

        mask = False
        for arg in args:
            if isinstance(arg, Scalar):
                mask = mask | arg.mask
            if isinstance(arg, ma.MaskedArray) and arg.mask != ma.nomask:
                mask = mask | arg.mask

        return Tuple(np.rollaxis(np.array(args), 0, len(args)), mask)

    @staticmethod
    def cross_scalars(*args):
        """Returns a new Tuple constructed by combining every possible set of
        components provided as a list of scalars. The returned Tuple will have a
        shape defined by concatenating the shapes of all the arguments.
        """

        scalars = []
        newshape = []
        dtype = "int"
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            scalars.append(scalar)
            newshape += scalar.shape
            if scalar.vals.dtype.kind == "f": dtype = "float"

        buffer = np.empty(newshape + [len(args)], dtype=dtype)

        newaxes = []
        count = 0
        for scalar in scalars[::-1]:
            newaxes.append(count)
            count += len(scalar.shape)

        newaxes.reverse()

        for i in range(len(scalars)):
            scalars[i] = scalars[i].reshape(scalars[i].shape + newaxes[i] * [1])

        reshaped = Array.broadcast_arrays(scalars)

        for i in range(len(reshaped)):
            buffer[...,i] = reshaped[i].vals

        return Tuple(buffer)

    @staticmethod
    def from_scalar_list(list):
        """Returns a new Tuple constructed by combining the Scalars or arrays
        given in a list.
        """

        return Tuple(np.rollaxis(np.array(list), 0, len(list)))

    def as_index(self):
        """Returns this object as a list of lists, which can be used to index a
        numpy ndarray, thereby returning an ndarray of the same shape as the
        Tuple object. Each value is rounded down to the nearest integer."""

        return list(np.rollaxis((self.vals // 1).astype("int"), -1, 0))

    def int(self):
        """Returns the integer (floor) component of each index."""

        return Tuple((self.vals // 1).astype("int"))

    def frac(self):
        """Returns the fractional component of each index."""

        return Tuple(self.vals % 1)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Tuple(unittest.TestCase):

    def runTest(self):

        foo = np.arange(24).reshape(3,4,2)

        test = Tuple(np.array([[[0,0,0], [0,0,1], [0,1,0]],
                               [[0,1,1], [0,2,0], [2,3,1]]]))
        self.assertEqual(test.shape, [2,3])
        self.assertEqual(test.item, [3])

        result = foo[test.as_index()]
        self.assertEqual(result.shape, (2,3))
        self.assertTrue(np.all(result == [[0, 1, 2],[3, 4, 23]]))

        self.assertEqual(test + (1,1,0), [[[1,1,0], [1,1,1], [1,2,0]],
                                          [[1,2,1], [1,3,0], [3,4,1]]])

        self.assertEqual((test + (0.5,0.5,0.5)).int(), test)

        self.assertTrue(np.all((test + (0.5,0.5,0.5)).frac().vals == 0.5))

        # cross_scalars()
        t = Tuple.cross_scalars(np.arange(5), np.arange(4), np.arange(3))
        self.assertEqual(t.shape, [5,4,3])
        self.assertTrue(np.all(t.vals[4,:,:,0] == 4))
        self.assertTrue(np.all(t.vals[:,3,:,1] == 3))
        self.assertTrue(np.all(t.vals[:,:,2,2] == 2))

        # cross_scalars()
        t = Tuple.cross_scalars(np.arange(5), np.arange(12).reshape(4,3),
                                np.arange(2))
        self.assertEqual(t.shape, [5,4,3,2])
        self.assertTrue(np.all(t.vals[4,:,:,:,0] ==  4))
        self.assertTrue(np.all(t.vals[:,3,2,:,1] == 11))
        self.assertTrue(np.all(t.vals[:,:,:,1,2] ==  1))

        # New tests 2/1/12 (MRS)

        test = Tuple(np.arange(6).reshape(3,2))
        self.assertEqual(str(test), "Tuple[[0 1]\n [2 3]\n [4 5]]")

        test.mask = np.array([False, False, True])
        self.assertEqual(str(test),   "Tuple[[0 1]\n [2 3]\n [-- --], mask]")
        self.assertEqual(str(test*2), "Tuple[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test/2), "Tuple[[0 0]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test%2), "Tuple[[0 1]\n [0 1]\n [-- --], mask]")

        self.assertEqual(str(test + (1,0)),
                         "Tuple[[1 1]\n [3 3]\n [-- --], mask]")
        self.assertEqual(str(test - (0,1)),
                         "Tuple[[0 0]\n [2 2]\n [-- --], mask]")
        self.assertEqual(str(test + test),
                         "Tuple[[0 2]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test + np.arange(6).reshape(3,2)),
                         "Tuple[[0 2]\n [4 6]\n [-- --], mask]")

        temp = Tuple(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(test + temp),
                         "Tuple[[-- --]\n [4 6]\n [-- --], mask]")
        self.assertEqual(str(test - 2*temp),
                         "Tuple[[-- --]\n [-2 -3]\n [-- --], mask]")
        self.assertEqual(str(test * temp),
                         "Tuple[[-- --]\n [4 9]\n [-- --], mask]")
        self.assertEqual(str(test / temp),
                         "Tuple[[-- --]\n [1 1]\n [-- --], mask]")
        self.assertEqual(str(test % temp),
                         "Tuple[[-- --]\n [0 0]\n [-- --], mask]")
        self.assertEqual(str(test / [[2,1],[1,0],[7,0]]),
                         "Tuple[[0 1]\n [-- --]\n [-- --], mask]")
        self.assertEqual(str(test % [[2,1],[1,0],[7,0]]),
                         "Tuple[[0 0]\n [-- --]\n [-- --], mask]")

        temp = Tuple(np.arange(6).reshape(3,2), [True, False, False])
        self.assertEqual(str(temp),      "Tuple[[-- --]\n [2 3]\n [4 5], mask]")
        self.assertEqual(str(temp[0]),   "Tuple[-- --, mask]")
        self.assertEqual(str(temp[1]),   "Tuple[2 3]")
        self.assertEqual(str(temp[0:2]), "Tuple[[-- --]\n [2 3], mask]")
        self.assertEqual(str(temp[0:1]), "Tuple[[-- --], mask]")
        self.assertEqual(str(temp[1:2]), "Tuple[[2 3]]")

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
