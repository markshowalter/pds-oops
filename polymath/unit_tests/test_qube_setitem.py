################################################################################
# Qube __setitem__() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

class Test_Qube_setitem(unittest.TestCase):

  def runTest(self):

    ############################################################################
    # Qube into Qube, no broadcast, unmasked, with integers, ellipses, colons
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1)

    a[0] = b[0]
    self.assertTrue(np.all(a.values[0] == b.values[0]))
    self.assertTrue(np.all(a.mask == b.mask))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.values[:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask == b.mask))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,:,0]))
    self.assertTrue(np.all(a.mask == b.mask))

    ############################################################################
    # Same as above, with matching masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    a[0] = b[0]
    self.assertTrue(np.all(a.values[0] == b.values[0]))
    self.assertTrue(np.all(a.mask[0] == b.mask[0]))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.values[:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask[:,0] == b.mask[:,0]))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,:,0]))
    self.assertTrue(np.all(a.mask[:,:,0] == b.mask[:,:,0]))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.values[0,:,0] == b.values[0,:,1]))
    self.assertTrue(np.all(a.mask[0,:,0] == b.mask[0,:,1]))

    a[...,::-1] = b
    self.assertTrue(np.all(a.values == b.values[:,:,::-1]))
    self.assertTrue(np.all(a.mask == b.mask[:,:,::-1]))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5]))
    self.assertTrue(np.all(a.mask[:,:,0:5:2] == b.mask[:,:,2:5]))

    ############################################################################
    # Same as above, requiring right mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)

    a[0] = b[0]
    self.assertTrue(np.all(a.values[0] == b.values[0]))
    self.assertTrue(np.all(a.mask[0] == True))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.values[:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask[:,0] == True))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,:,0]))
    self.assertTrue(np.all(a.mask[:,:,0] == True))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.values[0,:,0] == b.values[0,:,1]))
    self.assertTrue(np.all(a.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.values == b.values[:,:,::-1]))
    self.assertTrue(np.all(a.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5]))
    self.assertTrue(np.all(a.mask[:,:,0:5:2] == True))

    ############################################################################
    # Same as above, requiring left mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))

    a[0] = b[0]
    self.assertTrue(np.all(a.values[0] == b.values[0]))
    self.assertTrue(np.all(a.mask[0] == b.mask[0]))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.values[:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask[:,0] == b.mask[:,0]))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,:,0]))
    self.assertTrue(np.all(a.mask[:,:,0] == b.mask[:,:,0]))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.values[0,:,0] == b.values[0,:,1]))
    self.assertTrue(np.all(a.mask[0,:,0] == b.mask[0,:,1]))

    a[...,::-1] = b
    self.assertTrue(np.all(a.values == b.values[:,:,::-1]))
    self.assertTrue(np.all(a.mask == b.mask[:,:,::-1]))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5]))
    self.assertTrue(np.all(a.mask[:,:,0:5:2] == b.mask[:,:,2:5]))

    ############################################################################
    # Same as above, requiring left and right mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    self.assertEqual(type(a.mask), bool)
    self.assertEqual(type(b.mask), bool)

    a[0] = b[0]
    self.assertTrue(np.all(a.values[0] == b.values[0]))
    self.assertTrue(np.all(a.mask[0] == True))
    self.assertEqual(type(a.mask), np.ndarray)
    self.assertEqual(type(b.mask), bool)

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.values[:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask[:,0] == True))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,:,0]))
    self.assertTrue(np.all(a.mask[:,:,0] == True))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.values[0,:,0] == b.values[0,:,1]))
    self.assertTrue(np.all(a.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.values == b.values[:,:,::-1]))
    self.assertTrue(np.all(a.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5]))
    self.assertTrue(np.all(a.mask[:,:,0:5:2] == True))

    ############################################################################
    # Same as above, requiring right object broadcasting
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(6,3,2), drank=1, mask=True)

    a[0] = b
    self.assertTrue(np.all(a.values[0] == b.values))
    self.assertTrue(np.all(a.mask[0] == True))

    a[:,0] = b
    self.assertTrue(np.all(a.values[:,0] == b.values))
    self.assertTrue(np.all(a.mask[:,0] == True))

    b = Vector(np.random.randn(5,6,3,2), drank=1, mask=True)
    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.values[:,:,0] == b.values[:,0]))
    self.assertTrue(np.all(a.mask[:,:,0] == True))

    a[0,...,0] = b[...,1]
    self.assertTrue(np.all(a.values[0,:,0] == b.values[:,1]))
    self.assertTrue(np.all(a.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.values[:,:,::-1] == b.values))
    self.assertTrue(np.all(a.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.values[:,:,0:5:2] == b.values[:,2:5]))
    self.assertTrue(np.all(a.mask[:,:,0:5:2] == True))

    ############################################################################
    # Using boolean arrays as masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3), mask=True)

    mask = np.array([True,False,False,True])
    a[mask] = b[mask]
    self.assertTrue(np.all(a.values[mask] == b.values[mask]))
    self.assertTrue(np.all(a.mask[mask] == True))
    self.assertTrue( np.all(a.values[0] == b.values[0]))
    self.assertFalse(np.all(a.values[1] == b.values[1]))
    self.assertFalse(np.all(a.values[2] == b.values[2]))
    self.assertTrue( np.all(a.values[3] == b.values[3]))
    self.assertTrue( np.all(a.mask[0] == True))
    self.assertTrue( np.all(a.mask[3] == True))

    mask = np.array([True,False,False,True])
    a[mask] = (0,0,1)
    self.assertTrue(np.all(a.values[mask][...,0] == 0))
    self.assertTrue(np.all(a.values[mask][...,1] == 0))
    self.assertTrue(np.all(a.values[mask][...,2] == 1))
    self.assertTrue(np.all(a.mask[mask] == False))
    self.assertTrue( np.all(a.values[0] == (0,0,1)))
    self.assertFalse(np.all(a.values[1] == b.values[1]))
    self.assertFalse(np.all(a.values[2] == b.values[2]))
    self.assertTrue( np.all(a.values[3] == (0,0,1)))
    self.assertTrue( np.all(a.mask[0] == False))
    self.assertTrue( np.all(a.mask[3] == False))

    mask = np.array([True,False,False,True])
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    a[mask] = b
    self.assertTrue(np.all(a.values[mask] == b.values))
    self.assertTrue(np.all(a.mask[mask] == False))
    self.assertTrue( np.all(a.mask[0] == False))
    self.assertTrue( np.all(a.mask[3] == False))

    ############################################################################
    # Same as above, using Boolean subclasses
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3), mask=True)

    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = b[mask]
    self.assertTrue(np.all(a.values[mask.values] == b.values[mask.values]))
    self.assertTrue(np.all(a.mask[mask.values] == True))
    self.assertTrue( np.all(a.values[0] == b.values[0]))
    self.assertFalse(np.all(a.values[1] == b.values[1]))
    self.assertFalse(np.all(a.values[2] == b.values[2]))
    self.assertTrue( np.all(a.values[3] == b.values[3]))
    self.assertTrue( np.all(a.mask[0] == True))
    self.assertTrue( np.all(a.mask[3] == True))

    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = (0,0,1)
    self.assertTrue(np.all(a.values[mask.values][...,0] == 0))
    self.assertTrue(np.all(a.values[mask.values][...,1] == 0))
    self.assertTrue(np.all(a.values[mask.values][...,2] == 1))
    self.assertTrue(np.all(a.mask[mask.values] == False))
    self.assertTrue( np.all(a.values[0] == (0,0,1)))
    self.assertFalse(np.all(a.values[1] == b.values[1]))
    self.assertFalse(np.all(a.values[2] == b.values[2]))
    self.assertTrue( np.all(a.values[3] == (0,0,1)))
    self.assertTrue( np.all(a.mask[0] == False))
    self.assertTrue( np.all(a.mask[3] == False))

    mask = Boolean(np.array([True,False,False,True]))
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    a[mask] = b
    self.assertTrue(np.all(a.values[mask.values] == b.values))
    self.assertTrue(np.all(a.mask[mask.values] == False))
    self.assertTrue( np.all(a.mask[0] == False))
    self.assertTrue( np.all(a.mask[3] == False))

    ############################################################################
    # Using tuples, Vectors, Pairs
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=False)
    b = Vector(np.random.randn(3,6,3), mask=True)

    tuple = [(0,1,3),(0,1,3)]
    a[tuple] = b
    self.assertTrue(np.all(a.values[0,0] == b.values[0]))
    self.assertTrue(np.all(a.values[1,1] == b.values[1]))
    self.assertTrue(np.all(a.values[3,3] == b.values[2]))
    self.assertTrue(np.all(a.mask[0,0] == True))
    self.assertTrue(np.all(a.mask[1,1] == True))
    self.assertTrue(np.all(a.mask[3,3] == True))

    pair = Pair([(0,0),(1,1),(3,3)])
    a[pair] = b
    self.assertTrue(np.all(a.values[0,0] == b.values[0]))
    self.assertTrue(np.all(a.values[1,1] == b.values[1]))
    self.assertTrue(np.all(a.values[3,3] == b.values[2]))
    self.assertTrue(np.all(a.mask[0,0] == True))
    self.assertTrue(np.all(a.mask[1,1] == True))
    self.assertTrue(np.all(a.mask[3,3] == True))

    self.assertEqual(a[pair], a[tuple])

    b = Vector(np.random.randn(3,3), mask=True)

    tuple = [(0,1,3),(0,1,3),(0,0,0)]
    a[tuple] = b
    self.assertTrue(np.all(a.values[0,0,0] == b.values[0]))
    self.assertTrue(np.all(a.values[1,1,0] == b.values[1]))
    self.assertTrue(np.all(a.values[3,3,0] == b.values[2]))
    self.assertTrue(np.all(a.mask[0,0,0] == True))
    self.assertTrue(np.all(a.mask[1,1,0] == True))
    self.assertTrue(np.all(a.mask[3,3,0] == True))

    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    a[vector] = b
    self.assertTrue(np.all(a.values[0,0,0] == b.values[0]))
    self.assertTrue(np.all(a.values[1,1,0] == b.values[1]))
    self.assertTrue(np.all(a.values[3,3,0] == b.values[2]))
    self.assertTrue(np.all(a.mask[0,0,0] == True))
    self.assertTrue(np.all(a.mask[1,1,0] == True))
    self.assertTrue(np.all(a.mask[3,3,0] == True))

    self.assertEqual(a[vector], a[tuple])

    ############################################################################
    ############################################################################
    # All the same test as above objects with derivatives
    ############################################################################
    ############################################################################

    ############################################################################
    # Qube into Qube, no broadcast, unmasked, with integers, ellipses, colons
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    a.insert_deriv('v', Vector(np.random.randn(4,5,6,3,2,3), drank=2))

    b = Vector(np.random.randn(4,5,6,3,2), drank=1)

    self.assertRaises(ValueError, a.__setitem__, 0, b[0])

    b = Vector(np.random.randn(4,5,6,3,2), drank=1)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    b.insert_deriv('v', Vector(np.random.randn(4,5,6,3,2,3), drank=2))

    a[0] = b[0]
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.mask == b.d_dt.mask))
    self.assertTrue(np.all(a.d_dv.values[0] == b.d_dv.values[0]))
    self.assertTrue(np.all(a.d_dv.mask == b.d_dv.mask))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask == b.d_dt.mask))
    self.assertTrue(np.all(a.d_dv.values[:,0] == b.d_dv.values[:,0]))
    self.assertTrue(np.all(a.d_dv.mask == b.d_dv.mask))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0]))
    self.assertTrue(np.all(a.d_dt.mask == b.d_dt.mask))
    self.assertTrue(np.all(a.d_dv.values[:,:,0] == b.d_dv.values[:,:,0]))
    self.assertTrue(np.all(a.d_dv.mask == b.d_dv.mask))

    ############################################################################
    # Same as above, with matching masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=a.mask))
    b = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))

    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,3), drank=1))
    self.assertRaises(ValueError, a.__setitem__, 0, b[0])

    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=b.mask))

    a[0] = b[0]
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.mask[0] == b.d_dt.mask[0]))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,0] == b.d_dt.mask[:,0]))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0] == b.d_dt.mask[:,:,0]))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1]))
    self.assertTrue(np.all(a.d_dt.mask[0,:,0] == b.d_dt.mask[0,:,1]))

    a[...,::-1] = b
    self.assertTrue(np.all(a.d_dt.values == b.d_dt.values[:,:,::-1]))
    self.assertTrue(np.all(a.d_dt.mask == b.d_dt.mask[:,:,::-1]))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0:5:2] == b.d_dt.mask[:,:,2:5]))

    ############################################################################
    # Same as above, requiring right mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=a.mask))

    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=True))

    a[0] = b[0]
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.mask[0] == True))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,0] == True))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0] == True))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1]))
    self.assertTrue(np.all(a.d_dt.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.d_dt.values == b.d_dt.values[:,:,::-1]))
    self.assertTrue(np.all(a.d_dt.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0:5:2] == True))

    ############################################################################
    # Same as above, requiring left mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))

    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=b.mask))

    a[0] = b[0]
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.mask[0] == b.d_dt.mask[0]))

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,0] == b.d_dt.mask[:,0]))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0] == b.d_dt.mask[:,:,0]))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1]))
    self.assertTrue(np.all(a.d_dt.mask[0,:,0] == b.d_dt.mask[0,:,1]))

    a[...,::-1] = b
    self.assertTrue(np.all(a.d_dt.values == b.d_dt.values[:,:,::-1]))
    self.assertTrue(np.all(a.d_dt.mask == b.d_dt.mask[:,:,::-1]))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0:5:2] == b.d_dt.mask[:,:,2:5]))

    ############################################################################
    # Same as above, requiring left and right mask reshaping
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))

    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=True))

    a[0] = b[0]
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.mask[0] == True))
    self.assertEqual(type(a.d_dt.mask), np.ndarray)

    a[:,0] = b[:,0]
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,0] == True))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0] == True))

    a[0,...,0] = b[0,...,1]
    self.assertTrue(np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1]))
    self.assertTrue(np.all(a.d_dt.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.d_dt.values == b.d_dt.values[:,:,::-1]))
    self.assertTrue(np.all(a.d_dt.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0:5:2] == True))

    ############################################################################
    # Same as above, requiring right object broadcasting
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3)))

    b = Vector(np.random.randn(6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(6,3), mask=True))

    a[0] = b
    self.assertTrue(np.all(a.d_dt.values[0] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dt.mask[0] == True))

    a[:,0] = b
    self.assertTrue(np.all(a.d_dt.values[:,0] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dt.mask[:,0] == True))

    b = Vector(np.random.randn(5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(5,6,3), mask=True))

    a[...,0] = b[...,0]
    self.assertTrue(np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,0]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0] == True))

    a[0,...,0] = b[...,1]
    self.assertTrue(np.all(a.d_dt.values[0,:,0] == b.d_dt.values[:,1]))
    self.assertTrue(np.all(a.d_dt.mask[0,:,0] == True))

    a[...,::-1] = b
    self.assertTrue(np.all(a.d_dt.values[:,:,::-1] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dt.mask == True))

    a[...,0:5:2] = b[...,2:5]
    self.assertTrue(np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,2:5]))
    self.assertTrue(np.all(a.d_dt.mask[:,:,0:5:2] == True))

    ############################################################################
    # Using boolean arrays as masks
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=a.mask))

    b = Vector(np.random.randn(4,5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=True))

    mask = np.array([True,False,False,True])
    a[mask] = b[mask]
    self.assertTrue(np.all(a.d_dt.values[mask] == b.d_dt.values[mask]))
    self.assertTrue(np.all(a.d_dt.mask[mask] == True))
    self.assertTrue( np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertFalse(np.all(a.d_dt.values[1] == b.d_dt.values[1]))
    self.assertFalse(np.all(a.d_dt.values[2] == b.d_dt.values[2]))
    self.assertTrue( np.all(a.d_dt.values[3] == b.d_dt.values[3]))
    self.assertTrue( np.all(a.d_dt.mask[0] == True))
    self.assertTrue( np.all(a.d_dt.mask[3] == True))

    mask = np.array([True,False,False,True])
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    b.insert_deriv('t', Vector(np.random.randn(2,5,6,3), mask=False))

    a[mask] = b
    self.assertTrue(np.all(a.d_dt.values[mask] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dt.mask[mask] == False))
    self.assertTrue( np.all(a.d_dt.mask[0] == False))
    self.assertTrue( np.all(a.d_dt.mask[3] == False))

    ############################################################################
    # Same as above, using Boolean subclasses
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=a.mask))

    b = Vector(np.random.randn(4,5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=True))

    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = b[mask]
    self.assertTrue(np.all(a.d_dt.values[mask.values] ==
                           b.d_dt.values[mask.values]))
    self.assertTrue(np.all(a.d_dt.mask[mask.values] == True))
    self.assertTrue( np.all(a.d_dt.values[0] == b.d_dt.values[0]))
    self.assertFalse(np.all(a.d_dt.values[1] == b.d_dt.values[1]))
    self.assertFalse(np.all(a.d_dt.values[2] == b.d_dt.values[2]))
    self.assertTrue( np.all(a.d_dt.values[3] == b.d_dt.values[3]))
    self.assertTrue( np.all(a.d_dt.mask[0] == True))
    self.assertTrue( np.all(a.d_dt.mask[3] == True))

    mask = Boolean(np.array([True,False,False,True]))
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    b.insert_deriv('t', Vector(np.random.randn(2,5,6,3), mask=False))

    a[mask] = b
    self.assertTrue(np.all(a.d_dt.values[mask.values] == b.d_dt.values))
    self.assertTrue(np.all(a.d_dt.mask[mask.values] == False))
    self.assertTrue( np.all(a.d_dt.mask[0] == False))
    self.assertTrue( np.all(a.d_dt.mask[3] == False))

    ############################################################################
    # Using tuples, Vectors, Pairs
    ############################################################################

    a = Vector(np.random.randn(4,5,6,3), mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=False))

    b = Vector(np.random.randn(3,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(3,6,3), mask=True))

    tuple = [(0,1,3),(0,1,3)]
    a[tuple] = b
    self.assertTrue(np.all(a.d_dt.values[0,0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.values[1,1] == b.d_dt.values[1]))
    self.assertTrue(np.all(a.d_dt.values[3,3] == b.d_dt.values[2]))
    self.assertTrue(np.all(a.d_dt.mask[0,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[1,1] == True))
    self.assertTrue(np.all(a.d_dt.mask[3,3] == True))

    pair = Pair([(0,0),(1,1),(3,3)])
    a[pair] = b
    self.assertTrue(np.all(a.d_dt.values[0,0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.values[1,1] == b.d_dt.values[1]))
    self.assertTrue(np.all(a.d_dt.values[3,3] == b.d_dt.values[2]))
    self.assertTrue(np.all(a.d_dt.mask[0,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[1,1] == True))
    self.assertTrue(np.all(a.d_dt.mask[3,3] == True))

    self.assertEqual(a.d_dt[pair], a.d_dt[tuple])

    b = Vector(np.random.randn(3,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(3,3), mask=True))

    tuple = [(0,1,3),(0,1,3),(0,0,0)]
    a[tuple] = b
    self.assertTrue(np.all(a.d_dt.values[0,0,0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.values[1,1,0] == b.d_dt.values[1]))
    self.assertTrue(np.all(a.d_dt.values[3,3,0] == b.d_dt.values[2]))
    self.assertTrue(np.all(a.d_dt.mask[0,0,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[1,1,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[3,3,0] == True))

    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    a[vector] = b
    self.assertTrue(np.all(a.d_dt.values[0,0,0] == b.d_dt.values[0]))
    self.assertTrue(np.all(a.d_dt.values[1,1,0] == b.d_dt.values[1]))
    self.assertTrue(np.all(a.d_dt.values[3,3,0] == b.d_dt.values[2]))
    self.assertTrue(np.all(a.d_dt.mask[0,0,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[1,1,0] == True))
    self.assertTrue(np.all(a.d_dt.mask[3,3,0] == True))

    self.assertEqual(a.d_dt[vector], a.d_dt[tuple])

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################