################################################################################
# oops/obs_/pushbroom.py: Subclass Pushbroom of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation import Observation

class Pushbroom(Observation):
    """A Pushbroom is subclass of Observation consisting of a 2-D image
    generated by sweeping a strip of sensors across a field of view.

    The FOV object is assumed to define the entire field of view, even if the
    reality is that a 1-D array was swept in a (roughly) perpendicular
    direction. The virtual array of data is assumed to have a t-dimension of 1,
    while the number of time steps is equal to the number of samples in the u
    or v direction, depending on the direction of sweep. In effect, then, the
    virtual array samples a diagonal ramp through the cube.
    """

    def __init__(self, axes, uv_size,
                       cadence, fov, path_id, frame_id, **subfields):
        """Constructor for a Slit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of "u" or "ut"
                        should appear at the location of the array's u-axis;
                        "vt" or "v" should appear at the location of the array's
                        v-axis. The "t" suffix is used for the one of these axes
                        that is emulated by time-sampling the slit.
            uv_size     the size of the detector in FOV units along the (u,v)
                        axes. Default is (1,1), indicating no dead space between
                        the detectors. It will be < 1 if there are gaps.

            cadence     a Cadence object defining the start time and duration of
                        each consecutive position of the pushbroom.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).
            path_id     the registered ID of a path co-located with the
                        instrument.
            frame_id    the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.cadence = cadence
        self.fov = fov
        self.path_id = path_id
        self.frame_id = frame_id

        self.axes = list(axes)
        assert (("u" in self.axes and "vt" in self.axes) or
                ("v" in self.axes and "ut" in self.axes))

        if "ut" in self.axes:
            self.u_axis = self.axes.index("ut")
            self.v_axis = self.axes.index("v")
            self.t_axis = self.u_axis
            self.cross_slit_uv_index = 0
            self.along_slit_uv_index = 1
        else:
            self.u_axis = self.axes.index("u")
            self.v_axis = self.axes.index("vt")
            self.t_axis = self.v_axis
            self.cross_slit_uv_index = 1
            self.along_slit_uv_index = 0

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        assert len(self.cadence.shape) == 1
        assert (self.fov.uv_shape.vals[self.cross_slit_uv_index] ==
                self.cadence.shape[0])

        self.uv_shape = list(self.fov.uv_shape.vals)

        self.uv_size = Pair.as_pair(uv_size)
        self.uv_is_discontinuous = (self.uv_size != Pair.ONES)

        duv_dt_basis_vals = np.zeros(2)
        duv_dt_basis_vals[self.cross_slit_uv_index] = 1.
        self.duv_dt_basis = Pair(duv_dt_basis_vals)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return

    def uvt(self, indices, fovmask=False):
        """Returns the FOV coordinates (u,v) and the time in seconds TDB
        associated with the given indices into the data array. This method
        supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Tuple.as_tuple(indices)

        uv = indices.as_pair((self.u_axis,self.v_axis))
        if self.uv_is_discontinuous:
            uv_int = Pair.as_int(uv)
            uv = uv_int + (uv - uv_int) * self.uv_size

        tstep = indices.to_scalar(self.t_axis)
        time = self.cadence.time_at_tstep(tstep)

        if fovmask:
            is_inside = self.uv_is_inside(uv, inclusive=True)
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
                uv.mask = mask
                time.mask = mask

        return (uv, time)

    def uvt_range(self, indices, fovmask=False):
        """Returns the ranges of FOV coordinates (u,v) and the time range in
        seconds TDB associated with the given integer indices into the data
        array.

        Input:
            indices     a Tuple of integer array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        indices = Tuple.as_int(indices)

        uv_min = indices.as_pair((self.u_axis,self.v_axis))
        uv_max = uv_min + self.uv_size

        tstep = indices.to_scalar(self.t_axis)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep)

        if fovmask:
            is_inside = self.uv_is_inside(uv_min, inclusive=False)
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
                uv_min.mask = mask
                uv_max.mask = mask
                time_min.mask = mask
                time_max.mask = mask

        return (uv_min, uv_max, time_min, time_max)

    def times_at_uv(self, uv_pair, fovmask=False, extras=None):
        """Returns the start and stop times of the specified spatial pixel
        (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.
            extras      an optional tuple or dictionary containing any extra
                        parameters required for the conversion from (u,v) to
                        time.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        uv_pair = Pair.as_int(uv_pair)
        tstep = uv_pair.to_scalar(self.cross_slit_uv_index)
        (time0, time1) = self.cadence.time_range_at_tstep(tstep)

        if fovmask:
            is_inside = self.fov.uv_is_inside(uv_pair, inclusive=True)
            if not np.all(is_inside):
                mask = uv_pair.mask | np.logical_not(is_inside)
                time0.mask = mask
                time1.mask = mask

        return (time0, time1)

    def sweep_duv_dt(self, uv_pair, extras=None):
        """Returns the mean local sweep speed of the instrument in the (u,v)
        directions.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      an optional tuple or dictionary containing any extra
                        parameters required to define the timing of array
                        elements.

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        uv_pair = Pair.as_pair(uv_pair)
        tstep = uv_pair.to_scalar(self.cross_slit_uv_index)

        return self.duv_dt_basis / self.cadence.tstride_at_tstep(tstep)

    def time_shift(self, dtime):
        """Returns a copy of the observation object in which times have been
        shifted by a constant value.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Pushbroom(self.axes, self.uv_size,
                        self.cadence.time_shift(dtime),
                        self.fov, self.path_id, self.frame_id)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Pushbroom(unittest.TestCase):

    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        flatfov = FlatFOV((0.001,0.001), (10,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pushbroom(axes=("u","vt"), uv_size=(1,1),
                        cadence=cadence, fov=flatfov,
                        path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(uv, indices.as_pair())

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6], indices.as_pair()[:6])

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, indices.as_pair())
        self.assertEqual(uv_max, indices.as_pair() + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, indices.as_pair())
        self.assertEqual(uv_max, indices.as_pair() + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], indices.as_pair()[:2])
        self.assertEqual(uv_max[:2], indices.as_pair()[:2] + (1,1))
        self.assertEqual(time_min[:2], cadence.tstride *
                                       indices.to_scalar(1)[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.tstride * uv.to_scalar(1)[:4])
        self.assertEqual(time1[:4], time0[:4] + cadence.texp)

        # Alternative axis order ("ut","v")
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Pushbroom(axes=("ut","v"), uv_size=(1,1),
                        cadence=cadence, fov=flatfov,
                        path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.as_pair())
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min, indices.as_pair())
        self.assertEqual(uv_max, indices.as_pair() + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(0))
        self.assertEqual(time1, time0 + cadence.texp)

        # Alternative uv_size and texp for discontinuous indices
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Pushbroom(axes=("ut","v"), uv_size=(0.5,0.8),
                        cadence=cadence, fov=flatfov,
                        path_id="SSB", frame_id="J2000")

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1], 50.)
        self.assertEqual(obs.uvt((5,5))[1], 50.)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.,0.))
        self.assertEqual(obs.uvt((5,0))[0], (5.,0.))
        self.assertEqual(obs.uvt((5,5))[0], (5.,5.))

        self.assertTrue(abs(obs.uvt((6      ,0))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.1,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.2,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.3,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.4,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (6.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (4.,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (5.,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,5.0)) < delta)

        # Alternative with uv_size and texp and axes
        obs = Pushbroom(axes=("a","v","b","ut","c"), uv_size=(0.5,0.8),
                        cadence=cadence, fov=flatfov,
                        path_id="SSB", frame_id="J2000")

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],  0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.,0.))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (5.,0.))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (5.,5.))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.1,1.)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.2,2.)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.3,3.)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.4,4.)) < delta)
        self.assertTrue(abs(obs.uvt((1,5,4,7 - eps,7))[0] - (6.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,5.0)) < delta)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
