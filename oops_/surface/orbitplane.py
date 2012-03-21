################################################################################
# oops_/surface/orbitplane.py: OrbitPlane subclass of class Surface
#
# 3/ 18/12 MRS: Adapted from class RingPlane.
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.event import Event
from oops_.surface.surface_ import Surface
from oops_.surface.ringplane import RingPlane
import oops_.frame.all as frame_
import oops_.path.all as path_
import oops_.registry as registry

class OrbitPlane(Surface):
    """OrbitPlane is a subclass of the Surface class describing a flat surface
    sharing its geometric center and tilt with a body on an eccentric and/or
    inclined orbit. The orbit is described as circle offset from the center of
    the planet by a distance ae; this approximation is only accurate to first
    order in eccentricty.

    The coordinate system consists of cylindrical coordinates (a, theta, z)
    where a is the mean radius of the orbit. The zero of longitude is aligned
    with the pericenter.

    The system is masked outside the semimajor axis, but unmasked inside.
    However, coordinates and intercepts are calculated at all locations.
    """

    def __init__(self, elements, epoch, origin, frame, id=None):
        """Constructor for an OffsetPlane surface.

            elements    a tuple containing three, six or nine orbital elements:
                a           mean radius of orbit, km.
                lon         mean longitude at epoch of a reference object, in
                            radians. This is provided if the user wishes to
                            track a moving body in the plane. However, it does
                            not affect the surface or its coordinate system.
                n           mean motion of a body orbiting within the ring, in
                            radians/sec. This affects velocities returned by
                            the surface but not the surface or its coordinate
                            system.

                e           orbital eccentricty.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!

            epoch       the time TDB relative to which all orbital elements are
                        defined.
            origin      the path or ID of the orbit center.
            frame       the frame or ID of the frame in which the orbit is
                        defined. Should be inertial.
            id          the name under which to register a temporary path or
                        frame if it is needed. Not used for circular, equatorial
                        orbits. None to use temporary path and frame names.
        """

        # Save the initial center path and frame. The frame should be inertial.
        self.origin_id = registry.as_path_id(origin)
        self.frame_id = registry.as_frame_id(frame)

        # We will update the surface's actual path and frame as needed
        self.internal_origin_id = self.origin_id
        self.internal_frame_id = self.frame_id

        # Save the orbital elements
        self.a   = elements[0]
        self.lon = elements[1]
        self.n   = elements[2]

        self.epoch = Scalar.as_scalar(epoch)

        # Interpret the inclination
        self.has_inclination = len(elements) >= 9
        if self.has_inclination:
            self.i = elements[6]
            self.has_inclination = (self.i != 0)

        # If the orbit is inclined, define a special-purpose inclined frame
        if self.has_inclination:
            if id is None:
                self.inclined_frame_id = registry.temporary_frame_id()
            else:
                self.inclined_frame_id = id + "_INCLINATION"

            self.inclined_frame = frame_.InclinedFrame(
                                                elements[6],  # inclination
                                                elements[7],  # ascending node
                                                elements[8],  # regression rate
                                                self.epoch,
                                                self.frame_id,
                                                True,         # despin
                                                self.inclined_frame_id)
            self.internal_frame_id = self.inclined_frame_id
        else:
            self.inclined_frame = None
            self.inclined_frame_id = self.internal_frame_id

        # The inclined frame changes its tilt relative to the equatorial plane,
        # accounting for nodal regression, but does not change the reference
        # longitude from that used by the initial frame.

        # Interpret the eccentricity
        self.has_eccentricity = len(elements) >= 6
        if self.has_eccentricity:
            self.e = elements[3]
            self.has_eccentricity = (self.e != 0)

        # If the orbit is eccentric, define a special-purpose path defining the
        # center of the displaced ring
        if self.has_eccentricity:
            self.ae = self.a * self.e
            self.lon_sub_peri = self.lon - elements[4]
            self.n_sub_prec = self.n - elements[5]

            if id is None:
                self.peri_path_id = registry.temporary_path_id()
            else:
                self.peri_path_id = id + "_ECCENTRICITY"

            self.peri_path = path_.CirclePath(
                                    elements[0] * elements[3],  # a*e
                                    elements[4] + np.pi,        # apocenter
                                    elements[5],                # precession
                                    self.epoch,                 # epoch
                                    self.internal_origin_id,    # origin
                                    self.internal_frame_id,     # reference
                                    self.peri_path_id)          # id
            self.internal_origin_id = self.peri_path_id

            # The peri_path circulates around the initial origin but does not
            # rotate.

            if id is None:
                self.spin_frame_id = registry.temporary_frame_id()
            else:
                self.spin_frame_id = id + "_ECCENTRICITY"

            self.spin_frame = frame_.SpinFrame(elements[4],     # pericenter
                                            elements[5],        # precession
                                            self.epoch,         # epoch
                                            2,                  # z-axis
                                            self.internal_frame_id, # reference
                                            self.spin_frame_id)     # id
            self.internal_frame_id = self.spin_frame_id
        else:
            self.peri_path = None
            self.peri_path_id = self.internal_origin_id

            self.spin_frame = None
            self.spin_frame_id = self.internal_frame_id

        self.ringplane = RingPlane(self.internal_origin_id,
                                   self.internal_frame_id,
                                   radii=(0,self.a), gravity=None, elevation=0.)

    def as_coords(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         a Vector3 of observer positions, required for surfaces
                        that are defined in part by the position of the
                        observer; otherwise ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        return self.ringplane.as_coords(pos, obs, axes, derivs)

    def as_vector3(self, r, theta, z=Scalar(0.), obs=None, derivs=False):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            r           a Scalar of radius values, with optional units.
            theta       a Scalar of longitude values, with optional units.
            z           an optional Scalar of elevation values, with optional
                        units; default is Scalar(0.).
            obs         a Vector3 of observer positions. In some cases, a
                        surface is defined in part by the position of the
                        observer. In the case of a RingPlane, this argument is
                        ignored and can be omitted.
            derivs      if True, the partial derivatives of the returned vector
                        with respect to the coordinates are returned as well.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         a unitless Vector3 object of positions, in km.

                        If derivs is True, then the returned Vector3 object has
                        a subfield "d_dcoord", which contains the partial
                        derivatives d(x,y,z)/d(r,theta,z), as a MatrixN with
                        item shape [3,3].
        """

        return self.ringplane.as_vector3(r, theta, z, obs, derivs)

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

        Return:         (pos, t)
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar of scale factors t such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t. For purposes of differentiation, los
                        is assumed to have unit length.
        """

        return self.ringplane.intercept(obs, los, derivs)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            derivs      True to include a matrix of partial derivatives.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.

                        If derivs is True, then the normal vectors returned have
                        a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        return self.ringplane.normal(pos, derivs)

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal      a Vector3 of normal vectors, with optional units.
            derivs      true to return a matrix of partial derivatives.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dperp", which contains the partial
                        derivatives with respect to components of the normal
                        vector, as a MatrixN object with item shape [3,3].
        """

        return self.ringplane.intercept_with_normal(normal)

    def intercept_normal_to(self, pos):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        return self.ringplane.intercept_normal_to(pos)

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        if self.has_eccentricity:
            # For purposes of a first-order velocity calculation, we can assume
            # that the difference between mean longitude and true longitude, in
            # a planet-centered frame, is small.
            #
            # In an inertial, planet-centered frame:
            #
            # r = a - ae cos(lon - peri)
            # lon = lon0 + n * (time - epoch) + 2ae sin(lon - peri)
            #
            # dr/dt = ae sin(lon - peri) (n - prec)
            # dlon/dt = n + 2ae cos(n - peri) (n - prec)
            #
            # In a frame rotating at rate = prec:
            #
            # dr/dt = ae sin(lon - peri) (n - prec)
            # dlon/dt = (n - prec) + 2ae cos(lon - peri) (n - prec)
            #
            # x = r cos(lon)
            # y = r sin(lon)
            #
            # dx/dt = dr/dt * cos(lon) - r sin(lon) dlon/dt
            # dy/dy = dr/dt * sin(lon) + r cos(lon) dlon/dt

            (x,y,z) = pos.as_scalars()
            x += self.ae        # shift origin to center of planet

            r = (x**2 + y**2).sqrt()
            cos_lon_sub_peri = x/r
            sin_lon_sub_peri = y/r

            dr_dt = (self.ae * self.n_sub_prec) * sin_lon_sub_peri
            r_dlon_dt = self.n_sub_prec * r * (1 + 2*self.ae * cos_lon_sub_prec)

            dx_dt = dr_dt * cos_lon_sub_peri - r_dlon_dt * sin_lon_sub_peri
            dy_dt = dr_dt * sin_lon_sub_peri + r_dlon_dt * cos_lon_sub_peri

            return Vector3.from_scalars(dx_dt, dy_dt, 0.)

        else:
            return self.n * pos.cross((0,0,-1))

    ############################################################################
    # Overrides of event methods
    ############################################################################

    def as_event(self, time, coords, dcoords_dt=None, obs=None):
        """Converts a time and coordinates in the surface's internal coordinate
        system into an event object.

        Input:
            time        the Scalar of time values at which to evaluate the
                        coordinates.
            coords      a tuple containing the two or three coordinate values.
                        If only two are provided, the third is assumed to be
                        zero.
            dcoords_dt  an optional tuple containing rates of changes of the
                        coordinates. If provided, these values define the
                        velocity vector of the event object.
            obs         a Vector3 of observer positions. In some cases, a
                        surface is defined in part by the position of the
                        observer.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         an event object relative to the origin and frame of the
                        surface.
        """

        event = self.ringplane.as_event(time, coords, dcoords_dt, obs)
        return event.wrt(self.origin_id, self.frame_id)

    def event_as_coords(self, event, obs=None, axes=3, derivs=False):
        """Converts an event object to coordinates and, optionally, their
        time-derivatives.

        Input:
            event       an event object.
            obs         a Vector3 of observer positions. In some cases, a
                        surface is defined in part by the position of the
                        observer. In the case of a RingPlane, this argument is
                        ignored and can be omitted.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        time are returned as well. Using a tuple, you can
                        indicate whether to return time derivatives on a
                        coordinate-by-coordinate basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        event = event.wrt(self.internal_origin_id, self.internal_frame_id)
        return self.ringplane.event_as_coords(event, obs, axes, derivs)

    ############################################################################
    # Longitude-anomaly conversions
    ############################################################################

    def from_mean_anomaly(self, anom):
        """Returns the longitude in this coordinate frame based on the mean
        anomaly, and accurate to first order in eccentricity."""

        anom = Scalar.as_standard(anom)

        if not self.has_eccentricity:
            return anom
        else:
            return anom + (2*self.ae) * anom.sin()

    def to_mean_anomaly(self, lon, iters=4):
        """Returns the mean anomaly given a longitude in this frame, accurate
        to first order in eccentricity. Iteration is performed using Newton's
        method to ensure that this function is an exact inverse of
        from_mean_anomaly().
        """

        lon = Scalar.as_standard(lon)
        if not self.has_eccentricity: return lon

        # Solve lon = x + 2ae sin(x)
        #
        # Let
        #   y(x) = x + 2ae sin(x) - lon
        #
        #   dy/dx = 1 + 2ae cos(x)
        #
        # For x[n] as a guess at n,
        #   x[n+1] = x[n] - y(x[n]) / dy/dx

        x = lon.copy()
        ae_x2 = 2 * self.ae

        max_abs_dx = 4. # It's bigger than pi

        for iter in range(iters):
            dx = (lon - ae_x2 * x.sin()) / (1 + ae_x2 * x.cos())
            print iter, abs(dx).max()
            if dx == 0.: break

            x += dx

        return x

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_OrbitPlane(unittest.TestCase):

    def runTest(self):

        # elements = (a, lon, n)

        # Circular orbit, no derivatives, forward
        elements = (1, 0, 1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.as_coords(pos, axes=3, derivs=False)

        r_true = Scalar([1,2,1,1])
        l_true = Scalar([0, 0, np.pi, np.pi/2])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true) < 1.e-12)
        self.assertTrue(abs(l - l_true) < 1.e-12)
        self.assertTrue(abs(z - z_true) < 1.e-12)

        # Circular orbit, no derivatives, reverse
        pos2 = orbit.as_vector3(r, l, z, derivs=False)

        self.assertTrue(abs(pos - pos2) < 1.e-10)

        # Circular orbit, with derivatives, forward
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        eps = 1.e-6
        delta = 1.e-4

        for step in ([eps,0,0], [0,eps,0], [0,0,eps]):
            dpos = Vector3(step)
            (r,l,z) = orbit.as_coords(pos + dpos, axes=3, derivs=True)

            r_test = r + (r.d_dpos * dpos.as_column()).as_scalar()
            l_test = l + (l.d_dpos * dpos.as_column()).as_scalar()
            z_test = z + (z.d_dpos * dpos.as_column()).as_scalar()

            self.assertTrue(abs(r - r_test) < delta)
            self.assertTrue(abs(l - l_test) < delta)
            self.assertTrue(abs(z - z_test) < delta)

        # Circular orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.as_coords(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.as_vector3(r, l, z, derivs=True)

        pos1 = orbit.as_vector3(r + eps, l, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l + eps, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l, z + eps, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        # elements = (a, lon, n, e, peri, prec)

        # Eccentric orbit, no derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000", "TEST")
        eps = 1.e-6
        delta = 1.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        event = Event(0., pos, Vector3((0,0,0)),
                      orbit.origin_id, orbit.frame_id)
        (r,l,z) = orbit.event_as_coords(event, derivs=False)

        r_true = Scalar([1. + ae, 2. + ae, 1 - ae, np.sqrt(1. + ae**2)])
        l_true = Scalar([2*np.pi, 2*np.pi, np.pi, np.arctan2(1,ae)])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true) < delta)
        self.assertTrue(abs(l - l_true) < delta)
        self.assertTrue(abs(z - z_true) < delta)

        # Eccentric orbit, no derivatives, reverse
        event2 = orbit.as_event(event.time, (r,l,z))
        self.assertTrue(abs(pos - event.pos) < 1.e-10)
        self.assertTrue(abs(event.vel) < 1.e-10)

        # Eccentric orbit, with derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 3.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., pos, vel, orbit.origin_id, orbit.frame_id)
            (r,l,z) = orbit.event_as_coords(event, derivs=True)

            event = Event(eps, pos + vel*eps, vel,
                          orbit.origin_id, orbit.frame_id)
            (r1,l1,z1) = orbit.event_as_coords(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = (l1 - l) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).unmasked() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).unmasked() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).unmasked() < delta)

        # Eccentric orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.as_coords(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.as_vector3(r, l, z, derivs=True)

        pos1 = orbit.as_vector3(r + eps, l, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l + eps, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l, z + eps, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        # elements = (a, lon, n, e, peri, prec, i, node, regr)

        # Inclined orbit, no eccentricity, no derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -np.pi/2
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])
        event = Event(0., pos, Vector3((0,0,0)),
                      orbit.origin_id, orbit.frame_id)
        (r,l,z) = orbit.event_as_coords(event, derivs=False)

        r_true = Scalar([cosi, 2*cosi, cosi, np.sqrt(1 + (dz*sini)**2)])
        l_true = Scalar([2*np.pi, 2*np.pi, np.pi, np.arctan2(1,dz*sini)])
        z_true = Scalar([-sini, -2*sini, sini, dz*cosi])

        self.assertTrue(abs(r - r_true) < delta)
        self.assertTrue(abs(l - l_true) < delta)
        self.assertTrue(abs(z - z_true) < delta)

        # Inclined orbit, no derivatives, reverse
        event2 = orbit.as_event(event.time, (r,l,z))
        self.assertTrue(abs(pos - event.pos) < 1.e-10)
        self.assertTrue(abs(event.vel) < 1.e-10)

        # Inclined orbit, with derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -np.pi/2
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, "SSB", "J2000")
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., pos, vel, orbit.origin_id, orbit.frame_id)
            (r,l,z) = orbit.event_as_coords(event, derivs=True)

            event = Event(eps, pos + vel*eps, vel,
                          orbit.origin_id, orbit.frame_id)
            (r1,l1,z1) = orbit.event_as_coords(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = ((l1 - l + np.pi) % (2*np.pi) - np.pi) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).unmasked() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).unmasked() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).unmasked() < delta)

        # Inclined orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.as_coords(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        pos0 = orbit.as_vector3(r, l, z, derivs=True)

        pos1 = orbit.as_vector3(r + eps, l, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(0)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l + eps, z, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(1)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

        pos1 = orbit.as_vector3(r, l, z + eps, derivs=False)
        pos1_test = pos0 + (eps * pos0.d_dcoord.as_row(2)).as_vector3()
        self.assertTrue(abs(pos1_test - pos1) < delta)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
