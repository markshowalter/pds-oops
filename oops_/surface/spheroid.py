################################################################################
# oops_/surface/spheroid.py: Spheroid subclass of class Surface
#
# 2/15/12 Checked in (BSW)
# 2/17/12 Modified (MRS) - Inserted coordinate definitions; added use of trig
#   functions and sqrt() defined in Scalar class to enable cleaner algorithms.
#   Unit tests added.
# 3/4/12 MRS: cleaned up comments, added NotImplementedErrors for features still
#   TBD.
################################################################################

import numpy as np

from oops_.surface.surface_ import Surface
from oops_.array.all import *
import oops_.registry as registry

class Spheroid(Surface):
    """Spheroid defines a spheroidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the spheroid is
    oriented along the Z-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based
    on the assumption that a spherical body has been "squashed" along the
    Z-axis. The latitude defined in this manner is neither planetocentric nor
    planetographic; functions are provided to perform the conversion to either
    choice. Longitudes are measured in a right-handed manner, increasing toward
    the east. Values range from 0 to 2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, the
    direction of increasing elevation is not exactly normal to the surface.
    """

    UNIT_MATRIX = MatrixN([(1,0,0),(0,1,0),(0,0,1)])

    def __init__(self, origin, frame, radii):
        """Constructor for a Spheroid surface.

        Input:
            origin      the Path object or ID defining the center of the
                        spheroid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the spheroid is fixed, with the short axis along
                        the Z-coordinate.
            radii       a tuple (a,c), defining the long and short radii of the
                        spheroid.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.radii  = np.array((radii[0], radii[0], radii[1]))
        self.radii_sq = self.radii**2
        self.req    = radii[0]
        self.req_sq = self.req**2

        self.squash_z   = radii[1] / radii[0]
        self.unsquash_z = radii[0] / radii[1]

        self.squash    = Vector3((1., 1., self.squash_z))
        self.squash_sq = self.squash**2
        self.unsquash  = Vector3((1., 1., self.unsquash_z))
        self.unsquash_sq    = self.unsquash**2

    def to_coords(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate. The
                        coordinates are (longitude, latitude, elevation). Units
                        are radians and km; longitude ranges from 0 to 2*pi.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        unsquashed = Vector3.as_standard(pos) * self.unsquash

        r = unsquashed.norm()
        (x,y,z) = unsquashed.as_scalars()
        lat = (z/r).arcsin()
        lon = y.arctan2(x) % (2.*np.pi)

        if derivs is False: derivs = (False, False, False)
        if derivs is True: derivs = (True, True, True)

        if np.any(derivs):
            raise NotImplementedError("Spheroid.to_coords() " +
                                      " does not implement derivatives")

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.req)

    def from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                lon     longitude in radians.
                lat     latitude in radians
                elev    a rough measure of distance from the surface, in km;
            obs         position of the observer in the surface frame; ignored.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

                        If derivs is True, then pos is returned with subfields
                        "d_dobs" and "d_dcoords", where the former contains the
                        MatrixN of partial derivatives with respect to obs and
                        the latter is the MatrixN of partial derivatives with
                        respect to the coordinates. The MatrixN item shapes are
                        [3,3].
        """

        # Convert to Scalars in standard units
        lon = Scalar.as_standard(coords[0])
        lat = Scalar.as_standard(coords[1])

        if len(coords) == 2:
            r = Scalar(0.)
        else:
            r = Scalar.as_standard(coords[2]) + self.req

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin()
        z = r * lat.sin() * self.squash_z

        pos = Vector3.from_scalars(x,y,z)

        if derivs:
            raise NotImplementedError("Spheroid.from_coords() " +
                                      " does not implement derivatives")

        return pos

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

        Return:         a tuple (pos, t) where
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar such that:
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

        # Convert to standard units and un-squash
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_unsquashed = Vector3.as_standard(obs) * self.unsquash
        los_unsquashed = Vector3.as_standard(los) * self.unsquash

        # Solve for the intercept distance, masking lines of sight that miss

        # Use the quadratic formula...
        # The use of b.sign() below always selects the closer intercept point

        # a = los_unsquashed.dot(los_unsquashed)
        # b = los_unsquashed.dot(obs_unsquashed) * 2.
        # c = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        # d = b**2 - 4. * a * c
        #
        # t = (-b + b.sign() * d.sqrt()) / (2*a)
        # pos = obs + t*los

        # This is the same algorithm as is commented out above, but avoids a
        # few unnecessary math operations

        a      = los_unsquashed.dot(los_unsquashed)
        b_div2 = los_unsquashed.dot(obs_unsquashed)
        c      = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        d_div4 = b_div2**2 - a * c

        bsign_sqrtd_div2 = b_div2.sign() * d_div4.sqrt()
        t = (bsign_sqrtd_div2 - b_div2) / a
        pos = obs + t*los

        if derivs:
            # Using step-by-step differentiation of the equations above

            # da_dlos = 2 * los * self.unsquash_sq
            # db_dlos = 2 * obs * self.unsquash_sq
            # db_dobs = 2 * los * self.unsquash_sq
            # dc_dobs = 2 * obs * self.unsquash_sq

            da_dlos_div2 = los * self.unsquash_sq
            db_dlos_div2 = obs * self.unsquash_sq
            db_dobs_div2 = los * self.unsquash_sq
            dc_dobs_div2 = obs * self.unsquash_sq

            # dd_dlos = 2 * b * db_dlos - 4 * c * da_dlos
            # dd_dobs = 2 * b * db_dobs - 4 * a * dc_dobs

            dd_dlos_div8 = b_div2 * db_dlos_div2 - c * da_dlos_div2
            dd_dobs_div8 = b_div2 * db_dobs_div2 - a * dc_dobs_div2

            # dsqrt = d.sqrt()
            # d_dsqrt_dd = 0.5 / dsqrt
            # d_dsqrt_dlos = d_dsqrt_dd * dd_dlos
            # d_dsqrt_dobs = d_dsqrt_dd * dd_dobs

            # d[bsign_sqrtd]/d[x] = 1/2 / bsign_sqrtd * d[d]/d[x]
            #                     = 1/4 / bsign_sqrtd_div2 * d[d]/d[x]

            d_bsign_sqrtd_dlos_div2 = dd_dlos_div8 / bsign_sqrtd_div2
            d_bsign_sqrtd_dobs_div2 = dd_dobs_div8 / bsign_sqrtd_div2

            # inv2a = 0.5/a
            # d_inv2a_da = -2 * inv2a**2
            # 
            # dt_dlos = (inv2a * (b.sign()*d_dsqrt_dlos - db_dlos) +
            #           (b.sign()*dsqrt - b)*d_inv2a_da * da_dlos).as_vectorn()
            # dt_dobs = (inv2a * (b.sign()*d_dsqrt_dobs - db_dobs)).as_vectorn()
            # 
            # dpos_dobs = (los.as_column() * dt_dobs.as_row() +
            #              Spheroid.UNIT_MATRIX)
            # dpos_dlos = (los.as_column() * dt_dlos.as_row() +
            #              Spheroid.UNIT_MATRIX * t)

            dt_dlos = ((d_bsign_sqrtd_dlos_div2
                        - db_dlos_div2 - 2 * t * da_dlos_div2) / a).as_vectorn()
            dt_dobs = ((d_bsign_sqrtd_dobs_div2
                        - db_dobs_div2) / a).as_vectorn()

            dpos_dobs = (los.as_column() * dt_dobs.as_row() +
                         Spheroid.UNIT_MATRIX)
            dpos_dlos = (los.as_column() * dt_dlos.as_row() +
                         Spheroid.UNIT_MATRIX * t)

            los_norm = los.norm()
            pos.insert_subfield("d_dobs", dpos_dobs)
            pos.insert_subfield("d_dlos", dpos_dlos * los_norm)
            t.insert_subfield("d_dobs", dt_dobs.as_row())
            t.insert_subfield("d_dlos", dt_dlos.as_row() * los_norm)

        return (pos, t)

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

        perp = Vector3.as_standard(pos) * self.unsquash_sq

        if derivs:
            raise NotImplementedError("Spheroid.normal() " +
                                      "does not implement derivatives")

        return perp

    def intercept_with_normal(self, normal, derivs=False):
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

        result = Vector3.as_standard(normal) * self.squash_sq

        if derivs:
            raise NotImplementedError("Spheroid.intercept_with_normal() " +
                                      "does not implement derivatives")

        return result

    def intercept_normal_to(self, pos, derivs=False):
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

        def guess_intercept_normal_to(pos):
            sq_pos = pos * self.squash_sq
            cept = sq_pos.unit() * self.radii * self.squash_sq
            return cept

        def f(t, pos):
            """Compute F(t) = ( (x0 * a**2) / (t + a**2) )**2 +
                              ( (y0 * a**2) / (t + a**2) )**2 +
                              ( (z0 * c**2) / (t + c**2) )**2 - 1, where
                <x0,y0,z0> represents pos, and self.radii is <a,a,c>.
                
            Input:
                pos     a Vector3 of positions near the surface, with optional
                        units
                t       t in F(t).
                
            Return      Solution to F(t)
            """
            denom = np.array([t,]*3).transpose() + self.radii_sq
            v1 = Vector3(self.radii_sq * pos.vals)
            v = v1 / denom
            w1 = v**2
            # w = w1.vals.sum(axis=1) - 1.
            w = w1.vals.sum(axis=-1) - 1.
            return w

        def fprime(t, pos):
            """Compute F'(t) = (-2 * x0**2 * a**4) / (t + a**2)**3 +
                               (-2 * y0**2 * a**4) / (t + a**2)**3 +
                               (-2 * z0**2 * c**4) / (t + c**2)**3, where
                <x0,y0,z0> represents pos, and self.radii is <a,a,c>.
                
            Input:
                pos     a Vector3 of positions near the surface, with optional
                        units
                t       t in F(t).
                
            Return      Solution to F'(t)
            """
            denom = (np.array([t,]*3).transpose() + self.radii_sq)**3
            v = (-2. * pos**2 * self.radii**4) / denom
            # w = v.vals.sum(axis=1)
            w = v.vals.sum(axis=-1)
            return w

        def intercept_normal_close(pos, cept, norm, t):
            """Check if angle between the surface normal and the vector from the
            point on the surface to the point in question, pos, is very small.
            For some reason sep() was not working without creating unit vectors.
            
            Input:
                pos     a Vector3 of positions near the surface, with optional
                        units
                cept    intercept point on surface
                norm    surface normal
                t       multiple of unit surface normal to reach pos.
                
            Return:     Boolean whether close enough.
            """
                
            cept = pos - norm * t
            # test_vector = (pos - cept).unit()
            # sep = test_vector.sep(norm.unit())
            sep = (pos - cept).sep(norm)
            return sep < 1.e-9

        def newton_intercept_normal_to(pos, t):
            """Runs Newton numerical method until angle between normal vector
            and vector from surface intercept point and pos is close to zero.
            
            Input:
                pos     a Vector3 of positions near the surface, with optiona
                        units
                t       t such that F(t) = ( (x0 * a**2) / (t + a**2) )**2 +
                        ( (y0 * a**2) / (t + a**2) )**2 +
                        ( (z0 * c**2) / (t + c**2) )**2 - 1, where <x0,y0,z0>
                        represents pos, and self.radii is <a,a,c>.
            
            Return:     a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.
            """
            denom = np.array([t,]*3).transpose() + self.radii_sq
            numer = pos * self.radii_sq
            cept = numer / denom
            norm = self.normal(cept)
            if intercept_normal_close(pos, cept, norm, t):
                return cept
            else:
                f_of_t = self.f(t, pos)
                fprime_of_t = self.fprime(t, pos)
                t -= f_of_t / fprime_of_t
                return self.newton_intercept_normal_to(pos, t)

        cept = guess_intercept_normal_to(pos)
        norm = self.normal(cept).unit()
        pos_cept = pos - cept
        t = pos_cept.vals[...,0] / norm.vals[...,0]

        if derivs:
            raise NotImplementedError("Spheroid.intercept_normal_to() " +
                                      "does not implement derivatives")

        return newton_intercept_normal_to(pos, t)

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        return Vector3((0,0,0))

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetocentric equivalent.
        """

        return (lat.tan() * self.squash_z).arctan()

    def lat_to_graphic(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetographic equivalent.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_centric(self, lat):
        """Converts a latitude value given in planetocentric coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_graphic(self, lat):
        """Converts a latitude value given in planetographic coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.squash_z).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Spheroid(unittest.TestCase):

    def runTest(self):

        from oops_.frame.frame_ import Frame
        from oops_.path.path_ import Path

        REQ  = 60268.
        RPOL = 54364.
        planet = Spheroid("SSB", "J2000", (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        obs = (2 * np.random.rand(NPTS,3) - 1.) * REQ

        (lon,lat,elev) = planet.to_coords(obs,axes=3)
        test = planet.from_coords((lon,lat,elev))
        self.assertTrue(abs(test - obs) < 3.e-9)

        # Spheroid intercepts & normals
        obs[...,0] = np.abs(obs[...,0])
        obs[...,0] += REQ

        los = (2 * np.random.rand(NPTS,3) - 1.)
        los[...,0] = -np.abs(los[...,0])

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts) < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm()[~pts.mask] - REQ) < 1.e-8)

        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(abs(normals.unit() - pts.unit()) < 1.e-14)
        
        # Test intercept_with_normal()
        vector = Vector3(np.random.random((100,3)))
        intercept = planet.intercept_with_normal(vector)
        sep = vector.sep(planet.normal(intercept))
        self.assertTrue(sep < 1.e-14)

        # Test intercept_normal_to()
        obs = Vector3(np.random.random((10,3)) * 10.*REQ + REQ)
        intercept = planet.intercept_normal_to(obs)
        sep = (obs - intercept).sep(planet.normal(intercept))
        k = obs - intercept
        self.assertTrue(sep < 3.e-12)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
