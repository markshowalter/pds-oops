################################################################################
# oops/surface_/ellipsoid.py: Ellipsoid subclass of class Surface
################################################################################

from __future__ import print_function

import numpy as np
from polymath import *

from oops.surface_.surface import Surface
from oops.path_.path       import Path
from oops.frame_.frame     import Frame
from oops.config           import SURFACE_PHOTONS, LOGGING
from oops.constants        import *

class Ellipsoid(Surface):
    """Ellipsoid defines a ellipsoidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the ellipsoid is
    oriented along the Z-axis of the frame and the long radius is along the
    X-axis.

    The coordinates defining the surface grid are (longitude, latitude).
    Both are based on the assumption that a spherical body has been
    "squashed" along the Y- and Z-axes. The latitudes and longitudes defined in
    this manner are neither planetocentric nor planetographic; functions are
    provided to perform conversions to either choice. Longitudes are measured
    in a right-handed manner, increasing toward the east; values range from 0 to
    2*pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, note
    that with this definition, the gradient of the elevation value is not
    exactly normal to the surface.
    """

    COORDINATE_TYPE = "spherical"
    IS_VIRTUAL = False

    DEBUG = False       # True for convergence testing in intercept_normal_to()

    PACKRAT_ARGS = ['origin', 'frame', 'radii', 'exclusion']

    def __init__(self, origin, frame, radii, exclusion=0.95):
        """Constructor for an Ellipsoid object.

        Input:
            origin      the Path object or ID defining the center of the
                        ellipsoid.
            frame       the Frame object or ID defining the coordinate frame in
                        which the ellipsoid is fixed, with the shortest radius
                        of the ellipsoid along the Z-axis and the longest radius
                        along the X-axis.
            radii       a tuple (a,b,c) containing the radii from longest to
                        shortest, in km.
            exclusion   the fraction of the polar radius within which
                        calculations of intercept_normal_to() are suppressed.
                        Values of less than 0.9 are not recommended because
                        the problem becomes numerically unstable.
        """

        self.origin = Path.as_waypoint(origin)
        self.frame  = Frame.as_wayframe(frame)

        self.radii  = np.asfarray(radii)
        self.radii_sq = self.radii**2
        self.req    = self.radii[0]
        self.req_sq = self.req**2
        self.rpol   = self.radii[2]

        self.squash_y   = self.radii[1] / self.radii[0]
        self.unsquash_y = self.radii[0] / self.radii[1]

        self.squash_z   = self.radii[2] / self.radii[0]
        self.unsquash_z = self.radii[0] / self.radii[2]

        self.squash    = Vector3((1., self.squash_y, self.squash_z))
        self.squash_sq = self.squash.element_mul(self.squash)
        self.unsquash  = Vector3((1., 1./self.squash_y, 1./self.squash_z))
        self.unsquash_sq = self.unsquash.element_mul(self.unsquash)

        self.unsquash_sq_2d = Matrix(([1,0,0],
                                      [0,self.unsquash_y**2,0],
                                      [0,0,self.unsquash_z**2]))

        # This is the exclusion zone radius, within which calculations of
        # intercept_normal_to() are automatically masked due to the ill-defined
        # geometry.

        self.exclusion = exclusion * self.rpol

    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        pos = Vector3.as_vector3(pos, derivs)

        unsquashed = Vector3.as_vector3(pos).element_mul(self.unsquash)

        r = unsquashed.norm()
        (x,y,z) = unsquashed.to_scalars()
        lat = (z/r).arcsin()
#        lon_unsquashed = y.arctan2(x) % TWOPI
        lon = y.arctan2(x) % TWOPI

        # Convert longitude from "unsquashed" to planetocentric
#         lon = (lon_unsquashed.tan() * self.squash_y).arctan()   # -pi/2 to pi/2
#         lon += np.pi * ((lon_unsquashed.vals + np.pi/2) // np.pi)

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.req)

    def vector3_from_coords(self, coords, obs=None, time=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        # Convert to Scalars
        lon = Scalar.as_scalar(coords[0], derivs)
        lat = Scalar.as_scalar(coords[1], derivs)

        # Convert longitude from planetocentric to "unsquashed"
#         lon_unsquashed = (lon.tan() * self.unsquash_y).arctan()
#         lon_unsquashed = lon_unsquashed + PI * ((lon.vals + HALFPI) // PI)

        if len(coords) == 2:
            r = Scalar(self.req)
        else:
            r = Scalar(coords[2], derivs) + self.req

        r_coslat = r * lat.cos()
#         x = r_coslat * lon_unsquashed.cos()
#         y = r_coslat * lon_unsquashed.sin() * self.squash_y
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin() * self.squash_y
        z = r * lat.sin() * self.squash_z

        return Vector3.from_scalars(x,y,z)

    def intercept(self, obs, los, time=None, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       optional initial guess at the coefficient t such that:
                            intercept = obs + t * los

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        # Convert to Vector3 and un-squash
        obs = Vector3.as_vector3(obs, derivs)
        los = Vector3.as_vector3(los, derivs)

        obs_unsquashed = obs.element_mul(self.unsquash)
        los_unsquashed = los.element_mul(self.unsquash)

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
#         t = (bsign_sqrtd_div2 - b_div2) / a
        t = -c / (b_div2 + bsign_sqrtd_div2)     # more accurate!

        pos = obs + t*los
#         pos = self._apply_exclusion(pos)  ### why was this here?? -MRS
        return (pos, t)

    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        pos = Vector3.as_vector3(pos, derivs)
        return pos.element_mul(self.unsquash_sq)

    def intercept_with_normal(self, normal, time=None, derivs=False,
                                    guess=None):
        """Intercept point where the normal vector parallels the given vector.

        Input:
            normal      a Vector3 of normal vectors.
            derivs      True to propagate derivatives in the normal vector into
            time        a Scalar time at which to evaluate the surface; ignored.
                        the returned intercepts.
            guess       optional initial guess a coefficient array p such that:
                            pos = intercept + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a Vector3 of surface intercept points, in km. Where no
                        solution exists, the returned Vector3 will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            pos = intercept + p * normal(intercept).
        """

        normal = Vector3.as_vector3(normal, derivs)
        return normal.element_mul(self.squash).unit().element_mul(self.radii)

    def intercept_normal_to(self, pos, time=None, derivs=False, guess=None):
        """Intercept point whose normal vector passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface.
            time        a Scalar time at which to evaluateevaluate the surface; ignored.
            derivs      True to propagate derivatives in pos into the returned
                        intercepts.
            guess       optional initial guess a coefficient array p such that:
                            intercept = pos + p * normal(intercept);
                        use guess=False for the converged value of p to be
                        returned even if an initial guess was not provided.

        Return:         a vector3 of surface intercept points, in km. Where no
                        solution exists, the returned vector will be masked.

                        If guess is not None, then it instead returns a tuple
                        (intercepts, p), where p is the converged solution such
                        that 
                            intercept = pos + p * normal(intercept).
        """

        pos_with_derivs = Vector3.as_vector3(pos, derivs)
        pos_with_derivs = self._apply_exclusion(pos_with_derivs)

        pos = pos_with_derivs.wod

        # The intercept point satisfies:
        #   cept + p * perp(cept) = pos
        # where
        #   perp(cept) = cept * unsquash_sq
        #
        # Let B1 == unsquash_y
        # Let B2 == unsquash_y**2
        # Let C1 == unsquash_z
        # Let C2 == unsquash_z**2
        #
        # Expanding,
        #   pos_x = (1 + p) cept_x
        #   pos_y = (1 + B2 p) cept_y
        #   pos_z = (1 + C2 p) cept_z
        #
        # The intercept point must also satisfy
        #   |cept * unsquash| = req
        # or
        #   cept_x**2 + B2 cept_y**2 + C2 cept_z**2 = req_sq
        #
        # Solve:
        #   cept_x**2 + B2 cept_y**2 + C2 cept_z**2 - req_sq = 0
        #
        #   pos_x**2 / (1 + p)**2 +
        #   pos_y**2 / (1 + B2 p)**2 * B2 +
        #   pos_z**2 / (1 + C2 p)**2 * C2 - req_sq = 0
        #
        # f(p) = the above expression
        #
        # df/dt = -2 pos_x**2 / (1 + p)**3 +
        #       = -2 pos_y**2 / (1 + B2 p)**3 B2**2 +
        #       = -2 pos_z**2 / (1 + C2 p)**3 C2**2
        #
        # Let denom = [1 + p, 1 + B2 p, 1 + C2 p]
        # Let unsquash = [1, B1, C1]
        # Let unsquash_sq = [1, B2, C2]
        #
        # Let scale = [1, B1, C1] / [1 + p, 1 + B2 p, 1 + C2 p]
        #
        # f(p) = (pos * scale) dot (pos * scale) - req_sq
        #
        # df/dt = -2 (pos * scale) dot (pos * unsquash/denom)

        # Make an initial guess at p, if necessary
        if guess in (None, False):
            cept = pos.element_mul(self.unsquash).unit().element_mul(self.radii)
            p = (pos - cept).norm() / self.normal(cept).norm()
        else:
            p = guess.copy(readonly=False, recursive=False)

        # Terminate when accuracy stops improving by at least a factor of 2
        max_dp = 1.e99
        for iter in range(SURFACE_PHOTONS.max_iterations):
            denom = Vector3.ONES + p * self.unsquash_sq

            pos_scale = pos.element_mul(self.unsquash.element_div(denom))
            f = pos_scale.dot(pos_scale) - self.req_sq

            ratio = self.unsquash_sq.element_div(denom)
            df_dp_div_neg2 = pos_scale.dot(pos_scale.element_mul(ratio))

            dt = -0.5 * f/df_dp_div_neg2
            p -= dt

            prev_max_dp = max_dp
            max_dp = abs(dt).max()

            if LOGGING.surface_iterations or Ellipsoid.DEBUG:
                print(LOGGING.prefix, "Surface.spheroid.intercept_normal_to",
                                      iter, max_dp)

            if (np.all(Scalar.as_scalar(max_dp).mask) or
                max_dp <= SURFACE_PHOTONS.dlt_precision or
                max_dp >= prev_max_dp * 0.5): break

        denom = Vector3.ONES + p * self.unsquash_sq
        cept = pos.element_div(denom)

        if derivs:
            # First, we need dt/dpos
            #
            # pos_x**2 / (1 + p)**2 +
            # pos_y**2 / (1 + B2 p)**2 * B2 +
            # pos_z**2 / (1 + C2 p)**2 * C2 - req_sq = 0
            #
            # Derive dt/dpos_x...
            #
            # 2 pos_x / (1+p)**2
            #   + pos_x**2 * (-2)/(1+p)**3 dt/dpos_x
            #   + pos_y**2 * (-2)/(1+B2 p)**3 B2**2 dt/dpos_x
            #   + pos_z**2 * (-2)/(1+C2 p)**3 C2**2 dt/dpos_x = 0
            #
            # dt/dpos_x [pos_x**2 / (1+p)**3 +
            #            pos_y**2 * B2**2/(1 + B2 p)**3] +
            #            pos_z**2 * C2**2/(1 + C2 p)**3] = pos_x / (1+p)**2
            #
            # Derive dt/dpos_y...
            #
            # pos_x**2 * (-2)/(1+p)**3 dt/dpos_y
            #   + 2 pos_y * (1 + B2 p)**2 * B2 +
            #   + pos_y**2 * (-2)/(1 + B2 p)**3 B2**2 dt/dpos_y
            #   + pos_z**2 * (-2)/(1 + C2 p)**3 C2**2 dt/dpos_y = 0
            #
            # dt/dpos_y [pos_x**2 / (1+p)**3 +
            #            pos_y**2 * B2**2/(1 + B2 p)**3] +
            #            pos_z**2 * C2**2/(1 + C2 p)**3] =
            #                                       pos_y B2/(1 + B2 p)**2
            #
            # Derive dt/dpos_z using similar math...
            #
            # dt/dpos_z [pos_x**2 / (1+p)**3 +
            #            pos_y**2 * B2**2/(1 + B2 p)**3] +
            #            pos_z**2 * C2**2/(1 + C2 p)**3] =
            #                                       pos_z C2/(1 + C2 p)**2
            #
            # Let denom = [1 + p, 1 + B2 p, 1 + C2 p]
            # Let unsquash_sq = [1, B2, C2]
            #
            # Let denom1 = [pos_x**2 / (1+p)**3 +
            #               pos_y**2 * B2**2 / (1 + B2 p)**3]
            #               pos_z**2 * C2**2 / (1 + C2 p)**3]
            # in the expressions for dt/dpos. Note that this is identical to
            # df_dp_div_neg2 in the expressions above.
            #
            # dt/dpos_x * denom1 = pos_x / (1+p)**2
            # dt/dpos_y * denom1 = pos_y * B2 / (1 + B2 p)**2
            # dt/dpos_z * denom1 = pos_z * C2 / (1 + C2 p)**2

            stretch = self.unsquash_sq.element_div(denom.element_mul(denom))
            dp_dpos = pos.element_mul(stretch) / df_dp_div_neg2
            dp_dpos = dp_dpos.swap_items([Scalar])

            # Now we can proceed with dcept/dpos
            #
            # cept + perp(cept) * p = pos
            #
            # dcept/dpos + perp(cept) dt/dpos + p dperp/dcept dcept/dpos = I
            #
            # (I + p dperp/dcept) dcept/dpos = I - perp(cept) dt/dpos
            #
            # dcept/dpos = (I + p dperp/dcept)**(-1) * (I - perp dt/dpos)

            cept_with_derivs = cept.copy(readonly=False, recursive=False)
            cept_with_derivs.insert_deriv('cept', Vector3.IDENTITY)
            perp = self.normal(cept_with_derivs, derivs=True)

            mat = Matrix.as_matrix(Vector3.IDENTITY + p * perp.d_dcept)
            dcept_dpos = mat.reciprocal() * (Vector3.IDENTITY -
                                             perp.wod * dp_dpos)

            for (key,deriv) in pos_with_derivs.derivs.items():
                cept.insert_deriv(key, dcept_dpos.chain(deriv), override=True)
                p.insert_deriv(key, dp_dpos.chain(deriv), override=True)

        if guess is None:
            return cept
        else:
            return (cept, p)

    def _apply_exclusion(self, pos):
        """This internal method is used by intercept_normal_to() to exclude any
        positions that fall too close to the center of the surface. The math
        is poorly-behaved in this region.

        (1) It sets the mask on any of these points to True.
        (2) It sets the magnitude of any of these points to the edge of the
            exclusion zone, in order to avoid runtime errors in the math
            libraries.
        """

        replacement = Vector3((0.,0.,self.exclusion/2.))

        # Replace NaNs and zeros
        nans = np.any(np.isnan(pos.vals), axis=-1)
        zeros = np.all(pos.vals == 0, axis=-1)
        pos = pos.mask_where(nans | zeros, replace=replacement)

        # Define the exclusion zone as a NumPy boolean array
        pos_unsquashed = pos.element_mul(self.unsquash)
        pos_sq_vals = pos_unsquashed.dot(pos_unsquashed).vals
        mask = (pos_sq_vals <= self.exclusion**2)

        if not np.any(mask): return pos

        # Scale all masked vectors out to the exclusion radius
        if mask is True:
            return pos * self.exclusion / np.sqrt(pos_sq_vals)
        else:
            factor = np.ones(pos.shape)
            factor[mask] = self.exclusion / np.sqrt(pos_sq_vals[mask])
            return pos * Scalar(factor, mask)

    ############################################################################
    # Longitude conversions
    ############################################################################

    def lon_to_centric(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetocentric.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetocentric longitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.squash_y).arctan2(lon.cos())

    def lon_from_centric(self, lon, derivs=False):
        """Convert planetocentric longitude to internal coordinates.

        Input:
            lon         planetocentric longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.unsquash_y).arctan2(lon.cos())

    def lon_to_graphic(self, lon, derivs=False):
        """Convert longitude in internal coordinates to planetographic.

        Input:
            lon         squashed longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          planetographic longitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.unsquash_y).arctan2(lon.cos())

    def lon_from_graphic(self, lon, derivs=False):
        """Convert planetographic longitude to internal coordinates.

        Input:
            lon         planetographic longitude in radians.
            derivs      True to include derivatives in returned result.

        Return          squashed longitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        return (lon.sin() * self.squash_y).arctan2(lon.cos())

    ############################################################################
    # Latitude conversions
    ############################################################################

    def lat_to_centric(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetocentric.
        """

        lon = Scalar.as_scalar(lon, derivs)
        lat = Scalar.as_scalar(lat, derivs)

        denom = (lon.cos()**2 + (lon.sin() * self.squash_y)**2).sqrt()

        return (lat.tan() * self.squash_z / denom).arctan()

    def lat_from_centric(self, lat, lon, derivs=False):
        """Convert planetocentric latitude to internal ellipsoid latitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        lat = Scalar.as_scalar(lat, derivs)

        factor = (lon.cos()**2 + (lon.sin(derivs) * self.squash_y)**2).sqrt()

        return (lat.tan() * self.unsquash_z * factor).arctan()

    def lat_to_graphic(self, lat, lon, derivs=False):
        """Convert latitude in internal ellipsoid coordinates to planetographic.
        """

        lon = Scalar.as_scalar(lon, derivs)
        lat = Scalar.as_scalar(lat, derivs)

        denom = (lon.cos()**2 + (lon.sin() * self.unsquash_y)**2).sqrt()

        return (lat.tan() * self.unsquash_z / denom).arctan()

    def lat_from_graphic(self, lat, lon, derivs=False):
        """Convert planetographic latitude to internal ellipsoid latitude.
        """

        lon = Scalar.as_scalar(lon, derivs)
        lat = Scalar.as_scalar(lat, derivs)

        factor = (lon.cos()**2 + (lon.sin() * self.unsquash_y)**2).sqrt()

        return (lat.tan() * self.squash_z * factor).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Ellipsoid(unittest.TestCase):

    def runTest(self):

        from oops.frame_.frame import Frame
        from oops.path_.path import Path

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.
        planet = Ellipsoid("SSB", "J2000", (REQ, RMID, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        pos = (2 * np.random.rand(NPTS,3) - 1.) * REQ   # range is -REQ to REQ

        (lon,lat,elev) = planet.coords_from_vector3(pos, axes=3)
        test = planet.vector3_from_coords((lon,lat,elev))
        self.assertTrue(abs(test - pos).max() < 1.e-8)

        # Make sure longitudes convert to planetocentric and back
        test_lon = np.arctan2(pos[...,1], pos[...,0])
        centric_lon = planet.lon_to_centric(lon)
        diffs = (centric_lon - test_lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        # Make sure latitudes convert to planetocentric and back
        test_lat = np.arcsin(pos[...,2] / np.sqrt(np.sum(pos**2, axis=-1)))
        centric_lat = planet.lat_to_centric(lat,lon)
        self.assertTrue(abs(centric_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_centric(centric_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Make sure longitudes convert to planetographic and back
        normals = planet.normal(pos)
        test_lon = np.arctan2(normals.vals[...,1], normals.vals[...,0])
        graphic_lon = planet.lon_to_graphic(lon)
        diffs = (graphic_lon - test_lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        test_lon2 = planet.lon_from_centric(centric_lon)
        diffs = (test_lon2 - lon + HALFPI) % PI - HALFPI
        self.assertTrue(abs(diffs).max() < 1.e-8)

        # Make sure latitudes convert to planetographic and back
        test_lat = np.arcsin(normals.vals[...,2] / normals.norm().vals)
        graphic_lat = planet.lat_to_graphic(lat,lon)
        self.assertTrue(abs(graphic_lat - test_lat).max() < 1.e-8)

        test_lat2 = planet.lat_from_graphic(graphic_lat, lon)
        self.assertTrue(abs(test_lat2 - lat).max() < 1.e-8)

        # Ellipsoid intercepts & normals
        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts).max() < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,1] *= REQ/RMID
        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm() - REQ).max() < 1.e-8)

        normals.vals[...,1] *= RMID/REQ
        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(abs(normals.unit() - pts.unit()).max() < 1.e-14)

        # Intercept derivatives

        # Lines of sight with grazing incidence can have large numerical errors,
        # but this is not to be considered an error in the analytic calculation.
        # As a unit test, we ignore the largest 3% of the errors, but require
        # that the rest of the errors be very small.

        obs = REQ * (np.random.rand(NPTS,3) + 1.)       # range is REQ to 2*REQ
        los = -np.random.rand(NPTS,3)                   # range is -1 to 0

        obs = Vector3(obs)
        los = Vector3(los).unit()
        obs.insert_deriv('obs', Vector3.IDENTITY)
        los.insert_deriv('los', Vector3.IDENTITY)

        eps = 1.
        frac = 0.97     # Ignore errors above this cutoff
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs + dobs[i], los, derivs=False)
            (cept2,t2) = planet.intercept(obs - dobs[i], los, derivs=False)

            dcept_dobs = (cept1 - cept2) / (2*eps)
            ref = Vector3(cept.d_dobs.vals[...,i], cept.d_dobs.mask)

            errors = abs(dcept_dobs - ref) / abs(ref)
            sorted = np.sort(errors.vals[errors.antimask])
                        # mask=True where the line of sight missed the surface
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dobs = (t1 - t2) / (2*eps)
            ref = t.d_dobs.vals[...,i]

            errors = abs(dt_dobs/ref - 1)
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        eps = 1.e-6
        frac = 0.97
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (cept,t) = planet.intercept(obs, los, derivs=True)
            (cept1,t1) = planet.intercept(obs, los + dlos[i], derivs=False)
            (cept2,t2) = planet.intercept(obs, los - dlos[i], derivs=False)

            dcept_dlos = (cept1 - cept2) / (2*eps)
            ref = Vector3(cept.d_dlos.vals[...,i], cept.d_dlos.mask)

            errors = abs(dcept_dlos - ref) / abs(ref)
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

            dt_dlos = (t1 - t2) / (2*eps)
            ref = t.d_dlos.vals[...,i]

            errors = abs(dt_dlos/ref - 1)
            sorted = np.sort(errors.vals[errors.antimask])
            selected_error = sorted[int(sorted.size * frac)]
            self.assertTrue(selected_error < 1.e-5)

        # Test normal()
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        perp = planet.normal(cept)
        test1 = cept.element_mul(planet.unsquash).unit()
        test2 = perp.element_mul(planet.squash).unit()

        self.assertTrue(abs(test1 - test2).max() < 1.e-12)

        eps = 1.e-7
        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon+eps,lat,0.))
        cept2 = planet.vector3_from_coords((lon-eps,lat,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - HALFPI).max() < 1.e-8)

        (lon,lat) = planet.coords_from_vector3(cept, axes=2)
        cept1 = planet.vector3_from_coords((lon,lat+eps,0.))
        cept2 = planet.vector3_from_coords((lon,lat-eps,0.))

        self.assertTrue(abs((cept2 - cept1).sep(perp) - HALFPI).max() < 1.e-8)

        # Test intercept_with_normal()
        vector = Vector3(np.random.random((100,3)))
        cept = planet.intercept_with_normal(vector)
        sep = vector.sep(planet.normal(cept))
        self.assertTrue(sep.max() < 1.e-14)

        # Test intercept_normal_to()
        pos = Vector3(np.random.random((100,3)) * 4.*REQ + REQ)
        cept = planet.intercept_normal_to(pos)
        sep = (pos - cept).sep(planet.normal(cept))
        self.assertTrue(sep.max() < 3.e-12)
        self.assertTrue(abs(cept.element_mul(planet.unsquash).norm() -
                        planet.req).max() < 1.e-6)

        # Test normal() derivative
        cept = Vector3(np.random.random((100,3))).unit().element_mul(planet.radii)
        cept.insert_deriv('pos', Vector3.IDENTITY, override=True)
        perp = planet.normal(cept, derivs=True)
        eps = 1.e-5
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            perp1 = planet.normal(cept + dpos[i])
            dperp_dpos = (perp1 - perp) / eps

            ref = Vector3(perp.d_dpos.vals[...,i,:], perp.d_dpos.mask)
            self.assertTrue(abs(dperp_dpos - ref).max() < 1.e-4)

        # Test intercept_normal_to() derivative
        pos = Vector3(np.random.random((3,3)) * 4.*REQ + REQ)
        pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
        (cept,t) = planet.intercept_normal_to(pos, derivs=True, guess=False)
        self.assertTrue(abs(cept.element_mul(planet.unsquash).norm() -
                        planet.req).max() < 1.e-6)

        eps = 1.
        dpos = ((eps,0,0), (0,eps,0), (0,0,eps))
        perp = planet.normal(cept)
        for i in range(3):
            (cept1,t1) = planet.intercept_normal_to(pos + dpos[i], derivs=False,
                                                    guess=t)
            (cept2,t2) = planet.intercept_normal_to(pos - dpos[i], derivs=False,
                                                    guess=t)
            dcept_dpos = (cept1 - cept2) / (2*eps)
            self.assertTrue(abs(dcept_dpos.sep(perp) - HALFPI).max() < 1.e-5)

            ref = Vector3(cept.d_dpos.vals[...,i], cept.d_dpos.mask)
            self.assertTrue(abs(dcept_dpos - ref).max() < 1.e-5)

            dt_dpos = (t1 - t2) / (2*eps)
            ref = t.d_dpos.vals[...,i]
            self.assertTrue(abs(dt_dpos/ref - 1).max() < 1.e-5)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
