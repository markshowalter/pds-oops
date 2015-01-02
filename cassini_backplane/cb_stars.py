###############################################################################
# cb_stars.py
#
# Routines related to stars.
#
# Exported routines:
#    star_list_for_obs
#    star_create_model
#    star_perform_photometry
#    star_make_good_bad_overlay
#    star_find_offset
###############################################################################

import cb_logging
import logging

import copy

import numpy as np
import numpy.ma as ma
import polymath
import scipy.ndimage.filters as filt

import oops
from psfmodel.gaussian import GaussianPSF
from imgdisp import *
from starcat import (UCAC4StarCatalog,
                     SCLASS_TO_B_MINUS_V, SCLASS_TO_SURFACE_TEMP)
from starcat.starcatalog import (Star, SCLASS_TO_SURFACE_TEMP, 
                                 SCLASS_TO_B_MINUS_V)

from cb_config import *
from cb_correlate import *
from cb_rings import *
from cb_util_flux import *
from cb_util_image import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__

DEBUG_STARS_FILTER_IMGDISP = False

STAR_CATALOG = UCAC4StarCatalog(STAR_CATALOG_ROOT)

STARS_DEFAULT_CONFIG = {
    # Minimum number of stars that much photometrically match for an offset
    # to be considered good.
    'min_stars': 3,

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # PSF size for modeling a star (must be odd).
    'psf_size': 9,
        
    # The default star class when none is available.
    'default_star_class': 'G0',
    
    # The minimum DN that is guaranteed to be visible in the image.
    'min_brightness_guaranteed_vis': 200.,

    # The minimum DN count for a star to be detectable. These values are pretty
    # aggressively dim - there's no guarantee a star with this brightness can
    # actually be seen.
    ('min_detectable_dn', 'NAC'): 20,
    ('min_detectable_dn', 'WAC'): 150,

    # The range of vmags to use when determining the dimmest star visible.
    'min_vmag': 5.,
    'max_vmag': 15.,
    'vmag_increment': 0.5,

    # The size of the box to analyze vs. the predicted integrated DN.
    # XXX These numbers are just made up, but seem to work.
    # More study should be done.
    'photometry_boxsize_1': (500,11),
    'photometry_boxsize_2': (100,9),
    'photometry_boxsize_default': 7,
    
    # How far a star has to be from a major body before it is no longer
    # considered to conflict.
    'star_body_conflict_margin': 3,
}
    
#===============================================================================
#
# FIND STARS IN THE FOV AND RETURN THEM.
# 
#===============================================================================

def _aberrate_star(obs, star):
    """Update the RA,DEC position of a star with stellar aberration."""
    # XXX There must be a better way to do this.
    x = 1e30 * np.cos(star.ra) * np.cos(star.dec)
    y = 1e30 * np.sin(star.ra) * np.cos(star.dec) 
    z = 1e30 * np.sin(star.dec)
    pos = polymath.Vector3((x,y,z))

    path = oops.path.LinearPath((pos, polymath.Vector3.ZERO), obs.midtime,
                                'SSB')  
                      
    event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                     polymath.Vector3.ZERO),
                       obs.path, obs.frame)
    _, event = path.photon_to_event(event)
    abb_ra, abb_dec = event.ra_and_dec(apparent=True)

    star.ra = abb_ra.vals
    star.dec = abb_dec.vals

def _compute_dimmest_visible_star_vmag(obs, star_config):
    """Compute the VMAG of the dimmest star likely visible."""
    min_dn = star_config[('min_detectable_dn', obs.detector)] # photons / pixel
    fake_star = Star()
    fake_star.temperature = SCLASS_TO_SURFACE_TEMP['G0']
    min_mag = star_config['min_vmag']
    max_mag = star_config['max_vmag']
    mag_increment = star_config['vmag_increment']
    for mag in np.arange(min_mag, max_mag+1e-6, mag_increment):
        fake_star.unique_number = 0
        fake_star.johnson_mag_v = mag
        fake_star.johnson_mag_b = mag+SCLASS_TO_B_MINUS_V['G0']
        dn = compute_dn_from_star(obs, fake_star)
        if dn < min_dn:
            return mag # This is conservative
    return mag

def _star_list_for_obs(obs, ra_min, ra_max, dec_min, dec_max,
                       mag_min, mag_max, extend_fov, star_config, **kwargs):
    """Return a list of stars with the given constraints.
    
    See star_list_for_obs for full details."""
    logger = logging.getLogger(_LOGGING_NAME+'._star_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = star_config[('min_detectable_dn', obs.detector)]
    
    # Get a list of all reasonable stars with the given magnitude range.
    
    orig_star_list = [x for x in 
              STAR_CATALOG.find_stars(allow_double=True,
                                      allow_galaxy=False,
                                      ra_min=ra_min, ra_max=ra_max,
                                      dec_min=dec_min, dec_max=dec_max,
                                      vmag_min=mag_min, vmag_max=mag_max,
                                      **kwargs)]

    # Fake the temperature if it's not known, and eliminate stars we just
    # don't want to deal with.

    discard_class = 0
    discard_dn = 0
        
    default_star_class = star_config['default_star_class']
    
    star_list = []
    for star in orig_star_list:
        star.conflicts = None
        star.temperature_faked = False
        if star.temperature is None:
            star.temperature_faked = True
            star.temperature = SCLASS_TO_SURFACE_TEMP[default_star_class]
            star.spectral_class = default_star_class
            star.johnson_mag_v = (star.vmag-
                          SCLASS_TO_B_MINUS_V[default_star_class]/2.)
            star.johnson_mag_b = (star.vmag-
                          SCLASS_TO_B_MINUS_V[default_star_class]/2.)
        star.dn = compute_dn_from_star(obs, star)
        if star.dn < min_dn:
            discard_dn += 1
            continue
        if star.spectral_class[0] == 'M':
            # M stars are too dim and too red to be seen
            discard_class += 1
            continue
        _aberrate_star(obs, star)
        star_list.append(star)

    # Eliminate stars that are not actually in the FOV, including some
    # margin beyond the edge
    
    ra_dec_list = [x.ra_dec_with_pm(obs.midtime) for x in star_list]
    ra_list = [x[0] for x in ra_dec_list]
    dec_list = [x[1] for x in ra_dec_list]
    
    uv = obs.uv_from_ra_and_dec(ra_list, dec_list)
    u_list, v_list = uv.to_scalars()
    u_list = u_list.vals
    v_list = v_list.vals

    new_star_list = []

    discard_uv = 0        
    for star, u, v in zip(star_list, u_list, v_list):
        if (u < -extend_fov[0] or u > obs.data.shape[1]+extend_fov[0]-1 or
            v < -extend_fov[1] or v > obs.data.shape[0]+extend_fov[1]-1):
            discard_uv += 1
            continue
        
        star.u = u
        star.v = v

        new_star_list.append(star)

    logger.debug(
        'Found %d stars, discarded because of CLASS %d, LOW DN %d, BAD UV %d',
        len(orig_star_list), discard_class, discard_dn, discard_uv)

    return new_star_list

def star_list_for_obs(obs, extend_fov=(0,0), star_config=None,
                      **kwargs):
    """Return a list of stars in the FOV of the obs.

    Inputs:
        obs                The observation.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to return stars (a star's U/V value will be negative
                           or greater than the FOV shape).
        star_config        Configuration parameters.
        **kwargs           Passed to find_stars to restrict the types of stars
                           returned.
                           
    Returns:
        star_list          The list of Star objects with additional attributes
                           for each Star:
                           
           .u and .v           The U,V coordinate including stellar aberration.
           .faked_temperature  A bool indicating if the temperature and
                               spectral class had to be faked.
           .dn                 The estimated integrated DN count given the
                               star's class, magnitude, and the filters being
                               used.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.star_list_for_obs')

    if star_config is None:
        star_config = STARS_DEFAULT_CONFIG
        
    max_stars = star_config['max_stars']
    psf_size = star_config['psf_size']
    
    margin = psf_size//2
    
    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs,
                                             extend_fov=extend_fov)

    # Try to be efficient by limiting the magnitudes searched so we don't
    # return a huge number of dimmer stars and then only need a few of them.
    magnitude_list = [0., 12., 13., 14., 15., 16., 17.]
    
    mag_vmax = _compute_dimmest_visible_star_vmag(obs, star_config)+1
    logger.debug('Max detectable VMAG %.4f', mag_vmax)
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        if mag_min > mag_vmax:
            break
        mag_max = min(mag_max, mag_vmax)
        
        star_list = _star_list_for_obs(obs,
                                       ra_min, ra_max, dec_min, dec_max,
                                       mag_min, mag_max,
                                       extend_fov, star_config, **kwargs)
        full_star_list += star_list
        
        logger.debug('Got %d stars, total %d', len(star_list),
                     len(full_star_list))
        
        if len(full_star_list) >= max_stars:
            break

    # Sort the list with the brightest stars first.
    full_star_list.sort(key=lambda x: x.dn, reverse=True)
                
    full_star_list = full_star_list[:max_stars]

    logger.debug('Returned star list:')
    for star in full_star_list:
        logger.debug('Star %9d U %8.3f V %8.3f DN %7.2f MAG %6.3f BMAG %6.3f '+
                     'VMAG %6.3f SCLASS %3s TEMP %6d',
                     star.unique_number, star.u, star.v, star.dn, star.vmag,
                     0 if star.johnson_mag_b is None else star.johnson_mag_b,
                     0 if star.johnson_mag_v is None else star.johnson_mag_v,
                     'XX' if star.spectral_class is None else
                             star.spectral_class,
                     0 if star.temperature is None else star.temperature)
        
    return full_star_list

#===============================================================================
#
# CREATE A MODEL OR OVERLAY FOR STARS IN THE FOV.
# 
#===============================================================================
    
def star_create_model(obs, star_list, offset_u=0., offset_v=0.,
                      extend_fov=(0,0), star_config=None):
    """Create a model containing nothing but stars.
    
    Individual stars are modeled using a Gaussian PSF with a sigma based on
    the ISS camera. If the complete PSF doesn't fit in the image, the star
    is ignored.
    
    Inputs:
        obs                The observation.
        star_list          The list of Stars.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        extend_fov         The amount to extend the model beyond the limits of
                           the obs FOV. The returned model will be the shape of
                           the obs FOV plus two times the extend value in each
                           dimension.
        star_config        Configuration parameters.
        
    Returns:
        model              The model.
    """
    if star_config is None:
        star_config = STARS_DEFAULT_CONFIG
        
    psf_size = star_config['psf_size']
    
    model = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                      obs.data.shape[1]+extend_fov[0]*2),
                     dtype=np.float32)
    
    gausspsf = GaussianPSF(sigma=ISS_PSF_SIGMA[obs.detector])
    
    vmag_list = [x.vmag for x in star_list]
    star_mag_min = min(vmag_list)
    star_mag_max = max(vmag_list)
    
    margin = psf_size//2
    
    for star in star_list:
        u_idx = star.u+offset_u+extend_fov[0]
        v_idx = star.v+offset_v+extend_fov[1]
        u_int = int(u_idx)
        v_int = int(v_idx)

        if (u_int < margin or u_int >= model.shape[1]-margin or
            v_int < margin or v_int >= model.shape[0]-margin):
            continue

        u_frac = u_idx-u_int
        v_frac = v_idx-v_int
        psf = gausspsf.eval_rect((psf_size,psf_size),
                                 offset=(v_frac,u_frac),
                                 scale=star.dn)
        model[v_int-margin:v_int+margin+1, u_int-margin:u_int+margin+1] += psf
        
    return model

def star_make_good_bad_overlay(obs, star_list, offset_u, offset_v,
                               extend_fov=(0,0),
                               overlay_box_width=None,
                               overlay_box_thickness=None,
                               star_config=None):
    """Create an overlay with high and low confidence stars marked.
    
    Inputs:
        obs                The observation.
        star_list          The list of Star objects.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        extend_fov         The amount to extend the overlay beyond the limits
                           of the obs FOV. The returned model will be the shape
                           of the obs FOV plus two times the extend value in
                           each dimension.
        overlay_box_width  If None, draw a circle of radius 3.
                           If 0, draw a box of the size of the photometry
                                 measurement.
                           Otherwise, draw a box of the given size.
        overlay_box_thickness  If a box is drawn, this is the thickness of the
                               box sides.
        star_config        Configuration parameters.
                           
    Returns:
        overlay            The overlay.
        
        Star excluded by brightness or conflict: red
        Star bad photometry: blue
        Star good photometry: green
    """
    if star_config is None:
        star_config = STARS_DEFAULT_CONFIG
        
    overlay = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                        obs.data.shape[1]+extend_fov[0]*2, 3),
                       dtype=np.uint8)
    
    for star in star_list:
        # Should NOT be rounded for plotting, since all of coord
        # X to X+0.9999 is the same pixel
        u_idx = int(star.u+offset_u+extend_fov[0])
        v_idx = int(star.v+offset_v+extend_fov[1])
        
        if (not star.is_bright_enough or not star.is_dim_enough or
            star.conflicts):
            color = (255,0,0)
        else:
            if star.photometry_confidence > star_config['min_confidence']:
                color = (0,255,0)
            else:
                color = (0,0,255)
        if overlay_box_width is not None:
            width = overlay_box_width
            if width == 0:
                width = star.photometry_box_size // 2
            draw_rect(overlay, u_idx, v_idx, 
                      width, width,
                      color, overlay_box_thickness)
        else:
            draw_circle(overlay, u_idx, v_idx, 1, color, 3)

    return overlay

#===============================================================================
#
# PERFORM PHOTOMETRY.
# 
#===============================================================================

def _trust_star_dn(obs):
    return obs.filter1 == 'CL1' and obs.filter2 == 'CL2'

def _star_perform_photometry(obs, calib_data, star, offset_u, offset_v,
                             extend_fov, star_config):
    """Perform photometry on a single star.
    
    See star_perform_photometry for full details.
    """
    # calib_data is calibrated in DN and extended
    u = int(np.round(star.u)) + offset_u + extend_fov[0]
    v = int(np.round(star.v)) + offset_v + extend_fov[1]
    
    if star.dn > star_config['photometry_boxsize_1'][0]:
        boxsize = star_config['photometry_boxsize_1'][1]
    elif star.dn > star_config['photometry_boxsize_2'][0]:
        boxsize = star_config['photometry_boxsize_2'][1]
    else:
        boxsize = star_config['photometry_boxsize_default']
    
    star.photometry_box_width = boxsize
    
    box_halfsize = boxsize // 2

    # Don't process stars that are off the edge of the real (not extended)
    # data.
    if (u-extend_fov[0] < box_halfsize or
        u-extend_fov[0] > calib_data.shape[1]-2*extend_fov[0]-box_halfsize-1 or
        v-extend_fov[1] < box_halfsize or
        v-extend_fov[1] > calib_data.shape[0]-2*extend_fov[1]-box_halfsize-1):
        return None
        
    subimage = calib_data[v-box_halfsize:v+box_halfsize+1,
                          u-box_halfsize:u+box_halfsize+1]
    subimage = subimage.view(ma.MaskedArray)
    subimage[1:-1, 1:-1] = ma.masked # Mask out the center
    
    bkgnd = ma.mean(subimage)
    bkgnd_std = ma.std(subimage)
    
    subimage.mask = ~subimage.mask # Mask out the edge
    integrated_dn = np.sum(subimage-bkgnd)
    integrated_std = np.std(subimage-bkgnd)

    return integrated_dn, bkgnd, integrated_std, bkgnd_std
    
def star_perform_photometry(obs, calib_data, star_list, offset_u=0, offset_v=0,
                            extend_fov=(0,0), star_config=None):
    """Perform photometry on a list of stars.
    
    Inputs:
        obs                The observation.
        calib_data         obs.data calibrated as DN.
        star_list          The list of Star objects.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to perform photometry.
        star_config        Configuration parameters.
    
    Returns:
        good_stars         The number of good stars.
                           (star.photometry_confidence > STARS_MIN_CONFIDENCE)
        
        Each Star is populated with:
        
            .integrated_dn            The actual integrated dn measured.
            .photometry_confidence    The confidence in the result. Currently
                                      adds 0.5 for a non-noisy background and
                                      adds 0.5 for the DN within range. 
    """
    logger = logging.getLogger(_LOGGING_NAME+'.star_perform_photometry')

    if star_config is None:
        star_config = STARS_DEFAULT_CONFIG
        
    image = obs.data
    min_dn = star_config[('min_detectable_dn', obs.detector)]
    
    for star in star_list:
        u = int(np.round(star.u)) + offset_u
        v = int(np.round(star.v)) + offset_v
        if star.conflicts:
            # Stars that conflict with bodies are ignored
            star.integrated_dn = 0.
            star.photometry_confidence = 0.
            logger.debug('Star %9d %2s UV %4d %4d IGNORED %s',
                         star.unique_number, star.spectral_class,
                         u, v, star.conflicts)
            continue
        ret = _star_perform_photometry(obs, calib_data, star,
                                       offset_u, offset_v, extend_fov,
                                       star_config)
        if ret is None:
            integrated_dn = 0.
            confidence = 0.
        else:
            integrated_dn, bkgnd, integrated_std, bkgnd_std = ret
            if integrated_dn < 0:
                confidence = 0.    
            elif star.temperature_faked or not _trust_star_dn(obs):
                # Really the only thing we can do here is see if we detected
                # something at all, because we can't trust the photometry
                confidence = ((integrated_dn >= min_dn)*0.5 +
                              (integrated_std >= bkgnd_std*2)*0.5)
            else:  
                confidence = ((star.dn/3 < integrated_dn < star.dn*3)*0.5 +
                              (integrated_std >= bkgnd_std*1.5)*0.5)
                
        star.integrated_dn = integrated_dn
        star.photometry_confidence = confidence
        
        logger.debug(
            'Star %9d %2s UV %4d %4d PRED %7.2f MEAS %7.2f CONF %4.2f',
            star.unique_number, star.spectral_class,
            u, v, star.dn, star.integrated_dn, star.photometry_confidence)

    good_stars = 0
    for star in star_list:
        if star.photometry_confidence >= star_config['min_confidence']:
            good_stars += 1

    return good_stars

#===============================================================================
# 
# FIND THE IMAGE OFFSET BASED ON STARS.
#
#===============================================================================

def _star_mark_conflicts(obs, star, offset_u, offset_v, margin,
                         star_config):
    """Check if a star conflicts with known bodies or rings.
    
    Sets star.conflicts to a string describing why the Star conflicted.
    
    Returns True if the star conflicted, False if the star didn't.
    """
    logger = logging.getLogger(_LOGGING_NAME+'._star_mark_conflicts')

    # Check for off the edge
    if (not (margin < star.u+offset_u < obs.data.shape[1]-margin) or
        not (margin < star.v+offset_v < obs.data.shape[0]-margin)):
        logger.debug('Star %9d U %8.3f V %8.3f is off the edge',
                     star.unique_number, star.u, star.v)
        star.conflicts = 'EDGE'
        return True

    # Cache the body inventory
    if obs.star_body_list is None:
        obs.star_body_list = obs.inventory(LARGE_BODY_LIST)
        
    if len(obs.star_body_list):
        # Create a Meshgrid for the area around the star.
        # Give 3 pixels of slop on each side - we don't want a star to
        # even be close to a large object.
        star_slop = star_config['star_body_conflict_margin']
        meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(star.u+offset_u-star_slop,
                                                 star.v+offset_v-star_slop),
                                         limit =(star.u+offset_u+star_slop,
                                                 star.v+offset_v+star_slop))
        backplane = oops.Backplane(obs, meshgrid)
    
        # Check for planet and moons
        for body_name in obs.star_body_list:
            intercepted = backplane.where_intercepted(body_name)
            if np.any(intercepted):
                logger.debug('Star %9d U %8.3f V %8.3f conflicts with %s',
                             star.unique_number, star.u, star.v, body_name)
                star.conflicts = 'BODY:' + body_name
                return True

    # Check for rings
    ring_radius = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    ring_longitude = (obs.ext_bp.ring_longitude('saturn:ring').vals.
                      astype('float'))
    rad = ring_radius[star.v+obs.extend_fov[1], star.u+obs.extend_fov[0]]
    long = ring_longitude[star.v+obs.extend_fov[1], star.u+obs.extend_fov[0]]

    # XXX We might want to improve this to support the known position of the
    # F ring core.
    # C to A rings and F ring
    if ((oops.SATURN_C_RING[0] <= rad <= oops.SATURN_A_RING[1]) or
        (139890 <= rad <= 140550)): # F ring
        logger.debug('Star %9d U %8.3f V %8.3f conflicts with rings radius '
                     '%.1f',
                     star.unique_number, star.u, star.v, rad)
        star.conflicts = 'RINGS'
        return True
    
    return False

#    XXX THIS DOESN'T WORK because fov.sphere_falls_inside doesn't work with Subarray!
#
#            sub_fov = oops.fov.Subarray(obs.fov, (star.u, star.v), (7,7))
#            inv = obs.inventory(LARGE_BODY_LIST, fov=sub_fov)
#            if len(inv) > 0:
#                for body_name in inv:
#                    if star.use_for_model is False:
#                        logger.warn('YIKES Star %9d U %8.3f V %8.3f'
#                                    'DOUBLE CONFLICT',
#                                    star.unique_number, star.u, star.v)
#                    logger.debug('Star %9d U %8.3f V %8.3f conflicts with %s',
#                                 star.unique_number, star.u, star.v, body_name)
#                    star.use_for_model = False
#                    star.bad_reason = 'BODY:' + body_name
#                something_conflicted = True
        

def _optimize_offset_list(offset_list, tolerance=1):
    """Remove bad offsets.
    
    A bad offset is defined as an offset that makes a line with two other
    offsets in the list. We remove these because when 2-D correlation is
    performed on certain times of images, there is a 'line' of correlation
    peaks through the image, none of which are actually correct. When we
    are finding a limited number of peaks, they all get eaten up by this
    line and we never look elsewhere.
    """
    logger = logging.getLogger(_LOGGING_NAME+'._star_find_offset')

    mark_for_deletion = [False] * len(offset_list)
    for idx1 in xrange(len(offset_list)-2):
        for idx2 in xrange(idx1+1,len(offset_list)-1):
            for idx3 in xrange(idx2+1,len(offset_list)):
                u1 = offset_list[idx1][0]
                u2 = offset_list[idx2][0]
                u3 = offset_list[idx3][0]
                v1 = offset_list[idx1][1]
                v2 = offset_list[idx2][1]
                v3 = offset_list[idx3][1]
                if (u1 is None or u2 is None or u3 is None or
                    v1 is None or v2 is None or v3 is None):
                    continue
                if u1 == u2: # Vertical line
                    if abs(u3-u1) <= tolerance:
                        logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) '+
                                     'in a line',
                                     idx1+1, u1, v1, 
                                     idx2+1, u2, v2,
                                     idx3+1, u3, v3)
                        mark_for_deletion[idx2] = True
                        mark_for_deletion[idx3] = True                        
                else:
                    slope = float(v1-v2)/float(u1-u2)
                    if abs(slope) < 0.5:
                        v_intercept = slope * (u3-u1) + v1
                        diff = abs(v3-v_intercept)
                    else:
                        u_intercept = 1/slope * (v3-v1) + u1
                        diff = abs(u3-u_intercept)
                    if diff <= tolerance:
                        logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) '+
                                     'in a line',
                                     idx1+1, u1, v1,
                                     idx2+1, u2, v2,
                                     idx3+1, u3, v3)
                        mark_for_deletion[idx2] = True
                        mark_for_deletion[idx3] = True
    new_offset_list = []
    for i in xrange(len(offset_list)):
        if not mark_for_deletion[i]:
            new_offset_list.append(offset_list[i])
    
    return new_offset_list

def _star_find_offset(obs, filtered_data, star_list, margin, min_stars,
                      search_multiplier, max_offsets, already_tried,
                      debug_level, star_config):
    """Internal helper for star_find_offset so the loops don't get too deep."""
    # 1) Find an offset
    # 2) Remove any stars that are on top of a moon, planet, or opaque part of
    #    the rings
    # 3) Repeat until convergence

    logger = logging.getLogger(_LOGGING_NAME+'._star_find_offset')

    min_brightness_guaranteed_vis = star_config[
                                        'min_brightness_guaranteed_vis']
    min_confidence = star_config['min_confidence']
    
    # Restrict the search size    
    search_size_max_u, search_size_max_v = MAX_POINTING_ERROR[obs.detector]
    search_size_max_u = int(search_size_max_u*search_multiplier)
    search_size_max_v = int(search_size_max_v*search_multiplier)
    
    for star in star_list:
        star.conflicts = None

    peak_margin = 3 # Amount on each side of a correlation peak to black out
    # Make sure we have peaks that can cover 2 complete "lines" in the
    # correlation
    trial_max_offsets = (max(2*search_size_max_u+1, 2*search_size_max_v+1) //
                       (peak_margin*2+1)) + 4
        
    # Find the best offset using the current star list.
    # Then look to see if any of the stars correspond to known
    # objects like moons, planets, or opaque parts of the ring.
    # If so, get rid of those stars and iterate.
    
    model = star_create_model(obs, star_list, extend_fov=obs.extend_fov,
                              star_config=star_config)

    offset_list = find_correlation_and_offset(
                    filtered_data, model, search_size_min=0,
                    search_size_max=(search_size_max_u, search_size_max_v),
                    max_offsets=trial_max_offsets)

    offset_list = _optimize_offset_list(offset_list)
    offset_list = offset_list[:max_offsets]
    
    new_offset_u_list = []
    new_offset_v_list = []
    new_peak_list = []
    for i in xrange(len(offset_list)):
        if offset_list[i][:2] not in already_tried:
            new_offset_u_list.append(offset_list[i][0])
            new_offset_v_list.append(offset_list[i][1])
            new_peak_list.append(offset_list[i][2])
            # Nobody else gets to try these before we do
            already_tried.append(offset_list[i][:2])
        else:
            logger.debug('Offset %d,%d already tried (or reserved)', 
                         offset_list[i][0], offset_list[i][1])

    logger.debug('Final peak list:')
    for i in xrange(len(new_offset_u_list)):
        logger.debug('Peak %d U,V %d,%d VAL %f', i+1, 
                     new_offset_u_list[i], new_offset_v_list[i],
                     new_peak_list[i])

    if len(new_offset_u_list) == 0:
        # No peaks found at all - tell the top-level loop there's no point
        # in trying more
        return None, None, None, False, True
            
    for peak_num in xrange(len(new_offset_u_list)):
        #
        #            *** LEVEL 4+n ***
        #
        # At this level we actually find the offsets for the set of stars
        # previously restricted and the restricted search space.
        #
        # If one of those offsets gives good photometry, we're done.
        #
        # Otherwise, we mark conflicting stars and recurse with a new list
        # of non-conflicting stars.
        #
        offset_u = new_offset_u_list[peak_num]
        offset_v = new_offset_v_list[peak_num]
        peak = new_peak_list[peak_num]

        logger.debug('** LEVEL %d: Peak %d - Trial offset U,V %d,%d', 
                     debug_level, peak_num+1, offset_u, offset_v)

        # First try the star list as given to us.
                    
        for star in star_list:
            star.conflicts = None
        
        something_conflicted = False
        for star in star_list:
            res = _star_mark_conflicts(obs, star, offset_u, offset_v,
                                       margin, star_config)
            something_conflicted = something_conflicted or res

        good_stars = star_perform_photometry(obs,
                                             obs.calib_dn_ext_data,
                                             star_list,
                                             offset_u=offset_u,
                                             offset_v=offset_v,
                                             extend_fov=obs.extend_fov,
                                             star_config=star_config)
        logger.debug('Photometry found %d good stars', good_stars)
        
        # We have to see at least 2/3 of the really bright stars to
        # fully believe the result. If we don't see this many, it
        # triggers some more aggressive searching.
        bright_stars = 0
        seen_bright_stars = 0
        for star in star_list:
            if (star.dn >= min_brightness_guaranteed_vis and
                not star.conflicts and
                star.integrated_dn != 0.): # Photometry failed if == 0.
                bright_stars += 1
                if star.photometry_confidence >= min_confidence:
                    seen_bright_stars += 1
        if good_stars >= min_stars:
            if bright_stars > 0 and seen_bright_stars < bright_stars*2//3:
                logger.debug('***** Enough good stars, but only saw %d '+
                             'out of %d bright stars - possibly bad '+
                             'offset U,V %d,%d',
                             seen_bright_stars, bright_stars,
                             offset_u, offset_v)
                # Return True so the top-level loop keeps searching
                return offset_u, offset_v, good_stars, True, False
            logger.debug('***** Enough good stars - final offset U,V %d,%d',
                         offset_u, offset_v)
            # Return False so the top-level loop gives up
            return offset_u, offset_v, good_stars, False, False

        # OK that didn't work - get rid of the conflicting stars and
        # recurse until there are no conflicts

        if not something_conflicted:
            # No point in trying again - we'd just have the same stars!
            logger.debug('Nothing conflicted - continuing to next peak')
            continue
            
        # Create the current non-conflicting star list
        non_conf_star_list = [x for x in star_list if not x.conflicts]
    
        logger.debug('After conflict - # stars %d', len(non_conf_star_list))
                
        if len(non_conf_star_list) < min_stars:
            logger.debug('Fewer than %d stars left (%d)', min_stars,
                         len(non_conf_star_list))
            continue

        # And recurse using this limited star list
        ret = _star_find_offset(obs, filtered_data, non_conf_star_list, 
                                margin, min_stars, search_multiplier, 
                                max_offsets, already_tried, 
                                debug_level+1, star_config)
        if ret[0] is not None:
            return ret
        # We know that everything in non_conf_star_list is not 
        # conflicting at this level, but they were probably mutated by
        # _star_find_offset, so reset them
        for star in non_conf_star_list:
            star.conflicts = False
                
    logger.debug('Exhausted all peaks - No offset found')

    return None, None, None, False, False

def _star_refine_offset(obs, calib_data, star_list, offset_u, offset_v,
                        star_config):
    """Perform astrometry to refine the final offset."""
    logger = logging.getLogger(_LOGGING_NAME+'._star_refine_offset')

    gausspsf = GaussianPSF(sigma=ISS_PSF_SIGMA[obs.detector])
    
    delta_u_list = []
    delta_v_list = []
    dn_list = []
    for star in star_list:
        if star.conflicts:
            continue
        if star.photometry_confidence < star_config['min_confidence']:
            continue
        u = star.u + offset_u
        v = star.v + offset_v
        ret = gausspsf.find_position(calib_data, (5,5),
                      (v,u), search_limit=(1.5, 1.5),
                      bkgnd_degree=2, bkgnd_ignore_center=(2,2),
                      bkgnd_num_sigma=5,
                      tolerance=1e-5, num_sigma=5,
                      max_bad_frac=0.2)
        if ret is None:
            continue
        pos_v, pos_u, metadata = ret
        logger.debug('Star %9d UV %7.2f %7.2f refined to %7.2f %7.2f', 
                     star.unique_number, u, v, pos_u, pos_v)
        delta_u_list.append(pos_u-u)
        delta_v_list.append(pos_v-v)
        dn_list.append(star.dn)
        
    if len(delta_u_list) == 0:
        return offset_u, offset_v
    
    du_mean = np.mean(delta_u_list)
    dv_mean = np.mean(delta_v_list)
    
    logger.debug('Mean dU,dV %7.2f %7.2f', du_mean, dv_mean)
    
    return (int(np.round(offset_u+du_mean)),
            int(np.round(offset_v+dv_mean)))

def star_find_offset(obs, extend_fov=(0,0), star_config=None):
    """Find the image offset based on stars.

    Inputs:
        obs                The observation.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to model stars to find an offset.
        star_config        Configuration parameters. None uses the default.
                           
    Returns:
        offset_u, offset_v, metadata
        
        offset_u           The (integer) offsets in the U/V directions.
        offset_v
        metadata           A dictionary containing information about the
                           offset result:
            'full_star_list'    The list of Stars in the FOV.
            'num_good_stars'    The number of Stars that photometrically match.
            'offset_u'          The U offset.
            'offset_v'          The V offset.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.star_find_offset')

    if star_config is None:
        star_config = STARS_DEFAULT_CONFIG
        
    min_dn = star_config[('min_detectable_dn', obs.detector)]
    psf_size = star_config['psf_size']
    min_stars = star_config['min_stars']
    
    margin = psf_size // 2

    metadata = {}

    set_obs_ext_bp(obs, extend_fov)
    set_obs_ext_data(obs, extend_fov)
    obs.star_body_list = None # Body inventory cache
    obs.calib_dn_ext_data = None # DN-calibrated, extended data
    
    # Get the Star list and initialize our new fields
    star_list = star_list_for_obs(obs,
                                  extend_fov=obs.extend_fov,
                                  star_config=star_config)
    for star in star_list:
        star.photometry_confidence = 0.
        star.is_bright_enough = False
        star.is_dim_enough = True

    metadata['full_star_list'] = star_list
    metadata['num_good_stars'] = 0
    metadata['offset_u'] = None
    metadata['offset_v'] = None

    # A list of offsets that we have already tried so we don't waste time
    # trying them a second time.
    already_tried = []
    
    # A list of offset results so we can choose the best one at the very end.
    saved_offsets = []
        
    while True:
        #
        #            *** LEVEL 1 ***
        #
        # At this level we delete stars that are too bright. These stars may
        # "latch on" to portions of the image that aren't really stars.
        #
        star_count = 0
        first_good_star = None
        for star in star_list:
            if star.is_dim_enough:
                star_count += 1
                if first_good_star is None:
                    first_good_star = star
                
        logger.debug('** LEVEL 1: Trying star list with %d stars', star_count)

        if star_count < min_stars:
            # There's no point in continuing if there aren't enough stars
            logger.debug('FAILED to find a valid offset - too few total stars')
            return None, None, metadata
    
        if first_good_star.dn < min_dn:
            # There's no point in continuing if the brightest star is below the
            # detection threshold
            logger.debug(
                 'FAILED to find a valid offset - brightest star is too dim')
            return None, None, metadata
    
        if obs.calib_dn_ext_data is None:
            # First pass - don't do these things earlier outside the loop
            # because we'd just be wasting time if there were never enough
            # stars.
             
            # For star use only, need data in DN for photometry
            obs.calib_dn_ext_data = calibrate_iof_image_as_dn(
                                                      obs, data=obs.ext_data)
        
            # Filter that calibrated data.
            # 1) Subtract the local background (median)
            # 2) Eliminate anything that is < 0
            # 3) Eliminate any portions of the image that are near a pixel
            #    that is brighter than the maximum star DN
            #    Note that since a star's photons are spread out over the PSF,
            #    you have to have a really bright single pixel to be brighter
            #    than the entire integrated DN of the brightest star in the
            #    FOV.
            filtered_data = filter_sub_median(obs.calib_dn_ext_data, 
                                              median_boxsize=11)
 
# XXX Maybe we should use the maximum filter?       
#filtered_data = filter_local_maximum(obs.calib_dn_ext_data,
#                                     maximum_boxsize=11, median_boxsize=11,
#                                     maximum_blur=5)

            filtered_data[filtered_data < 0.] = 0.

            # If we trust the DN values, then we can eliminate any pixels that
            # are way too bright.            
#            if _trust_star_dn(obs):
#                max_dn = star_list[0].dn # Star list is sorted by DN            
#                mask = filtered_data > max_dn
#                mask = filt.maximum_filter(mask, 11)
#                filtered_data[mask] = 0.

            if DEBUG_STARS_FILTER_IMGDISP:
                toplevel = Tkinter.Tk()
                frame_toplevel = Tkinter.Frame(toplevel)
                imdisp = ImageDisp([filtered_data],
                                   parent=frame_toplevel,
                                   canvas_size=(512,512),
                                   allow_enlarge=True, enlarge_limit=10,
                                   auto_update=True)
                frame_toplevel.pack()
                Tkinter.mainloop()

        offset_u = None
        offset_v = None
        good_stars = 0
        confidence = 0.
    
        got_it = False
        
        # Try with the brightest stars, then go down to the dimmest
        for min_dn_gain in [1.]:#[4., 2.5, 1.]:
            #
            #            *** LEVEL 2 ***
            #
            # At this level we delete stars that are too dim. These stars may
            # have poor detections and just confused the correlation.
            #
            logger.debug('** LEVEL 2: Trying DN gain %.1f', min_dn_gain)
            new_star_list = []
            for star in star_list:
                star.photometry_confidence = 0.
                star.is_bright_enough = False
                if not star.is_dim_enough:
                    continue
                if star.dn < min_dn*min_dn_gain:
                    continue
                star.is_bright_enough = True
                new_star_list.append(star)

            logger.debug('Using %d stars', len(new_star_list))
                            
            if len(new_star_list) < min_stars:
                logger.debug('Not enough stars: %d (%d required)',
                             len(new_star_list), min_stars)
                continue
            
            # Try with a small search area, then enlarge
            for search_multipler in [0.25, 0.5, 0.75, 1.]:
                #
                #            *** LEVEL 3 ***
                #
                # At this level we restrict the search space so we first try
                # smaller offsets.
                #
                logger.debug('** LEVEL 3: Trying search multiplier %.2f', 
                             search_multipler)
                
                # The remaining search levels are inside the subroutine
                ret = _star_find_offset(obs, filtered_data, new_star_list,
                                        margin, min_stars, search_multipler,
                                        5, already_tried, 4, star_config) 
        
                # Save the offset and maybe continue iterating
                (offset_u, offset_v, good_stars, 
                 keep_searching, no_peaks) = ret

                if no_peaks and search_multipler == 1.:
                    logger.debug('No peaks found at largest search range - '+
                                 'aborting star offset finding')
                    got_it = True
                    break
                 
                if offset_u is None:
                    logger.debug('No valid offset found - iterating')
                    continue

                logger.debug('Found valid offset U,V %d,%d', 
                             offset_u, offset_v)
                saved_star_list = copy.deepcopy(star_list)
                saved_offsets.append((offset_u, offset_v, good_stars, 
                                      saved_star_list))
                if not keep_searching:
                    got_it = True
                    break
                
                # End of LEVEL 3 - restrict search area
                
            if got_it:
                break
    
            if len(new_star_list) == len(star_list):
                # No point in trying smaller DNs if we already are looking at
                # all stars
                logger.debug(
                     'Already looking at all stars - ignoring other DNs')
                break

            # End of LEVEL 2 - eliminate dim stars
            
        if got_it:
            break
        
        # Get rid of an unusually bright star - these sometimes get stuck
        # correlating with non-star objects, like parts of the F ring
        still_good_stars_left = False
        for i in xrange(len(star_list)):
            if not star_list[i].is_dim_enough:
                continue
            # First bright star we used last time
            if i == len(star_list)-1:
                # It was the last star - nothing to compare against
                logger.debug('No dim enough stars left - giving up')
                break
            if star_list[i].dn > 1000: #star_list[i].dn > star_list[i+1].dn*2: # XXX
                # Star is twice as bright as next star - get rid of it
#                logger.debug('Star %9d (DN %7.2f) is much brighter than star %9d (DN %7.2f) - ignoring and iterating',
#                             star_list[i].unique_number, star_list[i].dn,
#                             star_list[i+1].unique_number, star_list[i+1].dn)
                logger.debug('Star %9d (DN %7.2f) is too bright - '+
                             'ignoring and iterating',
                             star_list[i].unique_number, star_list[i].dn)
                star_list[i].is_dim_enough = False
                still_good_stars_left = True
            break
            
        if not still_good_stars_left:
            break
    
        # End of LEVEL 1 - eliminate bright stars
        
    if len(saved_offsets) == 0:
        good_stars = 0
        logger.debug('FAILED to find a valid offset')
    else:
        best_offset_u = None
        best_offset_v = None
        best_star_list = None
        best_good_stars = -1
        for offset_u, offset_v, good_stars, saved_star_list in saved_offsets:
            if len(saved_offsets) > 1:
                logger.debug('Saved offset U,V %d,%d / Good stars %d',
                             offset_u, offset_v, good_stars)
            if good_stars > best_good_stars:
                best_offset_u = offset_u
                best_offset_v = offset_v
                best_good_stars = good_stars
                best_star_list = saved_star_list
        offset_u = best_offset_u
        offset_v = best_offset_v
        good_stars = best_good_stars
        star_list = saved_star_list

        logger.debug('Trial final offset U,V %d,%d / Good stars %d',
                     offset_u, offset_v, good_stars)

        offset_u, offset_v = _star_refine_offset(obs, obs.data, star_list,
                                                 offset_u, offset_v, star_config)
        
        logger.debug('Returning final offset U,V %d,%d / Good stars %d',
                     offset_u, offset_v, good_stars)
            
    metadata['full_star_list'] = star_list
    metadata['num_good_stars'] = good_stars
    metadata['offset_u'] = offset_u
    metadata['offset_v'] = offset_v
    
    return offset_u, offset_v, metadata
