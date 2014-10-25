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

import numpy as np
import numpy.ma as ma
import polymath
import scipy.ndimage.filters as filt
import copy

import oops
from psfmodel.gaussian import GaussianPSF
from imgdisp import draw_circle, draw_rect
from starcat import (UCAC4StarCatalog,
                     SCLASS_TO_B_MINUS_V, SCLASS_TO_SURFACE_TEMP)
from cb_config import *
from cb_correlate import *
from cb_rings import *
from cb_util_flux import *
from cb_util_image import *
from cb_util_oops import *

STAR_CATALOG = UCAC4StarCatalog(STAR_CATALOG_ROOT)

LOGGING_NAME = 'cb.' + __name__

STAR_MIN_CONFIDENCE = 0.9
DEFAULT_STAR_CLASS = 'G0'
STAR_MIN_BRIGHTNESS_GUARANTEED_VIS = 200.

#===============================================================================
#
# FIND STARS IN THE FOV AND RETURN THEM.
# 
#===============================================================================

def _aberrate_star(obs, star):
    """Compute the RA,DEC position of a star with stellar aberration."""
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

def _star_list_for_obs(obs, ra_min, ra_max, dec_min, dec_max,
                       mag_min, mag_max, extend_fov,
                       **kwargs):
    """Return a list of stars with the given constraints.
    
    See star_list_for_obs for full details."""
    logger = logging.getLogger(LOGGING_NAME+'._star_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
    # Get a list of all reasonable stars with the given magnitude range.
    
    orig_star_list = [x for x in 
              STAR_CATALOG.find_stars(allow_double=True,
                                      allow_galaxy=True,
                                      ra_min=ra_min, ra_max=ra_max,
                                      dec_min=dec_min, dec_max=dec_max,
                                      vmag_min=mag_min, vmag_max=mag_max,
                                      **kwargs)]

    # Fake the temperature if it's not known, and eliminate stars we just
    # don't want to deal with.

    discard_class = 0
    discard_dn = 0
        
    star_list = []
    for star in orig_star_list:
        star.conflicts = None
        star.temperature_faked = False
        if star.temperature is None:
            star.temperature_faked = True
            star.temperature = SCLASS_TO_SURFACE_TEMP[DEFAULT_STAR_CLASS]
            star.spectral_class = DEFAULT_STAR_CLASS
            star.johnson_mag_v = (star.vmag-
                                  SCLASS_TO_B_MINUS_V[DEFAULT_STAR_CLASS]/2.)
            star.johnson_mag_b = (star.vmag-
                                  SCLASS_TO_B_MINUS_V[DEFAULT_STAR_CLASS]/2.)
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

    logger.debug('Found %d stars, discarded because of CLASS %d, LOW DN %d, BAD UV %d',
                 len(orig_star_list), discard_class, discard_dn, discard_uv)

    return new_star_list

def star_list_for_obs(obs, max_stars=30, psf_size=9, extend_fov=(0,0),
                      **kwargs):
    """Return a list of stars in the FOV of the obs.

    Inputs:
        obs                The observation.
        max_stars          The maximum number of stars to return.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd. This is just used to restrict
                           the list of stars to ones not too close to the edge
                           of the image.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to return stars (a star's U/V value will be negative
                           or greater than the FOV shape).
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
    logger = logging.getLogger(LOGGING_NAME+'.star_list_for_obs')
    
    margin = psf_size//2
    
    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs,
                                             extend_fov=extend_fov)

    # Try to be efficient by limiting the magnitudes searched so we don't
    # return a huge number of dimmer stars and then only need a few of them.
    magnitude_list = [0., 12., 13., 14., 15.]
    
    mag_vmax = compute_dimmest_visible_star_vmag(obs)+1

    logger.debug('Max detectable VMAG %.4f', mag_vmax)
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        if mag_min > mag_vmax:
            break
        mag_max = min(mag_max, mag_vmax)
        
        star_list = _star_list_for_obs(obs,
                                       ra_min, ra_max, dec_min, dec_max,
                                       mag_min, mag_max,
                                       extend_fov, **kwargs)
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
    
def star_create_model(obs, star_list, psf_size=9, offset_u=0., offset_v=0.,
                      extend_fov=(0,0)):
    """Create a model containing nothing but stars.
    
    Inputs:
        obs                The observation.
        star_list          The list of Stars.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        extend_fov         The amount to extend the model beyond the limits of
                           the obs FOV. The returned model will be the shape of
                           the obs FOV plus two times the extend value in each
                           dimension.
        
    Returns:
        model              The model.
    """
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

def _star_perform_photometry(obs, calib_data, star, offset_u, offset_v,
                             extend_fov):
    # Data is calibrated, non-extended
    u = int(np.round(star.u)) + offset_u + extend_fov[0]
    v = int(np.round(star.v)) + offset_v + extend_fov[1]
    
    if star.dn > 500:
        boxsize = 11
    elif star.dn > 100:
        boxsize = 9
    else:
        boxsize = 7 #XXX
    
    star.photometry_box_width = boxsize
    
    box_halfsize = boxsize // 2

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
                            extend_fov=(0,0)):
    """Perform photometry on a list of stars.
    
    Inputs:
        obs                The observation.
        star_list          The list of Star objects.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        confidence_sigma
        max_bkgnd
    
    Returns:
        good_stars, confidence
        
        good_stars         The number of good stars.
        confidence         The aggregate confidence.
        
        Each Star is populated with:
        
        integrated_dn
        photometry_confidence
    """
    logger = logging.getLogger(LOGGING_NAME+'.star_perform_photometry')

    image = obs.data
    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
#    logger.debug('OFFSET U,V %d %d BOXSIZE %d to %d MINCONF %f FITPSF %d',
#                 offset_u, offset_v, min_boxsize, max_boxsize, min_confidence,
#                 fit_psf)

    for star in star_list:
        u = int(np.round(star.u)) + offset_u
        v = int(np.round(star.v)) + offset_v
        if star.conflicts:
            star.integrated_dn = 0.
            star.photometry_confidence = 0.
            logger.debug('Star %9d %2s UV %4d %4d IGNORED %s',
                         star.unique_number, star.spectral_class,
                         u, v, star.conflicts)
            continue
        ret = _star_perform_photometry(obs, calib_data, star,
                                       offset_u, offset_v, extend_fov)
        if ret is None:
            integrated_dn = 0.
            confidence = 0.
        else:
            integrated_dn, bkgnd, integrated_std, bkgnd_std = ret
#            print integrated_dn, bkgnd, integrated_std, bkgnd_std
            if integrated_dn < 0:
                confidence = 0.    
            elif star.temperature_faked:
                # Really the only thing we can do here is see if we detected
                # something at all, because we can't trust the photometry
                confidence = ((integrated_dn >= min_dn)*0.5 +
                              (integrated_std >= bkgnd_std*2)*0.5)
            else:  
                confidence = ((star.dn/3 < integrated_dn < star.dn*3)*0.5 +
                              (integrated_std >= bkgnd_std*1.5)*0.5)
                
        star.integrated_dn = integrated_dn
        star.photometry_confidence = confidence
        
        logger.debug('Star %9d %2s UV %4d %4d PRED %7.2f MEAS %7.2f CONF %4.2f',
                     star.unique_number, star.spectral_class,
                     u, v, star.dn, star.integrated_dn, star.photometry_confidence)

    good_stars = 0
    for star in star_list:
        if star.photometry_confidence >= STAR_MIN_CONFIDENCE:
            good_stars += 1

    return good_stars

def _star_refine_offset(obs, calib_data, star_list, offset_u, offset_v):
    logger = logging.getLogger(LOGGING_NAME+'._star_refine_offset')

    gausspsf = GaussianPSF(sigma=ISS_PSF_SIGMA[obs.detector])
    
    delta_u_list = []
    delta_v_list = []
    dn_list = []
    for star in star_list:
        if star.conflicts:
            continue
        if star.photometry_confidence < STAR_MIN_CONFIDENCE:
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
    
#    du_mean = np.average(delta_u_list, weights=dn_list)
#    dv_mean = np.average(delta_v_list, weights=dn_list)
    
    du_mean = np.mean(delta_u_list)
    dv_mean = np.mean(delta_v_list)
    
    logger.debug('Mean dU,dV %7.2f %7.2f', du_mean, dv_mean)
    
    return (int(np.round(offset_u+du_mean)),
            int(np.round(offset_v+dv_mean)))

def star_make_good_bad_overlay(obs, star_list, offset_u, offset_v,
                               overlay_box_width=None,
                               overlay_box_thickness=None,
                               extend_fov=(0,0)):
    """Create an overlay with high and low confidence stars marked.
    
    Inputs:
        obs                The observation.
        star_list          The list of Star objects.
        min_confidence     The minimum confidence for a star to be considered
                           good.
                           
    Returns:
        overlay            The overlay.
    """
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
            if star.photometry_confidence > STAR_MIN_CONFIDENCE:
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
#===============================================================================

def _star_mark_conflicts(obs, star, offset_u, offset_v, margin):
    logger = logging.getLogger(LOGGING_NAME+'._star_mark_conflicts')

    # Check for off the edge
    if (not (margin < star.u+offset_u < obs.data.shape[1]-margin) or
        not (margin < star.v+offset_v < obs.data.shape[0]-margin)):
        logger.debug('Star %9d U %8.3f V %8.3f is off the edge',
                     star.unique_number, star.u, star.v)
        star.conflicts = 'EDGE'
        return True

#    # Check for planet and moons
#    for body_name in LARGE_BODY_LIST:
#        intercepted = obs.ext_bp.where_intercepted(body_name)
#        if intercepted.vals[star.v+obs.extend_fov[1],
#                            star.u+obs.extend_fov[0]]:
#            logger.debug('Star %9d U %8.3f V %8.3f conflicts with %s',
#                         star.unique_number, star.u, star.v, body_name)
#            star.use_for_model = False
#            star.bad_reason = 'BODY:' + body_name
#            return True

    if obs.star_body_list is None:
        obs.star_body_list = obs.inventory(LARGE_BODY_LIST)
        
    if len(obs.star_body_list) > 0:
        # Give 3 pixels of slop on each side - we don't want a star to
        # even be close to a large object.
        star_slop = 3
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
    ring_longitude = obs.ext_bp.ring_longitude('saturn:ring').vals.astype('float')
    rad = ring_radius[star.v+obs.extend_fov[1], star.u+obs.extend_fov[0]]
    long = ring_longitude[star.v+obs.extend_fov[1], star.u+obs.extend_fov[0]]
#    logger.debug('Star %9d U %8.3f V %8.3f rings radius '
#                 '%.1f',
#                 star.unique_number, star.u, star.v, rad)
#    logger.debug('F Ring radius %.1f', rings_fring_radius_at_longitude(obs, long))        

    if ((oops.SATURN_C_RING[0] <= rad <= oops.SATURN_A_RING[1]) or # C to A rings
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
    logger = logging.getLogger(LOGGING_NAME+'._star_find_offset')

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
                        logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) in a line',
                                     idx1+1, u1, v1, idx2+1, u2, v2, idx3+1, u3, v3)
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
                        logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) in a line',
                                     idx1+1, u1, v1, idx2+1, u2, v2, idx3+1, u3, v3)
                        mark_for_deletion[idx2] = True
                        mark_for_deletion[idx3] = True
    new_offset_list = []
    for i in xrange(len(offset_list)):
        if not mark_for_deletion[i]:
            new_offset_list.append(offset_list[i])
    
    return new_offset_list

def _star_find_offset(obs, filtered_data, star_list, margin, min_stars,
                      search_multiplier, num_peaks, debug_level=4, already_tried=None):
    # 1) Find an offset
    # 2) Remove any stars that are on top of a moon, planet, or opaque part of
    #    the rings
    # 3) Repeat until convergence

    logger = logging.getLogger(LOGGING_NAME+'._star_find_offset')

    if already_tried is None:
        already_tried = []
        
    search_size_max_u, search_size_max_v = MAX_POINTING_ERROR[obs.detector]
    search_size_max_u = int(search_size_max_u*search_multiplier)
    search_size_max_v = int(search_size_max_v*search_multiplier)
    
    for star in star_list:
        star.conflicts = None

    peak_margin = 3 # Amount on each side of a correlation peak to black out
    # Make sure we have peaks that can cover 2 complete "lines" in the
    # correlation
    trial_num_peaks = (max(2*search_size_max_u+1, 2*search_size_max_v+1) //
                       (peak_margin*2+1)) + 4
        
    # Find the best offset using the current star list.
    # Then look to see if any of the stars correspond to known
    # objects like moons, planets, or opaque parts of the ring.
    # If so, get rid of those stars and iterate.
    
    model = star_create_model(obs, star_list, extend_fov=obs.extend_fov)

    offset_list = find_correlation_and_offset(
                    filtered_data, model, search_size_min=0,
                    search_size_max=(search_size_max_u, search_size_max_v),
                    num_peaks=trial_num_peaks)

    offset_list = _optimize_offset_list(offset_list)
    offset_list = offset_list[:num_peaks]
    
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
            
    for peak_num in xrange(len(new_offset_u_list)):
        offset_u = new_offset_u_list[peak_num]
        offset_v = new_offset_v_list[peak_num]
        peak = new_peak_list[peak_num]

        if offset_u is None:
            logger.debug('** LEVEL %d: Peak # %d - Correlation FAILED', 
                         debug_level, peak_num+1)
            continue

        logger.debug('** LEVEL %d: Peak %d - Trial offset U,V %d,%d', debug_level, 
                     peak_num+1, offset_u, offset_v)

        for star in star_list:
            star.conflicts = None
        
        conflict_pass_num = 1
        something_conflicted = True
        while something_conflicted:
            something_conflicted = False
            logger.debug('** LEVEL %d: Conflict check %d', 
                         debug_level+1, conflict_pass_num)
            
            # Create a model for the current star list
            new_star_list = []
            for star in star_list:
                if not star.conflicts:
                    new_star_list.append(star)
        
            for star in new_star_list:
                res = _star_mark_conflicts(obs, star, offset_u, offset_v,
                                           margin)
                something_conflicted = something_conflicted or res

            good_stars = star_perform_photometry(obs,
                                                 obs.calib_dn_ext_data,
                                                 new_star_list,
                                                 offset_u=offset_u,
                                                 offset_v=offset_v,
                                                 extend_fov=obs.extend_fov)
            logger.debug('Photometry found %d good stars', good_stars)
            
            # We have to see at least 2/3 of the really bright stars to
            # fully believe the result. If we don't see this many, it
            # triggers some more aggressive searching.
            bright_stars = 0
            seen_bright_stars = 0
            for star in new_star_list:
                if (star.dn >= STAR_MIN_BRIGHTNESS_GUARANTEED_VIS and
                    not star.conflicts and
                    star.integrated_dn != 0.): # Photometry failed if == 0.
                    bright_stars += 1
                    if star.photometry_confidence >= STAR_MIN_CONFIDENCE:
                        seen_bright_stars += 1
            if good_stars >= min_stars:
                if bright_stars > 0 and seen_bright_stars < bright_stars*2//3:
                    logger.debug('***** Enough good stars, but only saw %d out of %d bright stars - possibly bad offset U,V %d,%d',
                                 seen_bright_stars, bright_stars, offset_u, offset_v)
                    return offset_u, offset_v, good_stars, True
                logger.debug('***** Enough good stars - final offset U,V %d,%d',
                             offset_u, offset_v)
                return offset_u, offset_v, good_stars, False

            # OK that didn't work - get rid of the conflicting stars and do
            # another correlation - iterating until there are no conflicts

            if not something_conflicted:
                # No point in trying again - we'd just have the same stars!
                logger.debug('Nothing conflicted - continuing to next peak')
                break
            
            # Create a model for the current star list
            new_star_list = []
            for star in star_list:
                if not star.conflicts:
                    new_star_list.append(star)
        
            logger.debug('After conflict - # stars %d', len(new_star_list))
                    
            if len(new_star_list) < min_stars:
                logger.debug('Fewer than %d stars left (%d)', min_stars,
                             len(new_star_list))
                break
            
            model = star_create_model(obs, new_star_list, extend_fov=obs.extend_fov)
    
            conf_offset_list = find_correlation_and_offset(
                            filtered_data, model, search_size_min=0,
                            search_size_max=(search_size_max_u, search_size_max_v),
                            num_peaks=1)

            new_conf_offset_u_list = []
            new_conf_offset_v_list = []
            new_conf_peak_list = []
            for i in xrange(len(conf_offset_list)):
                if conf_offset_list[i][:2] not in already_tried:
                    new_conf_offset_u_list.append(conf_offset_list[i][0])
                    new_conf_offset_v_list.append(conf_offset_list[i][1])
                    new_conf_peak_list.append(conf_offset_list[i][2])
                    # Nobody else gets to try these before we do
                    already_tried.append(conf_offset_list[i][:2])
                else:
                    logger.debug('Offset %d,%d already tried (or reserved)',
                                 conf_offset_list[i][0], conf_offset_list[i][0])

            for conf_peak_num in xrange(len(new_conf_offset_u_list)):
                offset_u = new_conf_offset_u_list[conf_peak_num]
                offset_v = new_conf_offset_v_list[conf_peak_num]
                if offset_u is None:
                    logger.debug('** LEVEL %d: After conflict check %d peak %d NO OFFSET', 
                                 debug_level+1, conflict_pass_num,
                                 conf_peak_num+1)
                    continue
                already_tried.append((offset_u,offset_v))
                logger.debug('** LEVEL %d: Conflict check %d peak %d offset U,V %d,%d', 
                             debug_level+1, conflict_pass_num, conf_peak_num+1,
                             offset_u, offset_v)
                conf_star_list = [x for x in new_star_list if not x.conflicts]
                ret = _star_find_offset(obs, filtered_data, conf_star_list, margin, min_stars,
                                        search_multiplier, num_peaks, debug_level+2,
                                        already_tried)
                if ret is not None:
                    return ret
                
            conflict_pass_num += 1

    logger.debug('Exhausted all peaks - No offset found')

    return None

def star_find_offset(obs, min_stars=3, psf_size=9, extend_fov=(0,0)):
    logger = logging.getLogger(LOGGING_NAME+'.star_find_offset')

    metadata = {}

    min_dn = MIN_DETECTABLE_DN[obs.detector]

    set_obs_ext_bp(obs, extend_fov)
    set_obs_ext_data(obs, extend_fov)
    obs.star_body_list = None # Body inventory cache
    obs.calib_dn_ext_data = None
    
    star_list = star_list_for_obs(obs, max_stars=30, 
                                  extend_fov=obs.extend_fov)
    for star in star_list:
        star.photometry_confidence = 0.
        star.is_bright_enough = False
        star.is_dim_enough = True

    metadata['full_star_list'] = star_list
    metadata['num_good_stars'] = 0

    already_tried = []
    saved_offsets = []
        
    while True:
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
    
        if first_good_star.dn < MIN_DETECTABLE_DN[obs.detector]:
            # There's no point in continuing if the brightest star is below the
            # detection threshold
            logger.debug('FAILED to find a valid offset - brightest star is too dim')
            return None, None, metadata
    
        if obs.calib_dn_ext_data is None:
            # First pass 
            # For star use only, need data in DN for photometry
            obs.calib_dn_ext_data = calibrate_iof_image_as_dn(obs, data=obs.ext_data)
        
            margin = psf_size // 2
            
            filtered_data = filter_sub_median(obs.calib_dn_ext_data, 
                                              median_boxsize=11)
        
#            filtered_data = filter_local_maximum(obs.calib_dn_ext_data,
#                                                 maximum_boxsize=11, median_boxsize=11,
#                                                 maximum_blur=5)

            filtered_data[filtered_data < 0.] = 0.
            
            max_dn = star_list[0].dn # Star list is sorted by DN
            
            mask = filtered_data > max_dn
            
            mask = filt.maximum_filter(mask, 11)
            filtered_data[mask] = 0.
    
        offset_u = None
        offset_v = None
        good_stars = 0
        confidence = 0.
    
        got_it = False
        
        # Try with the brightest stars, then go down to the dimmest
        for min_dn_gain in [1.]:#[4., 2.5, 1.]:
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
                logger.debug('** LEVEL 3: Trying search multiplier %.2f', search_multipler)
                
                ret = _star_find_offset(obs, filtered_data, new_star_list, margin, 
                                        min_stars, search_multipler, 5,
                                        already_tried=already_tried) 
        
                if ret is None:
                    logger.debug('No valid offset found - iterating')
                    continue
                
                offset_u, offset_v, good_stars, keep_searching = ret
                logger.debug('Found valid offset U,V %d,%d', offset_u, offset_v)
                saved_star_list = copy.deepcopy(star_list)
                saved_offsets.append((offset_u, offset_v, good_stars, saved_star_list))
                if not keep_searching:
                    got_it = True
                    break
                
            if got_it:
                break
    
            if len(new_star_list) == len(star_list):
                # No point in trying smaller DNs if we already are looking at
                # all stars
                logger.debug('Already looking at all stars - ignoring other DNs')
                break

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
                logger.debug('Star %9d (DN %7.2f) is too bright - ignoring and iterating',
                             star_list[i].unique_number, star_list[i].dn)
                star_list[i].is_dim_enough = False
                still_good_stars_left = True
            break
            
        if not still_good_stars_left:
            break
    
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
                                                 offset_u, offset_v)
        
        logger.debug('Returning final offset U,V %d,%d / Good stars %d',
                     offset_u, offset_v, good_stars)
            
    metadata['full_star_list'] = star_list
    metadata['num_good_stars'] = good_stars
    
    return offset_u, offset_v, metadata
