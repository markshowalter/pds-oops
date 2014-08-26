import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filt

import oops
from psfmodel.gaussian import GaussianPSF
from imgdisp import draw_circle
from starcat import (UCAC4StarCatalog,
                     SCLASS_TO_B_MINUS_V, SCLASS_TO_SURFACE_TEMP)
from cb_config import (STAR_CATALOG_FILENAME, ISS_PSF_SIGMA,
                       MAX_POINTING_ERROR, LARGE_BODY_LIST)
from cb_correlate import *
from cb_util_image import *
from cb_util_flux import *
from cb_util_oops import *


STAR_CATALOG = UCAC4StarCatalog(STAR_CATALOG_FILENAME)
#STAR_CATALOG.debug_level = 100

LOGGING_NAME = 'cb.' + __name__

STAR_MIN_CONFIDENCE = 0.65
DEFAULT_STAR_CLASS = 'G0'

#===============================================================================
# 
#===============================================================================

# These values are pretty aggressively dim - there's no guarantee a star with 
# this brightness can actually be seen.
MIN_DETECTABLE_DN = {'NAC': 20, 'WAC': 150}

def _star_list_for_obs(obs, ra_min, ra_max, dec_min, dec_max,
                       mag_min, mag_max, extend_fov_u, extend_fov_v,
                       **kwargs):
    logger = logging.getLogger(LOGGING_NAME+'._star_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
    # Get a list of all reasonable stars with the given magnitude range
    
    orig_star_list = [x for x in 
              STAR_CATALOG.find_stars(allow_double=True,
                                      allow_galaxy=True,
                                      ra_min=ra_min, ra_max=ra_max,
                                      dec_min=dec_min, dec_max=dec_max,
                                      vmag_min=mag_min, vmag_max=mag_max,
                                      **kwargs)]

    # Fake the temperature if it's not known, and eliminate stars
    # we just don't want to deal with.

    discard_class = 0
    discard_dn = 0
        
    star_list = []
    for star in orig_star_list:
        star.use_for_model = True
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
        if (u < -extend_fov_u or u > obs.data.shape[1]+extend_fov_u-1 or
            v < -extend_fov_v or v > obs.data.shape[0]+extend_fov_v-1):
            discard_uv += 1
            continue
        
        star.u = u
        star.v = v

        new_star_list.append(star)

    logger.debug('Found %d stars, discarded because of CLASS %d, LOW DN %d, BAD UV %d',
                 len(orig_star_list), discard_class, discard_dn, discard_uv)

    return new_star_list

def star_list_for_obs(obs, max_stars=30, psf_size=9, 
                      extend_fov_u=0, extend_fov_v=0, **kwargs):
    """Return a list of stars in the FOV of the obs.

    Inputs:
        obs                The observation.
        max_stars          The maximum number of stars to return.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd. This is just used to restrict
                           the list of stars to ones not too close to the edge
                           of the image.
        extend_fov_u       The amount beyond the image in the U dimension to
                           return stars (U value will be negative or greater
                           than the FOV shape)
        extend_fov_v       The amount beyond the image in the V dimension to
                           return stars (V value will be negative or greater
                           than the FOV shape)
        **kwargs           Passed to find_stars to restrict the types of stars
                           returned.
                           
    Returns:
        star_list          The list of Star objects.
                           .u and .v give the U,V coordinate.
                           .faked_temperature is a bool indicating if the
                               temperature and spectral class had to be faked. 
    """
    logger = logging.getLogger(LOGGING_NAME+'.star_list_for_obs')
    
    margin = psf_size//2
    
    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs,
                                             extend_fov_u=extend_fov_u,
                                             extend_fov_v=extend_fov_v)

    # XXX THIS COULD BE MADE MUCH MORE EFFICIENT
    # PAY ATTENTION TO WHERE RA/DEC IS POINTING - SGR? OUT OF PLANE?
    # ESTIMATE MAG NEEDED ON FIRST TRY
    
    magnitude_list = [0., 12., 12.5, 13., 13.5, 14., 14.3, 14.5, 14.7, 14.9]
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        star_list = _star_list_for_obs(obs,
                                       ra_min, ra_max, dec_min, dec_max,
                                       mag_min, mag_max,
                                       extend_fov_u, extend_fov_v,
                                       **kwargs)
#        if len(star_list) == 0:
#            # Didn't find anything so just give up on finding anything dimmer
#            break
        
        full_star_list += star_list
        
        logger.debug('Got %d stars, total %d', len(star_list),
                     len(full_star_list))
        
        if len(full_star_list) >= max_stars:
            break

    full_star_list.sort(key=lambda x: x.dn, reverse=True)
            
    if len(full_star_list) > max_stars:
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
                      extend_fov_u=0, extend_fov_v=0):
    """Create a model containing nothing but stars.
    
    Inputs:
        obs                The observation.
        star_list          The list of Stars.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        extend_fov_u       The amount to extend the model beyond the limits of
        extend_fov_v       the obs FOV. The returned model will be the shape of
                           the obs FOV plus two times the extend value in each
                           dimension.
        
    Returns:
        model              The model.
    """
    model = np.zeros((obs.data.shape[0]+extend_fov_v*2,
                      obs.data.shape[1]+extend_fov_u*2))
    
    gausspsf = GaussianPSF(sigma=ISS_PSF_SIGMA[obs.detector])
    
    vmag_list = [x.vmag for x in star_list]
    star_mag_min = min(vmag_list)
    star_mag_max = max(vmag_list)
    
    margin = psf_size//2
    
    for star in star_list:
        u_idx = star.u+offset_u+extend_fov_u
        v_idx = star.v+offset_v+extend_fov_v
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

def _star_perform_photometry(obs, star, offset_u=0, offset_v=0):
    u = int(np.round(star.u)) + offset_u
    v = int(np.round(star.v)) + offset_v
    
    if star.dn > 100:
        boxsize = 7
    else:
        boxsize = 5
    
    box_halfsize = boxsize // 2

    if (u < box_halfsize or
        u > obs.data.shape[1]-box_halfsize-1 or
        v < box_halfsize or
        v > obs.data.shape[0]-box_halfsize-1):
        return None
        
    subimage = obs.data[v-box_halfsize:v+box_halfsize+1,
                        u-box_halfsize:u+box_halfsize+1]
    subimage = subimage.view(ma.MaskedArray)
    subimage[1:-1, 1:-1] = ma.masked # Mask out the center
    
    bkgnd = ma.mean(subimage)
    bkgnd_std = ma.std(subimage)
    
    subimage.mask = ~subimage.mask # Mask out the edge
    integrated_dn = np.sum(subimage-bkgnd)

    return integrated_dn, bkgnd, bkgnd_std
    
def star_perform_photometry(obs, star_list, offset_u=0, offset_v=0):
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
        if not star.use_for_model:
            star.integrated_dn = 0.
            star.photometry_confidence = 0.
            logger.debug('Star %9d %2s UV %4d %4d IGNORED %s',
                         star.unique_number, star.spectral_class,
                         u, v, star.bad_reason)
            continue
        ret = _star_perform_photometry(obs, star, offset_u, offset_v)
        if ret is None:
            integrated_dn = 0.
            confidence = 0.
        else:
            integrated_dn, bkgnd, bkgnd_std = ret
            if integrated_dn < 0:
                confidence = 0.    
            elif star.temperature_faked:
                # Really the only thing we can do here is see if we detected
                # something at all, because we can't trust the photometry
                confidence = float(integrated_dn >= min_dn)
            else:  
                confidence = float((star.dn/3 < integrated_dn < star.dn*3))
                
        star.integrated_dn = integrated_dn
        star.photometry_confidence = confidence
        
        logger.debug('Star %9d %2s UV %4d %4d PRED %7.2f MEAS %7.2f CONF %d',
                     star.unique_number, star.spectral_class,
                     u, v, star.dn, star.integrated_dn, star.photometry_confidence)

    good_stars = 0
    for star in star_list:
        if star.photometry_confidence > 0:
            good_stars += 1

    return good_stars

def star_make_good_bad_overlay(obs, star_list, offset_u, offset_v,
                               extend_fov_u=0, extend_fov_v=0):
    """Create an overlay with high and low confidence stars marked.
    
    Inputs:
        obs                The observation.
        star_list          The list of Star objects.
        min_confidence     The minimum confidence for a star to be considered
                           good.
                           
    Returns:
        overlay            The overlay.
    """
    overlay = np.zeros((obs.data.shape[0]+extend_fov_v*2,
                        obs.data.shape[1]+extend_fov_u*2, 3))
    
    for star in star_list:
        u_idx = int(np.round(star.u+offset_u+extend_fov_u))
        v_idx = int(np.round(star.v+offset_v+extend_fov_v))
        
        if not star.use_for_model:
            color = (1,0,0)
        else:
            if star.photometry_confidence:
                color = (1,1,0)
            else:
                color = (.5,.5,1)
        draw_circle(overlay, u_idx, v_idx, 1, color, 3)

    return overlay

#===============================================================================
# 
#===============================================================================

def _star_mark_conflicts(obs, star, offset_u, offset_v, margin,
                         extend_fov_u, extend_fov_v):
    logger = logging.getLogger(LOGGING_NAME+'._star_mark_conflicts')

    # Check for off the edge
    if (not (margin-extend_fov_u < star.u+offset_u <
             obs.data.shape[1]+extend_fov_u-margin) or
        not (margin-extend_fov_v < star.v+offset_v <
             obs.data.shape[0]+extend_fov_v-margin)):
        logger.debug('Star %9d U %8.3f V %8.f is off the edge',
                     star.unique_number, star.u, star.v)
        star.use_for_model = False
        star.bad_reason = 'EDGE'
        return True
        
    # Give 3 pixels of slop on each side - we don't want a star to
    # even be close to a large object.
    star_slop = 3
    meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                     origin=(star.u+offset_u-star_slop,
                                             star.v+offset_v-star_slop),
                                     limit=(star.u+offset_u+star_slop,
                                            star.v+offset_v+star_slop))
    backplane = oops.Backplane(obs, meshgrid)

    # Check for planet and moons
    for body_name in LARGE_BODY_LIST:
        intercepted = backplane.where_intercepted(body_name)
        if np.any(intercepted.vals):
            logger.debug('Star %9d U %8.3f V %8.3f conflicts with %s',
                         star.unique_number, star.u, star.v, body_name)
            star.use_for_model = False
            star.bad_reason = 'BODY:' + body_name
            return True
    
    # Check for rings
    ring_radius = backplane.ring_radius('saturn:ring').vals.astype('float')
    min_rad = np.min(ring_radius)
    max_rad = np.max(ring_radius)
    if ((min_rad < 136775 and max_rad > 74658) or # C to A rings
        (min_rad < 140550 and max_rad > 139890)): # F ring
        logger.debug('Star %9d U %8.3f V %8.3f conflicts with rings radii '
                     '%.1f to %.1f',
                     star.unique_number, star.u, star.v, min_rad, max_rad)
        star.use_for_model = False
        star.bad_reason = 'RINGS'
        return True
    
#    # Check for bad photometry
#    ret = _star_perform_photometry(obs, star, offset_u, offset_v)
#    if ret is not None:
#        integrated_dn, bkgnd, bkgnd_std = ret
#        min_dn = MIN_DETECTABLE_DN[obs.detector]
#    if ((star.temperature_faked and integrated_dn < min_dn) or
#        (not star.temperature_faked and
#         integrated_dn < star.dn/2)):
#        logger.debug('Star %9d U %8.3f V %8.3f INTDN %.2f less than minimum',
#                     star.unique_number, star.u, star.v, integrated_dn)
#        star.use_for_model = False
#        star.bad_reason = 'PHOTOMETRY:LOW BRIGHTNESS'
#        something_conflicted = True
#    if integrated_dn < bkgnd_std*3:
#        logger.debug('Star %9d U %8.3f V %8.3f INTDN %.2f BKGNDSTD %.2f '
#                     'background too noisy',
#                     star.unique_number, star.u, star.v, integrated_dn,
#                     bkgnd_std)
#        star.use_for_model = False
#        star.bad_reason = 'PHOTOMETRY:BKGNDSTD'
#        something_conflicted = True
#        if bkgnd_std > min_dn/2:
#            logger.debug('Star %9d U %8.3f V %8.3f INTDN %.2f BKGNDSTD %.2f '
#                         'background too noisy',
#                         star.unique_number, star.u, star.v, integrated_dn,
#                         bkgnd_std)
#            star.use_for_model = False
#            star.bad_reason = 'PHOTOMETRY:BKGNDSTD'
#            return True

    return False

def _star_find_offset(obs, filtered_data, star_list, margin, min_stars,
                      extend_fov_u, extend_fov_v):
    # 1) Find an offset
    # 2) Remove any stars that are on top of a moon, planet, or opaque part of
    #    the rings
    # 3) Repeat until convergence

    logger = logging.getLogger(LOGGING_NAME+'._star_find_offset')

    search_size_max_u, search_size_max_v = MAX_POINTING_ERROR[obs.detector]
    
    for star in star_list:
        star.use_for_model = True
        star.bad_reason = None
        
    pass_number = 1
    while True:
        # Create a model for the current star list
        new_star_list = []
        for star in star_list:
            if star.use_for_model:
                new_star_list.append(star)

        logger.debug('Pass %d # stars %d', pass_number, len(new_star_list))
                
        if len(new_star_list) < min_stars:
            logger.debug('Fewer than %d stars left (%d)', min_stars,
                         len(new_star_list))
            return None
        
        model = star_create_model(obs, new_star_list,
                                  extend_fov_u=extend_fov_u,
                                  extend_fov_v=extend_fov_v)

        # Find the best offset using the current star list.
        # Then look to see if any of the stars correspond to known
        # objects like moons, planets, or opaque parts of the ring.
        # If so, get rid of those stars and iterate.
        
        offset_u, offset_v, peak = find_correlation_and_offset(
                        filtered_data, model, search_size_min=0,
                        search_size_max=(search_size_max_u, search_size_max_v))
        
        logger.debug('Trial offset U,V %d %d', offset_u, offset_v)
        
        something_conflicted = False
        
        for star in new_star_list:
            res = _star_mark_conflicts(obs, star, offset_u, offset_v, margin,
                                       extend_fov_u, extend_fov_v)
            something_conflicted = something_conflicted or res

        if not something_conflicted:
            break

        pass_number += 1

#    if pass_number == 1:
        # If we exited on pass 1, then there were no conflicts at all
    new_offset_u = offset_u
    new_offset_v = offset_v
    logger.debug('No conflicts - final offset U,V %d %d',
                 offset_u, offset_v)
#    else:
#        # Otherwise we have to redo everything to be safe
#        logger.debug('No conflicts - last trial offset U,V %d %d - '
#                     'starting final pass', offset_u, offset_v)
#    
#        # Now reset everything, get rid of stars that conflict with objects,
#        # and recompute the offset. It had better be about the same!
#            
#        for star in star_list:
#            star.use_for_model = True
#            star.bad_reason = None
#        
#        new_star_list = []
#        for star in star_list:
#            _star_mark_conflicts(obs, star, offset_u, offset_v, margin)
#            if star.use_for_model:
#                new_star_list.append(star)
#            
#        if len(new_star_list) < min_stars:
#            logger.warn('Somehow the reset list has fewer than %d stars! (%d)',
#                        min_stars, len(new_star_list))
#            return None
#    
#        model = star_create_model(obs, new_star_list,
#                                  extend_fov_u=search_size_max_u,
#                                  extend_fov_v=search_size_max_v)
#    
#        new_offset_u, new_offset_v, peak = find_correlation_and_offset(
#                        data, model, search_size_min=0,
#                        search_size_max=(search_size_max_u, search_size_max_v))
#    
#        logger.debug('Revised offset U,V %d %d', new_offset_u, new_offset_v)
#        
#        if abs(offset_u-new_offset_u) > 1 or abs(offset_v-new_offset_v) > 1:
#            # Unstable result
#            logger.warn('The reset list has an offset that differs by more than 1')
#            return None

    return new_offset_u, new_offset_v

def star_find_offset(obs, ext_data=None, min_stars=3, psf_size=9,
                     extend_fov_u=0, extend_fov_v=0):
    logger = logging.getLogger(LOGGING_NAME+'._star_find_offset')

    orig_star_list = star_list_for_obs(obs, max_stars=30, 
                                       extend_fov_u=extend_fov_u,
                                       extend_fov_v=extend_fov_v)
    if len(orig_star_list) < min_stars:
        # There's no point in continuing if there aren't enough stars
        return None, None, orig_star_list, 0

    if orig_star_list[0].dn < MIN_DETECTABLE_DN[obs.detector]:
        # There's no point in continuing if the brightest star is below the
        # detection threshold
        return None, None, [], 0
 
    if ext_data is None:
        ext_data = pad_image(obs.data, extend_fov_u, extend_fov_v)

    margin = psf_size // 2
    
    star_list = orig_star_list[:] # Copy
    
    filtered_data = filter_local_maximum(ext_data, area_size=7)

    filtered_data[filtered_data < 0.] = 0.
    
    max_dn = orig_star_list[0].dn # Star list is sorted by DN
    
    mask = filtered_data > max_dn
    
    mask = filt.maximum_filter(mask, 11)
    filtered_data[mask] = 0.

    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
    offset_u = None
    offset_v = None
    good_stars = 0
    confidence = 0.
    
    # Try with the brightest stars, then go down to the dimmest
    for min_dn_gain in [4., 2.5, 1.]:
        logger.debug('Trying DN gain %.1f', min_dn_gain)
        
        new_star_list = []
        for star in star_list:
            if star.dn < min_dn*min_dn_gain:
                continue
            new_star_list.append(star)
            
        if len(new_star_list) < min_stars:
            logger.debug('Not enough stars: %d (%d required)',
                         len(new_star_list), min_stars)
            continue
        
        ret = _star_find_offset(obs, filtered_data, new_star_list, margin, 
                                min_stars, extend_fov_u, extend_fov_v) 

        if ret is None:
            logger.debug('No valid offset found - iterating')
            continue
        
        offset_u, offset_v = ret
        logger.debug('Found valid offset U,V %d %d', offset_u, offset_v)
        good_stars = star_perform_photometry(obs,
                                             new_star_list,
                                             offset_u=offset_u,
                                             offset_v=offset_v)
        logger.debug('Photometry found %d good stars', good_stars)
        if good_stars < min_stars:
            logger.debug('Insufficient good stars')
            offset_u = None
            if len(new_star_list) == len(star_list):
                # No point in trying smaller DNs if we already are looking at
                # all stars
                break
            continue
    if offset_u is None:
        good_stars = 0
        for star in new_star_list:
            star.photometry_confidence = 0.
        logger.debug('FAILED to find a valid offset')
    else:
        logger.debug('Returning final offset U,V %d %d / Good stars %d',
                     offset_u, offset_v, good_stars)
            
    return offset_u, offset_v, new_star_list, good_stars


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
        
