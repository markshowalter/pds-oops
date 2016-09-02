###############################################################################
# cb_stars.py
#
# Routines related to stars.
#
# Exported routines:
#    stars_list_for_obs
#    stars_create_model
#    stars_perform_photometry
#    stars_make_good_bad_overlay
#    stars_find_offset
###############################################################################

import cb_logging
import logging

import copy
import time

import numpy as np
import numpy.ma as ma
import polymath
import scipy.ndimage.filters as filt
from PIL import Image, ImageDraw, ImageFont

import oops
from polymath import *
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
DEBUG_STARS_MODEL_IMGDISP = False

STAR_CATALOG = UCAC4StarCatalog(STAR_CATALOG_ROOT)

    
#===============================================================================
#
# FIND STARS IN THE FOV AND RETURN THEM.
# 
#===============================================================================

def _aberrate_star(obs, star):
    """Update the RA,DEC position of a star with stellar aberration."""
    event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                     polymath.Vector3.ZERO),
                       obs.path, obs.frame)

    event.neg_arr_j2000 = polymath.Vector3.from_ra_dec_length(
                              star.ra, star.dec, 1., recursive=False)
    (abb_ra, abb_dec, _) = event.neg_arr_ap_j2000.to_ra_dec_length(
                                                     recursive=False)

    star.ra = abb_ra.vals
    star.dec = abb_dec.vals

def _compute_dimmest_visible_star_vmag(obs, stars_config):
    """Compute the VMAG of the dimmest star likely visible."""
    min_dn = stars_config[('min_detectable_dn', obs_detector(obs))] # photons / pixel
    if min_dn == 0:
        return 1000
    fake_star = Star()
    fake_star.temperature = SCLASS_TO_SURFACE_TEMP['G0']
    min_mag = stars_config['min_vmag']
    max_mag = stars_config['max_vmag']
    mag_increment = stars_config['vmag_increment']
    for mag in np.arange(min_mag, max_mag+1e-6, mag_increment):
        fake_star.unique_number = 0
        fake_star.johnson_mag_v = mag
        fake_star.johnson_mag_b = mag+SCLASS_TO_B_MINUS_V['G0']
        dn = compute_dn_from_star(obs, fake_star)
        if dn < min_dn:
            return mag # This is conservative
    return mag

def _stars_list_for_obs(obs, ra_min, ra_max, dec_min, dec_max,
                        mag_min, mag_max, radec_movement,
                        extend_fov, stars_config, **kwargs):
    """Return a list of stars with the given constraints.
    
    See stars_list_for_obs for full details."""
    logger = logging.getLogger(_LOGGING_NAME+'._stars_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = stars_config[('min_detectable_dn', obs_detector(obs))]
    
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
        
    default_star_class = stars_config['default_star_class']
    
    star_list = []
    for star in orig_star_list:
        star.conflicts = None
        star.temperature_faked = False
        star.integrated_dn = 0.
        star.overlay_box_width = 0
        star.overlay_box_thickness = 0
        if star.temperature is None:
            star.temperature_faked = True
            star.temperature = SCLASS_TO_SURFACE_TEMP[default_star_class]
            star.spectral_class = default_star_class
            star.johnson_mag_v = (star.vmag-
                          SCLASS_TO_B_MINUS_V[default_star_class]/2.)
            star.johnson_mag_b = (star.vmag-
                          SCLASS_TO_B_MINUS_V[default_star_class]/2.)
        if min_dn == 0:
            star.dn = 1000-star.vmag
        else:
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
    ra_list = Scalar([x[0] for x in ra_dec_list])
    dec_list = Scalar([x[1] for x in ra_dec_list])
    
    uv = obs.uv_from_ra_and_dec(ra_list, dec_list, apparent=True)
    u_list, v_list = uv.to_scalars()
    u_list = u_list.vals
    v_list = v_list.vals

    if radec_movement is not None:
        uv1 = obs.uv_from_ra_and_dec(ra_list-radec_movement[0],
                                     dec_list-radec_movement[1], 
                                     apparent=True)
    else:
        uv1 = obs.uv_from_ra_and_dec(ra_list, dec_list, time_frac=0., 
                                     apparent=True)
    u1_list, v1_list = uv1.to_scalars()
    u1_list = u1_list.vals
    v1_list = v1_list.vals

    if radec_movement is not None:
        uv2 = obs.uv_from_ra_and_dec(ra_list+radec_movement[0],
                                     dec_list+radec_movement[1], 
                                     apparent=True)
    else:
        uv2 = obs.uv_from_ra_and_dec(ra_list, dec_list, time_frac=1., 
                                     apparent=True)
    u2_list, v2_list = uv2.to_scalars()
    u2_list = u2_list.vals
    v2_list = v2_list.vals

    new_star_list = []

    discard_uv = 0        
    for star, u, v, u1, v1, u2, v2 in zip(
                          star_list, 
                          u_list, v_list,
                          u1_list, v1_list,
                          u2_list, v2_list):
        if (u < -extend_fov[0] or u > obs.data.shape[1]+extend_fov[0]-1 or
            v < -extend_fov[1] or v > obs.data.shape[0]+extend_fov[1]-1):
            discard_uv += 1
            continue
        
        star.u = u
        star.v = v                
        star.move_u = u2-u1
        star.move_v = v2-v1

        new_star_list.append(star)

    logger.debug(
        'Found %d stars, discarded because of CLASS %d, LOW DN %d, BAD UV %d',
        len(orig_star_list), discard_class, discard_dn, discard_uv)

    return new_star_list

def stars_list_for_obs(obs, radec_movement,
                       extend_fov=(0,0), stars_config=None,
                      **kwargs):
    """Return a list of stars in the FOV of the obs.

    Inputs:
        obs                The observation.
        radec_movement     A tuple (dra,ddec) that gives the movement of the
                           camera in each half of the exposure. None if
                           no movement is available.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to return stars (a star's U/V value will be negative
                           or greater than the FOV shape).
        stars_config        Configuration parameters.
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
    logger = logging.getLogger(_LOGGING_NAME+'.stars_list_for_obs')

    if stars_config is None:
        stars_config = STARS_DEFAULT_CONFIG
        
    max_stars = stars_config['max_stars']
    
    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs,
                                             extend_fov=extend_fov)

    # Try to be efficient by limiting the magnitudes searched so we don't
    # return a huge number of dimmer stars and then only need a few of them.
    magnitude_list = [0., 12., 13., 14., 15., 16., 17.]
    
    mag_vmax = _compute_dimmest_visible_star_vmag(obs, stars_config)+1
    logger.debug('Max detectable VMAG %.4f', mag_vmax)
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        if mag_min > mag_vmax:
            break
        mag_max = min(mag_max, mag_vmax)
        
        star_list = _stars_list_for_obs(obs,
                                        ra_min, ra_max, dec_min, dec_max,
                                        mag_min, mag_max,
                                        radec_movement,
                                        extend_fov, stars_config, **kwargs)
        full_star_list += star_list
        
        logger.debug('Got %d stars, total %d', len(star_list),
                     len(full_star_list))
        
        if len(full_star_list) >= max_stars:
            break

    # Sort the list with the brightest stars first.
    full_star_list.sort(key=lambda x: x.dn, reverse=True)
                
    full_star_list = full_star_list[:max_stars]

    logger.info('Star list:')
    for star in full_star_list:
        logger.info('Star %9d U %8.3f+%7.3f V %8.3f+%7.3f DN %7.2f MAG %6.3f BMAG %6.3f '+
                    'VMAG %6.3f SCLASS %3s TEMP %6d',
                    star.unique_number, 
                    star.u, abs(star.move_u), 
                    star.v, abs(star.move_v),
                    star.dn, star.vmag,
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
    
def stars_create_model(obs, star_list, offset=None, ignore_conflicts=False,
                       extend_fov=(0,0), stars_config=None):
    """Create a model containing nothing but stars.
    
    Individual stars are modeled using a Gaussian PSF with a sigma based on
    the ISS camera. If the complete PSF doesn't fit in the image, the star
    is ignored.
    
    Inputs:
        obs                The observation.
        star_list          The list of Stars.
        offset             The amount to offset a star's position in the (U,V)
                           directions.
        extend_fov         The amount to extend the model beyond the limits of
                           the obs FOV. The returned model will be the shape of
                           the obs FOV plus two times the extend value in each
                           dimension.
        stars_config        Configuration parameters.
        
    Returns:
        model              The model.
    """
    if stars_config is None:
        stars_config = STARS_DEFAULT_CONFIG
        
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset

    psf_size = stars_config['min_psf_size']
    max_move_steps = stars_config['max_movement_steps']
    
    model = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                      obs.data.shape[1]+extend_fov[0]*2),
                     dtype=np.float32)
    
    for star in star_list:
        if ignore_conflicts and star.conflicts:
            continue
        u_idx = star.u+offset_u+extend_fov[0]
        v_idx = star.v+offset_v+extend_fov[1]
        u_int = int(u_idx)
        v_int = int(v_idx)
        u_frac = u_idx-u_int
        v_frac = v_idx-v_int
        
        psf_size_half_u = (psf_size + np.round(abs(star.move_u))) // 2
        psf_size_half_v = (psf_size + np.round(abs(star.move_v))) // 2

        move_gran = max(abs(star.move_u)/max_move_steps,
                        abs(star.move_v)/max_move_steps)
        move_gran = np.clip(move_gran, 0.1, 1.0)
        
        gausspsf = GaussianPSF(sigma=PSF_SIGMA[obs_detector(obs)],
                               movement=(star.move_v,star.move_u),
                               movement_granularity=move_gran)
        
        if (u_int < psf_size_half_u or u_int >= model.shape[1]-psf_size_half_u or
            v_int < psf_size_half_v or v_int >= model.shape[0]-psf_size_half_v):
            continue

        psf = gausspsf.eval_rect((psf_size_half_v*2+1,psf_size_half_u*2+1),
                                 offset=(v_frac,u_frac),
                                 scale=star.dn)
        model[v_int-psf_size_half_v:v_int+psf_size_half_v+1, 
              u_int-psf_size_half_u:u_int+psf_size_half_u+1] += psf

    if DEBUG_STARS_MODEL_IMGDISP:
        imdisp = ImageDisp([model],
                           canvas_size=(1024,1024),
                           allow_enlarge=True, enlarge_limit=10,
                           auto_update=True)
        Tkinter.mainloop()
        
    return model

def stars_make_good_bad_overlay(obs, star_list, offset,
                                extend_fov=(0,0),
                                show_streaks=False,
                                label_avoid_mask=None,
                                stars_config=None):
    """Create an overlay with high and low confidence stars marked.
    
    Inputs:
        obs                    The observation.
        star_list              The list of Star objects.
        offset                 The amount to offset a star's position in the 
                               (U,V) directions.
        extend_fov             The amount to extend the overlay beyond the 
                               limits of the obs FOV. The returned model will 
                               be the shape of the obs FOV plus two times the 
                               extend value in each dimension.
        show_streaks           If True, draw the streak from the star's PSF in
                               addition to the box or circle.
        label_avoid_mask       A mask giving places where text labels should
                               not be placed (i.e. labels from another
                               model are already there). None if no mask.
        stars_config           Configuration parameters.
                           
    Returns:
        overlay                The overlay.
        
        Star excluded by brightness or conflict: circle
        Star bad photometry: thin square
        Star good photometry: thick square
    """
    if stars_config is None:
        stars_config = STARS_DEFAULT_CONFIG
    
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset
        
    overlay = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                        obs.data.shape[1]+extend_fov[0]*2),
                       dtype=np.uint8)
    text = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                     obs.data.shape[1]+extend_fov[0]*2),
                    dtype=np.bool)
    text_im = Image.frombuffer('L', (text.shape[1], text.shape[0]), text,
                               'raw', 'L', 0, 1)
    text_draw = ImageDraw.Draw(text_im)
#    font = ImageFont('')
    if show_streaks:
        psf_size = stars_config['min_psf_size']
        max_move_steps = stars_config['max_movement_steps']
        
        for star in star_list:
            u_idx = star.u+offset_u+extend_fov[0]
            v_idx = star.v+offset_v+extend_fov[1]
            u_int = int(u_idx)
            v_int = int(v_idx)
            u_frac = u_idx-u_int
            v_frac = v_idx-v_int
                
            psf_size_half_u = (psf_size + np.round(abs(star.move_u))) // 2
            psf_size_half_v = (psf_size + np.round(abs(star.move_v))) // 2
    
            move_gran = max(abs(star.move_u)/max_move_steps,
                            abs(star.move_v)/max_move_steps)
            move_gran = np.clip(move_gran, 0.1, 1.0)
            
            gausspsf = GaussianPSF(sigma=PSF_SIGMA[obs_detector(obs)],
                                   movement=(star.move_v,star.move_u),
                                   movement_granularity=move_gran)
            
            if (u_int < psf_size_half_u or u_int >= overlay.shape[1]-psf_size_half_u or
                v_int < psf_size_half_v or v_int >= overlay.shape[0]-psf_size_half_v):
                continue
    
            psf = gausspsf.eval_rect((psf_size_half_v*2+1,psf_size_half_u*2+1),
                                     offset=(v_frac,u_frac),
                                     scale=1.)
            psf = psf / np.max(psf) * 255
            psf = psf.astype('uint8')
            
            overlay[v_int-psf_size_half_v:v_int+psf_size_half_v+1, 
                    u_int-psf_size_half_u:u_int+psf_size_half_u+1] += psf

    # First go through and draw the circles and squares. We have to put the
    # squares in the right place. There's no way to avoid anything in
    # the label_avoid_mask.
    for star in star_list:
        # Should NOT be rounded for plotting, since all of coord
        # X to X+0.9999 is the same pixel
        u_idx = int(star.u+offset_u+extend_fov[0])
        v_idx = int(star.v+offset_v+extend_fov[1])
        star.overlay_box_width = 0
        star.overlay_box_thickness = 0
        if (not star.is_bright_enough or not star.is_dim_enough or
            star.conflicts):
            width = 3
            if (width < u_idx < overlay.shape[1]-width and
                width < v_idx < overlay.shape[0]-width):
                star.overlay_box_width = width
                star.overlay_box_thickness = 1
                draw_circle(overlay, u_idx, v_idx, width, 255)
        else:
            if star.integrated_dn == 0:
                width = 3
            else:
                width = star.photometry_box_size // 2 + 1
            thickness = 1
            if star.photometry_confidence > stars_config['min_confidence']:
                thickness = 3
            if (width+thickness-1 <= u_idx < 
                overlay.shape[1]-width-thickness+1 and
                width+thickness-1 <= v_idx < 
                overlay.shape[0]-width-thickness+1):
                star.overlay_box_width = width
                star.overlay_box_thickness = thickness
                draw_rect(overlay, u_idx, v_idx, 
                          width, width, 255, thickness)
    
    # Now go through a second time to do the text labels. This way the labels
    # can avoid overlapping with the squares.
    for star in star_list:
        # Should NOT be rounded for plotting, since all of coord
        # X to X+0.9999 is the same pixel
        u_idx = int(star.u+offset_u+extend_fov[0])
        v_idx = int(star.v+offset_v+extend_fov[1])
    
        width = star.overlay_box_width
        thickness = star.overlay_box_thickness
        if width == 0:
            continue

        width += thickness-1
        if (not width <= u_idx < overlay.shape[1]-width or
            not width <= v_idx < overlay.shape[0]-width):
            continue

        star_str1 = '%09d' % (star.unique_number)
        star_str2 = '%.3f %s' % (star.vmag,
            'XX' if star.spectral_class is None else star.spectral_class)
        text_size = text_draw.textsize(star_str1)

        locations = []
        v_text = v_idx-text_size[1] # Whole size because we're doing two lines
        if u_idx >= overlay.shape[1]//2:
            # Star is on right side of image - default to label on left
            if v_text+text_size[1]*2 < overlay.shape[0]:
                locations.append((u_idx-width-6-text_size[0], v_text))
            if v_text >= 0:
                locations.append((u_idx+width+6, v_text))
        else:
            # Star is on left side of image - default to label on right
            if v_text >= 0:
                locations.append((u_idx+width+6, v_text))
            if v_text+text_size[1]*2 < overlay.shape[0]:
                locations.append((u_idx-width-6-text_size[0], v_text))
        # Next try below star
        u_text = u_idx-text_size[0]//2
        v_text = v_idx+width+6
        if v_text+text_size[1]*2 < overlay.shape[0]:
            locations.append((u_text, v_text))
        # And above the star
        v_text = v_idx-width-3-text_size[1]*2
        if v_text >= 0:
            locations.append((u_text, v_text))
        # One last gasp effort...try a little further above or below
        u_text = u_idx-text_size[0]//2
        v_text = v_idx+width+12
        if v_text+text_size[1]*2 < overlay.shape[0]:
            locations.append((u_text, v_text))
        v_text = v_idx-width-12-text_size[1]
        if v_text >= 0:
            locations.append((u_text, v_text))
        # And to the side but further above
        v_text = v_idx-text_size[1]-6
        if v_text+text_size[1]*2 < overlay.shape[0]:
            locations.append((u_idx-width-6-text_size[0], v_text))
        if v_text >= 0:
            locations.append((u_idx+width+6, v_text))
        # And to the side but further below
        v_text = v_idx-text_size[1]+6
        if v_text+text_size[1]*2 < overlay.shape[0]:
            locations.append((u_idx-width-6-text_size[0], v_text))
        if v_text >= 0:
            locations.append((u_idx+width+6, v_text))
        
        good_u = None
        good_v = None
        preferred_u = None
        preferred_v = None
        text = np.array(text_im.getdata()).reshape(text.shape)
        for u, v in locations:
            if (not np.any(
               text[
                       max(v-3,0):
                       min(v+text_size[1]*2+3, overlay.shape[0]),
                       max(u-3,0):
                       min(u+text_size[0]+3, overlay.shape[1])]) and
                (label_avoid_mask is None or 
                 not np.any(
               label_avoid_mask[
                       max(v-3,0):
                       min(v+text_size[1]*2+3, overlay.shape[0]),
                       max(u-3,0):
                       min(u+text_size[0]+3, overlay.shape[1])]))):
                if good_u is None:
                    # Give precedence to earlier choices - they're prettier
                    good_u = u
                    good_v = v
                # But we'd really rather the text not overlap with the squares
                # either, if possible
                if (preferred_u is None and not np.any(
                   overlay[max(v-3,0):
                           min(v+text_size[1]*2+3, overlay.shape[0]),
                           max(u-3,0):
                           min(u+text_size[0]+3, overlay.shape[1])])):
                    preferred_u = u
                    preferred_v = v
                    break
        
        if preferred_u is not None:
            good_u = preferred_u
            good_v = preferred_v
            
        if good_u is not None:
            text_draw.text((good_u,good_v), star_str1, 
                           fill=1)
            text_draw.text((good_u,good_v+text_size[1]), star_str2, 
                           fill=1)
    
    text = np.array(text_im.getdata()).astype('bool').reshape(text.shape)
    
    return overlay, text

#===============================================================================
#
# PERFORM PHOTOMETRY.
# 
#===============================================================================

def _trust_star_dn(obs):
    return obs.filter1 == 'CL1' and obs.filter2 == 'CL2'

#def _stars_perform_photometry(obs, calib_data, star, offset,
#                              extend_fov, stars_config):
#    """Perform photometry on a single star.
#    
#    See star_perform_photometry for full details.
#    """        
#    # calib_data is calibrated in DN and extended
#    u = int(np.round(star.u)) + offset[0] + extend_fov[0]
#    v = int(np.round(star.v)) + offset[1] + extend_fov[1]
#    
#    if star.dn > stars_config['photometry_boxsize_1'][0]:
#        boxsize = stars_config['photometry_boxsize_1'][1]
#    elif star.dn > stars_config['photometry_boxsize_2'][0]:
#        boxsize = stars_config['photometry_boxsize_2'][1]
#    else:
#        boxsize = stars_config['photometry_boxsize_default']
#    
#    star.photometry_box_width = boxsize
#    
#    box_halfsize = boxsize // 2
#
#    # Don't process stars that are off the edge of the real (not extended)
#    # data.
#    if (u-extend_fov[0] < box_halfsize or
#        u-extend_fov[0] > calib_data.shape[1]-2*extend_fov[0]-box_halfsize-1 or
#        v-extend_fov[1] < box_halfsize or
#        v-extend_fov[1] > calib_data.shape[0]-2*extend_fov[1]-box_halfsize-1):
#        return None
#        
#    subimage = calib_data[v-box_halfsize:v+box_halfsize+1,
#                          u-box_halfsize:u+box_halfsize+1]
#    subimage = subimage.view(ma.MaskedArray)
#    subimage[1:-1, 1:-1] = ma.masked # Mask out the center
#    
#    bkgnd = ma.mean(subimage)
#    bkgnd_std = ma.std(subimage)
#    
#    subimage.mask = ~subimage.mask # Mask out the edge
#    integrated_dn = np.sum(subimage-bkgnd)
#    integrated_std = np.std(subimage-bkgnd)
#
#    return integrated_dn, bkgnd, integrated_std, bkgnd_std

def _stars_perform_photometry(obs, calib_data, star, offset,
                              extend_fov, stars_config):
    """Perform photometry on a single star.
    
    See star_perform_photometry for full details.
    """        
    # calib_data is calibrated in DN and extended
    u = int(np.round(star.u)) + offset[0] + extend_fov[0]
    v = int(np.round(star.v)) + offset[1] + extend_fov[1]

    if star.dn > stars_config['photometry_boxsize_1'][0]:
        boxsize = stars_config['photometry_boxsize_1'][1]
    elif star.dn > stars_config['photometry_boxsize_2'][0]:
        boxsize = stars_config['photometry_boxsize_2'][1]
    else:
        boxsize = stars_config['photometry_boxsize_default']

    star.photometry_box_size = boxsize

    psf_size_half_u = (boxsize + np.round(abs(star.move_u))) // 2
    psf_size_half_v = (boxsize + np.round(abs(star.move_v))) // 2

    # Don't process stars that are off the edge of the real (not extended)
    # data.
    if (u-extend_fov[0] < psf_size_half_u or
        u-extend_fov[0] > calib_data.shape[1]-2*extend_fov[0]-psf_size_half_u-1 or
        v-extend_fov[1] < psf_size_half_v or
        v-extend_fov[1] > calib_data.shape[0]-2*extend_fov[1]-psf_size_half_v-1):
        return None

    max_move_steps = stars_config['max_movement_steps']
    move_gran = max(abs(star.move_u)/max_move_steps,
                    abs(star.move_v)/max_move_steps)
    move_gran = np.clip(move_gran, 0.1, 1.0)
    
    gausspsf = GaussianPSF(sigma=PSF_SIGMA[obs_detector(obs)],
                           movement=(star.move_v,star.move_u),
                           movement_granularity=move_gran)
        
    psf = gausspsf.eval_rect((psf_size_half_v*2+1,psf_size_half_u*2+1),
                             offset=(0.5,0.5))

    center_u = psf.shape[1] // 2
    center_v = psf.shape[0] // 2

    subpsf = psf[center_v-boxsize//2+1:center_v+boxsize//2,
                 center_u-boxsize//2+1:center_u+boxsize//2]

    min_allowed_val = np.min(subpsf)

    subpsf = psf[center_v-boxsize//2:center_v+boxsize//2+1,
                 center_u-boxsize//2:center_u+boxsize//2+1]

    min_bkgnd_val = np.min(subpsf)
    
    subimage = calib_data[v-psf_size_half_v:v+psf_size_half_v+1,
                          u-psf_size_half_u:u+psf_size_half_u+1]

    streak_bool = psf >= min_allowed_val
    streak_data = subimage[streak_bool]
    
    bkgnd_bool = np.logical_and(filt.maximum_filter(streak_bool, 3),
                                np.logical_not(streak_bool))
    bkgnd_data = subimage[bkgnd_bool]
    
    bkgnd = ma.mean(bkgnd_data)
    bkgnd_std = ma.std(bkgnd_data)
    
    integrated_dn = np.sum(streak_data-bkgnd)
    integrated_std = np.std(streak_data-bkgnd)

    return integrated_dn, bkgnd, integrated_std, bkgnd_std

def stars_perform_photometry(obs, calib_data, star_list, offset=None,
                             extend_fov=(0,0), stars_config=None):
    """Perform photometry on a list of stars.
    
    Inputs:
        obs                The observation.
        calib_data         obs.data calibrated as DN.
        star_list          The list of Star objects.
        offset             The amount to offset a star's position in the (U,V)
                           directions.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to perform photometry.
        stars_config        Configuration parameters.
    
    Returns:
        good_stars         The number of good stars.
                           (star.photometry_confidence > STARS_MIN_CONFIDENCE)
        
        Each Star is populated with:
        
            .integrated_dn            The actual integrated dn measured.
            .photometry_confidence    The confidence in the result. Currently
                                      adds 0.5 for a non-noisy background and
                                      adds 0.5 for the DN within range. 
    """
    logger = logging.getLogger(_LOGGING_NAME+'.stars_perform_photometry')

    if stars_config is None:
        stars_config = STARS_DEFAULT_CONFIG
        
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset
        
    image = obs.data
    min_dn = stars_config[('min_detectable_dn', obs_detector(obs))]
    
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
        ret = _stars_perform_photometry(obs, calib_data, star,
                                        (offset_u, offset_v), extend_fov,
                                        stars_config)
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
        if star.photometry_confidence >= stars_config['min_confidence']:
            good_stars += 1

    return good_stars

#===============================================================================
# 
# FIND THE IMAGE OFFSET BASED ON STARS.
#
#===============================================================================

def _stars_mark_conflicts(obs, star, offset, rings_can_conflict, stars_config):
    """Check if a star conflicts with known bodies or rings.
    
    Sets star.conflicts to a string describing why the Star conflicted.
    
    Returns True if the star conflicted, False if the star didn't.
    """
    logger = logging.getLogger(_LOGGING_NAME+'._stars_mark_conflicts')

    # Check for off the edge
    psf_size = stars_config['min_psf_size']
    psf_size_half_u = (psf_size + np.round(abs(star.move_u))) // 2
    psf_size_half_v = (psf_size + np.round(abs(star.move_v))) // 2
    
    if (star.u+offset[0] < psf_size_half_u or 
        star.u+offset[0] >= obs.data.shape[1]-psf_size_half_u or
        star.v+offset[1] < psf_size_half_v or 
        star.v+offset[1] >= obs.data.shape[0]-psf_size_half_v):
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
        star_slop = stars_config['star_body_conflict_margin']
        meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(star.u+offset[0]-star_slop,
                                                 star.v+offset[1]-star_slop),
                                         limit =(star.u+offset[0]+star_slop,
                                                 star.v+offset[1]+star_slop))
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
    if rings_can_conflict:
        ring_radius = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
        ring_longitude = (obs.ext_bp.ring_longitude('saturn:ring').vals.
                          astype('float'))
        rad = ring_radius[star.v+obs.extend_fov[1], star.u+obs.extend_fov[0]]
        long = ring_longitude[star.v+obs.extend_fov[1], 
                              star.u+obs.extend_fov[0]]
    
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
        

def _stars_optimize_offset_list(offset_list, tolerance=1):
    """Remove bad offsets.
    
    A bad offset is defined as an offset that makes a line with two other
    offsets in the list. We remove these because when 2-D correlation is
    performed on certain times of images, there is a 'line' of correlation
    peaks through the image, none of which are actually correct. When we
    are finding a limited number of peaks, they all get eaten up by this
    line and we never look elsewhere.
    """
    logger = logging.getLogger(_LOGGING_NAME+'._stars_optimize_offset_list')

    mark_for_deletion = [False] * len(offset_list)
    for idx1 in xrange(len(offset_list)-2):
        for idx2 in xrange(idx1+1,len(offset_list)-1):
            for idx3 in xrange(idx2+1,len(offset_list)):
                u1 = offset_list[idx1][0][0]
                u2 = offset_list[idx2][0][0]
                u3 = offset_list[idx3][0][0]
                v1 = offset_list[idx1][0][1]
                v2 = offset_list[idx2][0][1]
                v3 = offset_list[idx3][0][1]
                if (u1 is None or u2 is None or u3 is None or
                    v1 is None or v2 is None or v3 is None):
                    continue
                if u1 == u2: # Vertical line
                    if abs(u3-u1) <= tolerance:
#                         logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) '+
#                                      'in a line',
#                                      idx1+1, u1, v1, 
#                                      idx2+1, u2, v2,
#                                      idx3+1, u3, v3)
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
#                         logger.debug('Points %d (%d,%d) %d (%d,%d) %d (%d,%d) '+
#                                      'in a line',
#                                      idx1+1, u1, v1,
#                                      idx2+1, u2, v2,
#                                      idx3+1, u3, v3)
                        mark_for_deletion[idx2] = True
                        mark_for_deletion[idx3] = True
    new_offset_list = []
    for i in xrange(len(offset_list)):
        if not mark_for_deletion[i]:
            new_offset_list.append(offset_list[i])
    
    return new_offset_list

def _stars_find_offset(obs, filtered_data, star_list, min_stars,
                       search_multiplier, max_offsets, already_tried,
                       debug_level, perform_photometry,
                       rings_can_conflict, 
                       radec_movement, stars_config):
    """Internal helper for stars_find_offset so the loops don't get too deep.
    
    Returns:
        (offset, good_stars, corr, keep_searching, no_peaks)
        
        offset            The offset if found, otherwise None.
        good_stars        If the offset is found, the number of good stars.
        corr              The correlation value for this offset.
        keep_searching    Even if an offset was found, we don't entirely
                          trust it, so add it to the list and keep searching.
        no_peaks          The correlation utterly failed and there are no
                          peaks. There's no point in continuing to look.
    """
    # 1) Find an offset
    # 2) Remove any stars that are on top of a moon, planet, or opaque part of
    #    the rings
    # 3) Repeat until convergence

    logger = logging.getLogger(_LOGGING_NAME+'._stars_find_offset')

    min_brightness_guaranteed_vis = stars_config[
                                        'min_brightness_guaranteed_vis']
    min_confidence = stars_config['min_confidence']
    
    # Restrict the search size    
    search_size_max_u, search_size_max_v = MAX_POINTING_ERROR[obs.data.shape, 
                                                              obs_detector(obs)]
    search_size_max_u = int(search_size_max_u*search_multiplier)
    search_size_max_v = int(search_size_max_v*search_multiplier)
    
    for star in star_list:
        star.conflicts = None

    peak_margin = 3 # Amount on each side of a correlation peak to black out
    # Make sure we have peaks that can cover 2 complete "lines" in the
    # correlation
    if perform_photometry:
        trial_max_offsets = (max(2*search_size_max_u+1, 
                                 2*search_size_max_v+1) //
                             (peak_margin*2+1)) + 4
    else:
        # No point in doing more than one offset if we're not going to do
        # photometry
        trial_max_offsets = 1
        
    # Find the best offset using the current star list.
    # Then look to see if any of the stars correspond to known
    # objects like moons, planets, or opaque parts of the ring.
    # If so, get rid of those stars and iterate.
    
    model = stars_create_model(obs, star_list, extend_fov=obs.extend_fov,
                               stars_config=stars_config)

    offset_list = find_correlation_and_offset(
                    filtered_data, model, search_size_min=0,
                    search_size_max=(search_size_max_u, search_size_max_v),
                    max_offsets=trial_max_offsets,
                    extend_fov=obs.extend_fov)

    offset_list = _stars_optimize_offset_list(offset_list)
    offset_list = offset_list[:max_offsets]
    
    new_offset_list = []
    new_peak_list = []
    for i in xrange(len(offset_list)):
        if offset_list[i][0] not in already_tried:
            new_offset_list.append(offset_list[i][0])
            new_peak_list.append(offset_list[i][1])
            # Nobody else gets to try these before we do
            already_tried.append(offset_list[i][0])
        else:
            logger.debug('Offset %d,%d already tried (or reserved)', 
                         offset_list[i][0][0], offset_list[i][0][1])

    if len(new_offset_list):
        logger.debug('Final peak list:')
        for i in xrange(len(new_offset_list)):
            logger.debug('Peak %d U,V %d,%d VAL %f', i+1, 
                         new_offset_list[i][0], new_offset_list[i][1],
                         new_peak_list[i])

    if len(new_offset_list) == 0:
        # No peaks found at all - tell the top-level loop there's no point
        # in trying more
        return None, None, None, False, True
            
    for peak_num in xrange(len(new_offset_list)):
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
        offset = new_offset_list[peak_num]
        peak = new_peak_list[peak_num]

        logger.debug('** LEVEL %d: Peak %d - Trial offset U,V %d,%d', 
                     debug_level, peak_num+1, offset[0], offset[1])

        # First try the star list as given to us.
                    
        for star in star_list:
            star.conflicts = None
        
        something_conflicted = False
        for star in star_list:
            res = _stars_mark_conflicts(obs, star, offset, rings_can_conflict,
                                        stars_config)
            something_conflicted = something_conflicted or res

        if not perform_photometry:
            good_stars = 0
            for star in star_list:
                star.integrated_dn = 0.
                if star.conflicts:
                    star.photometry_confidence = 0.
                else:
                    star.photometry_confidence = 1.
                    good_stars += 1
            logger.debug('Photometry NOT performed')
            photometry_str = 'WITHOUT'
        else:
            good_stars = stars_perform_photometry(obs,
                                                  obs.calib_dn_ext_data,
                                                  star_list,
                                                  offset=offset,
                                                  extend_fov=obs.extend_fov,
                                                  stars_config=stars_config)
            logger.debug('Photometry found %d good stars', good_stars)
            photometry_str = 'with'
    
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
                logger.info('***** Enough good stars (%s photometry), '+
                            'but only saw %d '+
                            'out of %d bright stars - possibly bad '+
                            'offset U,V %d,%d',
                            photometry_str, 
                            seen_bright_stars, bright_stars,
                            offset[0], offset[1])
                # Return True so the top-level loop keeps searching
                return offset, good_stars, peak, True, False
            logger.info('***** Enough good stars (%s photometry) - '+
                        'final offset U,V %d,%d',
                        photometry_str,
                        offset[0], offset[1])
            # Return False so the top-level loop gives up
            return offset, good_stars, peak, False, False

        if not something_conflicted:
            # No point in trying again - we'd just have the same stars!
            logger.debug('Nothing conflicted and photometry failed - '+
                         'continuing to next peak')
            continue

        # Get rid of the conflicting stars and recurse until there are no 
        # conflicts

        # Create the current non-conflicting star list
        non_conf_star_list = [x for x in star_list if not x.conflicts]
    
        logger.debug('After conflict - # stars %d', len(non_conf_star_list))
                
        if len(non_conf_star_list) < min_stars:
            logger.debug('Fewer than %d stars left (%d)', min_stars,
                         len(non_conf_star_list))
            continue

        # And recurse using this limited star list
        ret = _stars_find_offset(obs, filtered_data, non_conf_star_list, 
                                 min_stars, search_multiplier, 
                                 max_offsets, already_tried, 
                                 debug_level+1, perform_photometry,
                                 rings_can_conflict, radec_movement,
                                 stars_config)
        if ret[0] is not None:
            return ret
        # We know that everything in non_conf_star_list is not 
        # conflicting at this level, but they were probably mutated by
        # _stars_find_offset, so reset them
        for star in non_conf_star_list:
            star.conflicts = False
                
    logger.debug('Exhausted all peaks - No offset found')

    return None, None, None, False, False

def _stars_refine_offset(obs, calib_data, star_list, offset,
                         stars_config):
    """Perform astrometry to refine the final offset."""
    logger = logging.getLogger(_LOGGING_NAME+'._stars_refine_offset')

    psf_size = stars_config['min_psf_size']
    
    if psf_size < 7:
        logger.error('Unable to refine star fit because PSF SIZE of %d is '+
                     'too small',
                     psf_size)
        return offset
    
    delta_u_list = []
    delta_v_list = []
    dn_list = []
    for star in star_list:
        if star.conflicts:
            continue
        if star.photometry_confidence < stars_config['min_confidence']:
            continue
        if abs(star.move_u) > 1 or abs(star.move_v) > 1:
            logger.info('Aborting refine fit due to excessive streaking')
            return (offset[0], offset[1])
        u = star.u + offset[0]
        v = star.v + offset[1]
        psf_size_u = psf_size + np.round(abs(star.move_u))
        psf_size_u = (psf_size_u // 2) * 2 + 1
        psf_size_v = psf_size + np.round(abs(star.move_v))
        psf_size_v = (psf_size_v // 2) * 2 + 1
        gausspsf = GaussianPSF(sigma=PSF_SIGMA[obs_detector(obs)],
                               movement=(star.move_v,star.move_u))
        ret = gausspsf.find_position(calib_data, (psf_size_v,psf_size_u),
                      (v,u), search_limit=(1.5, 1.5),
                      bkgnd_degree=2, bkgnd_ignore_center=(2,2),
                      bkgnd_num_sigma=5,
                      tolerance=1e-5, num_sigma=10,
                      max_bad_frac=0.2,
                      allow_nonzero_base=True)
        if ret is None:
            continue
        pos_v, pos_u, metadata = ret
        logger.info('Star %9d UV %7.2f %7.2f refined to %7.2f %7.2f', 
                    star.unique_number, u, v, pos_u, pos_v)
        delta_u_list.append(pos_u-u)
        delta_v_list.append(pos_v-v)
        dn_list.append(star.dn)
        
    if len(delta_u_list) == 0:
        return offset[0], offset[1]
    
    du_mean = np.mean(delta_u_list)
    dv_mean = np.mean(delta_v_list)
    
    logger.info('Mean dU,dV %7.2f %7.2f', du_mean, dv_mean)
    
    if stars_config['allow_fractional_offsets']:
        return (offset[0]+du_mean, offset[1]+dv_mean)
    
    return (int(np.round(offset[0]+du_mean)),
            int(np.round(offset[1]+dv_mean)))

def stars_find_offset(obs, ra_dec_predicted,
                      extend_fov=(0,0), stars_config=None):
    """Find the image offset based on stars.

    Inputs:
        obs                The observation.
        ra_dec_predicted   A tuple (ra, dec, dra/dt, ddec/dt) giving the 
                           navigation information from the more-precise
                           predicted kernels. This is used to make accurate
                           star streaks.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to model stars to find an offset.
        stars_config        Configuration parameters. None uses the default.
                           
    Returns:
        metadata           A dictionary containing information about the
                           offset result:
            'offset'            The (U,V) offset.
            'confidence'        The confidence (0-1) in the result.
            'full_star_list'    The list of Stars in the FOV.
            'num_stars',        The number of Stars in the FOV.
            'num_good_stars'    The number of Stars that photometrically match.
            'rings_subtracted'  True if the rings were subtracted from the
                                image.
            'start_time'        The time (s) when stars_find_offset was called.
            'end_time'          The time (s) when stars_find_offset returned.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.stars_find_offset')

    if stars_config is None:
        stars_config = STARS_DEFAULT_CONFIG

    min_dn = stars_config[('min_detectable_dn', obs_detector(obs))]
    min_stars, min_stars_conf = stars_config['min_stars_low_confidence']
    min_stars_hc, min_stars_hc_conf = stars_config['min_stars_high_confidence']
    perform_photometry = stars_config['perform_photometry']

    radec_movement = None
    
    if ra_dec_predicted is not None:
        radec_movement = (ra_dec_predicted[2] * obs.texp/2,
                          ra_dec_predicted[3] * obs.texp/2)
            
    metadata = {}

    set_obs_ext_bp(obs, extend_fov)
    set_obs_ext_data(obs, extend_fov)
    obs.star_body_list = None # Body inventory cache

    if stars_config['calibrated_data']:
        obs.calib_dn_ext_data = None # DN-calibrated, extended data
    else:
        obs.calib_dn_ext_data = obs.ext_data
        filtered_data = obs.calib_dn_ext_data
            

    # Get the Star list and initialize our new fields
    star_list = stars_list_for_obs(obs, radec_movement,
                                   extend_fov=obs.extend_fov,
                                   stars_config=stars_config)
    for star in star_list:
        star.photometry_confidence = 0.
        star.is_bright_enough = False
        star.is_dim_enough = True

    metadata['offset'] = None
    metadata['confidence'] = 0.
    metadata['full_star_list'] = star_list
    metadata['num_stars'] = len(star_list)
    metadata['num_good_stars'] = 0
    metadata['rings_subtracted'] = False

    if len(star_list) > 0:
        first_star = star_list[0]
        smear_amt = np.sqrt(first_star.move_u**2+first_star.move_v**2)
        if smear_amt > stars_config['max_smear']:
            logger.debug(
             'FAILED to find a valid offset - star smear is too great')
            return metadata

    # A list of offsets that we have already tried so we don't waste time
    # trying them a second time.
    already_tried = []
    
    # A list of offset results so we can choose the best one at the very end.
    saved_offsets = []
    
    rings_can_conflict = True
    
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
            return metadata
    
        if first_good_star.dn < min_dn:
            # There's no point in continuing if the brightest star is below the
            # detection threshold
            logger.debug(
                 'FAILED to find a valid offset - brightest star is too dim')
            return metadata
    
        if obs.calib_dn_ext_data is None:
            # First pass - don't do these things earlier outside the loop
            # because we'd just be wasting time if there were never enough
            # stars.
            
            # For star use only, need data in DN for photometry
            obs.calib_dn_ext_data = calibrate_iof_image_as_dn(
                                                      obs, data=obs.ext_data)
            filtered_data = obs.calib_dn_ext_data
            
            if False:
                calib_data = unpad_image(obs.calib_dn_ext_data, extend_fov)
                rings_radial_model = rings_create_model_from_image(
                                                       obs, data=calib_data,
                                                       extend_fov=extend_fov)
                if rings_radial_model is not None:
                    imdisp = ImageDisp([calib_data, rings_radial_model, 
                                        calib_data-rings_radial_model],
                                       canvas_size=(512,512),
                                       allow_enlarge=True, enlarge_limit=10,
                                       auto_update=True)
                    Tkinter.mainloop()
    
                    filtered_data = pad_image(calib_data-rings_radial_model, 
                                              extend_fov)
                    obs.data[:,:] = calib_data-rings_radial_model
                    obs.ext_data[:,:] = filtered_data
                    rings_can_conflict = False
                    
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
 
            filtered_data[filtered_data < 0.] = 0.

            # If we trust the DN values, then we can eliminate any pixels that
            # are way too bright.            
            if _trust_star_dn(obs):
                max_dn = star_list[0].dn # Star list is sorted by DN            
                mask = filtered_data > max_dn*2
                mask = filt.maximum_filter(mask, 11)
                filtered_data[mask] = 0.

            if DEBUG_STARS_FILTER_IMGDISP:
                imdisp = ImageDisp([filtered_data],
                                   canvas_size=(512,512),
                                   allow_enlarge=True, enlarge_limit=10,
                                   auto_update=True)
                Tkinter.mainloop()

        offset = None
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
            for search_multipler in stars_config['search_multipliers']:
                #
                #            *** LEVEL 3 ***
                #
                # At this level we restrict the search space so we first try
                # smaller offsets.
                #
                logger.debug('** LEVEL 3: Trying search multiplier %.2f', 
                             search_multipler)
                
                # The remaining search levels are inside the subroutine
                ret = _stars_find_offset(obs, filtered_data, new_star_list,
                                         min_stars, search_multipler,
                                         5, already_tried, 4, 
                                         perform_photometry,
                                         rings_can_conflict,
                                         radec_movement,
                                         stars_config) 
        
                # Save the offset and maybe continue iterating
                (offset, good_stars, corr,
                 keep_searching, no_peaks) = ret

                if no_peaks and search_multipler == 1.:
                    logger.debug('No peaks found at largest search range - '+
                                 'aborting star offset finding')
                    got_it = True
                    break
                 
                if offset is None:
                    logger.debug('No valid offset found - iterating')
                    continue

                logger.debug('Found valid offset U,V %d,%d STARS %d CORR %f', 
                             offset[0], offset[1], good_stars, corr)
                saved_star_list = copy.deepcopy(star_list)
                saved_offsets.append((offset, good_stars, corr, saved_star_list))
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
            too_bright_dn = stars_config['too_bright_dn']
            too_bright_factor = stars_config['too_bright_factor']
            if ((too_bright_dn and star_list[i].dn > too_bright_dn) or
                (too_bright_factor and
                 star_list[i].dn > star_list[i+1].dn*too_bright_factor)):
                # Star is too bright - get rid of it
                logger.debug('Star %9d (DN %7.2f) is too bright - '+
                             'ignoring and iterating',
                             star_list[i].unique_number, star_list[i].dn)
                star_list[i].is_dim_enough = False
                still_good_stars_left = True
            break
            
        if not still_good_stars_left:
            break
    
        # End of LEVEL 1 - eliminate bright stars

    used_photometry = perform_photometry
            
    if len(saved_offsets) == 0:
        offset = None
        good_stars = 0
        logger.info('FAILED to find a valid offset')
        if perform_photometry:
            logger.info('Trying again with photometry disabled')
            ret = _stars_find_offset(obs, filtered_data, star_list,
                                     min_stars, 1.,
                                     1, [], 4, 
                                     False, rings_can_conflict,
                                     radec_movement, stars_config) 
            (offset, good_stars, corr,
             keep_searching, no_peaks) = ret
            if no_peaks:
                offset = None
            used_photometry = False
    else:
        best_offset = None
        best_star_list = None
        best_good_stars = -1
        best_corr = -1
        for offset, good_stars, corr, saved_star_list in saved_offsets:
            if len(saved_offsets) > 1:
                logger.info('Saved offset U,V %d,%d / Good stars %d / Corr %f',
                            offset[0], offset[1], good_stars, corr)
            if (good_stars > best_good_stars or
                (good_stars == best_good_stars and 
                 corr > best_corr)):
                best_offset = offset
                best_good_stars = good_stars
                best_corr = corr
                best_star_list = saved_star_list
        offset = best_offset
        good_stars = best_good_stars
        corr = best_corr
        star_list = saved_star_list

        logger.info('Trial final offset U,V %d,%d / Good stars %d / Corr %f',
                     offset[0], offset[1], good_stars, corr)

        offset = _stars_refine_offset(obs, obs.data, star_list,
                                      offset, stars_config)

    if offset is None:
        confidence = 0.
    else:
        confidence = ((good_stars-min_stars) * 
                        (float(min_stars_hc_conf)-min_stars_conf)/
                        (float(min_stars_hc-min_stars)) +
                      min_stars_conf)
        if len(star_list) > 0:
            movement = np.sqrt(star_list[0].move_u**2 + star_list[0].move_v**2)
            if movement > 1:
                confidence /= (movement-1)/4+1
        if not used_photometry:
            confidence *= 0.1
        confidence = np.clip(confidence, 0., 1.)
        
    metadata['offset'] = offset
    metadata['confidence'] = confidence         
    metadata['full_star_list'] = star_list
    metadata['num_stars'] = len(star_list)
    metadata['num_good_stars'] = good_stars
    metadata['rings_subtracted'] = not rings_can_conflict

    if offset is not None:
        logger.info('Returning final offset U,V %.2f,%.2f / Good stars %d / '+
                    'Corr %f / Conf %f / Rings sub %s',
                    offset[0], offset[1], good_stars, corr, confidence,
                    str(not rings_can_conflict))


    offset_x = 0
    offset_y = 0
    if offset is not None:
        offset_x = offset[0]
        offset_y = offset[1]

    for star in star_list:
        _stars_mark_conflicts(obs, star, (offset_x, offset_y), rings_can_conflict,
                              stars_config)
        
    logger.info('Final star list after offset:')
    for star in star_list:
        logger.info('Star %9d U %8.3f+%7.3f V %8.3f+%7.3f DN %7.2f MAG %6.3f '+
                    'SCLASS %3s TEMP %6d PRED %7.2f MEAS %7.2f CONF %4.2f CONFLICTS %s',
                    star.unique_number, 
                    star.u+offset_x, abs(star.move_u), 
                    star.v+offset_y, abs(star.move_v),
                    star.dn, star.vmag,
                    'XX' if star.spectral_class is None else
                            star.spectral_class,
                    0 if star.temperature is None else star.temperature,
                    star.dn, star.integrated_dn, star.photometry_confidence,
                    str(star.conflicts))
    
    return metadata
