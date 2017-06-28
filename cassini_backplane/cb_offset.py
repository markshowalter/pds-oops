###############################################################################
# cb_offset.py
#
# Routines related to finding image offsets.
#
# Exported routines:
#    master_find_offset
#    offset_create_overlay_image
#    offset_result_str
###############################################################################

import cb_logging
import logging

import time

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filt

import oops

from cb_bodies import *
from cb_config import MAX_POINTING_ERROR, LARGE_BODY_LIST, FUZZY_BODY_LIST
from cb_correlate import *
from cb_rings import *
from cb_stars import *
from cb_titan import *
from cb_util_file import *
from cb_util_image import *

_LOGGING_NAME = 'cb.' + __name__


def _normalize(data, masked=False):
    """Normalize data to [0,1] but preserve zeros."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_min == data_max:
        if masked:
            return ma.zeros(data.shape, dtype=np.float32)
        else:
            return np.zeros(data.shape, dtype=np.float32)
    new_data = (data-data_min) / float(data_max-data_min)
    if masked:
        new_data[ma.filled(data, 1.) == 0.] = 0.
    else:
        new_data[data == 0.] = 0.
    return new_data

def _combine_models(model_list, solid=False, masked=False):
    """Combine models by normalizing the sum of the normalized models."""
    if masked:
        new_model = ma.zeros(model_list[0].shape, dtype=np.float32)
        new_model[:] = ma.masked
    else:
        new_model = np.zeros(model_list[0].shape, dtype=np.float32)
    for model in model_list:
        if solid:
            new_model[model != 0] = _normalize(model, 
                                               masked=masked)[model != 0]
        else:
            new_model += _normalize(model, masked=masked)

    return _normalize(new_model, masked=masked)

def _combine_text(text_list):
    ret = text_list[0]
    for text in text_list[1:]:
        if text is not None:
            ret = np.logical_or(ret, text)
    return ret

def _model_filter(image, gaussian_pre_blur, median_boxsize, 
                  gaussian_median_blur, masked=False):
    """The filter to use on an image when looking for moons and rings."""
    if gaussian_pre_blur:
        image = filt.gaussian_filter(image, gaussian_pre_blur)

    ret = filter_sub_median(image, median_boxsize=median_boxsize, 
                            gaussian_blur=gaussian_median_blur)
    if masked:
        ret = ret.view(ma.MaskedArray)
        ret.mask = image.mask
    return ret

def _iterate_secondary_correlation(obs, final_model, model_offset, peak,
                                   search_size_min, search_size_max,
                                   model_filter_func, 
                                   model_rings_radial_only,
                                   extend_fov, 
                                   offset_config, rings_config):
    logger = logging.getLogger(_LOGGING_NAME+'._iterate_secondary_correlation')

    confidence_factor = 1.
    
    sec_threshold = offset_config['secondary_corr_threshold']
    sec_frac = offset_config['secondary_corr_peak_threshold']
    tweak_frac = offset_config['secondary_corr_tweak_threshold']
    
    model_offset = (int(np.round(model_offset[0])),
                    int(np.round(model_offset[1])))

    # Try a secondary correlation using the subimage ONLY. Then use that
    # offset to try a new subimage. Iterate until the result gives a
    # delta of (0,0) or we get tired of trying, in which case we report
    # failure.
    offset_x_list = []
    offset_y_list = []
    offset_corr_list = []
    offset_x_list.append(model_offset[0])
    offset_y_list.append(model_offset[1])
    min_corr = 1e38
    for attempt in xrange(offset_config['secondary_max_attempts']):
        logger.info('Attempt #%d: Running secondary correlation on offset U,V %d,%d',
                    attempt+1, model_offset[0], model_offset[1])
        offset_model = final_model[extend_fov[1]-model_offset[1]:
                                   extend_fov[1]-model_offset[1]+
                                       obs.data.shape[0],
                                   extend_fov[0]-model_offset[0]:
                                   extend_fov[0]-model_offset[0]+
                                       obs.data.shape[1]]
        model_offset_list = find_correlation_and_offset(
                               obs.data,
                               offset_model, search_size_min=search_size_min,
                               search_size_max=search_size_max,
                               filter=model_filter_func)
        sec_model_offset = None
        if len(model_offset_list):
            (sec_model_offset, new_peak, new_corr_details) = model_offset_list[0]
        if sec_model_offset is None:
            logger.info('Secondary model correlation FAILED - ending attempts')
            if attempt == 0:
                return None, 0, None
            confidence_factor *= 0.25
            break
        logger.info('Secondary model correlation dU,dV %d,%d '+
                     'CORR %f', sec_model_offset[0], sec_model_offset[1], 
                     new_peak)
        if model_rings_radial_only:
            (sec_model_offset, _) = rings_offset_radial_projection(
                                              obs, sec_model_offset,
                                              extend_fov=extend_fov,
                                              rings_config=rings_config)
            logger.info('Secondary model correlation after radial reprojection'+
                        ' dU,dV %.2f,%.2f', 
                        sec_model_offset[0], sec_model_offset[1]) 
        if new_peak < peak*sec_frac:
            logger.info('Secondary model correlation offset does '+
                        'not meet peak criteria - ending attempts')
            if attempt == 0:
                return None, 0, None
            confidence_factor *= 0.25
            break
        model_offset = (model_offset[0]+sec_model_offset[0],
                        model_offset[1]+sec_model_offset[1])
        if (abs(model_offset[0]) > extend_fov[0] or 
            abs(model_offset[1]) > extend_fov[1]):
            logger.info('Resulting offset is beyond maximum '+
                        'allowable offset - ending attempts')
            if attempt == 0:
                return None, 0, None
            confidence_factor *= 0.25
            break
        if (abs(sec_model_offset[0]) < sec_threshold and
            abs(sec_model_offset[1]) < sec_threshold):
            logger.info('Secondary correlation succeeded')
            new_corr_psf_details = corr_analyze_peak(*new_corr_details)
            if new_corr_psf_details is None:
                logger.info('Correlation peak analysis failed')
            else:
                corr_log_sigma(logger, new_corr_psf_details)
            return model_offset, confidence_factor, new_corr_psf_details
        offset_x_list.append(model_offset[0])
        offset_y_list.append(model_offset[1])
        offset_corr_list.append(new_corr_details)
        min_corr = min(min_corr, new_peak)
        confidence_factor *= 0.9
        
    offset_x_min = np.min(offset_x_list)
    offset_x_max = np.max(offset_x_list)
    offset_y_min = np.min(offset_y_list)
    offset_y_max = np.max(offset_y_list)
    
    if offset_x_min == offset_x_max and offset_y_min == offset_y_max:
        # They are all the same, so might as well arbitrarily choose the first
        # one to analyze the PSF.
        new_corr_psf_details = corr_analyze_peak(*off_corr_list[0])
        if new_corr_psf_details is None:
            logger.info('Correlation peak analysis failed')
        else:
            corr_log_sigma(logger, new_corr_psf_details)
        return ((offset_x_min, offset_y_min), confidence_factor, 
                new_corr_psf_details)
    
    logger.info('No convergence - trying exhaustive search U %d to %d, '+
                'V %d to %d', offset_x_min, offset_x_max, offset_y_min,
                offset_y_max)
    confidence_factor *= 0.5
    
    if ((offset_x_max-offset_x_min+1) * (offset_y_max-offset_y_min+1) >
        offset_config['secondary_max_num_exhaustive']):
        logger.info('Range of offsets exceeds maximum allowable - '+
                    'giving up')
        return None, 0, None
    
    best_offset = None
    best_corr = None
    best_corr_val = -1e38

    image_filtered = model_filter_func(obs.data)
        
    for offset_x in xrange(int(np.floor(offset_x_min)), 
                           int(np.ceil(offset_x_max))+1):
        for offset_y in xrange(int(np.floor(offset_y_min)), 
                               int(np.ceil(offset_y_max))+1):
            d_offset_model = final_model[extend_fov[1]-offset_y:
                                         extend_fov[1]-offset_y+
                                           obs.data.shape[0],
                                         extend_fov[0]-offset_x:
                                         extend_fov[0]-offset_x+
                                           obs.data.shape[1]]
            model_filtered = model_filter_func(d_offset_model)
            corr = correlate2d(image_filtered, model_filtered, 
                               normalize=True, retile=True)
            corr_val = corr[corr.shape[0]//2, corr.shape[1]//2]
            if corr_val < 0 or corr_val < min_corr*tweak_frac:
                logger.debug('Tweaking offset U,V %d,%d CORR %.16f - Correlation too small',
                             offset_x, offset_y, corr_val)
                continue
            logger.debug('Tweaking offset U,V %d,%d CORR %.16f - OK',
                         offset_x, offset_y, corr_val)
            if corr_val > best_corr_val:
                best_corr_val = corr_val
                best_corr = corr
                best_offset = (offset_x, offset_y)

    if best_corr is None:
        logger.info('Tweaking failed - giving up')
        return None, 0, None

    details = corr_analyze_peak(best_corr, 
                                best_corr.shape[0]//2, best_corr.shape[1]//2)
    
    return best_offset, confidence_factor, details
        
def master_find_offset(obs, 
                   offset_config=None,
                   create_overlay=False,
                   
                   allow_stars=True,
                       stars_show_streaks=False,
                       stars_config=None,
                        
                   allow_saturn=True,
                   allow_moons=True,
                       bodies_cartographic_data=None,
                       bodies_config=None,
                       titan_config=None,
                   
                   allow_rings=True,
                       rings_model_source='voyager',
                       rings_config=None,

                   bootstrap_config=None,
                   
                   botsim_offset=None,
                   force_bootstrap_candidate=False,
                   
                   bootstrapped=False):
    """Reproject the moon into a rectangular latitude/longitude space.
    
    Inputs:
        obs                      The Observation.
        offset_config            Config parameters for master offset.
        create_overlay           True to create a visual overlay.

        allow_stars              True to allow finding the offset based on
                                 stars.
        stars_show_streaks       Include streaks in the overlay.
        stars_config             Config parameters for stars.

        allow_saturn             True to allow finding the offset based on
                                 Saturn.
        allow_moons              True to allow finding the offset based on
                                 moons.
        bodies_cartographic_data The metadata to use for cartographic
                                 surfaces (see cb_bodies).
        bodies_config            Config parameters for bodies.
        titan_config             Config parameters for Titan navigation.

        allow_rings              True to allow finding the offset based on
                                 rings.
        rings_config             Config parameters for rings.
        bootstrap_config         Config parameters for bootstrapping.

        botsim_offset            None to find the offset automatically or
                                 a tuple (U,V) to force the result offset.
                                 This is useful for creating an overlay 
                                 with a known offset, such as when processing
                                 pairs of BOTSIM images.
        force_bootstrap_candidate  True to force this image to be a bootstrap
                                 candidate even if we really could find a
                                 viable offset.
                                 
        bootstrapped             True if this offset pass is the result of
                                 bootstrapping.
                                 
    Returns:
        metadata           A dictionary containing information about the
                           offset result:

          Data about the image:

            'full_path'        The full path of the source image.
            'camera'           The name of the camera: 'NAC' or 'WAC'.
            'filter1'          The names of the filters used.
            'filter2'
            'image_shape'      A tuple indicating the shape (in pixels) of the
                               image.
            'midtime'          The midtime of the observation.
            'scet_midtime'     The SCET midtime of the observation.
            'texp'             The exposure duration of the observation.
            'description'      The DESCRIPTION field from the VICAR header.
            'ra_dec_corner_orig'
                               A tuple (ra_min, ra_max, dec_min, dec_max)
                               giving the corners of the FOV-extended image
                               (apparent) with the original SPICE navigation at
                               the image midtime.
            'ra_dec_center_pred'
                               A tuple (ra,dec,dra/dt,ddec/dt) for the center 
                               of the image (apparent) with the original SPICE 
                               predicted kernels at the image midtime.
            'ra_dec_center_orig'
                               A tuple (ra,dec) for the center of the image
                               (apparent) with the original SPICE navigation
                               at the image midtime.
            'ra_dec_center_offset'
                               A tuple (ra,dec) for the center of the image
                               (apparent) with the new navigation at the
                               image midtime.

          Data about the offset process:
          
            'status'           'ok' if the process went to completion (whether
                               or not an offset was found). Other values are
                               set by external drivers.
            'offset'           The final (U,V) offset. None if offset finding
                               failed.
            'corr_psf_details' The details of the 2-D Gaussian fit to the 
                               correlation peak for the final offset.
            'confidence'       The confidence 0-1 of the final offset.
            'model_confidence' The confidence 0-1 of the model offset.
            'model_blur_amount' The amount the model was blurred before 
                               correlating.
            'model_override_bodies_curvature'
            'model_override_bodies_limb'
            'model_override_rings_curvature'
            'model_override_fiducial_features'
            'model_rings_blur'
            'model_bodies_blur'
                               The assumptions that were broken in order to
                               create the model offset, if any.
            'model_rings_radial_gradient'
                               The direction of the rings radial gradient
                               if the model offset was projected onto
                               the radial direction vector due to insufficient
                               ring curvature. None otherwise.
            'stars_offset'     The offset from star matching. None is star
                               matching failed.
            'model_offset'     The offset from model (rings and bodies) 
                               matching. None is model matching failed.
            'model_corr_psf_details'
                               The details of the 2-D Gaussian fit to the 
                               correlation peak for the model offset.
            'titan_offset'     The offset from Titan photometric navigation.
                               None if navigation failed.
            'large_bodies'     A list of the large bodies present in the image
                               sorted in ascending order by range.
            'body_only'        False if the image doesn't consist entirely of
                               a single body and nothing else.
                               Otherwise the name of the body.
            'rings_only'       True if the image consists entirely of the main
                               rings and nothing else.
            'model_contents'   A list of objects used to create the
                               non-star model: 'RINGS' and body names.
                               If the contents is only "TITAN", then the
                               special photometry-based Titan navigation
                               was performed.
            'offset_winner'    Which method won for the final offset:
                                   None, STARS, MODEL, or TITAN.
            'stars_metadata'   The metadata from star matching. None if star
                               matching not performed. The star-matching offset
                               is included here.
            'bodies_metadata'  A dictionary containing the metadata for each
                               body in the large_bodies list.
            'rings_metadata'   The metadata from ring modeling. None if ring
                               modeling not performed.
            'secondary_corr_ok' True if secondary model correlation was
                               successful. None if it wasn't performed.
            'offset_path'      The path to the offset filename - filled in by
                               the top-level program.
            'start_time'       The start time (s) of the entire offset process.
            'end_time'         The end time (s) of the entire offset process.
            
          Data for bootstrapping:
            
            'bootstrap_candidate'
                               True if the image is a good candidate for 
                               future bootstrapping attempts.
            'bootstrapped'     True if the image was navigated using 
                               bootstrapping.
            'bootstrap_status' A string describing attempts at bootstrapping:
                                   None        Bootstrapping not attempted
                               
          Large data that might be too big to store in an offset file:
            
            'ext_data'         The original obs.data extended by the maximum
                               search size.
            'stars_overlay'    The 2-D stars overlay (no text).
            'stars_overlay_text'
                               The 2-D stars overlay (text only).
            'bodies_overlay'   The 2-D bodies overlay (no text).
            'bodies_overlay_text'
                               The 2-D bodies overlay (text only).
            'rings_overlay'    The 2-D rings overlay (no text).
            'rings_overlay_text'
                               The 2-D rings overlay (text only).
            'overlay'          The visual overlay (if create_overlay is True).
            'ext_overlay'      The visual overlay extended by the maximum
                               search size (if create_overlay is True).
            
    """
    
                ##################
                # INITIALIZATION #
                ##################
    
    start_time = time.time()
    
    logger = logging.getLogger(_LOGGING_NAME+'.master_find_offset')
    
    logger.info('Processing %s', obs.full_path)
    logger.info('Taken %s / %s / Size %d x %d / TEXP %.3f / %s+%s / '+
                'SAMPLING %s / GAIN %d',
                cspice.et2utc(obs.midtime, 'C', 0),
                obs.detector, obs.data.shape[1], obs.data.shape[0], obs.texp,
                obs.filter1, obs.filter2, obs.sampling,
                obs.gain_mode)
    logger.debug('allow_stars %d, allow_saturn %d, allow_moons %d, '+
                 'allow_rings %d',
                 allow_stars, allow_saturn, allow_moons, allow_rings)
    if botsim_offset is not None:
        logger.info('BOTSIM offset U,V %d,%d', botsim_offset[0],
                    botsim_offset[1])
    if bodies_cartographic_data is None:
        logger.info('No cartographic data provided')
    else:
        for body_name in sorted(bodies_cartographic_data.keys()):
            logger.info('Cartographic data provided for: %s = %s',
                         body_name.upper(),
                         bodies_cartographic_data[body_name]['full_path'])
        
    if offset_config is None:
        offset_config = OFFSET_DEFAULT_CONFIG
        
    if bootstrap_config is None:
        bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
        
    extend_fov = MAX_POINTING_ERROR[obs.data.shape, obs.detector]
    search_size_max_u, search_size_max_v = extend_fov
    
    set_obs_ext_data(obs, extend_fov)
    obs.ext_data = image_interpolate_missing_stripes(obs.ext_data)
    set_obs_ext_corner_bp(obs, extend_fov)
    set_obs_ext_bp(obs, extend_fov)
    
    bodies_model_list = []
    rings_model = None
    rings_overlay_text = None
    titan_body_metadata = None
    final_model = None
    
    if create_overlay:
        color_overlay = np.zeros(obs.ext_data.shape + (3,), dtype=np.uint8)
        label_avoid_mask = np.zeros(obs.ext_data.shape, dtype=np.bool)
    else:
        color_overlay = None
        label_avoid_mask = None
    stars_overlay = None
    stars_overlay_text = None
    bodies_overlay_text = None
    rings_overlay_text = None
        
    offset = None
    
    # Initialize the metadata to at least have something for each key
    metadata = {}
    metadata['status'] = 'ok'
    # Image
    metadata['full_path'] = obs.full_path
    metadata['camera'] = obs.detector
    metadata['filter1'] = obs.filter1
    metadata['filter2'] = obs.filter2
    metadata['image_shape'] = tuple(obs.data.shape)
    metadata['midtime'] = obs.midtime
    scet_start = float(obs.dict['SPACECRAFT_CLOCK_START_COUNT'])
    scet_end = float(obs.dict['SPACECRAFT_CLOCK_STOP_COUNT'])
    metadata['scet_midtime'] = (scet_start+scet_end)/2
    metadata['texp'] = obs.texp
    metadata['description'] = obs.dict['DESCRIPTION']
    logger.info('Image description: %s', metadata['description'])
    metadata['ra_dec_corner_orig'] = compute_ra_dec_limits(
                                           obs, extend_fov=extend_fov)
    set_obs_center_bp(obs)
    ra = obs.center_bp.right_ascension()
    dec = obs.center_bp.declination()
    metadata['ra_dec_center_orig'] = (ra.vals,dec.vals)
    pred_metadata = file_read_predicted_metadata(obs.full_path)
    if pred_metadata is None:
        metadata['ra_dec_center_pred'] = None
        logger.warn('No predicted kernel information available')
    else:
        metadata['ra_dec_center_pred'] = (
                  pred_metadata['ra_center_midtime'],
                  pred_metadata['dec_center_midtime'],
                  pred_metadata['dra_dt'],
                  pred_metadata['ddec_dt'])
    metadata['ra_dec_center_offset'] = None
    # Offset process
    metadata['start_time'] = start_time
    metadata['end_time'] = None
    metadata['offset'] = None
    metadata['confidence'] = 0.
    metadata['model_override_bodies_curvature'] = None
    metadata['model_override_bodies_limb'] = None
    metadata['model_override_rings_curvature'] = None
    metadata['model_override_fiducial_features'] = None
    metadata['model_rings_blur'] = None
    metadata['model_bodies_blur'] = None
    metadata['model_rings_radial_gradient'] = None
    metadata['stars_offset'] = None
    metadata['stars_confidence'] = 0.
    metadata['model_offset'] = None
    metadata['model_confidence'] = 0.
    metadata['model_blur_amount'] = None
    metadata['titan_offset'] = None
    metadata['titan_confidence'] = 0.
    metadata['body_only'] = False
    metadata['rings_only'] = False
    metadata['model_contents'] = []
    metadata['offset_winner'] = None
    metadata['stars_metadata'] = None
    bodies_metadata = {}
    metadata['bodies_metadata'] = bodies_metadata
    metadata['rings_metadata'] = None
    metadata['titan_metadata'] = None
    metadata['secondary_corr_ok'] = None
    # Large
    metadata['ext_data'] = obs.ext_data
    metadata['ext_overlay'] = color_overlay
    if create_overlay:
        metadata['overlay'] = unpad_image(color_overlay, extend_fov)
    else:
        metadata['overlay'] = None
    metadata['stars_overlay'] = None
    metadata['stars_overlay_text'] = None
    metadata['bodies_overlay'] = None
    metadata['bodies_overlay_text'] = None
    metadata['rings_overlay'] = None
    metadata['rings_overlay_text'] = None
    # Bootstrapping
    metadata['bootstrap_candidate'] = False
    metadata['bootstrapped'] = bootstrapped
    metadata['bootstrap_status'] = None
    
    
                ###########################
                # IMAGE CONTENTS ANALYSIS #
                ###########################
    
    #
    # FIGURE OUT WHAT KINDS OF THINGS ARE IN THIS IMAGE
    #
    # - Bodies; a single body completely covers the image
    # - Rings; the main rings completely cover the image
    #

    # Always compute the inventory so we can see if the entire image is a
    # closeup of a single body and so the metadata has this information
    
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    # Make a list sorted by range, with the closest body first
    large_bodies_by_range = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_range.sort(key=lambda x: x[1]['range'])

    logger.info('Large body inventory by increasing range: %s',
                 [x[0] for x in large_bodies_by_range])

    # Now find the ring radii and distance for the main rings
    # We use this in various places below
    rings_radii = obs.ext_bp.ring_radius('saturn:ring')
    main_rings_radii = rings_radii.mask_where(rings_radii < RINGS_MIN_RADIUS)
    main_rings_radii = main_rings_radii.mask_where(rings_radii > 
                                                   RINGS_MAX_RADIUS)
    main_rings_radii = main_rings_radii.mvals
    df_rings_radii = rings_radii.mask_where(rings_radii < RINGS_MIN_RADIUS_D)
    df_rings_radii = df_rings_radii.mask_where(rings_radii > 
                                               RINGS_MAX_RADIUS_F)
    df_rings_radii = df_rings_radii.mvals

    rings_dist = obs.ext_bp.distance('saturn:ring')
    rings_dist = rings_dist.mask_where(main_rings_radii.mask).mvals
    min_rings_dist = np.min(rings_dist)

    # See if the main rings take up the entire image AND are in front of
    # all bodies.
    # It could make sense in this case to empty the body list, but the rings
    # are sometimes transparent and we want to be able to handle things like
    # Pan and Daphnis embedded in the rings.
    entirely_rings = False
    has_df_rings = False
    if not np.any(main_rings_radii.mask): # All radii valid
        has_df_rings = True
        found_front_body = False
        if len(large_bodies_by_range) > 0:
            inv = large_bodies_by_range[0][1]
            if inv['range'] < min_rings_dist:
                found_front_body = True
        if found_front_body:
            logger.debug('Image is entirely rings but %s is in front',
                         inv['name'])
        else:
            logger.debug('Image is entirely main rings')
            entirely_rings = True
            metadata['rings_only'] = True
    else:
        logger.debug('Image is not entirely main rings')
        if not np.all(df_rings_radii.mask): # At least some radii valid
            logger.debug('... but image has at least some rings between D and F')
            has_df_rings = True

    front_body_name = None
    if len(large_bodies_by_range) > 0:
        body_name, inv = large_bodies_by_range[0]
        size = (inv['u_max']-inv['u_min'])*(inv['v_max']-inv['v_min']) 
        if size >= offset_config['min_body_area']:
            front_body_name = body_name
            logger.debug('Front body %s is sufficiently large', 
                         front_body_name)

    if front_body_name in FUZZY_BODY_LIST:
        front_body_name = None
        logger.debug('... but front body %s is fuzzy', front_body_name)

    entirely_body = False
        
    if front_body_name is not None:
        # See if the front-most actually visible body takes up the entire image
        set_obs_corner_bp(obs)    
        corner_body_mask = (
            obs.ext_corner_bp.where_intercepted(front_body_name).
                mvals.filled(0))
        if np.all(corner_body_mask):
            inv = large_body_dict[front_body_name]
            if np.any(rings_dist < inv['range']):
                logger.info(
                  'Image is covered by %s, which is hidden by rings', 
                  front_body_name)
                front_body_name = None
                # In this case there's no point at all in still having a body
                # list since the closest body fills the whole frame and is
                # also hidden by the rings. There won't be anything to see or
                # anything to navigate on. This should really only happen
                # with Saturn.
                large_bodies_by_range = []
            else:
                entirely_body = front_body_name
                metadata['body_only'] = front_body_name
                logger.info('Image is covered by %s, which is not occulted '+
                            'by anything', front_body_name)

    if entirely_body and len(large_bodies_by_range) != 1:
        logger.warn('Something is very wrong - image is entirely %s but there '+
                    'are additional bodies in the body list!', entirely_body)

    metadata['large_bodies'] = [x[0] for x in large_bodies_by_range]


                    ######################
                    # OFFSET USING STARS #
                    ######################
    
    #
    # TRY TO FIND THE OFFSET USING STARS ONLY.
    #    XXX EVEN IF STAR PHOTOMETRY FAILS, WE COULD COMBINE THE STAR MODEL 
    #    XXX WITH LATER MODELS!
    #

    stars_offset = None
    stars_confidence = 0.
    stars_corr_psf_details = None
    stars_metadata = None
    
    if (not entirely_body and
        allow_stars and (not botsim_offset or create_overlay) and
        not force_bootstrap_candidate):
        stars_only = (len(large_bodies_by_range) == 0 and
                      not has_df_rings)
        stars_metadata = stars_find_offset(obs, metadata['ra_dec_center_pred'],
                                           stars_only=stars_only,
                                           extend_fov=extend_fov,
                                           stars_config=stars_config)
        metadata['stars_metadata'] = stars_metadata
        stars_offset = stars_metadata['offset']
        metadata['stars_offset'] = stars_offset
        stars_confidence = stars_metadata['confidence']
        metadata['stars_confidence'] = stars_confidence
        stars_corr_psf_details = stars_metadata['corr_psf_details']
        if stars_offset is None:
            logger.info('Final star offset N/A')
        else:
            logger.info('Final star offset U,V %.2f,%.2f good stars %d', 
                        stars_offset[0], stars_offset[1],
                        stars_metadata['num_good_stars'])
        if create_overlay:
            # Make the extended overlay with no offset because we shift the 
            # overlay later
            stars_overlay, stars_overlay_text = stars_make_good_bad_overlay(
                              obs,
                              stars_metadata['full_star_list'], (0,0),
                              show_streaks=stars_show_streaks,
                              extend_fov=extend_fov,
                              label_avoid_mask=label_avoid_mask,
                              stars_config=stars_config)
            if label_avoid_mask is not None:
                label_avoid_mask = np.logical_or(label_avoid_mask, 
                                                 stars_overlay_text)


                    ######################
                    # CREATE BODY MODELS #
                    ######################
    
    #
    # MAKE MODELS FOR THE BODIES IN THE IMAGE, EVEN THE FUZZY ONES
    # (BECAUSE WE'LL WANT THEM IN THE OVERLAY LATER)
    #
    if (allow_saturn or allow_moons):
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            # If the whole image is a single unobstructed body, then
            # we will want to bootstrap later, so don't bother making
            # fancy models.
            no_model = (entirely_body == body_name and
                        (bodies_cartographic_data is None or
                         body_name not in bodies_cartographic_data))
            body_model, body_metadata, body_overlay_text = bodies_create_model(
                    obs, body_name, inventory=inv,
                    extend_fov=extend_fov,
                    cartographic_data=bodies_cartographic_data,
                    always_create_model=create_overlay,
                    label_avoid_mask=label_avoid_mask,
                    bodies_config=bodies_config,
                    no_model=no_model)
            bodies_metadata[body_name] = body_metadata
            if body_model is not None:
                bodies_model_list.append((body_model, body_metadata, 
                                          body_overlay_text))
            if label_avoid_mask is not None:
                label_avoid_mask = np.logical_or(label_avoid_mask,
                                                 body_overlay_text)

    # We have all the body models and all the rings information. Go through
    # the bodies and see if they are partially occulted by anything else
    # and mark appropriately.
    # Note that this includes being occulted by a moon or rings that will
    # actually be outside the image once the final offset is found. It's
    # a pain to get around this behavior, and it's unlikely to ever matter.
    for body_idx in xrange(len(bodies_model_list)-1, -1, -1):
        # Start with the backmost body and work forward
        (body_model, body_metadata, 
         body_overlay_text) = bodies_model_list[body_idx]
        body_metadata['occulted_by'] = None
        if body_idx > 0: # Not the frontmost body
            # See if any bodies in front occult this body
            for body_idx2 in xrange(body_idx):
                (body_model2, body_metadata2, 
                 body_overlay_text2) = bodies_model_list[body_idx2]
                if np.any(np.logical_and(body_model != 0, body_model2 != 0)):
                    if body_metadata['occulted_by'] is None:
                        body_metadata['occulted_by'] = []
                    body_metadata['occulted_by'].append(body_metadata2['body_name'])
            # See if the rings occult this body
            body_rings_dist = rings_dist[body_model != 0]
            inv = body_metadata['inventory']
            if np.any(body_rings_dist < inv['range']):
                if body_metadata['occulted_by'] is None:
                    body_metadata['occulted_by'] = []
                body_metadata['occulted_by'].append('RINGS')
        if body_metadata['occulted_by'] is not None:
            logger.info('%s is occulted by %s', body_metadata['body_name'],
                        str(body_metadata['occulted_by']))                
    
    if (entirely_body and 
        bodies_metadata[entirely_body]['occulted_by'] is not None):
        logger.warn('Something is very wrong - Image marked as entirely body '+
                    'but body is occulted by something!')
        logger.debug('Marking as not entirely body %s due to occulting body',
                     entirely_body)
        entirely_body = None
 
    if entirely_body:
        if (botsim_offset is None and
            entirely_body not in FUZZY_BODY_LIST and
            (bodies_cartographic_data is None or
             entirely_body not in bodies_cartographic_data)):
            # Nothing we can do here except bootstrap
            metadata['bootstrap_candidate'] = True
            logger.info('Single body without cartographic data - '+
                        'bootstrap candidate and returning')
            # Go through and update all the body metadata
            for body_name in bodies_metadata:
                if body_name in bootstrap_config['body_list']: 
                    bodies_add_bootstrap_info(obs, bodies_metadata[body_name],
                                              None, 
                                              bodies_config=bodies_config)
            metadata['end_time'] = time.time()
            return metadata

    bad_body = False
    
    navigable_bodies_model_list = []
    for body_model, body_metadata, body_text in bodies_model_list:
        body_name = body_metadata['body_name']
        if body_name in FUZZY_BODY_LIST:
            # Fuzzy bodies can't be used for navigation, but can be used
            # later to create the overlay
            logger.debug('Model: %s is fuzzy - ignoring', body_name)
            continue
        if body_name == 'TITAN':
            # Titan can't be used for primary model navigation
            logger.debug('Model: %s - ignoring', body_name)
            titan_body_metadata = body_metadata
            continue
        navigable_bodies_model_list.append(
                           (body_model, body_metadata, body_text))
        if ((bodies_cartographic_data is None or
             body_name not in bodies_cartographic_data) and
            body_metadata['size_ok'] and
            (not body_metadata['curvature_ok'] or
             not body_metadata['limb_ok'])):
            bad_body = True

    # See if Saturn is behind everything and takes up the whole image. This
    # means the background will be bright, not black, which affects navigation
    # using bodies with bad limbs later (since the limbs are now actually
    # visible).
    saturn_behind_body_model = None
    saturn_behind_body_metadata = None
    if (len(large_bodies_by_range) > 1 and         # More than just Saturn
        large_bodies_by_range[-1][0] == 'SATURN'): # Saturn in back
        set_obs_corner_bp(obs)    
        corner_body_mask = (
            obs.ext_corner_bp.where_intercepted('SATURN').
                mvals.filled(0))
        if np.all(corner_body_mask):
            for body_model, body_metadata, body_text in navigable_bodies_model_list:
                body_name = body_metadata['body_name']
                if body_name == 'SATURN':
                    saturn_behind_body_model = body_model
                    saturn_behind_body_metadata = body_metadata
                    break
        

                    #####################
                    # CREATE RING MODEL #
                    #####################
    
    if allow_rings:
        rings_model, rings_metadata, rings_overlay_text = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         always_create_model=True,
                                         label_avoid_mask=label_avoid_mask,
                                         bodies_model_list=bodies_model_list,
                                         rings_config=rings_config)
        metadata['rings_metadata'] = rings_metadata
        if label_avoid_mask is not None:
            label_avoid_mask = np.logical_or(label_avoid_mask,
                                             rings_overlay_text)

    rings_curvature_ok = (rings_metadata is not None and
                          rings_metadata['curvature_ok'])
    rings_features_ok = (rings_metadata is not None and
                         rings_metadata['fiducial_features_ok'])
    rings_any_features = (rings_metadata is not None and
                          len(rings_metadata['fiducial_features'])>0)
    rings_features_blurred = None
    if (rings_metadata is not None and 
        rings_metadata['fiducial_blur'] is not None and
        rings_metadata['fiducial_blur'] != 1.):
        rings_features_blurred = rings_metadata['fiducial_blur']
    
    if force_bootstrap_candidate:
        metadata['bootstrap_candidate'] = True
        logger.info('Forcing bootstrap candidate and returning')
        # Go through and update all the body metadata
        for body_name in bodies_metadata:
            if body_name in bootstrap_config['body_list']: 
                bodies_add_bootstrap_info(obs, bodies_metadata[body_name],
                                          None, 
                                          bodies_config=bodies_config)
        metadata['end_time'] = time.time()
        return metadata
        
        
                #####################################################
                # MERGE ALL THE MODELS TOGETHER AND FIND THE OFFSET #
                #####################################################

    model_offset = None
    model_corr_psf_details = None
    model_confidence = 0.
    model_blur_amount = None

    previously_used_model_contents = []
            
    # Try relaxing varying constraints on the bodies until we find one that
    # works. Constraints are:
    #     override_bodies_curvature_ok
    #     override_bodies_limb_ok
    #     bodies_allow_blur
    #     override_bodies_size_ok
    #     override_rings_curvature_ok
    #     override_fiducial_features_ok
    #     rings_allow_blur

    can_override_body_curvature = False
    for body_model, body_metadata, body_text in navigable_bodies_model_list:
        if ((saturn_behind_body_model is None or
             body_metadata['body_name'] != 'SATURN') and
            not body_metadata['curvature_ok']):
            can_override_body_curvature = True
            break
    
    can_override_ring_curvature = False
    can_override_ring_features_ok = False
    if rings_any_features:
        if not rings_curvature_ok:
            can_override_ring_curvature = True
        if not rings_features_ok and rings_any_features:
            can_override_ring_features_ok = True

    can_override_body_limb = False
    for body_model, body_metadata, body_text in navigable_bodies_model_list:
        if ((saturn_behind_body_model is None or
             body_metadata['body_name'] != 'SATURN') and
            not body_metadata['limb_ok']):
            can_override_body_limb = True
            break

    can_override_body_size = False
    for body_model, body_metadata, body_text in navigable_bodies_model_list:
        if ((saturn_behind_body_model is None or
             body_metadata['body_name'] != 'SATURN') and
            not body_metadata['size_ok'] and
            np.count_nonzero(body_model)):
            # We still require there to be SOMETHING in the model
            can_override_body_size = True
            break

    can_allow_bodies_blur = False
    for body_model, body_metadata, body_text in navigable_bodies_model_list:
        if ((saturn_behind_body_model is None or
             body_metadata['body_name'] != 'SATURN') and
            body_metadata['body_blur'] is not None and
            body_metadata['body_blur'] <= offset_config['maximum_blur']):
            can_allow_bodies_blur = True
            break
    
    model_phase_info_list = []
    
    if saturn_behind_body_model is None:
        saturn_behind_list = [False]
    else:
        saturn_behind_list = [True, False]
    
    for saturn_behind_flag in saturn_behind_list:
        # Try everything conservatively
        model_phase_info_list.append((0.95, False, False, False, 
                                            False, False, False,
                                            False, 
                                            saturn_behind_flag))
        
        # If we have both bodies and rings in the image, then we can relax the
        # various constraints and be more likely to get a good result because
        # the body and rings are likely to constrain different directions.
        if len(navigable_bodies_model_list) > 0 and rings_any_features:
            for const_body_curve in [False,True]:
                if const_body_curve and not can_override_body_curvature:
                    continue
                for const_body_limb in [False,True]:
                    if const_body_limb and not can_override_body_limb:
                        continue
                    for const_body_blur in [False,True]:
                        if const_body_blur and not can_allow_bodies_blur:
                            continue
                        for const_body_size in [False,True]:
                            if const_body_size and not can_override_body_size:
                                continue
                            for const_ring_curve in [False,True]:
                                if (const_ring_curve and 
                                    not can_override_ring_curvature):
                                    continue
                                for const_ring_features in [False,True]:
                                    if (const_ring_features and 
                                        not can_override_ring_features_ok):
                                        continue
                                    if (not const_body_curve and
                                        not const_body_limb and
                                        not const_body_blur and
                                        not const_ring_curve and
                                        not const_ring_features):
                                        # Already did this case
                                        continue
                                    score = (0.9 - 0.1*const_body_curve -
                                                   0.3*const_body_limb -
                                                   0.1*const_body_blur -
                                                   0.2*const_body_size -
                                                   0.1*const_ring_curve -
                                                   0.1*const_ring_features)
                                    model_phase_info_list.append(
                                                         (score,
                                                          const_body_curve,
                                                          const_body_limb,
                                                          const_body_blur,
                                                          const_body_size,
                                                          const_ring_curve,
                                                          const_ring_features,
                                                          False,
                                                          saturn_behind_flag))
        else:
            # Try overriding just body curvature and/or the body blur
            if can_override_body_curvature:
                model_phase_info_list.append((0.35,   True, False, False,
                                                     False, False, False,
                                                     False,
                                                     saturn_behind_flag))
            if can_allow_bodies_blur:
                model_phase_info_list.append((0.33,  False, False,  True,
                                                     False, False, False,
                                                     False,
                                                     saturn_behind_flag))
            if can_override_body_curvature and can_allow_bodies_blur:
                model_phase_info_list.append((0.31,   True, False,  True,
                                                     False, False, False,
                                                     False,
                                                     saturn_behind_flag))
    
            # Try overriding just body size
            if can_override_body_size:
                model_phase_info_list.append((0.20,  False, False, False,
                                                      True, False, False,
                                                     False,
                                                     saturn_behind_flag))
                
            # Try overriding just body limbs. This is too dangerous in the general
            # case, but should work if Saturn is filling the FOV behind so we see
            # the moon in relief instead of reflected light. In fact it should
            # work well enough that we aren't even going to penalize the
            # confidence level much.
            if saturn_behind_body_model is not None and can_override_body_limb:
                model_phase_info_list.append((0.25, False,  True, False, 
                                                    False, False, False,
                                                    False,
                                                    saturn_behind_flag))
                if can_override_body_curvature:
                    model_phase_info_list.append((0.23,  True,  True, False,
                                                        False, False, False,
                                                        False,
                                                        saturn_behind_flag))
                if can_allow_bodies_blur:
                    model_phase_info_list.append((0.21, False,  True,  True,
                                                        False, False, False,
                                                        False,
                                                        saturn_behind_flag))
                if can_override_body_curvature and can_allow_bodies_blur:
                    model_phase_info_list.append((0.19,  True,  True,  True,
                                                        False, False, False,
                                                        False,
                                                        saturn_behind_flag))
    
            if rings_curvature_ok:
                # Try ring blurring
                if rings_features_blurred is not None:
                    model_phase_info_list.append((0.40, False, False, False,
                                                        False, False, False,
                                                         True,
                                                        saturn_behind_flag))
                
                # Try too few features
                if can_override_ring_features_ok:
                    model_phase_info_list.append((0.10, False, False, False,
                                                        False, False,  True,
                                                        False,
                                                        saturn_behind_flag))
                    
                # Try ring blurring but with too few features - this is likely
                # to be an awful result
                if (rings_features_blurred is not None and 
                    can_override_ring_features_ok):
                    model_phase_info_list.append((0.05, False, False, False,
                                                        False, False,  True,
                                                         True,
                                                        saturn_behind_flag))
            else:
                # Try overriding just ring curvature
                if can_override_ring_curvature:
                    ### If these confidence levels are changed, look at the if 
                    ### statement in the model loop
                    model_phase_info_list.append((0.30, False, False, False,
                                                        False,  True, False, 
                                                        False,
                                                        saturn_behind_flag))
                    if rings_features_blurred is not None:
                        # ... and blurring
                        model_phase_info_list.append((0.20, 
                                                      False, False, False,
                                                      False,  True, False, 
                                                      True,
                                                      saturn_behind_flag))
                    if can_override_ring_features_ok:
                        # ... and # of ring features
                        model_phase_info_list.append((0.15,
                                                      False, False, False,
                                                      False,  True,  True,
                                                      False,
                                                      saturn_behind_flag))
                        if rings_features_blurred is not None:
                            # ... and # of ring features plus blurring
                            model_phase_info_list.append((0.11,
                                                          False, False, False,
                                                          False,  True,  True,
                                                          True,
                                                          saturn_behind_flag))

    model_phase_info_list.sort(reverse=True)
    
    model_rings_radial_only = False
    model_rings_radial_gradient = None
    
    for model_phase_info in model_phase_info_list:
        (model_confidence, 
         override_bodies_curvature_ok, 
         override_bodies_limb_ok,
         bodies_allow_blur,
         override_bodies_size_ok,
         override_rings_curvature_ok,
         override_rings_features_ok,
         rings_allow_blur,
         allow_saturn_behind) = model_phase_info

        if botsim_offset is not None:
            # This is inside the loop simply because I don't want to add 
            # another level of indentation!
            break
        
        if bodies_allow_blur and not bodies_allow_blur:
            # If no body blurring is required, then skip over assumptions that
            # include body blurring
            continue
         
        if rings_allow_blur and rings_features_blurred is None:
            # If no ring blurring is required, then skip over assumptions that
            # include ring blurring
            continue
        
# This is not really true - because we might be able to navigate on a body
# WITHOUT the rings in this case.
#         if not rings_allow_blur and rings_features_blurred is not None:
#             # If ring blurring is required, then any model that includes
#             # the rings MUST be blurred
#             continue

        logger.info('*** Trying model offset with:')
        logger.info('   override_bodies_curvature_ok %s', 
                    str(override_bodies_curvature_ok))
        logger.info('   override_bodies_limb_ok      %s', 
                    str(override_bodies_limb_ok))
        logger.info('   bodies_allow_blur            %s', 
                    str(bodies_allow_blur))
        logger.info('   override_bodies_size_ok      %s', 
                    str(override_bodies_size_ok))
        logger.info('   override_rings_curvature_ok  %s', 
                    str(override_rings_curvature_ok))
        logger.info('   override_rings_features_ok   %s', 
                    str(override_rings_features_ok))
        logger.info('   rings_allow_blur             %s', 
                    str(rings_allow_blur))
        logger.info('   allow_saturn_behind          %s', 
                    str(allow_saturn_behind))
        logger.info('Initial confidence %.2f', model_confidence)
        
        body_model_list = []
        used_model_str_list = []

        model_blur_amount = None
        
        confidence_list = []
        
        # Deal with bodies first
                
        for body_model, body_metadata, body_text in navigable_bodies_model_list:
            body_name = body_metadata['body_name']
            # For Lambert models, pay attention to curvature and limb
            if ((bodies_cartographic_data is None or
                 body_name not in bodies_cartographic_data) and
                (not (override_bodies_curvature_ok or 
                      body_metadata['curvature_ok']) or
                 not (override_bodies_limb_ok or
                      body_metadata['limb_ok']))):
                continue
            # For all models, pay attention to blurring and size
            if (not ((bodies_allow_blur and
                      body_metadata['body_blur'] is not None and
                      body_metadata['body_blur'] <= offset_config['maximum_blur']) or
                     body_metadata['body_blur'] is None) or
                not (override_bodies_size_ok or
                     body_metadata['size_ok'])):
                continue
            body_model_list.append(body_model)
            used_model_str_list.append(body_name)
            confidence_list.append(body_metadata['confidence'])
            if body_metadata['body_blur'] is not None:
                if model_blur_amount is None:
                    model_blur_amount = body_metadata['body_blur']
                else:
                    model_blur_amount = max(model_blur_amount,
                                            body_metadata['body_blur'])
                logger.info('Blurring model by at least %f because of %s',
                            body_metadata['body_blur'], body_name)


        if (override_rings_curvature_ok and len(body_model_list) == 0 and
            model_confidence > 0.3):
            # We only allow flat rings when there is also a body to use;
            # otherwise we're pretty much guaranteed the navigation will be
            # bad.
            # But if the confidence is low enough, then we KNOW it will be
            # bad and we'll do the radial projection in the end anyway.
            continue

        model_rings_radial_only = (len(body_model_list) == 0 and
                                   override_rings_curvature_ok)

        ### Now deal with rings
        
        use_rings_model = False

        # If the rings are occluded by a bumpy body, give up now because
        # the correlation will produce bad results. See for example
        # N1511727503_2.
        rings_occluded_ok = True
#         if rings_model is not None:
#             for body_name in rings_metadata['occluded_by']:
#                 for body_model, body_metadata, body_text in bodies_model_list:
#                     if (body_metadata['body_name'] == body_name and
#                         body_metadata['body_blur'] is not None):
#                         rings_occluded_ok = False
#                         logger.info('Ignoring rings because they are occluded by'+
#                                     ' a bumpy %s', body_name)
#                         break
#                 if not rings_occluded_ok:
#                     break

        if (rings_model is not None and 
            rings_occluded_ok and
            (override_rings_curvature_ok or rings_curvature_ok) and 
            (override_rings_features_ok or rings_features_ok) and
            (rings_allow_blur or rings_features_blurred is None)):
            if rings_metadata['fiducial_blur'] > offset_config['maximum_blur']:
                continue
            use_rings_model = True 
            used_model_str_list.append('RINGS')
            confidence_list.append(rings_metadata['confidence'])
            if rings_features_blurred is not None:
                logger.info('Blurring model by at least %f because of rings',
                            rings_features_blurred)
            if model_blur_amount is None:
                model_blur_amount = rings_features_blurred
            else:
                model_blur_amount = max(model_blur_amount,
                                        rings_features_blurred)
    
        # Degrade the confidence as needed
        model_confidence *= np.sqrt(np.sum(np.array(confidence_list)**2))
        
        logger.info('Final confidence %.2f', model_confidence)
        
        metadata['model_contents'] = used_model_str_list
        logger.info('Model contains %s', str(used_model_str_list))
        final_model = None
        if len(body_model_list) == 0 and not use_rings_model:
            logger.info('Nothing to model - no offset found')
            continue

        # This is a horrible hack to deal with Saturn filling the FOV behind
        # everything else.
        # We want to add Saturn to the background after everything else
        # has been done so we don't count Saturn as a usefully navigable
        # body in the above tests.
        if (allow_saturn_behind and
            saturn_behind_body_model is not None):
            body_model_list.append(saturn_behind_body_model)
            used_model_str_list.append('SATURN')
            logger.info('Adding SATURN as whole-image background object')
            # Don't bother adding anything to the confidence list
            # Also we don't care about blurring for Saturn in this case since
            # there's no limb.
            metadata['model_contents'] = used_model_str_list
            logger.info('Model now contains %s', str(used_model_str_list))
                
        if (used_model_str_list, 
            model_blur_amount) in previously_used_model_contents:
            logger.info('Ignoring configuration - already tried')
            continue
        
        previously_used_model_contents.append((used_model_str_list,
                                               model_blur_amount))

        # We have at least one viable component of the model,
        # so combine everything together and try to find the offset
        body_model_list.reverse()
        if len(body_model_list) == 0:
            assert use_rings_model
            final_model = _normalize(rings_model)
        else:
            final_model = _combine_models(body_model_list, solid=True)
            if use_rings_model:
                final_model = _combine_models([final_model, rings_model])

        gaussian_blur = offset_config['default_gaussian_blur']
        if model_blur_amount is not None:
            gaussian_blur = max(model_blur_amount, gaussian_blur)
            logger.info('Blurring model by %f', model_blur_amount)
        else:
            logger.info('Blurring model by default amount %f',
                        gaussian_blur)
        model_filter_func = (lambda image, masked=False:
                 _model_filter(image, 
                               gaussian_blur, 
                               offset_config['median_filter_size'],
                               offset_config['median_filter_blur'],
                               masked=masked))
        model_offset_list = find_correlation_and_offset(
                                   obs.ext_data,
                                   final_model, search_size_min=0,
                                   search_size_max=(search_size_max_u, 
                                                    search_size_max_v),
                                   extend_fov=extend_fov,
                                   filter=model_filter_func)
        model_offset = None
        model_corr_psf_details = None
        peak = None
        if len(model_offset_list) > 0:
            (model_offset, peak, model_corr_psf_details) = model_offset_list[0]
        
        if model_offset is not None:
            logger.info('Primary correlation U,V %d,%d',
                        model_offset[0], model_offset[1])
            if model_rings_radial_only:
                (model_offset, 
                 model_rings_radial_gradient) = rings_offset_radial_projection(
                                                  obs, model_offset,
                                                  extend_fov=extend_fov,
                                                  rings_config=rings_config)
                logger.info('Correlation after ring '+
                            'radial reprojection U,V %.2f,%.2f',
                            model_offset[0], model_offset[1])
                if (abs(model_offset[0]) >= extend_fov[0] or
                    abs(model_offset[1]) >= extend_fov[1]):
                    logger.info('Radially reprojected offset is larger than search limits')
                    model_offset = None
        if model_offset is not None:
            model_offset_int = (int(np.round(model_offset[0])),
                                int(np.round(model_offset[1])))
            # Run it again to make sure it works with a fully sliced
            # model.
            # No point in doing this if we allow_masked, since we've
            # already used each model slice independently.
            sec_search_size = offset_config['secondary_corr_search_size']
            if model_rings_radial_only:
                sec_search_size = (search_size_max_u, 
                                   search_size_max_v)
            (model_offset, secc_conf_factor, 
             model_corr_psf_details) = _iterate_secondary_correlation(
                 obs, final_model, model_offset_int, peak,
                 0, sec_search_size,
                 model_filter_func, model_rings_radial_only,
                 extend_fov, offset_config, rings_config)
            if model_offset is None:
                metadata['secondary_corr_ok'] = False
            else:
                metadata['secondary_corr_ok'] = True
                model_offset_int = (int(np.round(model_offset[0])),
                                    int(np.round(model_offset[1])))
                model_confidence *= secc_conf_factor
                logger.info('Confidence after secondary correlation %.2f',
                            model_confidence)
                
        if model_offset is None:
            logger.info('Resulting model offset N/A')
        else:
            logger.info('Resulting model offset U,V %.2f,%.2f', 
                        model_offset[0], model_offset[1])
            if model_corr_psf_details is not None:
                corr_log_sigma(logger, model_corr_psf_details)
    
            shifted_model = shift_image(final_model, model_offset_int[0], 
                                        model_offset_int[1])
            shifted_model = unpad_image(shifted_model, extend_fov)

            # Only trust a model if it has at least a reasonable number of 
            # pixels and parts of the model are not right along the edge.
            cov_threshold = offset_config['model_cov_threshold']
            edge_pixels = offset_config['model_edge_pixels']
            model_count = np.count_nonzero(
                         shifted_model[edge_pixels:-edge_pixels+1,
                                       edge_pixels:-edge_pixels+1])
            if model_count < cov_threshold:
                logger.info('Final shifted model has too little coverage '+
                            '- reducing confidence')
                model_confidence *= float(model_count) / cov_threshold
        if model_confidence < offset_config['lowest_confidence']:
            logger.info('Resulting confidence is absurdly low - aborting')
            model_offset = None
            model_corr_psf_details = None
    
        if model_offset is not None:
            break
        
    # Out of the big model testing loop!

# XXX IMPLEMENT RING BOOTSTRAPPING
#    if model_offset is None:
#        if (rings_curvature_ok and not rings_features_ok and
#            not metadata['bootstrap_body']): # XXX
#            logger.info('Ring curvature OK but not enough fiducial '+
#                        'features - candidate for bootstrapping')
#            metadata['bootstrapping_candidate'] = True
#            metadata['bootstrap_body'] = 'RINGS'
#        if rings_features_ok and not rings_curvature_ok:
#            logger.info('Candidate for ring radial scan') # XXX

    if model_offset is not None:
        metadata['model_override_bodies_curvature'] = override_bodies_curvature_ok 
        metadata['model_override_bodies_limb'] = override_bodies_limb_ok
        metadata['model_override_rings_curvature'] = override_rings_curvature_ok
        metadata['model_override_fiducial_features'] = override_rings_features_ok
        metadata['model_rings_blur'] = rings_allow_blur
        metadata['model_bodies_blur'] = bodies_allow_blur
        metadata['model_confidence'] = model_confidence
        metadata['model_blur_amount'] = model_blur_amount
        metadata['model_rings_radial_gradient'] = model_rings_radial_gradient
        if ('RINGS' not in used_model_str_list and
            len(used_model_str_list) < 
                offset_config['num_bodies_threshold'] and
            np.count_nonzero(final_model) < 
                offset_config['bodies_cov_threshold']):
            logger.info('Too few moons, no rings, model has too little '+
                        'coverage - reducing confidence')
            model_confidence *= 0.25

    metadata['model_offset'] = model_offset
    metadata['model_corr_psf_details'] = model_corr_psf_details
    
    if model_offset is None:
        logger.info('Final model offset N/A')
    else:   
        logger.info('Final model offset U,V %.2f,%.2f (conf %.2f)',
                    model_offset[0], model_offset[1], model_confidence)
        if model_corr_psf_details is not None:
            corr_log_sigma(logger, model_corr_psf_details)


                #####################################
                # DEAL WITH TITAN AS A SPECIAL CASE #
                #####################################

    titan_offset = None
    titan_confidence = 0.
    
    if (botsim_offset is None and 
        titan_body_metadata is not None and
        titan_body_metadata['size_ok']): 
        unpadded_final_model = None
        if final_model is not None:
            unpadded_final_model = unpad_image(final_model, extend_fov)
        titan_metadata = titan_navigate(obs, unpadded_final_model,
                                        extend_fov=extend_fov, 
                                        titan_config=titan_config)
        metadata['titan_metadata'] = titan_metadata
        titan_offset = titan_metadata['offset']
        titan_confidence = titan_metadata['confidence']
        metadata['titan_offset'] = titan_offset
        metadata['titan_confidence'] = titan_confidence

        
                ########################################
                # COMPARE STARS OFFSET TO MODEL OFFSET #
                ########################################

    offset = None
    corr_psf_details = None
    # BOTSIM offset wins over everything
    if botsim_offset is not None:
        offset = botsim_offset
        metadata['offset_winner'] = 'BOTSIM'
        metadata['confidence'] = 1.
    elif (titan_offset is not None and
          (model_offset is None or titan_confidence > model_confidence) and
          (stars_offset is None or titan_confidence > stars_confidence)):
        # Titan wins over everything else
        offset = titan_offset
        metadata['offset_winner'] = 'TITAN'
        metadata['confidence'] = titan_confidence
    elif (stars_offset is not None and
          (model_offset is None or stars_confidence > model_confidence) and
          (titan_offset is None or stars_confidence > titan_confidence)):
        offset = stars_offset
        corr_psf_details = stars_corr_psf_details
        metadata['offset_winner'] = 'STARS'
        metadata['confidence'] = stars_confidence
    elif model_offset is not None:
        offset = model_offset
        corr_psf_details = model_corr_psf_details
        metadata['offset_winner'] = 'MODEL'
        metadata['confidence'] = model_confidence
    else:
        metadata['offset_winner'] = None
        metadata['confidence'] = 0.
        
        if model_offset is not None and stars_offset is not None:
            disagree_threshold = offset_config['stars_model_diff_threshold']
            stars_threshold = offset_config['stars_override_threshold']
            if (abs(stars_offset[0]-model_offset[0]) >= disagree_threshold or
                abs(stars_offset[1]-model_offset[1]) >= disagree_threshold):
                if (stars_metadata['num_good_stars'] >= stars_threshold):
                    logger.info('Star and model offsets disagree by too '+
                                'much - trusting star result')
                else:
                    logger.info('Star and model offsets disagree by too '+
                                'much - ignoring star result')
                    star_list = stars_metadata['full_star_list']
                    for star in star_list:
                        star.photometry_confidence = 0. # We changed the offset
                        star.use_for_correlation = False
                    star_overlay = stars_make_good_bad_overlay(obs,
                              star_list, (0,0),
                              extend_fov=extend_fov,
                              stars_config=stars_config)
                    offset = model_offset
                    corr_psf_details = model_corr_psf_details

    if (offset is not None and
        (abs(offset[0]) > extend_fov[0] or 
         abs(offset[1]) > extend_fov[1])):
        logger.info('Final offset is beyond maximum allowable offset!')
        offset = None
        corr_psf_details = None
        metadata['offset_winner'] = None 
        
    logger.info('Summary:')
    if stars_offset is None:
        logger.info('  Final star offset     N/A')
    else:
        logger.info('  Final star offset     U,V %.2f,%.2f (conf %.2f) good stars %d', 
                    stars_offset[0], stars_offset[1],
                    stars_confidence, stars_metadata['num_good_stars'])

    if model_offset is None:
        logger.info('  Final model offset    N/A')
    else:
        logger.info('  Final model offset    U,V %.2f,%.2f (conf %.2f)', 
                    model_offset[0], model_offset[1], model_confidence)

    if titan_offset is None:
        logger.info('  Final Titan offset    N/A')
    else:
        logger.info('  Final Titan offset    U,V %d,%d (conf %.2f)', 
                    titan_offset[0], titan_offset[1], titan_confidence)

    if botsim_offset is not None:
        logger.info('  FINAL OFFSET SET BY BOTSIM')

    if offset is None:
        logger.info('  Final combined offset FAILED')
    else:
        logger.info('  Final combined offset U,V %.2f,%.2f (conf %.2f)', 
                    offset[0], offset[1], metadata['confidence'])
    if corr_psf_details is not None:
        corr_log_sigma(logger, corr_psf_details)

    metadata['offset'] = offset
    metadata['corr_psf_details'] = corr_psf_details

    if offset is not None:
        orig_fov = obs.fov
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
        set_obs_center_bp(obs, force=True)
        ra = obs.center_bp.right_ascension()
        dec = obs.center_bp.declination()
        metadata['ra_dec_center_offset'] = (ra.vals,dec.vals)
        obs.fov = orig_fov
        set_obs_center_bp(obs, force=True)

    # Go through and update all the body metadata
    for body_name in bodies_metadata:
        if body_name in bootstrap_config['body_list']: 
            bodies_add_bootstrap_info(obs, bodies_metadata[body_name],
                                      offset, bodies_config=bodies_config)


                ######################
                # CREATE THE OVERLAY #
                ######################

    if create_overlay:
        ## BODIES ##
        # We recreate the model lists because there are models we might
        # include on the overlay that we didn't use for correlation.
        o_bodies_model_list = [x[0] for x in bodies_model_list]
        o_bodies_text_list = [x[2] for x in bodies_model_list]
        label_avoid_mask = np.zeros(obs.data.shape, dtype=np.bool)
        if len(o_bodies_model_list) > 0:
            o_bodies_model_list.reverse()
            bodies_combined = _combine_models(o_bodies_model_list, solid=True)
            bodies_overlay = _normalize(bodies_combined) * 255
            bodies_overlay_text = _combine_text(o_bodies_text_list)
            if offset is not None:
                bodies_overlay = shift_image(bodies_overlay, 
                                             int(np.round(offset[0])), 
                                             int(np.round(offset[1])))
                if bodies_overlay_text is not None:
                    bodies_overlay_text = shift_image(bodies_overlay_text, 
                                                 int(np.round(offset[0])), 
                                                 int(np.round(offset[1])))
            metadata['bodies_overlay'] = unpad_image(bodies_overlay, 
                                                     extend_fov)
            if bodies_overlay_text is None:
                metadata['bodies_overlay_text'] = None
            else:
                metadata['bodies_overlay_text'] = unpad_image(
                                                      bodies_overlay_text,
                                                      extend_fov)
                label_avoid_mask = np.logical_or(label_avoid_mask,
                                                 metadata['bodies_overlay_text'])
                color_overlay[...,1] = np.clip(bodies_overlay+
                                               bodies_overlay_text*255,
                                               0, 255)
        
        ## RINGS ##
        if rings_model is not None:
            # The rings model is usually 0-1
            # The rings overlay text is Bool
            rings_overlay = _normalize(rings_model) * 255
            if offset is not None:
                rings_overlay = shift_image(rings_overlay, 
                                            int(np.round(offset[0])), 
                                            int(np.round(offset[1])))
                if rings_overlay_text is not None:
                    rings_overlay_text = shift_image(rings_overlay_text, 
                                                     int(np.round(offset[0])), 
                                                     int(np.round(offset[1])))
            metadata['rings_overlay'] = unpad_image(rings_overlay,
                                                    extend_fov)
            if rings_overlay_text is None:
                metadata['rings_overlay_text'] = None
            else:
                metadata['rings_overlay_text'] = unpad_image(rings_overlay_text,
                                                             extend_fov)
                label_avoid_mask = np.logical_or(label_avoid_mask,
                                                 metadata['rings_overlay_text'])
                color_overlay[...,2] = np.clip(rings_overlay+
                                               rings_overlay_text*255,
                                               0, 255)
        
        ## STARS ##
        if stars_overlay is not None:
            # The stars overlay is already 0-255. We don't normalize the stars
            # overlay because it's already set up exactly how we want it.
            # However, we do need to normalize the text, which is a Bool.
            (new_stars_overlay, 
             new_stars_overlay_text) = stars_make_good_bad_overlay(
                              obs,
                              stars_metadata['full_star_list'], offset,
                              label_avoid_mask=label_avoid_mask,
                              show_streaks=stars_show_streaks,
                              stars_config=stars_config)
            if offset is not None:
                # We need to shift the original extended overlay, but not the
                # new non-extended one since it was created with an offset
                # in place
                stars_overlay = shift_image(stars_overlay, 
                                            int(np.round(offset[0])), 
                                            int(np.round(offset[1])))
                if stars_overlay_text is not None:
                    stars_overlay_text = shift_image(stars_overlay_text, 
                                                     int(np.round(offset[0])), 
                                                     int(np.round(offset[1])))
            metadata['stars_overlay'] = new_stars_overlay
            metadata['stars_overlay_text'] = new_stars_overlay_text
            if new_stars_overlay_text is not None:
                label_avoid_mask = np.logical_or(label_avoid_mask,
                                                 metadata['stars_overlay_text'])
                color_overlay[...,0] = np.clip(stars_overlay+
                                               stars_overlay_text*255,
                                               0, 255)

    metadata['ext_overlay'] = color_overlay
    if color_overlay is not None:
        metadata['overlay'] = unpad_image(color_overlay, extend_fov)

                #########################################
                # FIGURE OUT BOOTSTRAPPING IMPLICATIONS #
                #########################################
    
    # For moons, we mark a bootstrap candidate if any usable body
    # is "bad" - bad limb or curvature, no cartographic data -
    # but is still a good size
    if offset is None and bad_body:
        logger.info('Marking as bootstrap candidate')
        metadata['bootstrap_candidate'] = True
    
    metadata['end_time'] = time.time()

    logger.info('Total elapsed time %d seconds', 
                metadata['end_time']-metadata['start_time'])
    
    return metadata

def _scale_image(img, blackpoint, whitepoint, gamma):
    """Scale a 2-D image based on blackpoint, whitepoint, and gamma.
    
    Inputs:
    
    img                The 2-D image.
    blackpoint         Any element below the blackpoint will be black.
    whitepoint         Any element above the whitepoint will be white.
    gamma              Non-linear stretch (1.0 = linear stretch).
    """ 
    if whitepoint < blackpoint:
        whitepoint = blackpoint
        
    if whitepoint == blackpoint:
        whitepoint += 0.00001
    
    greyscale_img = np.floor((np.maximum(img-blackpoint, 0)/
                              (whitepoint-blackpoint))**gamma*256)
    greyscale_img = np.clip(greyscale_img, 0, 255) # Clip black and white
    return greyscale_img

def offset_create_overlay_image(obs, metadata,
                                blackpoint=None, whitepoint=None,
                                whitepoint_ignore_frac=1., 
                                gamma=None,
                                stars_blackpoint=None, stars_whitepoint=None,
                                stars_whitepoint_ignore_frac=1., 
                                stars_gamma=0.5,
                                font=None,
                                interpolate_missing_stripes=False):
    img = obs.data
    if interpolate_missing_stripes:
        img = image_interpolate_missing_stripes(img)
    offset = metadata['offset']
    if offset is None:
        offset = (0,0)
    stars_overlay = metadata['stars_overlay']
    stars_overlay_text = metadata['stars_overlay_text']
    bodies_overlay = metadata['bodies_overlay']
    bodies_overlay_text = metadata['bodies_overlay_text']
    rings_overlay = metadata['rings_overlay']
    rings_overlay_text = metadata['rings_overlay_text']
    bodies_metadata = metadata['bodies_metadata']

    image_font = None
    if font is not None:
        image_font = ImageFont.truetype(font[0], font[1])
        
    # Contrast stretch the main image
    if blackpoint is None:
        blackpoint = np.min(img)

    if whitepoint is None:
        img_sorted = sorted(list(img.flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
    if gamma is None:
        gamma = 0.5
        
    greyscale_img = _scale_image(img, blackpoint, whitepoint, gamma)

    stars_metadata = metadata['stars_metadata']
    titan_metadata = metadata['titan_metadata']
    
    if stars_metadata is not None:
        star_list = stars_metadata['full_star_list']
        if star_list is not None:
            for star in star_list:
                width = star.overlay_box_width
                if width == 0 or star.conflicts:
                    continue
                width -= 1
                u_idx = int(star.u+offset[0])
                v_idx = int(star.v+offset[1])
                stars_data = greyscale_img[max(v_idx-width,0):
                                           min(v_idx+width+1,img.shape[0]),
                                           max(u_idx-width,0):
                                           min(u_idx+width+1,img.shape[1])]
                stars_bp = stars_blackpoint
                stars_wp = stars_whitepoint
                if stars_blackpoint is None:
                    stars_bp = np.min(stars_data)
                if stars_whitepoint is None:
                    stars_wp = np.max(stars_data)
                stars_data[:,:] = _scale_image(stars_data, stars_bp, stars_wp,
                                               stars_gamma)
        
    mode = 'RGB'
    combined_data = np.zeros(greyscale_img.shape + (3,), dtype=np.uint8)
    combined_data[:,:,1] = greyscale_img[:,:]
    
    overlay = np.zeros(obs.data.shape, dtype=np.float)
    
    if stars_overlay is not None:
        overlay += stars_overlay
    if rings_overlay is not None:
        overlay += rings_overlay
    if bodies_overlay is not None:
        overlay += bodies_overlay

    overlay = np.clip(overlay, 0, 255)
    combined_data[:,:,0] = overlay

    if stars_overlay_text is not None:
        text_ok = stars_overlay_text != 0
        combined_data[text_ok, 0] = stars_overlay_text[text_ok] * 255 # Bool
        combined_data[text_ok, 1] = stars_overlay_text[text_ok] * 255
        combined_data[text_ok, 2] = stars_overlay_text[text_ok] * 255
    if rings_overlay_text is not None:
        rings_ok = rings_overlay_text != 0
        combined_data[rings_ok, 0] = rings_overlay_text[rings_ok] * 255 # Bool
        combined_data[rings_ok, 1] = rings_overlay_text[rings_ok] * 255
        combined_data[rings_ok, 2] = rings_overlay_text[rings_ok] * 255
    if bodies_overlay_text is not None:
        bodies_ok = bodies_overlay_text != 0
        combined_data[bodies_ok, 0] = bodies_overlay_text[bodies_ok] * 255 # Bool
        combined_data[bodies_ok, 1] = bodies_overlay_text[bodies_ok] * 255
        combined_data[bodies_ok, 2] = bodies_overlay_text[bodies_ok] * 255

    text_im = Image.frombuffer('RGB', (combined_data.shape[1], 
                                     combined_data.shape[0]), 
                               combined_data, 'raw', 'RGB', 0, 1)
    text_draw = ImageDraw.Draw(text_im)

    data_lines = []
    data_lines.append('%s %s %.2f %s %s' % (
                    obs.filename[:13], 
                    cspice.et2utc(obs.midtime, 'C', 0).replace(' ','-'),
                    obs.texp, obs.filter1, obs.filter2))
    offset_winner = metadata['offset_winner']
    if offset_winner == 'BOTSIM':
        data_line = 'BOTSIM %d,%d' % offset
        data_lines.append(data_line)
    else:
        stars_offset = metadata['stars_offset']
        model_offset = metadata['model_offset']
        titan_offset = metadata['titan_offset']
        
        offsets = []
        
        star_confidence = 0.
        if stars_metadata is None:
            data_line = 'Stars N/A'
        elif stars_offset is None:
            data_line = 'Stars None'
        else:
            if offset_winner == 'STARS':
                data_line = 'STARS'
            else:
                data_line = 'Stars'
            data_line += ' %.2f,%.2f' % stars_offset
            if ('stars_metadata' in metadata and
                'confidence' in metadata['stars_metadata']):
                star_confidence = metadata['stars_metadata']['confidence']
                data_line += '/%.2f' % star_confidence 
        offsets.append((star_confidence, data_line))
        
        model_confidence = 0.
        if model_offset is None:
            data_line = 'Model N/A'
        else:
            if (offset_winner != 'STARS' and
                offset_winner != 'TITAN'):
                data_line = 'MODEL'
            else:
                data_line = 'Model'
            data_line += ' %.2f,%.2f' % model_offset
            model_confidence = metadata['model_confidence']
            data_line += '/%.2f' % model_confidence
            if metadata['model_rings_radial_gradient'] is not None:
                data_line += ' NOLONG'
        offsets.append((model_confidence, data_line))
        
        titan_confidence = 0.
        if titan_metadata is None:
            data_line = 'Titan N/A'
        elif titan_offset is None:
            data_line = 'Titan None'
        else:
            if offset_winner == 'TITAN':
                data_line = 'TITAN'
            else:
                data_line = 'Titan'
            data_line += ' %d,%d' % titan_offset
            if ('titan_metadata' in metadata and
                'confidence' in metadata['titan_metadata']):
                titan_confidence = metadata['titan_metadata']['confidence']
                data_line += '/%.2f' % titan_confidence
        offsets.append((titan_confidence, data_line))
        offsets.sort()
        
        data_line = ' | '.join([x[1] for x in offsets])
        data_lines.append(data_line)

        model_contents = metadata['model_contents']
        if model_contents is not None and model_contents != []:
            new_model_contents = []
            for model_content in sorted(model_contents):
                sfx = ''
                if (bodies_metadata and model_content in bodies_metadata and
                    bodies_metadata[model_content]['cartographic_data_source']):
                    sfx = '(CART)'
                new_model_contents.append(model_content+sfx)
            model_contents_str = '+'.join(new_model_contents)
            data_lines.append('Model: '+model_contents_str)

        if metadata['bootstrap_candidate']:
            data_line = 'Bootstrap Cand'
            if (metadata['large_bodies'] is not None and
                len(metadata['large_bodies']) > 0):
                data_line += ' ' + metadata['large_bodies'][0]
            data_lines.append(data_line)
        
    text_size_h_list = []
    text_size_v_list = []
    for data_line in data_lines:
        text_size = text_draw.textsize(data_line, font=image_font)
        text_size_h_list.append(text_size[0])
        text_size_v_list.append(text_size[1])

    text_size_h = np.max(text_size_h_list)+6
    text_size_v = np.sum(text_size_v_list)+6       

    # Look for the emptiest corner - no text
    best_count = 1e38
    best_u = None
    best_v = None
    
    for v in [[0, text_size_v+1],
              [combined_data.shape[0]-text_size_v,
               combined_data.shape[0]]]:
        for u in [[0, text_size_h+1],
                  [combined_data.shape[1]-text_size_h,
                   combined_data.shape[1]]]:
            count = np.count_nonzero(combined_data[v[0]:v[1],u[0]:u[1],2])
            if count < best_count:
                best_u = u[0]
                best_v = v[0]
                best_count = count
            
    best_u += 3
    best_v += 3
    
    for data_line, v_inc in zip(data_lines, text_size_v_list):
        text_draw.text((best_u,best_v), data_line, fill=(255,255,255),
                       font=image_font)
        best_v += v_inc

    combined_data = np.array(text_im.getdata()).reshape(combined_data.shape)

    combined_data = np.cast['uint8'](combined_data)

    return combined_data

def offset_result_str(metadata):
    ret = ''
    if metadata is None:
        ret += 'No offset file written'
        return ret

    # Fix up the metadata for old files - eventually this should
    # be removed! XXX
    if 'error' in metadata:
        metadata['status'] = 'error'
        metadata['status_detail1'] = metadata['error']
        metadata['status_detail2'] = metadata['error_traceback']
    elif 'status' not in metadata:
        metadata['status'] = 'ok'

    status = metadata['status']
    if status == 'error':
        ret += 'ERROR: '
        error = metadata['status_detail1']
        if error.startswith('SPICE(NOFRAMECONNECT)'):
            ret += 'SPICE KERNEL MISSING DATA AT ' + error[34:53]
        else:
            ret += error 
        return ret
    
    if status == 'skipped':
        ret += 'SKIPPED: '+metadata['status_detail1']
        return ret
        
    assert status == 'ok'
    
    offset = metadata['offset']
    if offset is None:
        offset_str = '  N/A  '
    else:
        offset_str = '%3d,%3d' % tuple(offset)
    stars_offset = metadata['stars_offset']
    if stars_offset is None:
        stars_offset_str = '  N/A  '
    else:
        stars_offset_str = '%3d,%3d' % tuple(stars_offset)
    model_offset = metadata['model_offset']
    if model_offset is None:
        model_offset_str = '  N/A  '
    else:
        model_offset_str = '%3d,%3d' % tuple(model_offset)
    titan_offset = metadata['titan_offset']
    if titan_offset is None:
        titan_offset_str = '  N/A  '
    else:
        titan_offset_str = '%3d,%3d' % tuple(titan_offset)
    filter1 = metadata['filter1']
    filter2 = metadata['filter2']
    the_size = '%dx%d' % tuple(metadata['image_shape'])
    the_size = '%9s' % the_size
    the_time = cspice.et2utc(metadata['midtime'], 'C', 0)
    single_body_str = None
    if metadata['body_only']:
        single_body_str = 'Filled with ' + metadata['body_only']
    if metadata['rings_only']:
        single_body_str = 'Filled with rings'
    bootstrap_str = None
    if metadata['bootstrapped']:
        bootstrap_str = 'Bootstrapped'
    elif metadata['bootstrap_candidate']:
        bootstrap_str = 'Bootstrap cand'
        if (metadata['large_bodies'] is not None and
            len(metadata['large_bodies']) > 0):
            bootstrap_str += ' ' + metadata['large_bodies'][0]
        
    ret += the_time + ' ' + ('%4s'%filter1) + '+' + ('%-5s'%filter2) + ' '
    ret += the_size
    ret += ' Final ' + offset_str
    offset_winner = metadata['offset_winner']
    if offset_winner == 'BOTSIM':
        ret += '  BOTSIM'
    else:
        if offset_winner == 'STARS':
            ret += '  STAR ' + stars_offset_str
        else:
            ret += '  Star ' + stars_offset_str
        if offset_winner == 'MODEL':
            ret += '  MODEL ' + model_offset_str
        else:
            ret += '  Model ' + model_offset_str
        if offset_winner == 'TITAN':
            ret += '  TITAN ' + titan_offset_str
        else:
            ret += '  Titan ' + titan_offset_str
        if bootstrap_str:
            ret += ' ' + bootstrap_str
        if bootstrap_str and single_body_str:
            ret += ' '
        if single_body_str:
            ret += ' ' + single_body_str
        model_blur = metadata['model_blur_amount']
        if model_blur is not None:
            ret += (' (Blur %.5f)' % model_blur)
        
    return ret
