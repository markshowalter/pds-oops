###############################################################################
# cb_offset.py
#
# Routines related to finding image offsets.
#
# Exported routines:
#    master_find_offset
#    offset_create_overlay_image
###############################################################################

import cb_logging
import logging

import time

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filt

import imgdisp
import oops

from cb_bodies import *
from cb_config import MAX_POINTING_ERROR, LARGE_BODY_LIST, FUZZY_BODY_LIST
from cb_correlate import *
from cb_rings import *
from cb_stars import *
from cb_titan import *
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
            new_model[model != 0] = model[model != 0]
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

                   force_offset=None,
                   force_bootstrap_candidate=False):
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

        force_offset             None to find the offset automatically or
                                 a tuple (U,V) to force the result offset.
                                 This is useful for creating an overlay 
                                 with a known offset.
        force_bootstrap_candidate  True to force this image to be a bootstrap
                                 candidate even if we really could find a
                                 viable offset.

    Returns:
        metadata           A dictionary containing information about the
                           offset result:

          Data about the image:

            'camera'           The name of the camera: 'NAC' or 'WAC'.
            'filter1'          The names of the filters used.
            'filter2'
            'image_shape'      A tuple indicating the shape (in pixels) of the
                               image.
            'midtime'          The midtime of the observation.
            'ra_dec_corner_orig'
                               A tuple (ra_min, ra_max, dec_min, dec_max)
                               giving the corners of the FOV-extended image
                               with the original SPICE navigation.
            'ra_dec_center_orig'
                               A tuple (ra,dec) for the center of the image
                               with the original SPICE navigation.
            'ra_dec_center_offset'
                               A tuple (ra,dec) for the center of the image
                               with the new navigation.

          Data about the offset process:
          
            'offset'           The final (U,V) offset. None if offset finding
                               failed.
            'large_bodies'     A list of the large bodies present in the image
                               sorted in descending order by range.
            'body_only'        False if the image doesn't consist entirely of
                               a single body and nothing else.
                               Otherwise the name of the body.
            'rings_only'       True if the image consists entirely of the main
                               rings and nothing else.
            'used_objects_type' The type of objects used for the final offset:
                               None, 'stars', or 'model'.
            'model_contents'   A list of objects used to create the
                               non-star model: 'rings' and body names.
                               If the contents is only "TITAN", then the
                               special photometry-based Titan navigation
                               was performed.
            'model_overrides_stars' True if the non-star model was more trusted
                               than the star model.
            'stars_metadata'   The metadata from star matching. None if star
                               matching not performed. The star-matching offset
                               is included here.
            'bodies_metadata'  A dictionary containing the metadata for each
                               body in the large_bodies list.
            'rings_metadata'   The metadata from ring modeling. None if ring
                               modeling not performed.
            'model_offset'     The (U,V) offset determined by model matching.
                               None if model matching failed.
            'secondary_corr_ok' True if secondary model correlation was
                               successful. None if it wasn't performed.
            'start_time'       The start time (s) of the entire offset process.
            'end_time'         The end time (s) of the entire offset process.
            
          Data for bootstrapping:
            
            'bootstrap_candidate'
                               True if the image is a good candidate for 
                               future bootstrapping attempts.
            'bootstrap_body'   The largest (pixel size) body in the image.
            'bootstrapped'     True if the image was navigated using 
                               bootstrapping.
            'bootstrap_mosaic_path'
                               The file path where the mosaic data that was
                               used to attempt bootstrapping this image
                               is stored.
            'bootstrap_mosaic_filenames'
                               The list of filenames used to create the mosaic.
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
    if bodies_cartographic_data is None:
        logger.info('No cartographic data provided')
    else:
        logger.info('Cartographic data provided for: %s',
                     str(sorted(bodies_cartographic_data.keys())))
        
    if offset_config is None:
        offset_config = OFFSET_DEFAULT_CONFIG
        
    masked_model = False
    
    extend_fov = MAX_POINTING_ERROR[obs.data.shape, obs.detector]
    search_size_max_u, search_size_max_v = extend_fov
    
    set_obs_ext_data(obs, extend_fov)
    set_obs_ext_corner_bp(obs, extend_fov)
    
    bodies_model_list = []
    rings_model = None
    rings_overlay_text = None
    titan_model = None
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
    # Image
    metadata['camera'] = obs.detector
    metadata['filter1'] = obs.filter1
    metadata['filter2'] = obs.filter2
    metadata['image_shape'] = obs.data.shape
    metadata['midtime'] = obs.midtime
    metadata['ra_dec_corner_orig'] = compute_ra_dec_limits(
                                           obs, extend_fov=extend_fov)
    set_obs_center_bp(obs)
    ra = obs.center_bp.right_ascension()
    dec = obs.center_bp.declination()
    metadata['ra_dec_center_orig'] = (ra.vals,dec.vals)
    metadata['ra_dec_center_offset'] = None
    # Offset process
    metadata['start_time'] = start_time
    metadata['end_time'] = None
    metadata['offset'] = None
    metadata['body_only'] = False
    metadata['rings_only'] = False
    metadata['used_objects_type'] = None
    metadata['model_contents'] = []
    metadata['model_overrides_stars'] = False
    metadata['stars_metadata'] = None
    bodies_metadata = {}
    metadata['bodies_metadata'] = bodies_metadata
    metadata['rings_metadata'] = None
    metadata['titan_metadata'] = None
    metadata['model_offset'] = None
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
    metadata['bootstrap_body'] = None
    metadata['bootstrapped'] = False
    metadata['bootstrap_mosaic_path'] = None
    metadata['bootstrap_mosaic_filenames'] = None
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
    large_bodies_by_range = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_range.sort(key=lambda x: x[1]['range'])
#    large_bodies_by_size = [(x, large_body_dict[x]) for x in large_body_dict]
#    # Sort by area of the enclosing square
#    large_bodies_by_size.sort(key=lambda x: 
#                  reverse=True)
    
    logger.info('Large body inventory %s', [x[0] for x in large_bodies_by_range])

    metadata['large_bodies'] = [x[0] for x in large_bodies_by_range]
        
    for front_body_name, inv in large_bodies_by_range:
        size = (inv['u_max']-inv['u_min'])*(inv['v_max']-inv['v_min']) 
        if size >= offset_config['min_body_area']:
            break
    else:
        front_body_name = None

    entirely_body = False
        
    if front_body_name is not None:
        # See if the front-most actually visible body takes up the entire image    
        if ((front_body_name == 'SATURN' and allow_saturn) or
            (front_body_name != 'SATURN' and allow_moons)):
            corner_body_mask = (
                obs.ext_corner_bp.where_intercepted(front_body_name).
                    vals)
            if np.all(corner_body_mask):
                logger.info('Image appears to be covered by %s', front_body_name)
                entirely_body = front_body_name
                metadata['body_only'] = front_body_name

    if front_body_name in FUZZY_BODY_LIST:
        front_body_name = None

    metadata['bootstrap_body'] = front_body_name
    if front_body_name is not None:
        logger.info('Bootstrap body %s', metadata['bootstrap_body'])


    # See if the main rings take up the entire image
    # XXX THIS IS NOT A VALID TEST    
    entirely_rings = False
    radii = obs.ext_corner_bp.ring_radius('saturn:ring').vals.astype('float')
    if len(large_bodies_by_range) == 0:
        radii_good = np.logical_and(radii > RINGS_MIN_RADIUS,
                                    radii < RINGS_MAX_RADIUS)
        if np.all(radii_good):  
            logger.info('Image appears to be entirely rings')
            entirely_rings = True
            metadata['rings_only'] = True
#            entirely_rings = False # XXX
    
    
                    ######################
                    # OFFSET USING STARS #
                    ######################
    
    #
    # TRY TO FIND THE OFFSET USING STARS ONLY. STARS ARE ALWAYS OUR BEST
    # CHOICE.
    #

    star_offset = None
    stars_metadata = None
    
    if (not entirely_body and
        allow_stars and (not force_offset or create_overlay) and
        not force_bootstrap_candidate):
        stars_metadata = stars_find_offset(obs,
                                           extend_fov=extend_fov,
                                           stars_config=stars_config)
        metadata['stars_metadata'] = stars_metadata
        star_offset = stars_metadata['offset']
        if star_offset is None:
            logger.info('Final star offset N/A')
        else:
            logger.info('Final star offset U,V %.2f %.2f good stars %d', 
                        star_offset[0], star_offset[1],
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


                    ###############################
                    # CREATE RING AND BODY MODELS #
                    ###############################
    
    set_obs_ext_bp(obs, extend_fov)
    model_offset = None
    
    #
    # MAKE MODELS FOR THE BODIES IN THE IMAGE, EVEN THE FUZZY ONES
    #
    if (allow_saturn or allow_moons) and not entirely_rings:
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            mask_only = (entirely_body == body_name and
                         (bodies_cartographic_data is None or
                          body_name not in bodies_cartographic_data))
            body_model, body_metadata, body_overlay_text = bodies_create_model(
                    obs, body_name, inventory=inv,
                    extend_fov=extend_fov,
                    cartographic_data=bodies_cartographic_data,
                    always_create_model=create_overlay,
                    label_avoid_mask=label_avoid_mask,
                    bodies_config=bodies_config,
                    mask_only=mask_only)
            bodies_metadata[body_name] = body_metadata
            if body_model is not None:
                bodies_model_list.append((body_model, body_metadata, 
                                          body_overlay_text))
            if label_avoid_mask is not None:
                label_avoid_mask = np.logical_or(label_avoid_mask,
                                                 body_overlay_text)

    if entirely_body:
        if (entirely_body not in FUZZY_BODY_LIST and
            (bodies_cartographic_data is None or
             entirely_body not in bodies_cartographic_data)):
            # Nothing we can do here except bootstrap
            metadata['bootstrap_candidate'] = True
            logger.info('Single body without cartographic data - '+
                         'bootstrap candidate and returning')
            metadata['end_time'] = time.time()
            return metadata

    #
    # MAKE A MODEL FOR THE RINGS IN THE IMAGE
    #
    if allow_rings:
        rings_model, rings_metadata, rings_overlay_text = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         always_create_model=create_overlay,
                                         label_avoid_mask=label_avoid_mask,
                                         rings_config=rings_config)
        metadata['rings_metadata'] = rings_metadata
        if label_avoid_mask is not None:
            label_avoid_mask = np.logical_or(label_avoid_mask,
                                             rings_overlay_text)

    rings_curvature_ok = (metadata['rings_metadata'] is not None and
                          metadata['rings_metadata']['curvature_ok'])
    rings_features_ok = (metadata['rings_metadata'] is not None and
                         metadata['rings_metadata']['fiducial_features_ok'])
    rings_features_blurred = (metadata['rings_metadata'] is not None and
                              metadata['rings_metadata']['fiducial_blur'])
    
    if force_bootstrap_candidate:
        metadata['bootstrap_candidate'] = True
        logger.info('Forcing bootstrap candidate and returning')
        metadata['end_time'] = time.time()
        return metadata
        
        
                #####################################################
                # MERGE ALL THE MODELS TOGETHER AND FIND THE OFFSET #
                #####################################################

    model_list = []
    used_model_str_list = []
    titan_body_metadata = None

    # XXX Deal with moons on the far side of the rings
    if (rings_model is not None and rings_curvature_ok and 
        rings_features_ok and
        (rings_features_blurred is None or 
         len(bodies_model_list) == 0)):
        # Only include the rings if they are going to provide a valid
        # navigation reference
        # Only use blurred rings if there are no bodies to use
        model_list = model_list + [rings_model]
        used_model_str_list.append('RINGS')
        if rings_features_blurred is not None:
            logger.info('Using rings model blurred by %f because there are no'+
                        ' bodies', rings_features_blurred)

    good_body = False # True if we have cartographic data OR
                      #         the limb and curvature are both OK
    bad_body = False  # True if the closest useable body is not a good body

    if len(bodies_model_list) > 0:
        # Sorted by range
        for body_model, body_metadata, body_text in bodies_model_list:
            body_name = body_metadata['body_name']
            if body_name in FUZZY_BODY_LIST:
                # Fuzzy bodies can't be used for navigation, but can be used
                # later to create the overlay
                continue
            if body_name == 'TITAN':
                # Titan can't be used for primary model navigation
                titan_body_metadata = body_metadata
                continue
            if ((bodies_cartographic_data is None or
                 body_name not in bodies_cartographic_data) and
                (not body_metadata['curvature_ok'] or
                 not body_metadata['limb_ok'])):
                if not good_body:
                    # Only if there isn't a good closer body
                    bad_body = True
                continue
            good_body = True
            model_list.append(body_model)
            used_model_str_list.append(body_name)
    metadata['model_contents'] = used_model_str_list
    logger.info('Model contains %s', str(used_model_str_list))
    final_model = None
    if force_offset:
        model_offset = force_offset
        logger.info('FORCING OFFSET U,V %.2f,%.2f',
                     model_offset[0], model_offset[1])
    elif len(model_list) == 0:
        logger.info('Nothing to model - no offset found')
        if (rings_curvature_ok and not rings_features_ok and
            not metadata['bootstrap_body']): # XXX
            logger.info('Ring curvature OK but not enough fiducial features '+
                        '- candidate for bootstrapping')
            metadata['bootstrapping_candidate'] = True
            metadata['bootstrap_body'] = 'RINGS'
        if rings_features_ok and not rings_curvature_ok:
            logger.info('Candidate for ring radial scan') # XXX
    else:
        # We have at least one viable component of the model
        final_model = _combine_models(model_list, solid=True, 
                                      masked=masked_model)

        if (rings_model is None and 
            len(bodies_model_list) < offset_config['num_bodies_threshold'] and
            np.count_nonzero(final_model) < 
              offset_config['bodies_cov_threshold']):
            logger.info('Too few moons, no rings, model has too little '+
                        'coverage - model not used')
            model_offset = None
            peak = None
        else:
            gaussian_blur = offset_config['default_gaussian_blur']
            if rings_features_blurred is not None:
                gaussian_blur = rings_features_blurred
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
                                       filter=model_filter_func,
                                       masked=masked_model)
            model_offset = None
            peak = None
            if len(model_offset_list) > 0:
                (model_offset, peak) = model_offset_list[0]
        
            if model_offset is not None and not masked_model:
                # Run it again to make sure it works with a fully sliced
                # model.
                # No point in doing this if we allow_masked, since we've
                # already used each model slice independently.
                logger.info(
                     'Running secondary correlation on offset U,V %d,%d',
                     model_offset[0], model_offset[1])
                offset_model = final_model[extend_fov[1]-model_offset[1]:
                                           extend_fov[1]-model_offset[1]+
                                               obs.data.shape[0],
                                           extend_fov[0]-model_offset[0]:
                                           extend_fov[0]-model_offset[0]+
                                               obs.data.shape[1]]
                sec_search_size = offset_config['secondary_corr_search_size']
                model_offset_list = find_correlation_and_offset(
                                           obs.data,
                                           offset_model, search_size_min=0,
                                           search_size_max=(sec_search_size,
                                                            sec_search_size),
                                           filter=model_filter_func,
                                           masked=masked_model)
                new_model_offset = None
                if len(model_offset_list):
                    (new_model_offset, new_peak) = model_offset_list[0]
                metadata['secondary_corr_ok'] = False
                if new_model_offset is None:
                    logger.info('Secondary model correlation FAILED - '+
                                 'Not trusting result')
                    model_offset = None
                    peak = None
                else:
                    logger.info('Secondary model correlation dU,dV %d,%d '+
                                 'CORR %f', new_model_offset[0],
                                 new_model_offset[1], new_peak)
                    sec_threshold = offset_config['secondary_corr_threshold']
                    sec_frac = offset_config['secondary_corr_peak_threshold']
                    if (abs(new_model_offset[0]) >= sec_threshold or
                        abs(new_model_offset[1]) >= sec_threshold or
                        new_peak < peak*sec_frac):
                        logger.info('Secondary model correlation offset does '+
                                    'not meet criteria - Not trusting result')
                        model_offset = None
                        peak = None
                    else:
                        model_offset = (model_offset[0]+new_model_offset[0],
                                        model_offset[1]+new_model_offset[1])
                        peak = new_peak
                        metadata['secondary_corr_ok'] = True
                
        if model_offset is None:
            logger.info('Final model offset N/A')
        else:
            logger.info('Final model offset    U,V %d %d', 
                         model_offset[0], model_offset[1])
    
            shifted_model = shift_image(final_model, model_offset[0], 
                                        model_offset[1])
            shifted_model = unpad_image(shifted_model, extend_fov)

            # Only trust a model if it has at least a reasonable number of 
            # pixels and parts of the model are not right along the edge.
            cov_threshold = offset_config['model_cov_threshold']
            edge_pixels = offset_config['model_edge_pixels']
            if (np.count_nonzero(shifted_model[edge_pixels:-edge_pixels+1,
                                               edge_pixels:-edge_pixels+1]) <
                cov_threshold):
                logger.info('Final shifted model has too little coverage - '+
                            'Offset rejected')
                model_offset = None
                peak = None

    if (model_offset is None and star_offset is None and 
        titan_body_metadata is not None and 
        titan_body_metadata['curvature_ok']):
        logger.info('Model contains only TITAN and no stars - '+
                    'performing Titan photometric navigation')
        main_final_model = None
        if final_model is not None:
            main_final_model = unpad_image(final_model, extend_fov)
        titan_metadata = titan_navigate(obs, main_final_model, 
                                        titan_config=titan_config)
        metadata['titan_metadata'] = titan_metadata
        titan_offset = titan_metadata['offset']
        if titan_offset is not None:
            model_offset = titan_offset


    metadata['model_offset'] = model_offset
    
    
                ########################################
                # COMPARE STARS OFFSET TO MODEL OFFSET #
                ########################################

    offset = None    
    if star_offset is None:
        if model_offset is not None:
            offset = model_offset
            metadata['used_objects_type'] = 'model'
    else:
        # Assume stars are good until proven otherwise
        offset = star_offset
        metadata['used_objects_type'] = 'stars'
    if model_offset is not None and star_offset is not None:
        disagree_threshold = offset_config['stars_model_diff_threshold']
        stars_threshold = offset_config['stars_override_threshold']
        if (abs(star_offset[0]-model_offset[0]) >= disagree_threshold or
            abs(star_offset[1]-model_offset[1]) >= disagree_threshold):
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
                metadata['model_overrides_stars'] = True
                metadata['used_objects_type'] = 'model'

    if star_offset is None:
        logger.info('Final star offset N/A')
    else:
        logger.info('Final star offset     U,V %.2f %.2f good stars %d', 
                    star_offset[0], star_offset[1],
                    stars_metadata['num_good_stars'])

    if offset is None:
        logger.info('Final combined offset FAILED')
    else:
        logger.info('Final combined offset U,V %.2f %.2f', offset[0], offset[1])

    metadata['offset'] = offset

    if offset is not None:
        orig_fov = obs.fov
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
        set_obs_center_bp(obs, force=True)
        ra = obs.center_bp.right_ascension()
        dec = obs.center_bp.declination()
        metadata['ra_dec_center_offset'] = (ra.vals,dec.vals)
        obs.fov = orig_fov
        set_obs_center_bp(obs, force=True)


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
            bodies_combined = _combine_models(o_bodies_model_list, solid=True,
                                              masked=masked_model)
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
    
    # For moons, we mark a bootstrap candidate if the closest usable body
    # is "bad" - bad limb or curvature, no cartographic data
    if (offset is None and bad_body and not good_body and
        metadata['bootstrap_body'] is not None):
        logger.info('Marking as bootstrap candidate')
        metadata['bootstrap_candidate'] = True
    
    metadata['end_time'] = time.time()

    logger.info('Total elapsed time %d seconds', 
                metadata['end_time']-metadata['start_time'])
    
    return metadata

def offset_create_overlay_image(obs, metadata,
                                blackpoint=None, whitepoint=None,
                                whitepoint_ignore_frac=1., 
                                gamma=0.5,
                                stars_blackpoint=None, stars_whitepoint=None,
                                stars_whitepoint_ignore_frac=1., 
                                stars_gamma=0.5):
    img = obs.data
    offset = metadata['offset']
    if offset is None:
        offset = (0,0)
    stars_overlay = metadata['stars_overlay']
    stars_overlay_text = metadata['stars_overlay_text']
    bodies_overlay = metadata['bodies_overlay']
    bodies_overlay_text = metadata['bodies_overlay_text']
    rings_overlay = metadata['rings_overlay']
    rings_overlay_text = metadata['rings_overlay_text']
    
    gamma = 0.5
    
    # Contrast stretch the main image
    if blackpoint is None:
        blackpoint = np.min(img)

    if whitepoint is None:
        img_sorted = sorted(list(img.flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
    greyscale_img = ImageDisp.scale_image(img,
                                          blackpoint,
                                          whitepoint,
                                          gamma)

    stars_metadata = metadata['stars_metadata']
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
                stars_data[:,:] = ImageDisp.scale_image(stars_data,
                                                        stars_bp,
                                                        stars_wp,
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
    data_lines.append('%s %s' % (obs.filename[:13], 
                                 cspice.et2utc(obs.midtime, 'C', 0)))
    data_lines.append('%.2f %s %s' % (obs.texp, obs.filter1, obs.filter2))
    data_line = ''
    stars_ok = False
    model_ok = False
    if (metadata['stars_metadata'] is not None and
        metadata['stars_metadata']['offset'] is not None):
        data_line += 'Stars OK'
        stars_ok = True
    else:
        data_line += 'Stars FAIL'
    if metadata['model_offset'] is not None:
        model_ok = True
        data_line += ' / Model OK'
    else:
        data_line += ' / Model FAIL'
    if stars_ok and model_ok:
        if metadata['model_overrides_stars']:
            data_line += ' / Model WINS'
        else:
            data_line += ' / Stars WIN'
    data_lines.append(data_line)

    if metadata['bootstrap_candidate']:
        data_line = 'Bootstrap Cand ' + metadata['bootstrap_body']
        data_lines.append(data_line)
    
    text_size_h_list = []
    text_size_v_list = []
    for data_line in data_lines:
        text_size = text_draw.textsize(data_line)
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
        text_draw.text((best_u,best_v), data_line, fill=(255,255,255))
        best_v += v_inc

    combined_data = np.array(text_im.getdata()).reshape(combined_data.shape)

    combined_data = np.cast['uint8'](combined_data)

    return combined_data

def offset_result_str(metadata):
    ret = ''
    if metadata is None:
        ret += 'No offset file written'
        return ret

    if 'error' in metadata:
        ret += 'ERROR: '
        error = metadata['error']
        if error.startswith('SPICE(NOFRAMECONNECT)'):
            ret += 'SPICE KERNEL MISSING DATA AT ' + error[34:53]
        else:
            ret += error 
        return ret
    
    offset = metadata['offset']
    if offset is None:
        offset_str = '  N/A  '
    else:
        offset_str = '%3d,%3d' % tuple(offset)
    star_offset_str = '  N/A  '
    if metadata['stars_metadata'] is not None:
        star_offset = metadata['stars_metadata']['offset']
        if star_offset is not None:
            star_offset_str = '%3d,%3d' % tuple(star_offset)
    model_offset = metadata['model_offset']
    if model_offset is None:
        model_offset_str = '  N/A  '
    else:
        model_offset_str = '%3d,%3d' % tuple(model_offset)
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
    if metadata['bootstrap_candidate']:
        bootstrap_str = 'Bootstrap cand ' + metadata['bootstrap_body']
        
    ret += the_time + ' ' + ('%-4s'%filter1) + '+' + ('%-5s'%filter2) + ' '
    ret += the_size
    ret += ' Final ' + offset_str
    if metadata['used_objects_type'] == 'stars':
        ret += '  STAR ' + star_offset_str
    else:
        ret += '  Star ' + star_offset_str
    if metadata['used_objects_type'] == 'model':
        ret += '  MODEL ' + model_offset_str
    else:
        ret += '  Model ' + model_offset_str
    if bootstrap_str:
        ret += ' ' + bootstrap_str
    if bootstrap_str and single_body_str:
        ret += ' '
    if single_body_str:
        ret += ' ' + single_body_str
        
    return ret
    