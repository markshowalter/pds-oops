###############################################################################
# cb_offset.py
#
# Routines related to finding image offsets.
#
# Exported routines:
#    master_find_offset
###############################################################################

import cb_logging
import logging

import time

import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filt

import oops

from cb_bodies import *
from cb_config import MAX_POINTING_ERROR, LARGE_BODY_LIST, FUZZY_BODY_LIST
from cb_correlate import *
from cb_rings import *
from cb_stars import *
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
    new_data = (data-data_min) / (data_max-data_min)
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

def _model_filter(image, masked=False):
    """The filter to use on an image when looking for moons and rings."""
    ret = filter_sub_median(image, median_boxsize=11, gaussian_blur=1.2)
    if masked:
        ret = ret.view(ma.MaskedArray)
        ret.mask = image.mask
    return ret

    
def master_find_offset(obs, 
                   offset_config=None,
                   create_overlay=False,
                   
                   allow_stars=True,
                       stars_overlay_box_width=None,
                       stars_overlay_box_thickness=None,
                       stars_config=None,
                        
                   allow_saturn=True,
                   allow_moons=True,
                       bodies_cartographic_data=None,
                       bodies_config=None,
                   
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
        stars_overlay_box_width      Parameters to pass to 
        stars_overlay_box_thickness  star_make_good_bad_overlay to make star
                                     overlay symbols.
        stars_config             Config parameters for stars.

        allow_saturn             True to allow finding the offset based on
                                 Saturn.
        allow_moons              True to allow finding the offset based on
                                 moons.
        bodies_cartographic_data The metadata to use for cartographic
                                 surfaces (see cb_bodies).
        bodies_config            Config parameters for bodies.

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
            'ra_dec_corner'    A tuple (ra_min, ra_max, dec_min, dec_max)
                               giving the corners of the FOV-extended image.
            'ra_dec_center'    A tuple (ra,dec) for the center of the
                               image.

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
            'model_contents'   A list of object types used to create the
                               non-star model: 'rings', 'bodies'.
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
        
    masked_model = False # XXX
    
    extend_fov = MAX_POINTING_ERROR[obs.data.shape, obs.detector]
    search_size_max_u, search_size_max_v = extend_fov
    
    set_obs_ext_data(obs, extend_fov)
    set_obs_ext_corner_bp(obs, extend_fov)
    
    bodies_model_list = []            
    rings_model = None
    final_model = None
    
    if create_overlay:
        overlay = np.zeros(obs.ext_data.shape + (3,), dtype=np.uint8)
        star_overlay = None
    else:
        overlay = None
        
    offset = None
    
    # Initialize the metadata to at least have something for each key
    metadata = {}
    # Image
    metadata['camera'] = obs.detector
    metadata['filter1'] = obs.filter1
    metadata['filter2'] = obs.filter2
    metadata['image_shape'] = obs.data.shape
    metadata['midtime'] = obs.midtime
    metadata['ra_dec_corner'] = compute_ra_dec_limits(obs, 
                                                      extend_fov=extend_fov)    
    set_obs_center_bp(obs)
    ra = obs.center_bp.right_ascension()
    dec = obs.center_bp.declination()
    metadata['ra_dec_center'] = (ra.vals,dec.vals)
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
    metadata['model_offset'] = None
    metadata['secondary_corr_ok'] = None
    # Large
    metadata['ext_data'] = obs.ext_data
    metadata['ext_overlay'] = overlay
    if create_overlay:
        metadata['overlay'] = unpad_image(overlay, extend_fov)
    else:
        metadata['overlay'] = None
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
    # If the image is entirely rings then by definition we can't see any stars.
    # XXX Might want to allow stars to show through ring gaps.
    #

    star_offset = None
        
    if (not entirely_rings and not entirely_body and
        allow_stars and not force_offset and not force_bootstrap_candidate):
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
            # Make the overlay with no offset because we shift the overlay
            # later
            star_overlay = stars_make_good_bad_overlay(obs,
                              stars_metadata['full_star_list'], (0,0),
                              extend_fov=extend_fov,
                              overlay_box_width=stars_overlay_box_width,
                              overlay_box_thickness=stars_overlay_box_thickness,
                              stars_config=stars_config)


                    ###############################
                    # CREATE RING AND BODY MODELS #
                    ###############################
    
    set_obs_ext_bp(obs, extend_fov)
    model_offset = None
    
    #
    # MAKE MODELS FOR THE BODIES IN THE IMAGE
    #
    if (allow_saturn or allow_moons) and not entirely_rings:
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            if body_name in FUZZY_BODY_LIST:
                continue
            mask_only = (entirely_body == body_name and
                         (bodies_cartographic_data is None or
                          body_name not in bodies_cartographic_data))
            body_model, body_metadata = bodies_create_model(
                    obs, body_name, inventory=inv,
                    extend_fov=extend_fov,
                    cartographic_data=bodies_cartographic_data,
                    always_create_model=create_overlay,
                    bodies_config=bodies_config,
                    mask_only=mask_only)
            bodies_metadata[body_name] = body_metadata
            if body_model is not None:
                bodies_model_list.append((body_model, body_metadata))
            
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
        rings_model, rings_metadata = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         always_create_model=create_overlay,
                                         rings_config=rings_config)
        metadata['rings_metadata'] = rings_metadata

    rings_curvature_ok = (metadata['rings_metadata'] is not None and
                          metadata['rings_metadata']['curvature_ok'])
    rings_features_ok = (metadata['rings_metadata'] is not None and
                         metadata['rings_metadata']['fiducial_features_ok'])

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

    # XXX Deal with moons on the far side of the rings
    if rings_model is not None and rings_curvature_ok and rings_features_ok:
        # Only include the rings if they are going to provide a valid
        # navigation reference
        model_list = model_list + [rings_model]
        used_model_str_list.append('rings')

    good_body = False # True if we have cartographic data OR
                      #         the limb and curvature are both OK
    bad_body = False  # True if the closest useable body is not a good body

    if len(bodies_model_list) > 0:
        for body_model, body_metadata in bodies_model_list: # Sorted by range
            body_name = body_metadata['body_name']
            if body_name in FUZZY_BODY_LIST:
                # Fuzzy bodies can't be used for navigation, but can be used
                # later to create the overlay
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
        used_model_str_list.append('bodies')
    metadata['model_contents'] = used_model_str_list
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
            float(np.count_nonzero(final_model)) / final_model.size <
                offset_config['bodies_cov_threshold']):
            logger.info('Too few moons, no rings, model has too little '+
                        'coverage')
            model_offset = None
            peak = None
        else:            
            model_offset_list = find_correlation_and_offset(
                                       obs.ext_data,
                                       final_model, search_size_min=0,
                                       search_size_max=(search_size_max_u, 
                                                        search_size_max_v),
                                       extend_fov=extend_fov,
                                       filter=_model_filter,
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
                                           filter=_model_filter,
                                           masked=masked_model)
                new_model_offset = None
                if len(model_offset_list):
                    (new_model_offset, new_peak) = model_offset_list[0]
                if new_model_offset is None:
                    metadata['secondary_corr_ok'] = False
                    logger.info('Secondary model correlation FAILED - '+
                                 'Not trusting result')
                    model_offset = None
                    peak = None
                else:
                    metadata['secondary_corr_ok'] = True
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
                        
        if model_offset is None:
            logger.info('Final model offset N/A')
        else:
            logger.info('Final model offset    U,V %d %d', 
                         model_offset[0], model_offset[1])
    
            shifted_model = shift_image(final_model, -model_offset[0], 
                                        -model_offset[1])
            shifted_model = unpad_image(shifted_model, extend_fov)

            # Only trust a model if it has at least a reasonable number of 
            # pixels and parts of the model are not right along the edge.
            cov_threshold = offset_config['model_cov_threshold']
            edge_pixels = offset_config['model_edge_pixels']
            if ((float(np.count_nonzero(shifted_model)) /
                 shifted_model.size < cov_threshold) or
                not np.any(shifted_model[edge_pixels:-edge_pixels+1,
                                         edge_pixels:-edge_pixels+1])):
                logger.info('Final shifted model has too little coverage - '+
                            'Offset rejected')
                model_offset = None
                peak = None

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
                          overlay_box_width=stars_overlay_box_width,
                          overlay_box_thickness=stars_overlay_box_thickness,
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


                ######################
                # CREATE THE OVERLAY #
                ######################

    if create_overlay:
        # We recreate the model lists because there are models we might
        # include on the overlay that we didn't use for correlation.
        o_bodies_model_list = [x[0] for x in bodies_model_list]
        if len(o_bodies_model_list) > 0:
            bodies_combined = _combine_models(o_bodies_model_list, solid=True,
                                              masked=masked_model)
            overlay[...,1] = _normalize(bodies_combined) * 255
        if rings_model is not None:
            overlay[...,2] = _normalize(rings_model) * 255
        if star_overlay is not None:
            overlay = np.clip(overlay+star_overlay, 0, 255)

        if offset is not None:
            overlay = shift_image(overlay, -int(np.round(offset[0])), -int(np.round(offset[1])))
    
    metadata['ext_overlay'] = overlay
    if overlay is not None:
        metadata['overlay'] = unpad_image(overlay, extend_fov)

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

    return metadata
