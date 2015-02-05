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
                       star_overlay_box_width=None,
                       star_overlay_box_thickness=None,
                       star_config=None,
                        
                   allow_saturn=True,
                   allow_moons=True,
                       bodies_use_lambert=True,
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
        star_overlay_box_width   Parameters to pass to 
        star_overlay_box_thickness  star_make_good_bad_overlay to make star
                                 overlay symbols.
        star_config              Config parameters for stars.

        allow_saturn             True to allow finding the offset based on
                                 Saturn.
        allow_moons              True to allow finding the offset based on
                                 moons.
        bodies_use_lambert       True to use Lambert shading for moons.
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
                               sorted in descending order by the size in 
                               pixels.
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
            
          Data for bootstrapping:
            
            'bootstrap_candidate'
                               True if the image is a good candidate for 
                               future bootstrapping attempts.
            'bootstrap_reference'
                               True if the image is a good candidate for
                               future use as a bootstrap reference.
            'bootstrap_body'   The largest (pixel size) body in the image.
                               
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
    
    logger = logging.getLogger(_LOGGING_NAME+'.master_find_offset')
    
    logger.debug('Date %s X %d Y %d TEXP %.3f %s+%s %s SAMPLING %s GAIN %d',
                 cspice.et2utc(obs.midtime, 'C', 0),
                 obs.data.shape[1], obs.data.shape[0], obs.texp,
                 obs.filter1, obs.filter2, obs.detector, obs.sampling,
                 obs.gain_mode)
    logger.debug('create_overlay %d, allow_stars %d, allow_saturn %d, '+
                 'allow_moons %d, allow_rings %d',
                 create_overlay, allow_stars, allow_saturn, allow_moons,
                 allow_rings)
    
    if offset_config is None:
        offset_config = OFFSET_DEFAULT_CONFIG
        
    masked_model = False # XXX
    
    extend_fov = MAX_POINTING_ERROR[obs.detector]
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
    metadata['ra_dec_center'] = (ra,dec)
    # Offset process
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
    metadata['overlay'] = unpad_image(overlay, extend_fov)
    # Bootstrapping
    metadata['bootstrap_candidate'] = False
    metadata['bootstrap_reference'] = False
    metadata['bootstrap_body'] = None
    
    
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
    large_bodies_by_range.sort(key=lambda x: x[1]['range'], reverse=True)
    large_bodies_by_size = [(x, large_body_dict[x]) for x in large_body_dict]
    # Sort by area of the enclosing square
    large_bodies_by_size.sort(key=lambda x: 
                  (x[1]['u_max_unclipped']-x[1]['u_min_unclipped'])*
                  (x[1]['v_max_unclipped']-x[1]['v_min_unclipped']), 
                  reverse=True)
    
    logger.debug('Large body inventory %s', [x[0] for x in large_bodies_by_size])

    if len(large_bodies_by_size) > 0:
        metadata['bootstrap_body'] = large_bodies_by_size[0][0]
        logger.debug('Bootstrap body %s', metadata['bootstrap_body'])
        
    metadata['large_bodies'] = [x[0] for x in large_bodies_by_size]
        
    # See if any of the bodies take up the entire image    
    entirely_body = False
    if len(large_bodies_by_range) > 0:
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            corner_body_mask = (obs.ext_corner_bp.where_intercepted(body_name).
                                vals)
            if np.all(corner_body_mask):
                logger.debug('Image appears to be covered by %s', body_name)
                entirely_body = body_name
                metadata['body_only'] = body_name

    # See if the main rings take up the entire image
    # XXX THIS IS NOT A VALID TEST    
    entirely_rings = False    
    radii = obs.ext_corner_bp.ring_radius('saturn:ring').vals.astype('float')
    if len(large_bodies_by_range) == 0:
        radii_good = np.logical_and(radii > RINGS_MIN_RADIUS,
                                    radii < RINGS_MAX_RADIUS)
        if np.all(radii_good):  
            logger.debug('Image appears to be entirely rings')
            entirely_rings = True
            metadata['rings_only'] = True
    
    
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
        stars_metadata = star_find_offset(obs,
                                          extend_fov=extend_fov,
                                          star_config=star_config)
        metadata['stars_metadata'] = stars_metadata
        star_offset = stars_metadata['offset']
        if star_offset is None:
            logger.debug('Final star offset N/A')
        else:
            logger.debug('Final star offset U,V %d %d good stars %d', 
                         star_offset[0], star_offset[1],
                         stars_metadata['num_good_stars'])
        if create_overlay:
            # Make the overlay with no offset because we shift the overlay
            # later
            star_overlay = star_make_good_bad_overlay(obs,
                              stars_metadata['full_star_list'], (0,0),
                              extend_fov=extend_fov,
                              overlay_box_width=star_overlay_box_width,
                              overlay_box_thickness=star_overlay_box_thickness)


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
            if body_name in FUZZY_BODY_LIST and not create_overlay:
                continue
            mask_only = (entirely_body == body_name and
                         (bodies_cartographic_data is None or
                          body_name not in bodies_cartographic_data))
            body_model, body_metadata = bodies_create_model(
                    obs, body_name, inventory=inv,
                    extend_fov=extend_fov,
                    lambert=bodies_use_lambert,
                    cartographic_data=bodies_cartographic_data,
                    bodies_config=bodies_config,
                    mask_only=mask_only)
            bodies_model_list.append((body_model, body_metadata))
            bodies_metadata[body_name] = body_metadata
            
    if entirely_body:
        if (bodies_cartographic_data is None or
            entirely_body not in bodies_cartographic_data):
            # Nothing we can do here except bootstrap
            metadata['bootstrap_candidate'] = True
            logger.debug('Single body without cartographic data - forcing '+
                         'bootstrap candidate and returning')
            return metadata

    #
    # MAKE A MODEL FOR THE RINGS IN THE IMAGE
    #
    if allow_rings:
        rings_model, rings_metadata = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         rings_config=rings_config)
        metadata['rings_metadata'] = rings_metadata

    rings_curvature_ok = (metadata['rings_metadata'] is not None and
                          metadata['rings_metadata']['curvature_ok'])
    rings_features_ok = (metadata['rings_metadata'] is not None and
                         metadata['rings_metadata']['fiducial_features_ok'])

    if force_bootstrap_candidate:
        metadata['bootstrap_candidate'] = True
        logger.debug('Forcing bootstrap candidate and returning')
        return metadata
        
        
                #####################################################
                # MERGE ALL THE MODELS TOGETHER AND FIND THE OFFSET #
                #####################################################

    model_list = []
    used_model_str_list = []
    good_body = False
    bad_body = False
    # XXX Deal with moons on the far side of the rings
    if rings_model is not None and rings_curvature_ok and rings_features_ok:
        # Only include the rings if they are going to provide a valid
        # navigation reference
        model_list = model_list + [rings_model]
        used_model_str_list.append('rings')
    if len(bodies_model_list) > 0:
        for body_model, body_metadata in bodies_model_list:
            if body_model in FUZZY_BODY_LIST:
                # Fuzzy bodies can't be used for navigation, but can be used
                # later to create the overlay
                continue
            body_name = body_metadata['body_name']
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
        logger.debug('FORCING OFFSET U,V %d,%d',
                     model_offset[0], model_offset[1])
    elif len(model_list) == 0:
        logger.debug('Nothing to model - no offset found')
        if rings_curvature_ok and not rings_features_ok:
            logger.debug('Ring curvature OK but not enough fiducial features '+
                         '- candidate for bootstrapping')
            metadata['bootstrapping_candidate'] = True
            metadata['bootstrap_body'] = 'RINGS'
        if rings_features_ok and not rings_curvature_ok:
            logger.debug('Candidate for ring radial scan') # XXX
    else:
        # We have at least one viable component of the model
        final_model = _combine_models(model_list, solid=True, 
                                      masked=masked_model)

        if (rings_model is None and 
            len(bodies_model_list) < offset_config['num_bodies_threshold'] and
            float(np.count_nonzero(final_model)) / final_model.size <
                offset_config['bodies_cov_threshold']):
            logger.debug('Too few moons, no rings, model has too little '+
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
                logger.debug(
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
                    logger.debug('Secondary model correlation FAILED - '+
                                 'Not trusting result')
                    model_offset = None
                    peak = None
                else:
                    metadata['secondary_corr_ok'] = True
                    logger.debug('Secondary model correlation dU,dV %d,%d '+
                                 'CORR %f', new_model_offset[0],
                                 new_model_offset[1], new_peak)
                    sec_threshold = offset_config['secondary_corr_threshold']
                    sec_frac = offset_config['secondary_corr_peak_threshold']
                    if (abs(new_model_offset[0]) > sec_threshold or
                        abs(new_model_offset[1]) > sec_threshold or
                        new_peak < peak*sec_frac):
                        logger.debug('Secondary model correlation offset bad'+
                                     ' - Not trusting result')
                        model_offset = None
                        peak = None
                    else:
                        model_offset = (model_offset[0]+new_model_offset[0],
                                        model_offset[1]+new_model_offset[1])
                        peak = new_peak
                        
        if model_offset is None:
            logger.debug('Final model offset FAILED')
        else:
            logger.debug('Final model offset U,V %d %d', 
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
                logger.debug('Final shifted model has too little coverage - '+
                             'FAILED')
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
                logger.debug('Star and model offsets disagree by too '+
                             'much - trusting star result')
            else:
                logger.debug('Star and model offsets disagree by too '+
                             'much - ignoring star result')
                star_list = stars_metadata['full_star_list']
                for star in star_list:
                    star.photometry_confidence = 0. # We changed the offset
                    star.use_for_correlation = False
                star_overlay = star_make_good_bad_overlay(obs,
                          star_list, (0,0),
                          extend_fov=extend_fov,
                          overlay_box_width=star_overlay_box_width,
                          overlay_box_thickness=star_overlay_box_thickness,
                          star_config=star_config)
                offset = model_offset
                metadata['model_overrides_stars'] = True
                metadata['used_objects_type'] = 'model'

    if offset is None:
        logger.debug('Final combined offset FAILED')
    else:
        logger.debug('Final combined offset U,V %d %d', offset[0], offset[1])

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
            overlay = shift_image(overlay, -offset[0], -offset[1])
    
    metadata['ext_overlay'] = overlay
    metadata['overlay'] = unpad_image(overlay, extend_fov)

                #########################################
                # FIGURE OUT BOOTSTRAPPING IMPLICATIONS #
                #########################################
    
    if offset is None and bad_body and not good_body:
        logger.debug('Marking as bootstrap candidate')
        metadata['bootstrap_candidate'] = True
    
    return metadata

