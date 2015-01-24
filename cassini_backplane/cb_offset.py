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
                           
          Data about the offset process:
          
            'offset'           The final (U,V) offset. None if offset finding
                               failed.
            'large_bodies'     A list of the large bodies present in the image.
            'body_only'        False if the image doesn't consist entirely of
                               a single body and nothing else.
                               Otherwise the name of the body.
            'rings_only'       True if the image consists entirely of the main
                               rings and nothing else.
            'used_objects_type' The type of objects used to create the final
                               model: None, 'stars', or 'model'.
            'model_contents'   A list of object types used to create the
                               non-star model: 'rings', 'bodies'.
            'model_overrides_stars' True if the non-star model was more trusted
                               than the star model.
            'star_metadata'    The metadata from star matching. None if star
                               matching not performed.
            'ring_metadata'    The metadata from ring modeling. None if ring
                               modeling not performed.
            'model_offset'     The (U,V) offset determined by model matching.
                               None if model matching failed.
            
          Data for bootstrapping:
            
            'bootstrap_candidate'
                               True if the image is a good candidate for 
                               future bootstrapping attempts.
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
            'bootstrap_body'   The largest (pixel size) body in the image.
                               
          Large data:
            
            'ext_data'         The original obs.data extended by the maximum
                               search size.
            'overlay'          The visual overlay (if create_overlay is True).
            'ext_overlay'      The visual overlay extended by the maximum
                               search size (if create_overlay is True).
    """
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
    
    masked_model = False # XXX
    
    extend_fov = MAX_POINTING_ERROR[obs.detector]
    search_size_max_u, search_size_max_v = extend_fov
    
    set_obs_ext_data(obs, extend_fov)

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
    metadata['ext_data'] = obs.ext_data
    metadata['ext_overlay'] = overlay
    metadata['overlay'] = unpad_image(overlay, extend_fov)
    metadata['used_objects_type'] = None
    metadata['model_contents'] = []
    metadata['model_overrides_stars'] = False
    metadata['body_only'] = False
    metadata['rings_only'] = False
    metadata['star_metadata'] = None
    metadata['ring_metadata'] = None
    metadata['model_offset'] = None
    metadata['offset'] = None
    metadata['bootstrap_candidate'] = False
    
    # Information useful for boostrapping
    metadata['camera'] = obs.detector
    metadata['midtime'] = obs.midtime
    metadata['filter1'] = obs.filter1
    metadata['filter2'] = obs.filter2
    metadata['image_shape'] = obs.data.shape
    metadata['ra_dec_corner'] = compute_ra_dec_limits(obs, 
                                                      extend_fov=extend_fov)    
    set_obs_center_bp(obs)
    ra = obs.center_bp.right_ascension()
    dec = obs.center_bp.declination()
    metadata['ra_dec_center'] = (ra,dec)
    
    #
    # FIGURE OUT WHAT KINDS OF THINGS ARE IN THIS IMAGE
    #
    # - A single body completely covers the image
    # - The main rings completely cover the image
    #

    # Always compute the inventory so we can see if the entire image is a
    # closeup of a single body and so the metadata has this information
    
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    large_bodies_by_range = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_range.sort(key=lambda x: x[1]['range'], reverse=True)
    large_bodies_by_size = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_size.sort(key=lambda x: 
                  (x[1]['u_max_unclipped']-x[1]['u_min_unclipped'])*
                  (x[1]['v_max_unclipped']-x[1]['v_min_unclipped']), 
                  reverse=True)
    
    logger.debug('Large body inventory %s', sorted(large_body_dict.keys()))

    metadata['bootstrap_body'] = None
    if len(large_bodies_by_size) > 0:
        metadata['bootstrap_body'] = large_bodies_by_size[0][0]
        logger.debug('Largest body %s', metadata['bootstrap_body'])
        
    set_obs_ext_corner_bp(obs, extend_fov)
    
    metadata['large_bodies'] = [x[0] for x in large_bodies_by_range]
        
    # See if any of the bodies take up the entire image    
    entirely_body = False
    if len(large_bodies_by_range) > 0:
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            body_mask = obs.ext_corner_bp.where_intercepted(body_name).vals
            if np.all(body_mask):
                logger.debug('Image appears to be covered by %s', body_name)
                entirely_body = body_name

    # See if the main rings take up the entire image    
    entirely_rings = False    
    radii = obs.ext_corner_bp.ring_radius('saturn:ring').vals.astype('float')
    if len(large_bodies_by_range) == 0:
        radii_good = np.logical_and(radii > RINGS_MIN_RADIUS,
                                    radii < RINGS_MAX_RADIUS)
        if np.all(radii_good):  
            logger.debug('Image appears to be entirely rings')
            entirely_rings = True
    
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
        star_metadata = star_find_offset(obs,
                                         extend_fov=extend_fov,
                                         star_config=star_config)
        metadata['star_metadata'] = star_metadata
        star_offset = star_metadata['offset']
        if star_offset is None:
            logger.debug('Final star offset N/A')
        else:
            logger.debug('Final star offset U,V %d %d good stars %d', 
                         star_offset[0], star_offset[1],
                         star_metadata['num_good_stars'])
        if create_overlay:
            # Make the overlay with no offset because we shift the overlay
            # later
            star_overlay = star_make_good_bad_overlay(obs,
                              star_metadata['full_star_list'], (0,0),
                              extend_fov=extend_fov,
                              overlay_box_width=star_overlay_box_width,
                              overlay_box_thickness=star_overlay_box_thickness)

    if force_bootstrap_candidate:
        metadata['bootstrap_candidate'] = True
        logger.debug('Forcing bootstrap candidate and returning')
        return metadata
        
    if entirely_body:
        if (bodies_cartographic_data is None or
            bodies_cartographic_data['body_name'] != entirely_body):
            # Nothing we can do here except bootstrap
            metadata['bootstrap_candidate'] = True
            logger.debug('Single body without cartographic data - forcing '+
                         'bootstrap candidate and returning')
            return metadata

    ####                               ###
    #### RING AND BODY COMBINED MODELS ###
    ####                               ###
    
    set_obs_ext_bp(obs, extend_fov)
    model_offset = None
    
    #
    # TRY TO MAKE A MODEL FROM THE BODIES IN THE IMAGE
    #
    if ((allow_saturn or allow_moons) and 
        not entirely_rings and not entirely_body):
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            if body_name in FUZZY_BODY_LIST and not create_overlay:
                continue
            
            bodies_model, bodies_metadata = bodies_create_model(
                    obs, body_name, inventory=inv,
                    extend_fov=extend_fov,
                    lambert=bodies_use_lambert,
                    cartographic_data=bodies_cartographic_data,
                    bodies_config=bodies_config)
            bodies_model_list.append((bodies_model, bodies_metadata))

    #
    # TRY TO MAKE A MODEL FROM THE RINGS IN THE IMAGE
    #
    if allow_rings:
        rings_model, ring_metadata = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         rings_config=rings_config)
        metadata['ring_metadata'] = ring_metadata

    rings_curvature_ok = (metadata['ring_metadata'] is not None and
                          metadata['ring_metadata']['curvature_ok'])
    rings_features_ok = (metadata['ring_metadata'] is not None and
                         metadata['ring_metadata']['fiducial_features_ok'])

    #
    # MERGE ALL THE MODELS TOGETHER AND FIND THE OFFSET
    #

    model_list = []
    used_model_str_list = []
    # XXX Deal with moons on the far side of the rings
    if rings_model is not None and rings_curvature_ok and rings_features_ok:
        model_list = model_list + [rings_model]
        used_model_str_list.append('rings')
    if len(bodies_model_list) > 0:
        for bodies_model, bodies_metadata in bodies_model_list:
            if bodies_model in FUZZY_BODY_LIST:
                continue
            if (not bodies_metadata['curvature_ok'] or
                not bodies_metadata['limb_ok']):
                continue
            model_list.append(bodies_model)
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
        if not rings_curvature_ok:
            logger.debug('Candidate for ring radial scan') # XXX
    else:
        final_model = _combine_models(model_list, solid=True, 
                                      masked=masked_model)

        # XXX
        if (rings_model is None and len(bodies_model_list) < 3 and
            float(np.count_nonzero(final_model)) / final_model.size < 0.0005):
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
                                       extend_fov=extend_fov, # XXX
                                       filter=_model_filter,
                                       masked=masked_model)  # XXX
            if len(model_offset_list) > 0:
                (model_offset, peak) = model_offset_list[0]
            else:
                model_offset = None
                peak = None
        
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
                model_offset_list = find_correlation_and_offset(
                                           obs.data,
                                           offset_model, search_size_min=0,
                                           search_size_max=(15,15),
#                                           search_size_max=(search_size_max_u, 
#                                                            search_size_max_v), # XXX
                                           filter=_model_filter,
                                           masked=masked_model)
                new_model_offset = None
                if len(model_offset_list):
                    (new_model_offset, new_peak) = model_offset_list[0]
                if new_model_offset is None:
                    logger.debug('Secondary model correlation FAILED - '+
                                 'Not trusting result')
                    model_offset = None
                    peak = None
                else:
                    logger.debug('Secondary model correlation dU,dV %d,%d '+
                                 'CORR %f', new_model_offset[0],
                                 new_model_offset[1], new_peak)
                    if (abs(new_model_offset[0]) > 2 or
                        abs(new_model_offset[1]) > 2 or
                        new_peak < peak*.75): # Need slush for roundoff error
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

            # Only trust a model enough to override stars if it has at least a
            # reasonable number of pixels and parts of the model are not right
            # along the edge.
            if ((float(np.count_nonzero(shifted_model)) /
                 shifted_model.size < 0.0005) or
                not np.any(shifted_model[5:-4,5:-4])): # XXX
                logger.debug('Final shifted model has too little coverage - '+
                             'FAILED')
                model_offset = None
                peak = None

    metadata['model_offset'] = model_offset
    
    # 
    # FIGURE OUT WHICH RESULT WE BELIEVE
    #
    
    offset = star_offset
    metadata['used_objects_type'] = 'stars'
    metadata['model_overrides_stars'] = False
    if model_offset is not None:
        if star_offset is not None:
            if (abs(star_offset[0]-model_offset[0]) > 5 or    # XXX CONSTS
                abs(star_offset[1]-model_offset[1]) > 5):
                if metadata['star_metadata']['num_good_stars'] >= 6:
                    logger.debug('Star and model offsets disagree by too '+
                                 'much - trusting star result')
                else:
                    logger.debug('Star and model offsets disagree by too '+
                                 'much - ignoring star result')
                    star_list = star_metadata['full_star_list']
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
        else:
            offset = model_offset
            metadata['used_objects_type'] = 'model'

    if offset is None:
        logger.debug('Final combined offset FAILED')
    else:
        logger.debug('Final combined offset U,V %d %d', offset[0], offset[1])

    metadata['offset'] = offset
                
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
    
    return metadata

def  _master_find_offset_one_single_body(obs, metadata, body_name, extend_fov):
    metadata['body_only'] = body_name
    return metadata

