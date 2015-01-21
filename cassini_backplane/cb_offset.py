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


def _normalize(data):
    """Normalize data to [0,1] but preserve zeros."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_min == data_max:
        return np.zeros(data.shape, dtype=np.float32)
    new_data = (data-data_min) / (data_max-data_min)
    new_data[data == 0.] = 0.
    return new_data

def _combine_models(model_list, solid=False):
    """Combine models by normalizing the sum of the normalized models."""
    new_model = np.zeros(model_list[0].shape, dtype=np.float32)
    for model in model_list:
        if solid:
            new_model[model != 0] = model[model != 0]
        else:
            new_model += _normalize(model)

    return _normalize(new_model)

def _model_filter(image):
    """The filter to use on an image when looking for moons and rings."""
    return filter_sub_median(image, median_boxsize=11, gaussian_blur=1.2)
    
def master_find_offset(obs, 
                   create_overlay=False,
                   
                   allow_stars=True,
                       star_overlay_box_width=None,
                       star_overlay_box_thickness=None,
                       star_config=None,
                        
                   allow_saturn=True,

                   allow_moons=True,
                       moons_use_lambert=True,
                       moons_use_cartographic=False,
                       moons_cartographic_source='iss',
                   
                   allow_rings=True,
                       rings_model_source='voyager',

                   force_offset=None):
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
        moons_use_lambert        True to use Lambert shading for moons.
        moons_use_cartographic   True to allow the use of cartographic surface
                                 maps for moons.
        moons_cartographic_source  The source data to use for cartographic
                                 surfaces (see cb_moons).

        allow_rings              True to allow finding the offset based on
                                 rings.
        rings_model_source       The source for ring data (see cb_rings).

        force_offset             None to find the offset automatically or
                                 a tuple (offset_u,offset_v) to force the
                                 result offset. This is useful for creating
                                 an overlay with a known offset.

    Returns:
        metadata           A dictionary containing information about the
                           offset result:
            'offset_u'   The final (U,V) offset. 
            'offset_v'   None if offset finding failed.
            'ext_data'         The original obs.data extended by the maximum
                               search size.
            'overlay'          The visual overlay (if create_overlay is True).
            'ext_overlay'      The visual overlay extended by the maximum
                               search size (if create_overlay is True).
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
            'model_offset_u'   The (U,V) offset determined by model matching.
            'model_offset_v'   None if model matching failed.
            
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
        
    offset_u = None
    offset_v = None
    
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
    metadata['model_offset_u'] = None
    metadata['model_offset_v'] = None
    metadata['offset_u'] = None
    metadata['offset_v'] = None
    
    #
    # FIGURE OUT WHAT KINDS OF THINGS ARE IN THIS IMAGE
    #
    # - A single body completely covers the image
    # - The main rings completely cover the image
    #

    # Always compute the inventory so we can see if the entire image is a
    # closeup of a single body
    
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    large_bodies_by_range = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_range.sort(key=lambda x: x[1]['range'], reverse=True)
    
    logger.debug('Large body inventory %s', large_body_dict.keys())

    set_obs_ext_corner_bp(obs, extend_fov)
    
    metadata['large_bodies'] = [x[0] for x in large_bodies_by_range]
    
    entirely_body = False
    if len(large_bodies_by_range) > 0:
        # See if any of the bodies take up the entire image    
        for body_name, inv in large_bodies_by_range:
            body_mask = obs.ext_corner_bp.where_intercepted(body_name).vals
            if np.all(body_mask):
                logger.debug('Image appears to be entirely %s', body_name)
                return _master_find_offset_entirely_body(obs, metadata, body_name)

    # See if the main rings take up the entire image    
    entirely_rings = False    
    radii = obs.ext_corner_bp.ring_radius('saturn:ring').vals.astype('float')
    radii_good = np.logical_and(radii > RINGS_MIN_RADIUS,
                                radii < RINGS_MAX_RADIUS)
    if np.all(radii_good) and len(large_bodies_by_range) == 0:
        logger.debug('Image appears to be entirely rings')
        return _master_find_offset_entirely_rings(obs, metadata)
    
    #
    # TRY TO FIND THE OFFSET USING STARS ONLY. STARS ARE ALWAYS OUR BEST
    # CHOICE.
    #
    # If the image is entirely rings then by definition we can't see any stars.
    # XXX Might want to allow stars to show through ring gaps.
    #

    star_offset_u = None
    star_offset_v = None
        
    if (allow_stars and not force_offset):
        star_offset_u, star_offset_v, star_metadata = star_find_offset(obs,
                                                       extend_fov=extend_fov,
                                                       star_config=star_config)
        metadata['star_metadata'] = star_metadata
        if star_offset_u is None:
            logger.debug('Final star offset N/A')
        else:
            logger.debug('Final star offset U,V %d %d good stars %d', 
                         star_offset_u, star_offset_v,
                         star_metadata['num_good_stars'])
        if create_overlay:
            # Make the overlay with no offset because we shift the overlay
            # later
            star_overlay = star_make_good_bad_overlay(obs,
                              star_metadata['full_star_list'], 0, 0,
                              extend_fov=extend_fov,
                              overlay_box_width=star_overlay_box_width,
                              overlay_box_thickness=star_overlay_box_thickness)

    set_obs_ext_bp(obs, extend_fov)
    model_offset_u = None
    model_offset_v = None
    
    #
    # TRY TO MAKE A MODEL FROM THE BODIES IN THE IMAGE
    #
    if (allow_saturn or allow_moons) and not entirely_rings:
        for body_name, inv in large_bodies_by_range:
            if body_name == 'SATURN' and not allow_saturn:
                continue
            if body_name != 'SATURN' and not allow_moons:
                continue
            if body_name in FUZZY_BODY_LIST:
                continue
            
            model = bodies_create_model(obs, body_name,
                                        u_min=inv['u_min'],
                                        u_max=inv['u_max'],
                                        v_min=inv['v_min'],
                                        v_max=inv['v_max'],
                                        extend_fov=extend_fov,
                                        lambert=moons_use_lambert,
                                        force_spherical=False,
                                        use_cartographic=moons_use_cartographic,
                                        source=moons_cartographic_source)
            bodies_model_list.append(model)
    
    #
    # TRY TO MAKE A MODEL FROM THE RINGS IN THE IMAGE
    #
    if allow_rings:
        rings_model, ring_metadata = rings_create_model(
                                         obs, extend_fov=extend_fov,
                                         source=rings_model_source)
        metadata['ring_metadata'] = ring_metadata

    #
    # MERGE ALL THE MODELS TOGETHER AND FIND THE OFFSET
    #

    model_list = []
    used_model_str_list = []
    # XXX Deal with moons on the far side of the rings
    if rings_model is not None:
        model_list = model_list + [rings_model]
        used_model_str_list.append('rings')
    if len(bodies_model_list) > 0:
        for model in bodies_model_list:
            model_list.append(model)
        used_model_str_list.append('bodies')
    metadata['model_contents'] = used_model_str_list
    if len(model_list) == 0:
        logger.debug('Nothing to model - no offset found')
    elif force_offset:
        model_offset_u, model_offset_v = force_offset
        logger.debug('FORCING OFFSET U,V %d,%d', model_offset_u, model_offset_v)
    else:
        final_model = _combine_models(model_list, solid=True)

        # XXX
        if (rings_model is None and len(bodies_model_list) < 3 and
            float(np.count_nonzero(final_model)) / final_model.size < 0.0005):
            logger.debug('Too few moons, no rings, model has too little coverage')
            model_offset_u = None
            model_offset_v = None
            peak = None
        else:            
            model_offset_list = find_correlation_and_offset(
                                       obs.ext_data,
                                       final_model, search_size_min=0,
                                       search_size_max=(search_size_max_u, 
                                                        search_size_max_v),
                                       extend_fov=extend_fov, # XXX
                                       filter=_model_filter) # XXX
            if len(model_offset_list) > 0:
                (model_offset_u, model_offset_v,
                 peak) = model_offset_list[0]
            else:
                model_offset_u = None
                model_offset_v = None
                peak = None
        
            if model_offset_u is not None:
                # Run it again
                logger.debug(
                     'Running secondary correlation on offset U,V %d,%d',
                     model_offset_u, model_offset_v)
                offset_model = final_model[extend_fov[1]-model_offset_v:
                                           extend_fov[1]-model_offset_v+
                                               obs.data.shape[0],
                                           extend_fov[0]-model_offset_u:
                                           extend_fov[0]-model_offset_u+
                                               obs.data.shape[1]]
                model_offset_list = find_correlation_and_offset(
                                           obs.data,
                                           offset_model, search_size_min=0,
                                           search_size_max=(15,15),
#                                           search_size_max=(search_size_max_u, 
#                                                            search_size_max_v), # XXX
                                           filter=_model_filter) # XXX
                new_model_offset_u = None
                if len(model_offset_list):
                    (new_model_offset_u, new_model_offset_v,
                     new_peak) = model_offset_list[0]
                if new_model_offset_u is None:
                    logger.debug('Secondary model correlation FAILED - '+
                                 'Not trusting result')
                    model_offset_u = None
                    model_offset_v = None
                    peak = None
                else:
                    logger.debug('Secondary model correlation dU,dV %d,%d '+
                                 'CORR %f', new_model_offset_u,
                                 new_model_offset_v, new_peak)
                    if (abs(new_model_offset_u) > 2 or
                        abs(new_model_offset_v) > 2 or
                        new_peak < peak*.75): # Need slush for roundoff error
                        logger.debug('Secondary model correlation offset bad'+
                                     ' - Not trusting result')
                        model_offset_u = None
                        model_offset_v = None
                        peak = None
                    else:
                        model_offset_u += new_model_offset_u
                        model_offset_v += new_model_offset_v
                        peak = new_peak
                        
        if model_offset_u is None:
            logger.debug('Final model offset FAILED')
        else:
            logger.debug('Final model offset U,V %d %d', 
                         model_offset_u, model_offset_v)
    
            shifted_model = shift_image(final_model, -model_offset_u, 
                                        -model_offset_v)
            shifted_model = unpad_image(shifted_model, extend_fov)

            # Only trust a model enough to override stars if it has at least a
            # reasonable number of pixels and parts of the model are not right
            # along the edge.
            if ((float(np.count_nonzero(shifted_model)) /
                 shifted_model.size < 0.0005) or
                not np.any(shifted_model[5:-4,5:-4])): # XXX
                logger.debug('Final shifted model has too little coverage - '+
                             'FAILED')
                model_offset_u = None
                model_offset_v = None
                peak = None

    metadata['model_offset_u'] = model_offset_u
    metadata['model_offset_v'] = model_offset_v
    
    # 
    # FIGURE OUT WHICH RESULT WE BELIEVE
    #
    
    offset_u = star_offset_u
    offset_v = star_offset_v
    metadata['used_objects_type'] = 'stars'
    metadata['model_overrides_stars'] = False
    if model_offset_u is not None:
        if star_offset_u is not None:
            if (abs(star_offset_u-model_offset_u) > 5 or    # XXX CONSTS
                abs(star_offset_v-model_offset_v) > 5):
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
                              star_list, 0, 0,
                              extend_fov=extend_fov,
                              overlay_box_width=star_overlay_box_width,
                              overlay_box_thickness=star_overlay_box_thickness,
                              star_config=star_config)
                    offset_u = model_offset_u
                    offset_v = model_offset_v
                    metadata['model_overrides_stars'] = True
                    metadata['used_objects_type'] = 'model'
                    metadata['num_good_stars'] = 0
        else:
            offset_u = model_offset_u
            offset_v = model_offset_v
            metadata['used_objects_type'] = 'model'

    if offset_u is None:
        logger.debug('Final combined offset FAILED')
    else:
        logger.debug('Final combined offset U,V %d %d', offset_u, offset_v)

    metadata['offset_u'] = offset_u
    metadata['offset_v'] = offset_v
                
    if create_overlay:        
        if len(bodies_model_list) > 0:
            bodies_combined = _combine_models(bodies_model_list, solid=True)
            overlay[...,1] = (_normalize(bodies_combined) * 255)
        if rings_model is not None:
            overlay[...,2] = _normalize(rings_model) * 255
        if star_overlay is not None:
            overlay = np.clip(overlay+star_overlay, 0, 255)

        if offset_u is not None:
            overlay = shift_image(overlay, -offset_u, -offset_v)
    
    metadata['ext_overlay'] = overlay
    metadata['overlay'] = unpad_image(overlay, extend_fov)
    
    return metadata

def  _master_find_offset_entirely_body(obs, metadata, body_name):
    metadata['body_only'] = body_name
    return metadata

def  _master_find_offset_entirely_rings(obs, metadata):
    metadata['rings_only'] = True
    return metadata
