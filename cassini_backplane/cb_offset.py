import cb_logging
import logging

import numpy as np
import numpy.ma as ma

import oops

from cb_config import MAX_POINTING_ERROR, LARGE_BODY_LIST, FUZZY_BODY_LIST
from cb_correlate import *
from cb_moons import *
from cb_rings import *
from cb_stars import *
from cb_util_image import *

LOGGING_NAME = 'cb.' + __name__


def _normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_min == data_max:
        return np.zeros(data.shape)
    new_data = (data-data_min) / (data_max-data_min)
    new_data[data == 0.] = 0.
    return new_data

def _combine_models(model_list):
    new_model = np.zeros(model_list[0].shape)
    for model in model_list:
        new_model += _normalize(model)

    return _normalize(new_model)
    
def master_find_offset(obs, create_overlay=False,
                       star_overlay_box_width=None,
                       star_overlay_box_thickness=None, 
                       allow_stars=True,
                       allow_saturn=True,
                       allow_moons=True,
                       allow_rings=True):
#    allow_saturn = False
#    allow_moons = False
#    allow_rings = False
    
    logger = logging.getLogger(LOGGING_NAME+'.find_offset')

    extend_fov = MAX_POINTING_ERROR[obs.detector]
    search_size_max_u, search_size_max_v = extend_fov
    
    set_obs_ext_data(obs, extend_fov)

    moons_model_list = []            
    saturn_model = None
    rings_model = None
    final_model = None
    
    if create_overlay:
        overlay = np.zeros(obs.ext_data.shape + (3,), dtype=np.uint8)
        star_overlay = None
    else:
        overlay = None
        
    offset_u = None
    offset_v = None
    metadata = {}
    metadata['ext_data'] = obs.ext_data
    metadata['ext_overlay'] = overlay
    metadata['overlay'] = unpad_image(overlay, extend_fov)
    
    #
    # FIGURE OUT WHAT KINDS OF THINGS ARE IN THIS IMAGE
    #
    # - A single body completely covers the image
    # - The main rings completely cover the image
    #

    # Always compute the inventory so we can see if the entire image is a
    # closeup of a single body
    
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    
    logger.debug('Large body inventory %s', large_body_dict.keys())

    set_obs_ext_corner_bp(obs, extend_fov)
    
    entirely_body = False
    if len(large_body_dict) > 0:
        # See if any of the bodies take up the entire image    
        for body_name in large_body_dict:
            body_mask = obs.ext_corner_bp.where_intercepted(body_name).vals
            if np.all(body_mask):
                logger.debug('Image appears to be entirely %s - aborting', body_name) # XXX
                entirely_body = True
                return None, None, metadata

        # Now remove the fuzzy bodies - they aren't good to model on
        for body_name in FUZZY_BODY_LIST:
            if body_name in large_body_dict:
                del large_body_dict[body_name]

    # See if the main rings take up the entire image    
    entirely_rings = False    
    radii = obs.ext_corner_bp.ring_radius('saturn:ring').vals.astype('float')
    radii_good = np.logical_and(radii > RINGS_MIN_RADIUS,
                                radii < RINGS_MAX_RADIUS)
    if np.all(radii_good):
        logger.debug('Image appears to be entirely rings')
        entirely_rings = True
        
    #
    # TRY TO FIND THE OFFSET USING STARS ONLY. STARS ARE ALWAYS OUR BEST
    # CHOICE.
    #
    # If the image is entirely a single body or rings then by definition we
    # can't see any stars.
    #
    
    if allow_stars and not entirely_rings and not entirely_body:
        offset_u, offset_v, star_metadata = star_find_offset(obs,
                                                 extend_fov=extend_fov)
        metadata.update(star_metadata)
        if offset_u is None:
            logger.debug('Final star offset N/A')
        else:
            logger.debug('Final star offset U,V %d %d good stars %d', 
                         offset_u, offset_v, star_metadata['num_good_stars'])
            if create_overlay:
                star_overlay = star_make_good_bad_overlay(obs,
                                  star_metadata['used_star_list'], 0, 0,
                                  star_overlay_box_width,
                                  star_overlay_box_thickness,
                                  extend_fov=extend_fov)

    # If stars failed (or we need to force the creation of an image overlay for
    # debugging)...
    
    if offset_u is None or create_overlay:
        set_obs_ext_bp(obs, extend_fov)

        #
        # TRY TO MAKE A MODEL FROM THE MOONS IN THE IMAGE
        #
        if (allow_saturn or allow_moons) and not entirely_rings:
            for body_name in large_body_dict:
                if body_name == 'SATURN' and not allow_saturn:
                    continue
                if body_name != 'SATURN' and not allow_moons:
                    continue
                body = large_body_dict[body_name]
                
                model = moons_create_model(obs, body_name, lambert=True,
                                           u_min=body['u_min'], u_max=body['u_max'],
                                           v_min=body['v_min'], v_max=body['v_max'],
                                           extend_fov=extend_fov,
                                           force_spherical=True,
                                           use_cartographic=False)
                if body_name == 'SATURN':
                    saturn_model = model
                else:
                    moons_model_list.append(model)
        
        #
        # TRY TO MAKE A MODEL FROM THE RINGS IN THE IMAGE
        #
        if allow_rings and not entirely_body:
            rings_model = rings_create_model(obs, extend_fov=extend_fov)
    
        #
        # ONLY TRY TO FIND A NEW OFFSET IF STAR MATCHING FAILED
        #
        if offset_u is None:
            model_list = moons_model_list
            if saturn_model is not None:
                model_list = model_list + [saturn_model]
            if rings_model is not None:
                model_list = model_list + [rings_model]
            if len(model_list) == 0:
                logger.debug('Nothing to model - no offset found')
            else:
                final_model = _combine_models(model_list)
    
                offset_u, offset_v, peak = find_correlation_and_offset(obs.ext_data,
                                           final_model, search_size_min=0,
                                           search_size_max=(search_size_max_u, 
                                                            search_size_max_v))
    
                if offset_u is None:
                    logger.debug('Final model offset FAILED')
                else:
                    logger.debug('Final model offset U,V %d %d', offset_u, offset_v)

    if create_overlay:        
        if saturn_model is not None:
            overlay[...,0] = _normalize(saturn_model) * 255
        if len(moons_model_list) > 0:
            overlay[...,1] = _normalize(_combine_models(moons_model_list)) * 255
        if rings_model is not None:
            overlay[...,2] = _normalize(rings_model) * 255
        if star_overlay is not None:
            overlay = np.clip(overlay+star_overlay, 0, 255)

        if offset_u is not None:
            overlay = shift_image(overlay, -offset_u, -offset_v)
    
    metadata['ext_overlay'] = overlay
    metadata['overlay'] = unpad_image(overlay, extend_fov)
    
    return offset_u, offset_v, metadata
