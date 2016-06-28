###############################################################################
# cb_bodies.py
#
# Routines related to bodies.
#
# Exported routines:
#    bodies_create_model
#
#    bodies_interpolate_missing_stripes
#
#    bodies_generate_latitudes
#    bodies_generate_longitudes
#    bodies_latitude_longitude_to_pixels
#
#    bodies_reproject
#    bodies_mosaic_init
#    bodies_mosaic_add
###############################################################################

import cb_logging
import logging

import os
import time

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import oops
import polymath
import gravity
import cspice
from pdstable import PdsTable

from cb_config import *
from cb_util_image import *
from cb_util_oops import *


_LOGGING_NAME = 'cb.' + __name__


# Sometimes the bounding box returned by "inventory" is not quite big enough
BODIES_POSITION_SLOP_FRAC = 0.05

_BODIES_DEFAULT_REPRO_LAT_RESOLUTION = 0.1 * oops.RPD
_BODIES_DEFAULT_REPRO_LON_RESOLUTION = 0.1 * oops.RPD
_BODIES_DEFAULT_REPRO_ZOOM = 2
_BODIES_DEFAULT_REPRO_ZOOM_ORDER = 1
_BODIES_LONGITUDE_SLOP = 1e-6 # Must be smaller than any longitude or radius
_BODIES_LATITUDE_SLOP = 1e-6  # resolution we will be using
_BODIES_MIN_LATITUDE = -oops.HALFPI
_BODIES_MAX_LATITUDE = oops.HALFPI-_BODIES_LATITUDE_SLOP*2
_BODIES_MAX_LONGITUDE = oops.TWOPI-_BODIES_LONGITUDE_SLOP*2

# Lambert is set mainly based on the depth of craters - if the incidence angle
# is too large the crater interior is too shadowed and we don't get a good
# reprojection.
_BODIES_REPRO_MAX_INCIDENCE = 70. * oops.RPD
_BODIES_REPRO_MAX_EMISSION = 70. * oops.RPD
_BODIES_REPRO_MAX_RESOLUTION = None
_BODIES_REPRO_EDGE_MARGIN = 3


def _bodies_create_cartographic(bp, body_data):
    logger = logging.getLogger(_LOGGING_NAME+'._bodies_create_cartographic')

    lat_res = body_data['lat_resolution']
    lon_res = body_data['lon_resolution']
    data = body_data['img']
    min_lat_pixel = body_data['lat_idx_range'][0]
    min_lon_pixel = body_data['lon_idx_range'][0]
    body_name = body_data['body_name']
    latlon_type = body_data['latlon_type']
    lon_direction = body_data['lon_direction']
    
    bp_latitude = bp.latitude(body_name, lat_type=latlon_type)
    body_mask_inv = ma.getmaskarray(bp_latitude.mvals)
    ok_body_mask = np.logical_not(body_mask_inv)

    bp_latitude = bp_latitude.vals.astype('float')

    bp_longitude = bp.longitude(body_name, direction=lon_direction,
                                lon_type=latlon_type)
    bp_longitude = bp_longitude.vals.astype('float')

    latitude_pixels = (bp_latitude + oops.HALFPI) / lat_res
    latitude_pixels = np.round(latitude_pixels).astype('int')
    latitude_pixels -= min_lat_pixel
    
    longitude_pixels = bp_longitude / lon_res
    longitude_pixels = np.round(longitude_pixels).astype('int')
    longitude_pixels -= min_lon_pixel

    ok_lat = np.logical_and(latitude_pixels >= 0,
                            latitude_pixels <= data.shape[0]-1)
    ok_lon = np.logical_and(longitude_pixels >= 0,
                            longitude_pixels <= data.shape[1]-1)
    ok_latlon = np.logical_and(ok_lat, ok_lon)
    ok_pixel_mask = np.logical_and(ok_body_mask, ok_latlon)
    inv_ok_pixel_mask = np.logical_not(ok_pixel_mask)
    
    latitude_pixels = np.clip(latitude_pixels, 0, data.shape[0]-1)
    longitude_pixels = np.clip(longitude_pixels, 0, data.shape[1]-1)

    logger.debug('Lat Type %s, Lon Direction %s', latlon_type, lon_direction)
    logger.debug('Latitude range %8.2f %8.2f', 
                 np.min(bp_latitude[ok_body_mask])*oops.DPR, 
                 np.max(bp_latitude[ok_body_mask])*oops.DPR)
    logger.debug('Longitude range %6.2f %6.2f', 
                 np.min(bp_longitude[ok_body_mask])*oops.DPR, 
                 np.max(bp_longitude[ok_body_mask])*oops.DPR)

    model = data[latitude_pixels, longitude_pixels]
    model[inv_ok_pixel_mask] = 0

    return model

def _bodies_make_label(obs, body_name, model, label_avoid_mask, extend_fov,
                       bodies_config):
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    dict_entry = large_body_dict[body_name]

    u_size = dict_entry['u_pixel_size']
    v_size = dict_entry['v_pixel_size']
    bb_area = u_size * v_size
    if bb_area < bodies_config['min_text_area']:
        return None

    if label_avoid_mask is None:
        label_avoid_mask = np.zeros(model.shape, dtype=np.bool)
    else:
        label_avoid_mask = label_avoid_mask.copy()
    model_text = np.zeros(model.shape, dtype=np.bool)
    text_im = Image.frombuffer('L', (model_text.shape[1], 
                                     model_text.shape[0]), 
                               model_text, 'raw', 'L', 0, 1)
    text_draw = ImageDraw.Draw(text_im)

    body_u = dict_entry['center_uv'][0]+extend_fov[0]
    body_v = dict_entry['center_uv'][1]+extend_fov[1]
    body_v_size = dict_entry['v_pixel_size']/2

    # Create an array of distances from the center pixel - always try to put
    # text labels as close to the center as possible to minimize the chance
    # of going off the edge. But give preference to labeling bodies 
    # horizontally to make the labels prettier.
    axis1 = np.tile(np.arange(float(model.shape[1]))-body_u,
                    model.shape[0]).reshape(model.shape)
    axis2 = np.repeat(np.arange(float(model.shape[0]))-body_v,
                      model.shape[1]).reshape(model.shape)
    dist_from_center = axis1**2+5*axis2**2
    # Don't ever label bodies above or below their extent
    body_v_size_text = max(body_v_size, 5)
    if body_v > body_v_size_text:
        dist_from_center[:body_v-body_v_size_text+1,:] = 1e38
    if body_v <= obs.ext_data.shape[0]-body_v_size_text:
        dist_from_center[body_v+body_v_size_text:,:] = 1e38

    # Don't do anything too close to the edges (and text
    # has height)
    dist_from_center[:extend_fov[1]*2+5,:] = 1e38
    dist_from_center[-extend_fov[1]*2-5:,:] = 1e38
    dist_from_center[:,:extend_fov[0]*2+1] = 1e38
    dist_from_center[:,-extend_fov[0]*2:] = 1e38
    
    # Mask out the areas where bodies are sitting so we don't draw text over
    # them
    for name in large_body_dict:
        dict_entry2 = large_body_dict[name]
        u = dict_entry2['center_uv'][0]+extend_fov[0]
        v = dict_entry2['center_uv'][1]+extend_fov[1]
        u_size = dict_entry2['u_pixel_size']/2
        v_size = dict_entry2['v_pixel_size']/2

        axis1 = np.tile(np.arange(float(model.shape[1]))-u,
                        model.shape[0]).reshape(model.shape)
        axis2 = np.repeat(np.arange(float(model.shape[0]))-v,
                          model.shape[1]).reshape(model.shape)

        dist_body = (axis1/u_size)**2+(axis2/v_size)**2
        if not np.all(dist_body <= 1.):
            dist_from_center[dist_body <= 1.] = 1e38
        
    # Give us a few pixels margin from the edge of a moon
    dist_from_center = filt.maximum_filter(dist_from_center, 3)

    label_avoid_mask = np.logical_or(label_avoid_mask, 
                                     dist_from_center == 1e38)
    
    # Now find a good place for each label that doesn't overlap text from 
    # previous model steps or actual moon data
    text_name = body_name.capitalize()
    # Have to add the leading space because there seems to be a bug in ImageDraw
    # that cuts of part of the leading character in some cases
    text_size = text_draw.textsize(' '+text_name+'->')
    u = None
    v = None
    first_u = None
    first_v = None
    first_text = None
    while np.any(dist_from_center != 1e38):
        # Find the closest good pixel to the image center
        v,u = np.unravel_index(np.argmin(dist_from_center), 
                               dist_from_center.shape)
        raw_v, raw_u = v,u
        if (text_size[1]/2 <= v <
            model_text.shape[0]-text_size[1]/2):
            v -= text_size[1]/2
            if ((u < body_u and u > extend_fov[0]+text_size[0]) or
                (u >= body_u and 
                 u+text_size[0] > model_text.shape[1]-extend_fov[0])):
                u = u-text_size[0]
                text = ' ' + text_name + '->'
            else:
                text = ' <-' + text_name
            if first_u is None:
                first_u = u
                first_v = v
                first_text = text
            if (not np.any(
              label_avoid_mask[
                       max(v-3,0):
                       min(v+text_size[1]+3, model_text.shape[0]),
                       max(u-3,0):
                       min(u+text_size[0]+3, model_text.shape[1])])):
                break
        # We're overlapping with existing text! Or V is too close
        # to the edge.
        dist_from_center[max(raw_v-3,0):
                         min(raw_v+3,model_text.shape[0]),
                         max(raw_u-3,0):
                         min(raw_u+3,model_text.shape[1])] = 1e38
        u = None
    
    if u is None:
        # We give up - just use the first one we found
        u = first_u
        v = first_v
        text = first_text
    
    if u is None:
        # We never found anything good - go ahead and place it directly in the
        # center and ignore that it might overlap with other labels
        u = body_u+3
        v = body_v-text_size[1]//2
        text = ' <-'+text_name
        
    if u is not None:
        text_draw.text((u,v), text, fill=1)

    model_text = (np.array(text_im.getdata()).astype('bool').
                  reshape(model_text.shape))
    
    return model_text


def bodies_create_model(obs, body_name, inventory,
                        extend_fov=(0,0),
                        cartographic_data={},
                        always_create_model=False,
                        label_avoid_mask=None,
                        bodies_config=None,
                        mask_only=False):
    """Create a model for a body.
    
    Inputs:
        obs                    The Observation.
        body_name              The name of the moon.
        inventory              The entry returned from the inventory() method of
                               an Observation for this body. Used to find the
                               clipping rectangle.
        extend_fov             The amount beyond the image in the (U,V) 
                               dimension to generate the model.
        cartographic_data      A dictionary of body names containing 
                               cartographic
                               data in lat/lon format. Each entry contains 
                               metadata in the format returned by 
                               bodies_mosaic_init.
        always_create_model    True to always return a model even if the 
                               body is too small or the curvature or limb is 
                               bad.
                               This is overriden by mask_only.
        label_avoid_mask       A mask giving places where text labels should
                               not be placed (i.e. labels from another
                               model are already there). None if no mask.
        bodies_config          Configuration parameters.
        mask_only              Only compute the latlon mask and don't spent time
                               actually trying to make a model.
                                 
    Returns:
        model, metadata, text
        
        metadata is a dictionary containing

        'body_name'            The name of the body.
        'size_ok'              True if the body is large enough to bother with.
        'inventory'            The inventory entry for this body.
        'cartographic_data_source' The path of the mosaic used to provide the 
                               cartographic data. None if no data provided.
        'curvature_ok'         True if sufficient curvature is visible to permit
                               correlation.
        'limb_ok'              True if the limb is sufficiently sharp to permit
                               correlation.
        'entirely_visible'     True if the body is entirely visible even if 
                               shifted to the maximum extent specified by
                               extend_fov. This is based purely on geometry, 
                               not looking at whether other objects occlude it.
        'occluded_by'          A list of body names or 'RINGS' that occlude some
                               or all of this body. This is set in the main
                               offset loop, not in this procedure. None if
                               Nothing occludes it or it hasn't been processed
                               by the main loop yet.
        'in_saturn_shadow'     True if the body is in Saturn's shadow and only
                               illuminated by Saturn-shine.
        'sub_solar_lon'        The sub-solar longitude (IAU, West).
        'sub_solar_lat'        The sub-solar latitude.
        'sub_observer_lon'     The sub-observer longitude (IAU, West).
        'sub_observer_lat'     The sub-observer latitude.
        'phase_angle'          The phase angle at the body's center.

        'start_time'           The time (s) when bodies_create_model was called.
        'end_time'             The time (s) when bodies_create_model returned.

            These are used for bootstrapping:
            
        'latlon_mask'          A mask showing which lat/lon are visible in the
                               image.
        'mask_lat_resolution'  The latitude resolution in latlon_mask.
        'mask_lon_resolution'  The longitude resolution in latlon_mask.
        'mask_latlon_type'     The type of lat/lon ('centric', 'graphic', or
                               'squashed') in latlon_mask.
        'mask_lon_direction'   The direction of longitude ('east' or 'west')
                               in latlon_mask.
    """
    start_time = time.time()
    
    logger = logging.getLogger(_LOGGING_NAME+'.bodies_create_model')

    if bodies_config is None:
        bodies_config = BODIES_DEFAULT_CONFIG
        
    body_name = body_name.upper()

    set_obs_ext_bp(obs, extend_fov)
    
    metadata = {}
    metadata['body_name'] = body_name
    metadata['inventory'] = inventory
    metadata['cartographic_data_source'] = None
    if cartographic_data and body_name in cartographic_data:
        metadata['cartographic_data_source'] = (
                             cartographic_data[body_name]['full_path'])
    metadata['size_ok'] = None
    metadata['curvature_ok'] = None
    metadata['limb_ok'] = None
    metadata['entirely_visible'] = None
    metadata['occluded_by'] = None
    metadata['in_saturn_shadow'] = None
    metadata['latlon_mask'] = None
    metadata['mask_lat_resolution'] = None
    metadata['mask_lon_resolution'] = None
    metadata['mask_latlon_type'] = None
    metadata['mask_lon_direction'] = None
    metadata['has_bad_pixels'] = None
    metadata['is_obscured'] = False # XXX Subsolar LAT/LON? Obscured by rings?
    metadata['start_time'] = start_time 

    logger.info('*** Modeling %s ***', body_name)

    metadata['sub_solar_lon'] = obs.ext_bp.sub_solar_longitude(
                           body_name,
                           direction=bodies_config['mask_lon_direction']).vals
    metadata['sub_solar_lat'] = obs.ext_bp.sub_solar_latitude(body_name).vals
    metadata['sub_observer_lon'] = obs.ext_bp.sub_observer_longitude(
                           body_name,
                           direction=bodies_config['mask_lon_direction']).vals
    metadata['sub_observer_lat'] = obs.ext_bp.sub_observer_latitude(body_name).vals
    metadata['phase_angle'] = obs.ext_bp.center_phase_angle(body_name).vals

    logger.info('Sub-solar longitude %.2f', metadata['sub_solar_lon']*oops.DPR)
    logger.info('Sub-solar latitude %.2f', metadata['sub_solar_lat']*oops.DPR)
    logger.info('Sub-observer longitude %.2f', metadata['sub_observer_lon']*oops.DPR)
    logger.info('Sub-observer latitude %.2f', metadata['sub_observer_lat']*oops.DPR)
    logger.info('Phase angle %.2f', metadata['phase_angle']*oops.DPR)

    # Create a Meshgrid that only covers the center pixel of the body
    ctr_uv = inventory['center_uv']
    ctr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(ctr_uv[0]+.5, ctr_uv[1]+.5),
                                         limit =(ctr_uv[0]+.5, ctr_uv[1]+.5),
                                         swap  =True)
    ctr_bp = oops.Backplane(obs, meshgrid=ctr_meshgrid)
    saturn_shadow = ctr_bp.where_inside_shadow(body_name, 'saturn').vals
    if np.any(saturn_shadow):
        logger.info('Body is in Saturn\'s shadow')
        metadata['in_saturn_shadow'] = True
    else:
        metadata['in_saturn_shadow'] = False
        
    bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
    if bb_area >= bodies_config['min_bounding_box_area']:
        metadata['size_ok'] = True
    else:
        metadata['size_ok'] = False
        logger.info(
            'Bounding box (area %.3f pixels) is too small to bother with',
            bb_area)
        if not always_create_model and not mask_only:
            metadata['latlon_mask'] = None
            metadata['end_time'] = time.time()
            return None, metadata, None
        
    # Analyze the curvature
            
    u_min = inventory['u_min_unclipped']
    u_max = inventory['u_max_unclipped']
    v_min = inventory['v_min_unclipped']
    v_max = inventory['v_max_unclipped']

    logger.debug('Original bounding box U %d to %d V %d to %d',
                 u_min, u_max, v_min, v_max)
        
    entirely_visible = False
    if (u_min >= extend_fov[0] and 
        u_max <= obs.data.shape[1]-1-extend_fov[0] and
        v_min >= extend_fov[1] and 
        v_max <= obs.data.shape[0]-1-extend_fov[1]):
        # Body is entirely visible - no part is off the edge even when shifting
        # the extended FOV
        entirely_visible = True
    metadata['entirely_visible'] = entirely_visible
    
    curvature_threshold_frac = bodies_config['curvature_threshold_frac']
    curvature_threshold_pix = bodies_config['curvature_threshold_pixels']
    u_center = (u_min+u_max)/2
    v_center = (v_min+v_max)/2
    width = u_max-u_min+1
    height = v_max-v_min+1
    width_threshold = max(width * curvature_threshold_frac,
                          curvature_threshold_pix)
    height_threshold = max(height * curvature_threshold_frac,
                           curvature_threshold_pix)
    # Slightly more than half of the body must be visible in each of the
    # horizontal and vertical directions
    if (((u_center-width_threshold > extend_fov[0] and # Body is to left 
          u_center+width/2 < obs.data.shape[1]-1-extend_fov[0]) or
         (u_center+width_threshold < obs.data.shape[1]-1-extend_fov[0] and
          u_center-width/2 > extend_fov[0])) and # Body is to right
        ((v_center-height_threshold > extend_fov[1] and  # Body is at top
          v_center+height/2 < obs.data.shape[0]-1-extend_fov[1]) or
         (v_center+height_threshold < obs.data.shape[0]-1-extend_fov[1] and
          v_center-height/2 > extend_fov[1]))): # Body is at bottom
        metadata['curvature_ok'] = True
    else:
        metadata['curvature_ok'] = False
    
    u_min -= int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
    u_max += int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
    v_min -= int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)
    v_max += int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)
    
    u_min = np.clip(u_min, -extend_fov[0], obs.data.shape[1]+extend_fov[0]-1)
    u_max = np.clip(u_max, -extend_fov[0], obs.data.shape[1]+extend_fov[0]-1)
    v_min = np.clip(v_min, -extend_fov[1], obs.data.shape[0]+extend_fov[1]-1)
    v_max = np.clip(v_max, -extend_fov[1], obs.data.shape[0]+extend_fov[1]-1)
    
    # Things break if the moon is only a single pixel wide or tall
    if u_min == u_max and u_min == obs.data.shape[1]+extend_fov[0]-1:
        u_min -= 1
    if u_min == u_max and u_min == -extend_fov[0]:
        u_max += 1
    if v_min == v_max and v_min == obs.data.shape[0]+extend_fov[1]-1:
        v_min -= 1
    if v_min == v_max and v_min == -extend_fov[1]:
        v_max += 1
        
    logger.debug('Image size %d %d subrect U %d to %d V %d to %d',
                 obs.data.shape[1], obs.data.shape[0],
                 u_min, u_max, v_min, v_max)
    if entirely_visible:
        logger.info('All of body is visible')
    else:
        logger.info('Not all of body is visible')

    if metadata['curvature_ok']:
        logger.info('Curvature OK')
    else:
        logger.info('Curvature BAD')

    # Create a Meshgrid that only covers the extent of the body
    restr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                           origin=(u_min+.5, v_min+.5),
                                           limit =(u_max+.5, v_max+.5),
                                           swap  =True)
    restr_bp = oops.Backplane(obs, meshgrid=restr_meshgrid)

    if bb_area >= bodies_config['min_latlon_mask_area']:
        # Compute the lat/lon mask for bootstrapping
        
        latlon_mask = bodies_reproject(obs, body_name, 
            lat_resolution=bodies_config['mask_lat_resolution'], 
            lon_resolution=bodies_config['mask_lon_resolution'],
            zoom_amt=1,
            latlon_type=bodies_config['mask_latlon_type'],
            lon_direction=bodies_config['mask_lon_direction'],
            max_incidence=None,
            max_emission=None,
            max_resolution=None,
            mask_only=True, override_backplane=restr_bp,
            subimage_edges=(u_min,u_max,v_min,v_max))
    
        metadata['latlon_mask'] = latlon_mask
        metadata['mask_lat_resolution'] = bodies_config['mask_lat_resolution']
        metadata['mask_lon_resolution'] = bodies_config['mask_lon_resolution']
        metadata['mask_latlon_type'] = bodies_config['mask_latlon_type']
        metadata['mask_lon_direction'] = bodies_config['mask_lon_direction']

    if mask_only:
        metadata['end_time'] = time.time()
        return None, metadata, None

    # Analyze the limb
    
    incidence = restr_bp.incidence_angle(body_name)
    restr_body_mask_inv = ma.getmaskarray(incidence.mvals)
    restr_body_mask = np.logical_not(restr_body_mask_inv)

    # If the inv mask is true, but any of its neighbors are false, then
    # this is an edge
    limb_mask = restr_body_mask
    limb_mask_1 = shift_image(restr_body_mask_inv, -1,  0)
    limb_mask_2 = shift_image(restr_body_mask_inv,  1,  0)
    limb_mask_3 = shift_image(restr_body_mask_inv,  0, -1)
    limb_mask_4 = shift_image(restr_body_mask_inv,  0,  1)
    limb_mask_total = np.logical_or(limb_mask_1, limb_mask_2)
    limb_mask_total = np.logical_or(limb_mask_total, limb_mask_3)
    limb_mask_total = np.logical_or(limb_mask_total, limb_mask_4)
    limb_mask = np.logical_and(limb_mask, limb_mask_total)

    if not np.any(limb_mask):
        limb_incidence_min = 1e38
        limb_incidence_max = 1e38
        logger.info('No limb')
    else:
        limb_incidence_min = np.min(incidence[limb_mask].vals)
        limb_incidence_max = np.max(incidence[limb_mask].vals)
        logger.info('Limb incidence angle min %.2f max %.2f',
                     limb_incidence_min*oops.DPR, limb_incidence_max*oops.DPR)
    limb_threshold = bodies_config['limb_incidence_threshold']
    # If we can see the entire body, then we only need part of the limb to be
    # OK. If the body is partially off the edge, then we need the entire limb
    # to be OK. If we have cartographic data, then we don't need a limb
    # at all.
    if ((entirely_visible and limb_incidence_min < limb_threshold) or
        (not entirely_visible and limb_incidence_max < limb_threshold) or
        (cartographic_data and body_name in cartographic_data)):
        logger.info('Limb meets criteria')
        metadata['limb_ok'] = True
    else:
        metadata['limb_ok'] = False
        logger.info('Limb fails criteria')
        if not always_create_model:
            metadata['end_time'] = time.time()
            return None, metadata, None

    # Make the actual model

    restr_model = None    
    if (np.max(restr_body_mask) == 0 or 
        np.min(incidence[restr_body_mask].vals) >= oops.HALFPI):
        logger.debug('Looking only at back side - leaving model empty')
        # Make a slight glow even on the back side
        restr_model = np.zeros(restr_body_mask.shape)
        restr_model[restr_body_mask] = 0.01
    else:
        logger.debug('Making actual model')
    
        if bodies_config['use_lambert']:
            restr_model = restr_bp.lambert_law(body_name).vals.astype('float')
            # Make a slight glow even past the terminator
            restr_model = restr_model+0.01
            restr_model[restr_body_mask_inv] = 0.
        else:
            restr_model = restr_body_mask.astype('float')
    
        if cartographic_data:
            for cart_body in sorted(cartographic_data.keys()):
                if cart_body == body_name:
                    cart_body_data = cartographic_data[body_name]
                    cart_model = _bodies_create_cartographic(restr_bp, cart_body_data)
                    restr_model *= cart_model
                    logger.info('Cartographic data provided for %s - USING', cart_body)
                else:
                    logger.info('Cartographic data provided for %s', cart_body)

    # Take the full-resolution object and put it back in the right place in a
    # full-size image
    model = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                      obs.data.shape[1]+extend_fov[0]*2),
                     dtype=np.float32)
    if restr_model is not None:
        model[v_min+extend_fov[1]:v_max+extend_fov[1]+1,
              u_min+extend_fov[0]:u_max+extend_fov[0]+1] = restr_model
    
    model_text = _bodies_make_label(obs, body_name, model, label_avoid_mask,
                                    extend_fov, bodies_config)
    
    metadata['end_time'] = time.time()
    return model, metadata, model_text


#==============================================================================
# 
# BODY REPROJECTION UTILITIES
#
#==============================================================================

def bodies_interpolate_missing_stripes(data):
    """Interpolate missing horizontal data in an image.
    
    This routine handles an image that has the right side of some lines missing
    due to data transmission limitations. A pixel is interpolated if it is
    missing (zero) and the pixels immediately above and below are present
    (not zero)."""
    zero_mask = (data == 0.)
    data_up = np.zeros(data.shape)
    data_up[:-1,:] = data[1:,:]
    data_down = np.zeros(data.shape)
    data_down[1:,:] = data[:-1,:]
    up_mask = (data_up != 0.)
    down_mask = (data_down != 0.)
    good_mask = np.logical_and(zero_mask, up_mask)
    good_mask = np.logical_and(good_mask, down_mask)
    data_mean = (data_up+data_down)/2.
    ret_data = data.copy()
    ret_data[good_mask] = data_mean[good_mask]
    return ret_data

def bodies_generate_latitudes(latitude_start=_BODIES_MIN_LATITUDE,
                              latitude_end=_BODIES_MAX_LATITUDE,
                              lat_resolution=
                                    _BODIES_DEFAULT_REPRO_LAT_RESOLUTION):
    """Generate a list of latitudes.
    
    The list will be on lat_resolution boundaries and is guaranteed to
    not contain a latitude less than latitude_start or greater than
    latitude_end."""
    start_idx = int(np.ceil(latitude_start/lat_resolution))
    end_idx = int(np.floor(latitude_end/lat_resolution))
    return np.arange(start_idx, end_idx+1) * lat_resolution

def bodies_generate_longitudes(longitude_start=0.,
                               longitude_end=_BODIES_MAX_LONGITUDE,
                               lon_resolution=
                                   _BODIES_DEFAULT_REPRO_LON_RESOLUTION):
    """Generate a list of longitudes.
    
    The list will be on lon_resolution boundaries and is guaranteed to
    not contain a longitude less than longitude_start or greater than
    longitude_end."""
    start_idx = int(np.ceil(longitude_start/lon_resolution)) 
    end_idx = int(np.floor(longitude_end/lon_resolution))
    return np.arange(start_idx, end_idx+1) * lon_resolution

def bodies_latitude_longitude_to_pixels(obs, body_name, latitude, longitude,
                                        latlon_type='centric',
                                        lon_direction='east'):
    """Convert latitude,longitude pairs to U,V."""
    assert latlon_type in ('centric', 'graphic', 'squashed')
    assert lon_direction in ('east', 'west')

    latitude = polymath.Scalar.as_scalar(latitude)
    longitude = polymath.Scalar.as_scalar(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])

    moon_surface = oops.Body.lookup(body_name).surface

    # Get the 'squashed' latitude    
    if latlon_type == 'centric':
        longitude = moon_surface.lon_from_centric(longitude)
        latitude = moon_surface.lat_from_centric(latitude, longitude)
    elif latlon_type == 'graphic':
        longitude = moon_surface.lon_from_graphic(longitude)
        latitude = moon_surface.lat_from_graphic(latitude, longitude)

    # Internal longitude is always 'east'
    if lon_direction == 'west':
        longitude = (-longitude) % oops.TWOPI

    uv = obs.uv_from_coords(moon_surface, (longitude, latitude))
    
    return uv
    
    
#==============================================================================
# 
# BODY REPROJECTION MAIN ROUTINES
#
#==============================================================================

def bodies_reproject(
            obs, body_name, data=None, offset=None,
            offset_path=None,
            lat_resolution=_BODIES_DEFAULT_REPRO_LAT_RESOLUTION,
            lon_resolution=_BODIES_DEFAULT_REPRO_LON_RESOLUTION,
            max_incidence=_BODIES_REPRO_MAX_INCIDENCE,
            max_emission=_BODIES_REPRO_MAX_EMISSION,
            max_resolution=_BODIES_REPRO_MAX_RESOLUTION,
            edge_margin=_BODIES_REPRO_EDGE_MARGIN,
            zoom_amt=_BODIES_DEFAULT_REPRO_ZOOM, 
            zoom_order=_BODIES_DEFAULT_REPRO_ZOOM_ORDER,
            latlon_type='centric', lon_direction='east',
            mask_only=False, override_backplane=None,
            subimage_edges=None, mask_bad_areas=False):
    """Reproject the moon into a rectangular latitude/longitude space.
    
    Inputs:
        obs                      The Observation.
        body_name                The name of the moon.
        data                     The image data to use for the reprojection. If
                                 None, use obs.data.
        offset                   The offsets in (U,V) to apply to the image
                                 when computing the latitude and longitude
                                 values.
        offset_path              The full path of the OFFSET file used to
                                 determine the offset.
        lat_resolution      The latitude resolution of the new image
                                 (rad/pix).
        lon_resolution     The longitude resolution of the new image
                                 (rad/pix).
        max_incidence            The maximum incidence angle (rad) to permit
                                 for a valid pixel. None for no check.
        max_emission             The maximum emission angle (rad) to permit
                                 for a valid pixel. None for no check.
        max_resolution           The maximum resolution (km/pix) to permit for
                                 a valid pixel. None for no check.
        edge_margin              The number of pixels around the edge of the
                                 image to throw away because ISS images tend
                                 to have weird garbage there.
        zoom_amt                 The amount to magnify the original image for
                                 pixel value interpolation.
        zoom_order               The spline order to use for zooming.
        latlon_type              The coordinate system to use for latitude and
                                 longitude. One of 'centric', 'graphic', or
                                 'squashed'.
        lon_direction            The coordinate system to use for longitude.
                                 One of 'east' or 'west'.
        mask_only                If true, return only a complete mask of which
                                 longitudes and latitudes are visible in the
                                 image.
        override_backplane       A Backplane object to use instead of creating
                                 a fresh one based on obs. Note that if an 
                                 offset is supplied, this Backplane must have
                                 already been computed using that offset. This
                                 currently only works with mask_only True.
        subimage_edges           A tuple (u_min,u_max,v_min,v_max) describing
                                 the subimage used for the Meshgrid when
                                 override_backplane is used. None to indicate
                                 the Backplane is the size of data.
        mask_bad_areas           True to remove pixels nearby bad pixels before
                                 reprojecting. This is used to deal with images
                                 that have every other line contain zeroes due
                                 to a transmission error.
                                 
    Returns:
            If mask_only is False, a dictionary containing

        'body_name'        The name of the body.
        'path'             The full path from the Observation.
        'offset_path'      The full path of the OFFSET file.
        'lat_idx_range'    The range (min,max) of latitudes in the returned
                           image.
        'lat_resolution'   The resolution (rad/pix) in the latitude direction.
        'lon_idx_range'    The range (min,max) of longitudes in the returned
                           image.
        'lon_resolution'   The resolution (rad/pix) in the longitude direction.
        'latlon_type'      The latitude/longitude coordinate system (see
                           above).
        'lon_direction'    The longitude coordinate system (see above).
        'time'             The midtime of the observation (TDB).
        'offset'           The offset that was used on the image. Can be None
                           or a tuple (U,V).
                           
            The following only contain latitudes and longitudes in the ranges
            specified above. All angles are in radians.
            
        'img'              The reprojected image [latitude,longitude].
        'full_mask'        The mask of pixels that contain reprojected data.
                           True means the pixel is valid. If mask_only is True,
                           this mask is the full lat/lon size. Otherwise it
                           is restricted to the sub-set like everything
                           else.
        'resolution'       The radial resolution [latitude,longitude].
        'phase'            The phase angle [latitude,longitude].
        'emission'         The emission angle [latitude,longitude].
        'incidence'        The incidence angle [latitude,longitude].

        Note that the image data is taken from the zoomed, interpolated image,
        while the incidence, emission, phase, and resolution are taken from 
        the original non-interpolated data and thus will be slightly more 
        coarse-grained.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.bodies_reproject')

    if data is None:
        data = obs.data
        
    # We need to be careful not to use obs.bp from this point forward because
    # it will disagree with our current OffsetFOV
    orig_fov = None
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset
        orig_fov = obs.fov
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    
    # Get all the info for each pixel
    if override_backplane:
        bp = override_backplane
    else:
        bp = oops.Backplane(obs)
    
    if max_incidence is not None or not mask_only:
        lambert = bp.lambert_law(body_name).vals.astype('float32')
        bp_incidence = bp.incidence_angle(body_name).vals.astype('float32') 

    if max_emission is not None or not mask_only:
        bp_emission = bp.emission_angle(body_name).vals.astype('float32')
        
    if not mask_only:
        bp_phase = bp.phase_angle(body_name).vals.astype('float32')

        # Resolution takes into account the emission angle - the "along sight"
        # projection
        center_resolution = (bp.center_resolution(body_name).
                             vals.astype('float32'))
        resolution = center_resolution / np.cos(bp_emission)

    bp_latitude = bp.latitude(body_name, lat_type=latlon_type)
    body_mask_inv = ma.getmaskarray(bp_latitude.mvals)
    bp_latitude = bp_latitude.vals.astype('float32')

    bp_longitude = bp.longitude(body_name, direction=lon_direction,
                                lon_type=latlon_type)
    bp_longitude = bp_longitude.vals.astype('float32')

    if subimage_edges is not None:
        u_min, u_max, v_min, v_max = subimage_edges
        subimg = data[v_min:v_max+1,u_min:u_max+1]
    else:
        subimg = data

    if not mask_only:
        zero_mask = (subimg == 0.)    
        if mask_bad_areas:
            zero_mask = filt.maximum_filter(zero_mask, 3)
        
    # A pixel is OK if it falls on the body, the Lambert model is
    # bright enough, the emission angle is large enough, the resolution
    # is small enough, it's not on the edge, and the data isn't exactly ZERO.
    if edge_margin > 0:
        body_mask_inv[:edge_margin,:] = True
        body_mask_inv[-edge_margin:,:] = True
        body_mask_inv[:,:edge_margin] = True
        body_mask_inv[:,-edge_margin:] = True
        
    ok_body_mask_inv = body_mask_inv
    if max_incidence is not None:
        ok_body_mask_inv = np.logical_or(ok_body_mask_inv,
                                         bp_incidence > max_incidence)
    if max_emission is not None:
        ok_body_mask_inv = np.logical_or(ok_body_mask_inv, 
                                         bp_emission > max_emission)
    if max_resolution is not None:
        ok_body_mask_inv = np.logical_or(ok_body_mask_inv,
                                         resolution > max_resolution)
    if not mask_only:
        ok_body_mask_inv = np.logical_or(ok_body_mask_inv, zero_mask)
    ok_body_mask = np.logical_not(ok_body_mask_inv)

    empty_mask = not np.any(ok_body_mask)
    
    if not mask_only:
        # Divide the data by the Lambert model and multiply by the cosine of
        # the emission angle to account for projected illumination and viewing
        lambert[ok_body_mask_inv] = 1e38
        adj_data = data / lambert * np.cos(bp_emission)
        adj_data = adj_data.astype('float32')
        adj_data[ok_body_mask_inv] = 0.
        lambert[ok_body_mask_inv] = 0.
        bp_emission[ok_body_mask_inv] = 1e38
        resolution[ok_body_mask_inv] = 1e38

    # The number of pixels in the final reprojection 
    latitude_pixels = int(oops.PI / lat_resolution)
    longitude_pixels = int(oops.TWOPI / lon_resolution)

    if empty_mask:
        min_latitude_pixel = 0
        max_latitude_pixel = 1
        min_longitude_pixel = 0
        max_longitude_pixel = 1
    else:
        # Restrict the latitude and longitude ranges for some attempt at
        # efficiency
        min_latitude = np.min(bp_latitude[ok_body_mask]+oops.HALFPI)
        max_latitude = np.max(bp_latitude[ok_body_mask]+oops.HALFPI) 
        min_latitude_pixel = (np.floor(min_latitude / 
                                       lat_resolution)).astype('int')
        min_latitude_pixel = np.clip(min_latitude_pixel, 0, latitude_pixels-1)
        max_latitude_pixel = (np.ceil(max_latitude / 
                                      lat_resolution)).astype('int')
        max_latitude_pixel = np.clip(max_latitude_pixel, 0, latitude_pixels-1)
    
        min_longitude = np.min(bp_longitude[ok_body_mask])
        max_longitude = np.max(bp_longitude[ok_body_mask])  
        min_longitude_pixel = (np.floor(min_longitude / 
                                        lon_resolution)).astype('int')
        min_longitude_pixel = np.clip(min_longitude_pixel, 0, 
                                      longitude_pixels-1)
        max_longitude_pixel = (np.ceil(max_longitude / 
                                       lon_resolution)).astype('int')
        max_longitude_pixel = np.clip(max_longitude_pixel, 0, 
                                      longitude_pixels-1)

    num_latitude_pixel = max_latitude_pixel - min_latitude_pixel + 1
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    # Latitude bin numbers
    lat_bins = np.repeat(np.arange(min_latitude_pixel, max_latitude_pixel+1, 
                                   dtype=np.int), 
                         num_longitude_pixel)
    # Actual latitude (rad)
    lat_bins_act = lat_bins * lat_resolution - oops.HALFPI

    # Longitude bin numbers
    lon_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1,
                                 dtype=np.int), 
                       num_latitude_pixel)
    # Actual longitude (rad)
    lon_bins_act = lon_bins * lon_resolution

    logger.info('Offset U,V %d,%d / Lat Res %.3f / Lon Res %.3f / '+
                'LatLon Type %s / Lon Direction %s', 
                offset_u, offset_v, 
                lat_resolution*oops.DPR,
                lon_resolution*oops.DPR,
                latlon_type, lon_direction)
    if empty_mask:
        logger.info('Empty body mask')
    else:
        logger.info('Latitude range %8.2f %8.2f', 
                     np.min(bp_latitude[ok_body_mask])*oops.DPR, 
                     np.max(bp_latitude[ok_body_mask])*oops.DPR)
        logger.debug('Latitude bin range %8.2f %8.2f', 
                     np.min(lat_bins_act)*oops.DPR, 
                     np.max(lat_bins_act)*oops.DPR)
        logger.debug('Latitude pixel range %d %d', 
                     min_latitude_pixel, max_latitude_pixel) 
        logger.info('Longitude range %6.2f %6.2f', 
                     np.min(bp_longitude[ok_body_mask])*oops.DPR, 
                     np.max(bp_longitude[ok_body_mask])*oops.DPR)
        logger.debug('Longitude bin range %6.2f %6.2f', 
                     np.min(lon_bins_act)*oops.DPR,
                     np.max(lon_bins_act)*oops.DPR)
        logger.debug('Longitude pixel range %d %d', 
                     min_longitude_pixel, max_longitude_pixel)
        if not mask_only: 
            logger.info('Resolution range %7.2f %7.2f', 
                         np.min(resolution[ok_body_mask]),
                         np.max(resolution[ok_body_mask]))
            logger.debug('Data range %f %f', 
                         np.min(adj_data), 
                         np.max(adj_data))

    uv = bodies_latitude_longitude_to_pixels(
                    obs, body_name, lat_bins_act, lon_bins_act,
                    latlon_type=latlon_type, 
                    lon_direction=lon_direction)

    u, v = uv.to_scalars()
    pixmask = ma.getmaskarray(u.mvals)
    u_pixels = u.vals
    v_pixels = v.vals
    
    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= data.shape[0]-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    goodmask = np.logical_and(goodmask, ~pixmask)
    
    good_u = u_pixels[goodmask]
    good_v = v_pixels[goodmask]
    good_lat = lat_bins[goodmask]
    good_lon = lon_bins[goodmask]
    
    if mask_only:    
        repro_img = np.zeros((latitude_pixels, longitude_pixels), 
                             dtype=np.bool)
        repro_img[good_lat,good_lon] = True
    else:
        # Now get rid of bad data in the image. For some reason it's possible
        # to have bad data here that wasn't caught in the production of
        # ok_body_mask above.
        goodmask = adj_data[good_v.astype('int'), good_u.astype('int')] != 0.
        good_u = good_u[goodmask]
        good_v = good_v[goodmask]
        good_lat = good_lat[goodmask]
        good_lon = good_lon[goodmask]    

        if zoom_amt == 1:
            zoom_data = adj_data
        else:
            zoom_data = ndinterp.zoom(adj_data, zoom_amt, order=zoom_order)

        bad_data_mask = (adj_data == 0.)
        # Replicate the mask in each zoom x zoom block
        if zoom_amt == 1:
            bad_data_mask_zoom = bad_data_mask
        else:
            bad_data_mask_zoom = ndinterp.zoom(bad_data_mask, zoom_amt, 
                                               order=0)
        zoom_data[bad_data_mask_zoom] = 0.
    
        interp_data = zoom_data[(good_v*zoom_amt).astype('int'), 
                                (good_u*zoom_amt).astype('int')]

        repro_img = ma.zeros((latitude_pixels, longitude_pixels), 
                             dtype=np.float32)
        repro_img.mask = True
        repro_img[good_lat,good_lon] = interp_data

        good_u = good_u.astype('int')
        good_v = good_v.astype('int')
    
        repro_res = ma.zeros((latitude_pixels, longitude_pixels), 
                             dtype=np.float32)
        repro_res.mask = True
        repro_res[good_lat,good_lon] = resolution[good_v,good_u]
        full_mask = np.logical_not(ma.getmaskarray(repro_res))
    
        repro_phase = ma.zeros((latitude_pixels, longitude_pixels), 
                               dtype=np.float32)
        repro_phase.mask = True
        repro_phase[good_lat,good_lon] = bp_phase[good_v,good_u]
    
        repro_emission = ma.zeros((latitude_pixels, longitude_pixels), 
                                  dtype=np.float32)
        repro_emission.mask = True
        repro_emission[good_lat,good_lon] = bp_emission[good_v,good_u]
    
        repro_incidence = ma.zeros((latitude_pixels, longitude_pixels), 
                                   dtype=np.float32)
        repro_incidence.mask = True
        repro_incidence[good_lat,good_lon] = bp_incidence[good_v,good_u]

        full_mask = full_mask[min_latitude_pixel:max_latitude_pixel+1,
                              min_longitude_pixel:max_longitude_pixel+1]
        repro_img = ma.filled(
                  repro_img[min_latitude_pixel:max_latitude_pixel+1,
                            min_longitude_pixel:max_longitude_pixel+1], 0.)    
        repro_res = ma.filled(
                  repro_res[min_latitude_pixel:max_latitude_pixel+1,
                            min_longitude_pixel:max_longitude_pixel+1], 0.)
        repro_phase = ma.filled(
                  repro_phase[min_latitude_pixel:max_latitude_pixel+1,
                              min_longitude_pixel:max_longitude_pixel+1], 0.)
        repro_emission = ma.filled(
                  repro_emission[min_latitude_pixel:max_latitude_pixel+1,
                                 min_longitude_pixel:max_longitude_pixel+1], 0.)
        repro_incidence = ma.filled(
                  repro_incidence[min_latitude_pixel:max_latitude_pixel+1,
                                  min_longitude_pixel:max_longitude_pixel+1], 
                  0.)

    if orig_fov is not None:   
        obs.fov = orig_fov

    if mask_only:
        return repro_img
    
    ret = {}
    
    ret['body_name'] = body_name
    ret['path'] = obs.full_path
    ret['offset_path'] = offset_path
    ret['full_mask'] = full_mask.reshape((num_latitude_pixel, 
                                          num_longitude_pixel))
    ret['lat_idx_range'] = (min_latitude_pixel, max_latitude_pixel)
    ret['lat_resolution'] = lat_resolution
    ret['lon_idx_range'] = (min_longitude_pixel, max_longitude_pixel)
    ret['lon_resolution'] = lon_resolution
    ret['latlon_type'] = latlon_type
    ret['lon_direction'] = lon_direction
    ret['offset'] = offset
    ret['img'] = repro_img
    ret['resolution'] = repro_res
    ret['phase'] = repro_phase
    ret['emission'] = repro_emission
    ret['incidence'] = repro_incidence
    ret['time'] = obs.midtime
    
    return ret

def bodies_mosaic_init(body_name,
      lat_resolution=_BODIES_DEFAULT_REPRO_LAT_RESOLUTION,
      lon_resolution=_BODIES_DEFAULT_REPRO_LON_RESOLUTION,
      latlon_type='centric', lon_direction='east'):
    """Create the data structure for a moon mosaic.

    Inputs:
        lat_resolution      The latitude resolution of the new image
                                 (rad/pix).
        lon_resolution     The longitude resolution of the new image
                                 (rad/pix).
        latlon_type              The coordinate system to use for latitude and
                                 longitude. One of 'centric', 'graphic', or 
                                 'squashed'.
        lon_direction            The coordinate system to use for longitude.
                                 One of 'east' or 'west'.
                                 
    Returns:
        A dictionary containing an empty mosaic

        'body_name'        The name of the body.
        'img'              The full mosaic image.
        'full_mask'        The valid-pixel mask (all False). True means the
                           pixel is valid.
        'lat_idx_range'    The range (min,max) of latitudes in the returned
                           image.
        'lat_resolution'   The resolution (rad/pix) in the latitude direction.
        'lon_idx_range'    The range (min,max) of longitudes in the returned
                           image.
        'lon_resolution'   The resolution (rad/pix) in the longitude direction.
        'latlon_type'      The latitude/longitude coordinate system (see
                           above).
        'lon_direction'    The longitude coordinate system (see above).
        'resolution'       The per-pixel resolution.
        'phase'            The per-pixel phase angle.
        'emission'         The per-pixel emission angle.
        'incidence'        The per-pixel incidence angle.
        'image_number'     The per-pixel image number giving the image
                           used to fill the data for each pixel.
        'path_list'        The image paths indexed by image_number.
        'offset_path_list' The OFFSET paths indexed by image_number.
        'time'             The per-pixel time (TDB).
    """
    latitude_pixels = int(oops.PI / lat_resolution)
    longitude_pixels = int(oops.TWOPI / lon_resolution)
    
    ret = {}
    ret['body_name'] = body_name
    ret['img'] = np.zeros((latitude_pixels, longitude_pixels), 
                          dtype=np.float32)
    ret['full_mask'] = np.zeros((latitude_pixels, longitude_pixels), 
                                dtype=np.bool)
    ret['lat_idx_range'] = (0, latitude_pixels-1)
    ret['lat_resolution'] = lat_resolution
    ret['lon_idx_range'] = (0, longitude_pixels-1)
    ret['lon_resolution'] = lon_resolution
    ret['latlon_type'] = latlon_type
    ret['lon_direction'] = lon_direction
    ret['resolution'] = np.zeros((latitude_pixels, longitude_pixels), 
                                 dtype=np.float32)
    ret['phase'] = np.zeros((latitude_pixels, longitude_pixels), 
                            dtype=np.float32)
    ret['emission'] = np.zeros((latitude_pixels, longitude_pixels), 
                               dtype=np.float32)
    ret['incidence'] = np.zeros((latitude_pixels, longitude_pixels), 
                                dtype=np.float32)
    ret['image_number'] = np.zeros((latitude_pixels, longitude_pixels), 
                                   dtype=np.int32)
    ret['path_list'] = []
    ret['offset_path_list'] = []
    ret['time'] = np.zeros((latitude_pixels, longitude_pixels), 
                           dtype=np.float32)
    
    return ret

def bodies_mosaic_add(mosaic_metadata, repro_metadata, 
                      resolution_threshold=1.,
                      copy_slop=0):
    """Add a reprojected image to an existing mosaic.
    
    For each valid pixel in the reprojected image, it is copied to the
    mosaic if it has better resolution.
    
    Inputs:
        mosaic_metadata        The mosaic metadata.
        repro_metadata         The reprojected body metadata.
        resolution_threshold   The factor that the repro image resolution
                               must be greater than the mosaic resolution
                               to copy a pixel. Should be >= 1.
        copy_slop              The number of additional pixels around
                               a copied pixel to also copy. This is to
                               reduce the occurance of isolated pixels
                               from one image being surrounded by pixels
                               from another image.
    """
    mosaic_img = mosaic_metadata['img']
    mosaic_mask = mosaic_metadata['full_mask']
    
    assert mosaic_metadata['latlon_type'] == repro_metadata['latlon_type']
    assert mosaic_metadata['lon_direction'] == repro_metadata['lon_direction']
    assert mosaic_metadata['lat_resolution'] == repro_metadata['lat_resolution']
    assert mosaic_metadata['lon_resolution'] == repro_metadata['lon_resolution']
    
    repro_lat_idx_range = repro_metadata['lat_idx_range']
    repro_lon_idx_range = repro_metadata['lon_idx_range']

    repro_mask = np.zeros(mosaic_img.shape, dtype=np.bool) 
    repro_mask[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
               repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['full_mask']
        
    repro_img = np.zeros(mosaic_img.shape, dtype=np.float32) 
    repro_img[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
              repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['img']
    
    mosaic_res = mosaic_metadata['resolution']
    repro_res = np.zeros(mosaic_res.shape, dtype=np.float32)
    repro_res[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
              repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['resolution']
    
    mosaic_phase = mosaic_metadata['phase']
    repro_phase = np.zeros(mosaic_phase.shape, dtype=np.float32)
    repro_phase[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['phase']
    
    mosaic_emission = mosaic_metadata['emission']
    repro_emission = np.zeros(mosaic_emission.shape, dtype=np.float32)
    repro_emission[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                   repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['emission']
    
    mosaic_incidence = mosaic_metadata['incidence']
    repro_incidence = np.zeros(mosaic_incidence.shape, dtype=np.float32)
    repro_incidence[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                    repro_lon_idx_range[0]:repro_lon_idx_range[1]+1] = \
                                repro_metadata['incidence']
    
    mosaic_image_number = mosaic_metadata['image_number']
    mosaic_time = mosaic_metadata['time']
    
    # Calculate where the new resolution is better
    better_resolution_mask = (repro_res * resolution_threshold) < mosaic_res
#    plt.figure()
#    plt.imshow(better_resolution_mask)
#    if copy_slop > 0:
#        better_resolution_mask = filt.maximum_filter(better_resolution_mask,
#                                                     copy_slop*2+1)
#    plt.figure()
#    plt.imshow(better_resolution_mask)
    
    replace_mask = np.logical_and(repro_mask,
                                  np.logical_or(better_resolution_mask, 
                                                np.logical_not(mosaic_mask)))

    if copy_slop > 0:
        replace_mask = filt.maximum_filter(replace_mask, copy_slop*2+1)
        replace_mask = np.logical_and(repro_mask, replace_mask)

#    plt.figure()
#    plt.imshow(repro_mask)
#
#    plt.figure()
#    plt.imshow(replace_mask)
#    plt.show()

    mosaic_img[replace_mask] = repro_img[replace_mask]
    mosaic_res[replace_mask] = repro_res[replace_mask] 
    mosaic_phase[replace_mask] = repro_phase[replace_mask] 
    mosaic_emission[replace_mask] = repro_emission[replace_mask] 
    mosaic_incidence[replace_mask] = repro_incidence[replace_mask]
    mosaic_image_number[replace_mask] = len(mosaic_metadata['path_list']) 
    mosaic_metadata['path_list'].append(repro_metadata['path'])
    mosaic_metadata['offset_path_list'].append(repro_metadata['offset_path'])
    mosaic_time[replace_mask] = repro_metadata['time'] 
    mosaic_mask[replace_mask] = True


#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================

CARTOGRAPHIC_BODIES = {
    'DIONE': 'COISS_3003',
    'ENCELADUS': 'COISS_3002',
    'IPAETUS': 'COISS_3005',
    'MIMAS': 'COISS_3006',
    'PHOEBE': 'COISS_3001',
    'RHEA': 'COISS_3007',
    'TETHYS': 'COISS_3004',
}

CARTOGRAPHIC_FILE_CACHE = {}

#===============================================================================
# XXX ALL THIS IS TEMPORARY STUFF
#===============================================================================

def _spice_body_spherical(spice_id, radius):
    """Returns a Spheroid or Ellipsoid defining the path, orientation and shape
    of a body defined in the SPICE toolkit.

    Input:
        spice_id        the name or ID of the body as defined in the SPICE
                        toolkit.
        centric         True to use planetocentric latitudes, False to use
                        planetographic latitudes.
    """

    spice_body_id = oops.spice.body_id_and_name(spice_id)[0]
    origin_id = oops.spice.PATH_TRANSLATION[spice_body_id]

    spice_frame_name = oops.spice.frame_id_and_name(spice_id)[1]
    frame_id = oops.spice.FRAME_TRANSLATION[spice_frame_name]

    radii = cspice.bodvcd(spice_body_id, "RADII")

    return oops.surface.Spheroid(origin_id, frame_id, (radius, radius))

def _define_body_spherical(spice_id, parent, barycenter, radius):
    """Define the path, frame, surface for bodies by name or SPICE ID.

    All must share a common parent and barycenter."""

    # Define the body's path
    path = oops.SpicePath(spice_id, "SSB")

    # The name of the path is the name of the body
    name = path.path_id

    name_spherical = name + '_SPHERICAL'
    
    # If the body already exists, skip it
    if name_spherical in oops.Body.BODY_REGISTRY: return

    # Sometimes a frame is undefined for a new body; in this case any frame
    # will do.
    frame = oops.SpiceFrame(spice_id)

    # Define the planet's body
    # Note that this will overwrite any registered body of the same name
    body = oops.Body(name_spherical, name, frame.frame_id, parent, barycenter,
                     spice_name=name)

    # Add the gravity object if it exists
    try:
        body.apply_gravity(gravity.LOOKUP[name])
    except KeyError: pass

    # Add the surface object if shape information is available
    shape = _spice_body_spherical(spice_id, radius)
    body.apply_surface(shape, shape.req, shape.rpol)

    # Add a planet name to any satellite or barycenter
    if "SATELLITE" in body.keywords and parent is not None:
        body.add_keywords(parent)

    if "BARYCENTER" in body.keywords and parent is not None:
        body.add_keywords(parent)

# _define_body_spherical('MIMAS', 'SATURN', 'SATURN', 198.2)
# _define_body_spherical('ENCELADUS', 'SATURN', 'SATURN', 252.1)
# _define_body_spherical('TETHYS', 'SATURN', 'SATURN', 531.1)
# _define_body_spherical('DIONE', 'SATURN', 'SATURN', 561.4)
# _define_body_spherical('RHEA', 'SATURN', 'SATURN', 763.8)
# _define_body_spherical('IAPETUS', 'SATURN', 'SATURN BARYCENTER', 734.5)
# _define_body_spherical('PHOEBE', 'SATURN', 'SATURN BARYCENTER', 106.5)


#===============================================================================
# 
#===============================================================================

def _bodies_read_iss_map(body_name):
#        minimum_latitude = good_row['MINIMUM_LATITUDE']
#        maximum_latitude = good_row['MAXIMUM_LATITUDE']
#        westernmost_longitude = good_row['WESTERNMOST_LONGITUDE']
#        easternmost_longitude = good_row['EASTERNMOST_LONGITUDE']
#        map_projection_rotation = good_row['MAP_PROJECTION_ROTATION']
#        map_scale = good_row['MAP_SCALE']
    body_data = {}
    iss_dir = CARTOGRAPHIC_BODIES[body_name]
    img_index_filename = os.path.join(COISS_3XXX_ROOT, iss_dir, 'index',
                                      'img_index.lbl')
    img_index_table = PdsTable(img_index_filename)
    img_index_rows = img_index_table.dicts_by_row()
    good_row = None
    for img_index_row in img_index_rows:
        if img_index_row['MAP_PROJECTION_TYPE'] == 'SIMPLE CYLINDRICAL':
            assert good_row is None
            good_row = img_index_row
    assert good_row is not None
    assert good_row['TARGET_NAME'] == body_name
    assert good_row['COORDINATE_SYSTEM_NAME'] == 'PLANETOGRAPHIC'
    assert good_row['COORDINATE_SYSTEM_TYPE'] == 'BODY-FIXED ROTATING'
    
    body_data['center_longitude'] = good_row['CENTER_LONGITUDE']
    body_data['center_latitude'] = good_row['CENTER_LATITUDE']
    pos_long_direction = good_row['POSITIVE_LONGITUDE_DIRECTION'].lower()
    # XXX POSITIVE_LONGITUDE_DIRECTION should be a string, but it's marked
    # XXX as a float. This is bogus and has been reported.
    body_data['pos_long_direction'] = pos_long_direction[-4:]
    body_data['line_first_pixel'] = good_row['LINE_FIRST_PIXEL']-1
    assert body_data['line_first_pixel'] == 0 # Comp below would be wrong
    body_data['line_last_pixel'] = good_row['LINE_LAST_PIXEL']-1
    body_data['line_proj_offset'] = good_row['LINE_PROJECTION_OFFSET']
    body_data['sample_first_pixel'] = good_row['SAMPLE_FIRST_PIXEL']-1
    assert body_data['sample_first_pixel'] == 0 # Comp below would be wrong
    body_data['sample_last_pixel'] = good_row['SAMPLE_LAST_PIXEL']-1
    body_data['sample_proj_offset'] = good_row['SAMPLE_PROJECTION_OFFSET']
    body_data['map_resolution'] = good_row['MAP_RESOLUTION']
    map_filename = os.path.join(COISS_3XXX_ROOT, iss_dir,
                                good_row['FILE_SPECIFICATION_NAME'])
    map_data = np.fromfile(map_filename, dtype='uint8')
    body_data['nline'] = (body_data['line_last_pixel'] -
                          body_data['line_first_pixel'] + 1)
    body_data['nsamp'] = (body_data['sample_last_pixel'] -
                          body_data['sample_first_pixel'] + 1)
    nline = body_data['nline']
    nsamp = body_data['nsamp']
    read_nline = len(map_data) // nsamp
    map_data = map_data.reshape((read_nline, nsamp))
    body_data['map_data'] = map_data[read_nline-nline:,:]
    
    return body_data

def _bodies_read_schenk_jpg(body_name):
    body_data = {}
    img_filename = os.path.join(CB_SUPPORT_FILES_ROOT, body_name+'_MAP.jpg')
    img = Image.open(img_filename)
    nx, ny = img.size
    body_data['line_first_pixel'] = 0
    body_data['line_last_pixel'] = ny-1
    body_data['sample_first_pixel'] = 0
    body_data['sample_last_pixel'] = nx-1
    body_data['center_latitude'] = 0.
    body_data['center_longitude'] = 177.
    body_data['pos_long_direction'] = 'west'
    body_data['line_proj_offset'] = ny / 2.
    body_data['sample_proj_offset'] = nx / 2.
    body_data['map_resolution'] = nx / 360.
    body_data['nline'] = (body_data['line_last_pixel'] -
                          body_data['line_first_pixel'] + 1)
    body_data['nsamp'] = (body_data['sample_last_pixel'] -
                          body_data['sample_first_pixel'] + 1)
    data = np.asarray(img)
    data = data[:,:,1] # Green channel
    body_data['map_data'] = data

    return body_data
    
#def _bodies_create_cartographic(bp, body_name, force_spherical=True,
#                               source='schenk_jpg'):
#    if force_spherical:
#        body_name_spherical = body_name + '_SPHERICAL'
#    else:
#        body_name_spherical = body_name
#    
#    if (body_name_spherical, source) in CARTOGRAPHIC_FILE_CACHE:
#        body_data = CARTOGRAPHIC_FILE_CACHE[(body_name_spherical, source)]
#    else:
#        if source == 'iss':
#            body_data = _bodies_read_iss_map(body_name)
#        elif source == 'schenk_jpg':
#            body_data = _bodies_read_schenk_jpg(body_name)
#        CARTOGRAPHIC_FILE_CACHE[(body_name_spherical, source)] = body_data
#
#    nline = body_data['nline']
#    nsamp = body_data['nsamp']
#    map_data = body_data['map_data']
#    pos_long_direction = body_data['pos_long_direction']
#    line_proj_offset = body_data['line_proj_offset']
#    sample_proj_offset = body_data['sample_proj_offset']
#    center_longitude = body_data['center_longitude']
#    center_latitude = body_data['center_latitude']
#    map_resolution = body_data['map_resolution']
#    line_first_pixel = body_data['line_first_pixel']
#    line_last_pixel = body_data['line_last_pixel']
#    sample_first_pixel = body_data['sample_first_pixel']
#    sample_last_pixel = body_data['sample_last_pixel']
#    
#    latitude = bp.latitude(body_name_spherical,
#                           lat_type='centric').vals.astype('float')
#    longitude = bp.longitude(body_name_spherical,
#                             direction=pos_long_direction)
#    longitude = longitude.vals.astype('float')
#
#    print 'LAT RANGE', np.min(latitude), np.max(latitude)
#    print 'LONG RANGE', np.min(longitude), np.max(longitude)
#    
#    print 'LINE OFF', line_proj_offset, 'CTR LAT', center_latitude,
#    print 'MAP RES', map_resolution
#    line = (line_proj_offset -
#            (latitude - center_latitude) * map_resolution)
#
#    print 'SAMP OFF', sample_proj_offset, 'CTR LONG', center_longitude,
#    print 'MAP RES', map_resolution
#    sample = ((sample_proj_offset -
#               (longitude - center_longitude) * map_resolution) % nsamp)
#
#    line = line.astype('int')
#    sample = sample.astype('int')
#    
#    line_mask = np.logical_or(line < line_first_pixel,
#                              line > line_last_pixel)
#    sample_mask = np.logical_or(sample < sample_first_pixel,
#                                sample > sample_last_pixel)
#    mask = np.logical_or(line_mask, sample_mask)
#
#    line[mask] = 0
#    sample[mask] = 0
#    
#    model = map_data[line, sample] 
#    model[mask] = 0
#    
#    return model
