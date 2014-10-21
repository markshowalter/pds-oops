import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp

import os

import oops
import cspice

from pdstable import PdsTable
from tabulation import Tabulation
from cb_config import *
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

RING_OCCULTATION_TABLE = PdsTable(os.path.join(SUPPORT_FILES_ROOT,
                                               'IS2_P0001_V01_KM002.LBL'))
RING_OCCULTATION_DATA = Tabulation(RING_OCCULTATION_TABLE.column_dict['RING_INTERCEPT_RADIUS'],
                                   RING_OCCULTATION_TABLE.column_dict['I_OVER_F'])

RINGS_MIN_RADIUS = oops.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.SATURN_MAIN_RINGS[1]

RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.02
RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION = 5.
RINGS_DEFAULT_REPRO_ZOOM = 5

FRING_DEFAULT_REPRO_RADIUS_INNER = 139500. - 140220.
FRING_DEFAULT_REPRO_RADIUS_OUTER = 141000. - 140220.

#===============================================================================
# 
#===============================================================================

def rings_create_model(obs, extend_fov=(0,0)):
    logger = logging.getLogger(LOGGING_NAME+'.rings_create_model')

    set_obs_ext_bp(obs, extend_fov)
    
    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    logger.debug('Radii %.2f to %.2f', min_radius, max_radius)
    
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.debug('No main rings in image - returning null model')
        return None 

    radii[radii < RINGS_MIN_RADIUS] = 0
    radii[radii > RINGS_MAX_RADIUS] = 0
    
    model = RING_OCCULTATION_DATA(radii)

    saturn_shadow = obs.ext_bp.where_inside_shadow('saturn:ring', 'saturn').vals
    model[saturn_shadow] = 0
    
    if not np.any(model):
        logger.debug('Rings are entirely shadowed - returning null model')
        return None
    
    return model

"""
- Profiles at different lighting geometries, including phase, incidence, emission,
and lit/unlit.
- Occultation profiles for transparency.
- When combining models with the rings in front, use the occultation for transparency.
"""

#===============================================================================
# 
#===============================================================================

#===============================================================================
# 
#===============================================================================


def rings_restrict_longitude_radius_to_obs(obs, longitude, radius):
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    set_obs_bp(obs)
    bp = obs.bp

    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
    
    min_bp_radius = np.min(bp_radius)
    max_bp_radius = np.max(bp_radius)
    min_bp_longitude = np.min(bp_longitude)
    max_bp_longitude = np.max(bp_longitude)
    
    goodr = np.logical_and(radius >= min_bp_radius, radius <= max_bp_radius)
    goodl = np.logical_and(longitude >= min_bp_longitude, longitude <= max_bp_longitude)
    good = np.logical_and(goodr, goodl)
    
    radius = radius[good]
    longitude = longitude[good]
    
    return longitude, radius
    
#===========================================================================
# 
#===========================================================================

FRING_ROTATING_ET = cspice.utc2et("2007-1-1")
FRING_MEAN_MOTION = 581.964 # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 24.2             # deg
FRING_DW = 2.70025          # deg/day                

def _compute_longitude_shift(et): 
    return - (FRING_MEAN_MOTION * ((et - FRING_ROTATING_ET) / 86400.)) % 360.

def rings_fring_inertial_to_corotating(longitude, et):
    return (longitude + _compute_longitude_shift(et)) % 360.

def rings_fring_corotating_to_inertial(co_long, et):
    return (co_long - _compute_longitude_shift(et)) % 360.

def rings_longitude_radius_to_pixels(obs, longitude, radius, corotating=None):
    assert corotating in (None, 'F')
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    if corotating == 'F':
        longitude = rings_fring_corotating_to_inertial(longitude, obs.midtime)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    ring_surface = oops.Body.lookup('SATURN_RING_PLANE').surface
    obs_event = oops.Event(obs.midtime, (Vector3.ZERO,Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = ring_surface.photon_to_event_by_coords(obs_event, (radius, longitude*oops.RPD))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals

def rings_generate_longitudes(start_num=0,
                              end_num=None,
                              longitude_resolution=
                                    RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    if end_num is None:
        end_num = int(360. / longitude_resolution)-1
    return np.arange(start_num, end_num+1) * longitude_resolution

def rings_generate_radii(radius_inner, radius_outer,
                         radius_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
                         start_num=0, end_num=None):
    if end_num is None:
        end_num = int(np.ceil(((radius_outer-radius_inner+1) / radius_resolution)))-1
    return np.arange(start_num, end_num+1) * radius_resolution + radius_inner

    radius_pixels = int(np.ceil((radius_outer-radius_inner+1) / radius_resolution))

def rings_fring_radius_at_longitude(obs, longitude):        
    curly_w = FRING_W0 + FRING_DW*obs.midtime/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos((longitude-curly_w)*oops.RPD)))

    return radius
    
def rings_fring_longitude_radius(obs, longitude_step=0.01, corotating=False):
    num_longitudes = int(360. / longitude_step)
    longitudes = np.arange(num_longitudes) * longitude_step
    radius = rings_fring_radius_at_longitude(obs, longitudes)
    
    return longitudes, radius

def rings_fring_pixels(obs, offset_u=0, offset_v=0):
    longitude, radius = rings_fring_longitude_radius(obs)
    longitude, radius = rings_restrict_longitude_radius_to_obs(obs, longitude, radius)
    
    u_pixels, v_pixels = rings_longitude_radius_to_pixels(obs,
                                          longitude, radius)
    
    u_pixels += offset_u
    v_pixels += offset_v
    
    # This really shouldn't catch much since we already restricted
    # the longitude and radius
    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    good = np.logical_and(goodumask, goodvmask)
    
    u_pixels = u_pixels[good]
    v_pixels = v_pixels[good]
    
    return u_pixels, v_pixels

def rings_reproject(obs, offset_u=0, offset_v=0,
            longitude_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            radius_resolution=RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
            radius_inner=None,
            radius_outer=None,
            zoom=RINGS_DEFAULT_REPRO_ZOOM,
            fring=False):

    logger = logging.getLogger(LOGGING_NAME+'.rings_reproject')
    
    # We need to be careful not to use obs.bp from this point forward because
    # it will disagree with our current OffsetFOV
    orig_fov = obs.fov
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(offset_u, offset_v))
    
    # Get all the info for each pixel
    bp = oops.Backplane(obs)
    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = (bp.ring_longitude('saturn:ring').vals.astype('float') * 
                    oops.DPR)
    bp_resolution = (bp.ring_radial_resolution('saturn:ring')
                     .vals.astype('float'))
    bp_phase = bp.phase_angle('saturn:ring').vals.astype('float') * oops.DPR
    bp_emission = (bp.emission_angle('saturn:ring').vals.astype('float') * 
                   oops.DPR)
    bp_incidence = (bp.incidence_angle('saturn:ring').vals.astype('float') * 
                    oops.DPR)
    
    # The number of pixels in the final reprojection
    radius_pixels = int(np.ceil((radius_outer-radius_inner+1) / radius_resolution))
    longitude_pixels = int(360. / longitude_resolution)

    if fring:
        # Convert longitude to co-rotating
        bp_longitude = rings_fring_inertial_to_corotating(bp_longitude, 
                                                          obs.midtime)
    
    # Restrict the longitude range for some attempt at efficiency
    min_longitude_pixel = (np.floor(np.min(bp_longitude) / 
                                    longitude_resolution)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, longitude_pixels-1)
    max_longitude_pixel = (np.ceil(np.max(bp_longitude) / 
                                   longitude_resolution)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, longitude_pixels-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    # Longitude bin numbers
    long_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1), 
                        radius_pixels)
    # Actual longitude (deg)
    long_bins_act = long_bins * longitude_resolution

    # Radius bin numbers
    rad_bins = np.repeat(np.arange(radius_pixels), num_longitude_pixel)
    # Actual radius (km)
    if fring:
        rad_bins_offset = rings_fring_radius_at_longitude(obs,
                              rings_fring_corotating_to_inertial(long_bins_act, obs.midtime))        
        rad_bins_act = rad_bins * radius_resolution + radius_inner + rad_bins_offset
        logger.debug('Radius offset range %8.2f %8.2f', np.min(rad_bins_offset),
                     np.max(rad_bins_offset))
    else:
        rad_bins_act = rad_bins * radius_resolution + radius_inner

    logger.debug('Radius range %8.2f %8.2f', np.min(bp_radius), 
                 np.max(bp_radius))
    logger.debug('Radius bin range %8.2f %8.2f', np.min(rad_bins_act), 
                 np.max(rad_bins_act))
    logger.debug('Longitude range %6.2f %6.2f', np.min(bp_longitude), 
                 np.max(bp_longitude))
    logger.debug('Longitude bin range %6.2f %6.2f', np.min(long_bins_act),
                 np.max(long_bins_act))
    logger.debug('Resolution range %7.2f %7.2f', np.min(bp_resolution),
                 np.max(bp_resolution))
    logger.debug('Data range %f %f', np.min(obs.data), np.max(obs.data))

    if fring:
        corotating = 'F'
    else:
        corotating = None

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(obs, long_bins_act,
                                                          rad_bins_act,
                                                          corotating=corotating)
        
    zoom_data = ndinterp.zoom(obs.data, zoom) # XXX Default to order=3 - OK?

    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    good_u = u_pixels[goodmask]
    good_v = v_pixels[goodmask]
    
    good_rad = rad_bins[goodmask]
    good_long = long_bins[goodmask]
    
    interp_data = zoom_data[(good_v*zoom).astype('int'), (good_u*zoom).astype('int')]
    
    repro_img = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_img.mask = True
    repro_img[good_rad,good_long] = interp_data

    good_u = good_u.astype('int')
    good_v = good_v.astype('int')
    
    repro_res = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_res.mask = True
    repro_res[good_rad,good_long] = bp_resolution[good_v,good_u]
    repro_mean_res = ma.mean(repro_res, axis=0)
    # Mean will mask if ALL radii are masked are a particular longitude

    # All interpolated data should be masked the same, so we might as well
    # take one we've already computed.
    good_long_mask = np.logical_not(ma.getmaskarray(repro_mean_res))

    repro_phase = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_phase.mask = True
    repro_phase[good_rad,good_long] = bp_phase[good_v,good_u]
    repro_mean_phase = ma.mean(repro_phase, axis=0)

    repro_emission = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_emission.mask = True
    repro_emission[good_rad,good_long] = bp_emission[good_v,good_u]
    repro_mean_emission = ma.mean(repro_emission, axis=0)

    repro_incidence = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_incidence.mask = True
    repro_incidence[good_rad,good_long] = bp_incidence[good_v,good_u]
    repro_mean_incidence = ma.mean(repro_incidence) # scalar

    repro_img = ma.filled(repro_img[:,good_long_mask], 0.)
    repro_res = ma.filled(repro_res[:,good_long_mask], 0.)
    repro_phase = ma.filled(repro_phase[:,good_long_mask], 0.)
    repro_emission = ma.filled(repro_emission[:,good_long_mask], 0.)
    repro_incidence = ma.filled(repro_incidence[:,good_long_mask], 0.)
    
    repro_mean_res = repro_mean_res[good_long_mask]
    repro_mean_phase = repro_mean_phase[good_long_mask]
    repro_mean_emission = repro_mean_emission[good_long_mask]

    assert ma.count_masked(repro_mean_res) == 0
    assert ma.count_masked(repro_mean_phase) == 0
    assert ma.count_masked(repro_mean_emission) == 0
    
    obs.fov = orig_fov
    obs.bp = None

    ret = {}
    
    ret['long_mask'] = good_long_mask
    ret['img'] = repro_img
    ret['resolution'] = repro_res
    ret['phase'] = repro_phase
    ret['emission'] = repro_emission
    ret['incidence'] = repro_incidence
    ret['mean_resolution'] = repro_mean_res
    ret['mean_phase'] = repro_mean_phase
    ret['mean_emission'] = repro_mean_emission
    ret['mean_incidence'] = repro_mean_incidence
    ret['time'] = obs.midtime
    
    return ret

def rings_fring_mosaic_init(longitude_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
                            radius_resolution=RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
                            radius_inner=None,
                            radius_outer=None):

    radius_pixels = int((radius_outer-radius_inner) / radius_resolution)
    longitude_pixels = int(360. / longitude_resolution)
    
    ret = {}
    ret['img'] = np.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    ret['long_mask'] = np.zeros(longitude_pixels, dtype=np.bool)
    ret['mean_resolution'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_phase'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_emission'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_incidence'] = 0.
    ret['image_number'] = np.zeros(longitude_pixels, dtype=np.int32)
    ret['time'] = np.zeros(longitude_pixels, dtype=np.float32)
    
    return ret

def rings_fring_mosaic_add(mosaic_metadata, repro_metadata, image_number):
    mosaic_metadata['mean_incidence'] = repro_metadata['mean_incidence']
    
    radius_pixels = mosaic_metadata['img'].shape[0]
    mosaic_good_long = mosaic_metadata['long_mask']
    
    repro_good_long = repro_metadata['long_mask']
    mosaic_good_long = mosaic_metadata['long_mask']
    
    mosaic_img = mosaic_metadata['img']
    repro_img = np.zeros(mosaic_img.shape) 
    repro_img[:,repro_good_long] = repro_metadata['img']
    mosaic_res = mosaic_metadata['mean_resolution']
    repro_res = np.zeros(mosaic_res.shape) + 1e300
    repro_res[repro_good_long] = repro_metadata['mean_resolution']
    mosaic_phase = mosaic_metadata['mean_phase']
    repro_phase = np.zeros(mosaic_phase.shape)
    repro_phase[repro_good_long] = repro_metadata['mean_phase']
    mosaic_emission = mosaic_metadata['mean_emission']
    repro_emission = np.zeros(mosaic_emission.shape)
    repro_emission[repro_good_long] = repro_metadata['mean_emission']
    mosaic_image_number = mosaic_metadata['image_number']
    mosaic_time = mosaic_metadata['time']
    
    # Calculate number of good entries and where number is larger than before
    mosaic_valid_radius_count = radius_pixels-np.sum(mosaic_img == 0., axis=0)
    new_valid_radius_count = radius_pixels-np.sum(repro_img == 0., axis=0)
    valid_radius_count_better_mask = new_valid_radius_count > mosaic_valid_radius_count
    valid_radius_count_equal_mask = new_valid_radius_count == mosaic_valid_radius_count
    # Calculate where the new resolution is better
    better_resolution_mask = repro_res < mosaic_res
    # Final mask for which columns to replace mosaic values
    good_longitude_mask = np.logical_or(valid_radius_count_better_mask,
                np.logical_and(valid_radius_count_equal_mask, better_resolution_mask))

    mosaic_good_long[:] = np.logical_or(mosaic_good_long, good_longitude_mask)
    mosaic_img[:,good_longitude_mask] = repro_img[:,good_longitude_mask]
    mosaic_res[good_longitude_mask] = repro_res[good_longitude_mask] 
    mosaic_phase[good_longitude_mask] = repro_phase[good_longitude_mask] 
    mosaic_emission[good_longitude_mask] = repro_emission[good_longitude_mask] 
    mosaic_image_number[good_longitude_mask] = image_number 
     