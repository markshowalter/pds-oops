import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.interpolate as interp

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

DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.02
DEFAULT_REPRO_RADIUS_INNER = 139500.
DEFAULT_REPRO_RADIUS_OUTER = 141000.
DEFAULT_REPRO_RADIUS_RESOLUTION = 5.

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
FRING_MEAN_MOTION = 581.964

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
    
    set_obs_bp(obs)
    bp = obs.bp
    data = obs.data
    
    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
    if corotating == 'F':
        bp_longitude = rings_fring_inertial_to_corotating(bp_longitude, obs.midtime)
        
    flat_radius = bp_radius.flatten()
    flat_longitude = bp_longitude.flatten()

    flat_u = np.tile(np.arange(data.shape[1]), data.shape[0])
    flat_v = np.repeat(np.arange(data.shape[0]), data.shape[1])

    radlon_points = np.empty((longitude.shape[0], 2))
    radlon_points[:,0] = radius
    radlon_points[:,1] = longitude
    
    interp_u = interp.griddata((flat_radius, flat_longitude), flat_u,
                               radlon_points, fill_value=1e300)
    interp_v = interp.griddata((flat_radius, flat_longitude), flat_v,
                               radlon_points, fill_value=1e300)
    
    return interp_u, interp_v

def rings_generate_longitudes(start_num, end_num,
                              longitude_resolution=
                                    DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    return np.arange(start_num, end_num+1) * longitude_resolution
        
def rings_fring_longitude_radius(obs, longitude_step=0.01, corotating=False):
    num_longitudes = int(360. / longitude_step)
    longitudes = np.arange(num_longitudes) * longitude_step
    radius = np.empty(num_longitudes)
    radius[:] = 140220.
    
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
    goodu = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodv = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    good = np.logical_and(goodu, goodv)
    
    u_pixels = u_pixels[good]
    v_pixels = v_pixels[good]
    
    return u_pixels, v_pixels

def rings_fring_reproject(obs, offset_u=0, offset_v=0,
                  longitude_resolution=DEFAULT_REPRO_LONGITUDE_RESOLUTION,
                  radius_inner=DEFAULT_REPRO_RADIUS_INNER,
                  radius_outer=DEFAULT_REPRO_RADIUS_OUTER,
                  radius_resolution=DEFAULT_REPRO_RADIUS_RESOLUTION):

    logger = logging.getLogger(LOGGING_NAME+'.rings_fring_reproject')
    
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
    
    # Convert longitude to co-rotating
    bp_longitude = rings_fring_inertial_to_corotating(bp_longitude, 
                                                      obs.midtime)

    # The number of pixels in the final mosaic 
    radius_pixels = int((radius_outer-radius_inner) / radius_resolution)
    longitude_pixels = int(360. / longitude_resolution)

    # Restrict the longitude range for some attempt at efficiency
    min_longitude_pixel = (np.floor(np.min(bp_longitude) / 
                                    longitude_resolution)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, longitude_pixels-1)
    max_longitude_pixel = (np.ceil(np.max(bp_longitude) / 
                                   longitude_resolution)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, longitude_pixels-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    flat_radius = bp_radius.flatten()
    flat_longitude = bp_longitude.flatten()
    flat_data = obs.data.flatten()
    flat_resolution = bp_resolution.flatten() 
    flat_phase = bp_phase.flatten()
    flat_emission = bp_emission.flatten()
    flat_incidence = bp_incidence.flatten()

    # Radius bin numbers
    rad_bins = np.repeat(np.arange(radius_pixels), num_longitude_pixel)
    # Actual radius
    rad_bins_act = rad_bins * radius_resolution + radius_inner

    # Longitude bin numbers
    long_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1), 
                        radius_pixels)
    # Actual longitude
    long_bins_act = long_bins * longitude_resolution

    logger.debug('Radius range %8.2f %8.2f', np.min(flat_radius), 
                 np.max(flat_radius))
    logger.debug('Radius bin range %8.2f %8.2f', np.min(rad_bins_act), 
                 np.max(rad_bins_act))
    logger.debug('Longitude range %6.2f %6.2f', np.min(flat_longitude), 
                 np.max(flat_longitude))
    logger.debug('Longitude bin range %6.2f %6.2f', np.min(long_bins_act),
                 np.max(long_bins_act))
    logger.debug('Resolution range %7.2f %7.2f', np.min(flat_resolution),
                 np.max(flat_resolution))
    logger.debug('Data range %f %f', np.min(flat_data), np.max(flat_data))
    
    # The flat list of radius/longitude pairs covering the entire image
    radlon_points = np.empty((rad_bins.shape[0], 2))
    radlon_points[:,0] = rad_bins_act
    radlon_points[:,1] = long_bins_act

    # Interpolate data for each radius/longitude bin
    interp_data = interp.griddata((flat_radius, flat_longitude),
                                  flat_data,
                                  radlon_points, fill_value=1e300, method='nearest')
    interp_res = interp.griddata((flat_radius, flat_longitude), 
                                 flat_resolution,
                                 radlon_points, fill_value=1e300)
    interp_phase = interp.griddata((flat_radius, flat_longitude),
                                   flat_phase,
                                   radlon_points, fill_value=1e300)
    interp_emission = interp.griddata((flat_radius, flat_longitude), 
                                      flat_emission,
                                      radlon_points, fill_value=1e300)
    interp_incidence = interp.griddata((flat_radius, flat_longitude), 
                                       flat_incidence,
                                       radlon_points, fill_value=1e300)

    repro_mosaic = ma.zeros((radius_pixels, longitude_pixels))
    repro_mosaic.mask = True
    repro_mosaic[rad_bins,long_bins] = interp_data
    repro_mosaic = ma.masked_greater(repro_mosaic, 1e100)

    repro_res = ma.zeros((radius_pixels, longitude_pixels))
    repro_res.mask = True
    repro_res[rad_bins,long_bins] = interp_res
    repro_res = ma.masked_greater(repro_res, 1e100)
    repro_mean_res = ma.mean(repro_res, axis=0)
    # Mean will mask if ALL radii are masked are a particular longitude

    # All interpolated data should be masked the same, so we might as well
    # take one we've already computed.
    good_long_mask = np.logical_not(ma.getmaskarray(repro_mean_res))

    repro_phase = ma.zeros((radius_pixels, longitude_pixels))
    repro_phase[rad_bins,long_bins] = interp_phase
    repro_phase = ma.masked_greater(repro_phase, 1e100)
    repro_mean_phase = ma.mean(repro_phase, axis=0)

    repro_emission = ma.zeros((radius_pixels, longitude_pixels))
    repro_emission[rad_bins,long_bins] = interp_emission
    repro_emission = ma.masked_greater(repro_emission, 1e100)
    repro_mean_emission = ma.mean(repro_emission, axis=0)

    repro_incidence = ma.zeros((radius_pixels, longitude_pixels))
    repro_incidence[rad_bins,long_bins] = interp_incidence
    repro_incidence = ma.masked_greater(repro_incidence, 1e100)
    repro_mean_incidence = ma.mean(repro_incidence, axis=0)

    repro_mosaic = repro_mosaic[:,good_long_mask]
    repro_mean_res = repro_mean_res[good_long_mask]
    repro_mean_phase = repro_mean_phase[good_long_mask]
    repro_mean_emission = repro_mean_emission[good_long_mask]
    repro_mean_incidence = repro_mean_incidence[good_long_mask]

    full_longitudes = rings_generate_longitudes(0, longitude_pixels-1,
                                    longitude_resolution=longitude_resolution)
    full_longitudes = full_longitudes[good_long_mask]
    
    obs.fov = orig_fov
    obs.bp = None
        
    return (good_long_mask, full_longitudes, repro_mosaic, repro_mean_res,
            repro_mean_phase, repro_mean_emission, repro_mean_incidence)
