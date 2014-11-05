###############################################################################
# cb_rings.py
#
# Routines related to rings.
#
# Exported routines:
#    rings_create_model
#
#    rings_fring_inertial_to_corotating
#    rings_fring_corotating_to_inertial
#    rings_fring_radius_at_longitude
#    rings_fring_longitude_radius
#    rings_fring_pixels
#
#    rings_longitude_radius_to_pixels
#    rings_generate_longitudes
#    rings_generate_radii
#
#    rings_reproject
#    rings_mosaic_init
#    rings_mosaic_add
###############################################################################

import cb_logging
import logging

import os

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp

import polymath
import oops
import cspice
from pdstable import PdsTable
from tabulation import Tabulation

from cb_config import *
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

RING_VOYAGER_IF_TABLE = PdsTable(os.path.join(SUPPORT_FILES_ROOT,
                                              'IS2_P0001_V01_KM002.LBL'))
RING_VOYAGER_IF_DATA = Tabulation(
       RING_VOYAGER_IF_TABLE.column_dict['RING_INTERCEPT_RADIUS'],
       RING_VOYAGER_IF_TABLE.column_dict['I_OVER_F'])

RINGS_MIN_RADIUS = oops.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.SATURN_MAIN_RINGS[1]

RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.02
RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION = 5.
RINGS_DEFAULT_REPRO_ZOOM = 5

FRING_DEFAULT_REPRO_RADIUS_INNER = 139500. - 140220.
FRING_DEFAULT_REPRO_RADIUS_OUTER = 141000. - 140220.

#==============================================================================
# 
# RING MODELS
#
#==============================================================================

def rings_create_model(obs, extend_fov=(0,0)):
    """Create a model for the rings.
    
    If there are no rings in the image or they are entirely in shadow,
    return None.
    
    The rings model is created by interpolating from the Voyager I/F
    profile. Portions in Saturn's shadow are removed.
    """
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
    
    model = RING_VOYAGER_IF_DATA(radii)

    saturn_shadow = obs.ext_bp.where_inside_shadow('saturn:ring',
                                                   'saturn').vals
    model[saturn_shadow] = 0
    
    if not np.any(model):
        logger.debug('Rings are entirely shadowed - returning null model')
        return None
    
    return model


#==============================================================================
# 
# RING REPROJECTION UTILITIES
#
#==============================================================================

# F ring constants
FRING_ROTATING_ET = cspice.utc2et("2007-1-1")
FRING_MEAN_MOTION = 581.964 # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 24.2 # deg
FRING_DW = 2.70025 # deg/day                

##
# F ring routines
##

def _compute_fring_longitude_shift(et): 
    return - (FRING_MEAN_MOTION * ((et - FRING_ROTATING_ET) / 86400.)) % 360.

def rings_fring_inertial_to_corotating(longitude, et):
    """Convert inertial longitude (deg) to corotating."""
    return (longitude + _compute_fring_longitude_shift(et)) % 360.

def rings_fring_corotating_to_inertial(co_long, et):
    """Convert corotating longitude (deg) to inertial."""
    return (co_long - _compute_fring_longitude_shift(et)) % 360.

def rings_fring_radius_at_longitude(obs, longitude):
    """Return the radius (km) of the F ring core at a given inertial longitude
    (deg)."""
    curly_w = FRING_W0 + FRING_DW*obs.midtime/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos((longitude-curly_w)*oops.RPD)))

    return radius
    
def rings_fring_longitude_radius(obs, longitude_step=0.01):
    """Return  a set of longitude (deg),radius (km) pairs for the F ring
    core."""
    num_longitudes = int(360. / longitude_step)
    longitudes = np.arange(num_longitudes) * longitude_step
    radius = rings_fring_radius_at_longitude(obs, longitudes)
    
    return longitudes, radius

def rings_fring_pixels(obs, offset_u=0, offset_v=0, longitude_step=0.01):
    """Return a set of U,V pairs for the F ring in an image."""
    longitude, radius = rings_fring_longitude_radius(
                                     obs,
                                     longitude_step=longitude_step)
    longitude, radius = _rings_restrict_longitude_radius_to_obs(
                                     obs, longitude, radius)
    
    u_pixels, v_pixels = rings_longitude_radius_to_pixels(obs,
                                          longitude, radius)
    
    u_pixels += offset_u
    v_pixels += offset_v
    
    # Catch the cases that fell outside the image boundaries
    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    good = np.logical_and(goodumask, goodvmask)
    
    u_pixels = u_pixels[good]
    v_pixels = v_pixels[good]
    
    return u_pixels, v_pixels

##
# Non-F-ring-specific routines
##

def _rings_restrict_longitude_radius_to_obs(obs, longitude, radius):
    """Restrict the list of longitude (deg) and radius (km) to those present
    in the image.
    
    This only does an approximate job. It is still possible that
    longitude,radius pairs will be left that are outside the image bounds."""
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    set_obs_bp(obs)

    bp_radius = obs.bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = obs.bp.ring_longitude('saturn:ring').vals.astype('float')
    bp_longitude *= oops.DPR
    
    min_bp_radius = np.min(bp_radius)
    max_bp_radius = np.max(bp_radius)
    min_bp_longitude = np.min(bp_longitude)
    max_bp_longitude = np.max(bp_longitude)
    
    goodr = np.logical_and(radius >= min_bp_radius, radius <= max_bp_radius)
    goodl = np.logical_and(longitude >= min_bp_longitude,
                           longitude <= max_bp_longitude)
    good = np.logical_and(goodr, goodl)
    
    radius = radius[good]
    longitude = longitude[good]
    
    return longitude, radius
    
def rings_longitude_radius_to_pixels(obs, longitude, radius, corotating=None):
    """Convert longitude (deg),radius (km) pairs to U,V."""
    assert corotating in (None, 'F')
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    if corotating == 'F':
        longitude = rings_fring_corotating_to_inertial(longitude, obs.midtime)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    ring_surface = oops.Body.lookup('SATURN_RING_PLANE').surface
    obs_event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                         polymath.Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = ring_surface.photon_to_event_by_coords(obs_event,
                                                          (radius,
                                                           longitude*oops.RPD))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals

def rings_generate_longitudes(start_num=0,
                              end_num=None,
                              longitude_resolution=
                                    RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Generate a list of longitudes (deg)."""
    if end_num is None:
        end_num = int(360. / longitude_resolution)-1
    return np.arange(start_num, end_num+1) * longitude_resolution

def rings_generate_radii(radius_inner, radius_outer,
                         radius_resolution=
                             RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
                         start_num=0, end_num=None):
    """Generate a list of radii (km)."""
    if end_num is None:
        end_num = int(np.ceil(((radius_outer-radius_inner+1) /
                               radius_resolution)))-1
    return np.arange(start_num, end_num+1) * radius_resolution + radius_inner


#==============================================================================
# 
# RING REPROJECTION MAIN ROUTINES
#
#==============================================================================

def rings_reproject(
            obs, offset_u=0, offset_v=0,
            longitude_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            radius_resolution=RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
            radius_inner=None,
            radius_outer=None,
            zoom=RINGS_DEFAULT_REPRO_ZOOM,
            corotating=None):
    """Reproject the rings in an image into a rectangular longitude/radius
    space.
    
    Inputs:
        obs                      The Observation.
        offset_u                 The offsets in U,V to apply to the image
        offset_v                 when computing the longitude and radius
                                 values.
        longitude_resolution     The longitude resolution of the new image
                                 (deg/pix).
        radius_resolution        The radius resolution of the new image
                                 (km/pix).
        radius_inner             The radius closest to Saturn to reproject
                                 (km).
        radius_outer             The radius furthest from Saturn to reproject
                                 (km).
        zoom                     The amount to magnify the original image for
                                 pixel value interpolation.
        corotating               The name of the ring to use to compute
                                 co-rotating longitude. None if inertial
                                 longitude should be used.
                                 
    Returns:
        A dictionary containing
        
        'long_mask'        The mask of longitudes from the full 360-degree
                           set that contain reprojected data. This can be
                           used to recreate the list of actual longitudes
                           present.
        'time'             The midtime of the observation (TDB).
                           
            The following only contain longitudes with mask values of True
            above. All angles are in degrees.
        'img'              The reprojected image [radius,longitude].
        'resolution'       The radial resolution [radius,longitude].
        'phase'            The phase angle [radius,longitude].
        'emission'         The emission angle [radius,longitude].
        'incidence'        The incidence angle [radius,longitude].
        'mean_resolution'  The radial resolution averaged over all radii
                           [longitude].
        'mean_phase'       The phase angle averaged over all radii [longitude].
        'mean_emission'    The emission angle averaged over all radii
                           [longitude].
        'mean_incidence'   The incidence angle averaged over all radii AND
                           longitudes (it shouldn't change over the ring 
                           plane).
    """
    logger = logging.getLogger(LOGGING_NAME+'.rings_reproject')
    
    assert corotating in (None, 'F')
    
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
    radius_pixels = int(np.ceil((radius_outer-radius_inner+1) / 
                                radius_resolution))
    longitude_pixels = int(360. / longitude_resolution)

    if corotating == 'F':
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
    # Actual longitude for each bin (deg)
    long_bins_act = long_bins * longitude_resolution

    # Radius bin numbers
    rad_bins = np.repeat(np.arange(radius_pixels), num_longitude_pixel)
    # Actual radius for each bin (km)
    if corotating == 'F':
        rad_bins_offset = rings_fring_radius_at_longitude(obs,
                              rings_fring_corotating_to_inertial(long_bins_act,
                                                                 obs.midtime))        
        rad_bins_act = (rad_bins * radius_resolution + radius_inner +
                        rad_bins_offset)
        logger.debug('Radius offset range %8.2f %8.2f',
                     np.min(rad_bins_offset),
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

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                                  obs, long_bins_act,
                                                  rad_bins_act,
                                                  corotating=corotating)

    # Zoom the data and restrict the bins and pixels to ones actually in the
    # final reprojection.
    zoom_data = ndinterp.zoom(obs.data, zoom) # XXX Default to order=3 - OK?

    u_zoom = (u_pixels*zoom).astype('int')
    v_zoom = (v_pixels*zoom).astype('int')
    
    goodumask = np.logical_and(u_pixels >= 0, 
                               u_zoom <= obs.data.shape[1]*zoom-1)
    goodvmask = np.logical_and(v_pixels >= 0, 
                               v_zoom <= obs.data.shape[0]*zoom-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    u_pixels = u_pixels[goodmask].astype('int')
    v_pixels = v_pixels[goodmask].astype('int')
    u_zoom = u_zoom[goodmask]
    v_zoom = v_zoom[goodmask]
    good_rad = rad_bins[goodmask]
    good_long = long_bins[goodmask]
    
    interp_data = zoom_data[v_zoom, u_zoom]
    
    # Create the reprojected results.
    repro_img = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_img.mask = True
    repro_img[good_rad,good_long] = interp_data

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

    repro_emission = ma.zeros((radius_pixels, longitude_pixels), 
                              dtype=np.float32)
    repro_emission.mask = True
    repro_emission[good_rad,good_long] = bp_emission[good_v,good_u]
    repro_mean_emission = ma.mean(repro_emission, axis=0)

    repro_incidence = ma.zeros((radius_pixels, longitude_pixels), 
                               dtype=np.float32)
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

def rings_mosaic_init(
        longitude_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
        radius_resolution=RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
        radius_inner=None, radius_outer=None):
    """Create the data structure for a ring mosaic.

    Inputs:
        longitude_resolution     The longitude resolution of the new image
                                 (deg/pix).
        radius_resolution        The radius resolution of the new image
                                 (km/pix).
        radius_inner             The radius closest to Saturn to reproject
                                 (km).
        radius_outer             The radius furthest from Saturn to reproject
                                 (km).
                                 
    Returns:
        A dictionary containing an empty mosaic

        'img'              The full mosaic image.
        'long_mask'        The valid-longitude mask (all False).
        'mean_resolution'  The per-longitude mean resolution.
        'mean_phase'       The per-longitude mean phase angle.
        'mean_emission'    The per-longitude mean emission angle.
        'mean_incidence'   The scalar mean incidence angle.
        'image_number'     The per-longitude image number giving the image
                           used to fill the data for each longitude.
        'time'             The per-longitude time (TDB).
    """
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

def rings_mosaic_add(mosaic_metadata, repro_metadata, image_number):
    """Add a reprojected image to an existing mosaic.
    
    For each valid longitude in the reprojected image, it is copied to the
    mosaic if it has more valid radial data, or the same amount of radial
    data but the resolution is better.
    """
    mosaic_metadata['mean_incidence'] = repro_metadata['mean_incidence']
    
    radius_pixels = mosaic_metadata['img'].shape[0]
    repro_good_long = repro_metadata['long_mask']
    mosaic_good_long = mosaic_metadata['long_mask']
        
    # Create full-size versions of all the longitude-compressed reprojection
    # data.
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
    
    # Calculate the number of good entries and where the number is larger than
    # in the existing mosaic.
    mosaic_valid_radius_count = radius_pixels-np.sum(mosaic_img == 0., axis=0)
    new_valid_radius_count = radius_pixels-np.sum(repro_img == 0., axis=0)
    valid_radius_count_better_mask = (new_valid_radius_count > 
                                      mosaic_valid_radius_count)
    valid_radius_count_equal_mask = (new_valid_radius_count == 
                                     mosaic_valid_radius_count)
    
    # Calculate where the new resolution is better
    better_resolution_mask = repro_res < mosaic_res
    
    # Make the final mask for which columns to replace mosaic values in.
    good_longitude_mask = np.logical_or(valid_radius_count_better_mask,
        np.logical_and(valid_radius_count_equal_mask, better_resolution_mask))

    mosaic_good_long[:] = np.logical_or(mosaic_good_long, good_longitude_mask)
    mosaic_img[:,good_longitude_mask] = repro_img[:,good_longitude_mask]
    mosaic_res[good_longitude_mask] = repro_res[good_longitude_mask] 
    mosaic_phase[good_longitude_mask] = repro_phase[good_longitude_mask] 
    mosaic_emission[good_longitude_mask] = repro_emission[good_longitude_mask] 
    mosaic_image_number[good_longitude_mask] = image_number 
