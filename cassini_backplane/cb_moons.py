###############################################################################
# cb_moons.py
#
# Routines related to moons.
#
# Exported routines:
#    XXX
#    moons_create_model
#
#    moons_generate_latitudes
#    moons_generate_longitudes
#    moons_latitude_longitude_to_pixels
#
#    moons_reproject
#    moons_mosaic_init
#    moons_mosaic_add
###############################################################################

import cb_logging
import logging

import os

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp
import matplotlib.pyplot as plt
import PIL.Image

import oops
import polymath
import gravity
import cspice
from pdstable import PdsTable

from cb_config import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__

MOONS_DEFAULT_REPRO_LATITUDE_RESOLUTION = 0.1
MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.1
MOONS_DEFAULT_REPRO_ZOOM = 2
MOONS_REPRO_MIN_LAMBERT = 0.2
MOONS_REPRO_MAX_EMISSION = 60.

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

_define_body_spherical('MIMAS', 'SATURN', 'SATURN', 198.2)
_define_body_spherical('ENCELADUS', 'SATURN', 'SATURN', 252.1)
_define_body_spherical('TETHYS', 'SATURN', 'SATURN', 531.1)
_define_body_spherical('DIONE', 'SATURN', 'SATURN', 561.4)
_define_body_spherical('RHEA', 'SATURN', 'SATURN', 763.8)
_define_body_spherical('IAPETUS', 'SATURN', 'SATURN BARYCENTER', 734.5)
_define_body_spherical('PHOEBE', 'SATURN', 'SATURN BARYCENTER', 106.5)


#===============================================================================
# 
#===============================================================================

def mask_to_array(mask, shape):
    if np.shape(mask) == shape:
        return mask
    
    new_mask = np.empty(shape) #XXX dtype
    new_mask[:,:] = mask
    return new_mask

def _moons_read_iss_map(body_name):
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

def _moons_read_schenk_jpg(body_name):
    body_data = {}
    img_filename = os.path.join(SUPPORT_FILES_ROOT, body_name+'_MAP.jpg')
    img = PIL.Image.open(img_filename)
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
    
def _moons_create_cartographic(bp, body_name, force_spherical=True,
                               source='schenk_jpg'):
    if force_spherical:
        body_name_spherical = body_name + '_SPHERICAL'
    else:
        body_name_spherical = body_name
    
    if (body_name_spherical, source) in CARTOGRAPHIC_FILE_CACHE:
        body_data = CARTOGRAPHIC_FILE_CACHE[(body_name_spherical, source)]
    else:
        if source == 'iss':
            body_data = _moons_read_iss_map(body_name)
        elif source == 'schenk_jpg':
            body_data = _moons_read_schenk_jpg(body_name)
        CARTOGRAPHIC_FILE_CACHE[(body_name_spherical, source)] = body_data

    nline = body_data['nline']
    nsamp = body_data['nsamp']
    map_data = body_data['map_data']
    pos_long_direction = body_data['pos_long_direction']
    line_proj_offset = body_data['line_proj_offset']
    sample_proj_offset = body_data['sample_proj_offset']
    center_longitude = body_data['center_longitude']
    center_latitude = body_data['center_latitude']
    map_resolution = body_data['map_resolution']
    line_first_pixel = body_data['line_first_pixel']
    line_last_pixel = body_data['line_last_pixel']
    sample_first_pixel = body_data['sample_first_pixel']
    sample_last_pixel = body_data['sample_last_pixel']
    
    latitude = bp.latitude(body_name_spherical,
                           lat_type='centric').vals.astype('float') * oops.DPR # XXX
    longitude = bp.longitude(body_name_spherical,
                             direction=pos_long_direction)
    longitude = longitude.vals.astype('float') * oops.DPR

    print 'LAT RANGE', np.min(latitude), np.max(latitude)
    print 'LONG RANGE', np.min(longitude), np.max(longitude)
    
    print 'LINE OFF', line_proj_offset, 'CTR LAT', center_latitude,
    print 'MAP RES', map_resolution
    line = (line_proj_offset -
            (latitude - center_latitude) * map_resolution)

    print 'SAMP OFF', sample_proj_offset, 'CTR LONG', center_longitude,
    print 'MAP RES', map_resolution
    sample = ((sample_proj_offset -
               (longitude - center_longitude) * map_resolution) % nsamp)

    line = line.astype('int')
    sample = sample.astype('int')
    
    line_mask = np.logical_or(line < line_first_pixel,
                              line > line_last_pixel)
    sample_mask = np.logical_or(sample < sample_first_pixel,
                                sample > sample_last_pixel)
    mask = np.logical_or(line_mask, sample_mask)

    line[mask] = 0
    sample[mask] = 0
    
    model = map_data[line, sample] 
    model[mask] = 0
    
    return model

def moons_create_model(obs, body_name, lambert=True,
                       u_min=0, u_max=10000, v_min=0, v_max=10000,
                       extend_fov=(0,0),
                       force_spherical=True, use_cartographic=True):
    logger = logging.getLogger(_LOGGING_NAME+'.moons_create_model')

    body_name = body_name.upper()

    set_obs_ext_bp(obs, extend_fov)
    set_obs_ext_data(obs, extend_fov)
    
    u_min -= 1
    u_max += 1
    v_min -= 1
    v_max += 1
    
    u_min = np.clip(u_min, -extend_fov[0], obs.data.shape[1]+extend_fov[0]-1)
    u_max = np.clip(u_max, -extend_fov[0], obs.data.shape[1]+extend_fov[0]-1)
    v_min = np.clip(v_min, -extend_fov[1], obs.data.shape[0]+extend_fov[1]-1)
    v_max = np.clip(v_max, -extend_fov[1], obs.data.shape[0]+extend_fov[1]-1)
           
    logger.debug('"%s" image size %d %d extend %d %d subrect U %d to %d '
                 'V %d to %d',
                 body_name, obs.data.shape[1], obs.data.shape[0],
                 extend_fov[0], extend_fov[1], u_min, u_max, v_min, v_max)
    
    # Create a Meshgrid that only covers the extent of the body
    # We're making the assumption here that calling where_intercepted
    # on the entire image is more expensive than creating a new smaller
    # Blackplane.
    restr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                           origin=(u_min+.5, v_min+.5),
                                           limit =(u_max+.5, v_max+.5),
                                           swap  =True)

    restr_bp = oops.Backplane(obs, meshgrid=restr_meshgrid)

    restr_body_mask = restr_bp.where_intercepted(body_name).vals
    restr_body_mask = mask_to_array(restr_body_mask, restr_bp.shape)

    if lambert:
        restr_model = restr_bp.lambert_law(body_name).vals.astype('float')
        restr_model[np.logical_not(restr_body_mask)] = 0.
    else:
        restr_model = restr_body_mask.astype('float')

    if use_cartographic and body_name in CARTOGRAPHIC_BODIES:
        cart_model = _moons_create_cartographic(restr_bp, body_name,
                                                force_spherical=force_spherical)
        restr_model *= cart_model 

    # Take the full-resolution object and put it back in the right place in a
    # full-size image
    model = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                      obs.data.shape[1]+extend_fov[0]*2),
                     dtype=np.float32)
    model[v_min+extend_fov[1]:v_max+extend_fov[1]+1,
          u_min+extend_fov[0]:u_max+extend_fov[0]+1] = restr_model
        
    return model


#===============================================================================
# 
#===============================================================================

def moons_generate_latitudes(start_num=0,
                              end_num=None,
                              latitude_resolution=
                                    MOONS_DEFAULT_REPRO_LATITUDE_RESOLUTION):
    """Generate a list of latitudes (deg)."""
    if end_num is None:
        end_num = int(180. / latitude_resolution)-1
    return np.arange(start_num, end_num+1) * latitude_resolution - 90.

def moons_generate_longitudes(start_num=0,
                              end_num=None,
                              longitude_resolution=
                                    MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Generate a list of longitudes (deg)."""
    if end_num is None:
        end_num = int(360. / longitude_resolution)-1
    return np.arange(start_num, end_num+1) * longitude_resolution

def moons_latitude_longitude_to_pixels(obs, body_name, latitude, longitude,
                                       lat_type='centric'):
    """Convert latitude (deg),longitude (deg) pairs to U,V."""
    assert lat_type in ('centric', 'graphic')

    latitude = polymath.Scalar.as_scalar(latitude) * oops.RPD
    longitude = polymath.Scalar.as_scalar(longitude) * oops.RPD
    
    if len(longitude) == 0:
        return np.array([]), np.array([])

    moon_surface = oops.Body.lookup(body_name).surface
    
    if lat_type == 'centric':
        latitude = moon_surface.lat_from_centric(latitude)
    else:
        latitude = moon_surface.lat_from_graphic(latitude)
        
    obs_event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                         polymath.Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = moon_surface.photon_to_event_by_coords(
                                          obs_event, (longitude, latitude))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals
    
def moons_reproject(obs, body_name, offset_u=0, offset_v=0,
            latitude_resolution=MOONS_DEFAULT_REPRO_LATITUDE_RESOLUTION,
            longitude_resolution=MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            zoom=MOONS_DEFAULT_REPRO_ZOOM, lat_type='centric'):
    """Reproject the moon into a rectangular latitude/longitude space.
    
    Inputs:
        obs                      The Observation.
        body_name                The name of the moon.
        offset_u                 The offsets in U,V to apply to the image
        offset_v                 when computing the latitude and longitude
                                 values.
        latitude_resolution      The latitude resolution of the new image
                                 (deg/pix).
        longitude_resolution     The longitude resolution of the new image
                                 (deg/pix).
        zoom                     The amount to magnify the original image for
                                 pixel value interpolation.
                                 
    Returns:
        A dictionary containing
        
        'full_mask'        The mask of pixels that contain reprojected data.
        'lat_idx_range'    The range (min,max) of latitudes in the returned
                           image.
        'long_idx_range'   The range (min,max) of longitudes in the returned
                           image.
        'time'             The midtime of the observation (TDB).
                           
            The following only contain latitudes and longitudes in the ranges
            specified above. All angles are in degrees.
        'img'              The reprojected image [latitude,longitude].
        'resolution'       The radial resolution [latitude,longitude].
        'phase'            The phase angle [latitude,longitude].
        'emission'         The emission angle [latitude,longitude].
        'incidence'        The incidence angle [latitude,longitude].
    """
    logger = logging.getLogger(_LOGGING_NAME+'.moons_reproject')

    # We need to be careful not to use obs.bp from this point forward because
    # it will disagree with our current OffsetFOV
    orig_fov = obs.fov
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(offset_u, offset_v))
    
    # Get all the info for each pixel
    bp = oops.Backplane(obs)
    lambert = bp.lambert_law(body_name).vals.astype('float')    
    bp_emission = bp.emission_angle(body_name).vals.astype('float')
    bp_phase = bp.phase_angle(body_name).vals.astype('float') * oops.DPR
    bp_incidence = (bp.incidence_angle(body_name).vals.astype('float') * 
                    oops.DPR)

    # Resolution takes into account the emission angle - the "along sight"
    # projection
    center_resolution = bp.center_resolution(body_name).vals.astype('float')
    resolution = center_resolution / np.cos(bp_emission)

    bp_latitude = bp.latitude(body_name, lat_type='centric')
    body_mask_inv = mask_to_array(bp_latitude.mask, bp.shape)
    bp_latitude = bp_latitude.vals.astype('float') * oops.DPR

    bp_longitude = bp.longitude(body_name, direction='east')
    bp_longitude = bp_longitude.vals.astype('float') * oops.DPR

    bp_emission_deg = bp_emission * oops.DPR

    # A pixel is OK if it falls on the body, the lambert model is
    # bright enough and the emission angle is large enough
    ok_body_mask_inv = np.logical_or(body_mask_inv, 
                                     lambert < MOONS_REPRO_MIN_LAMBERT)
    ok_body_mask_inv = np.logical_or(ok_body_mask_inv, 
                                     bp_emission_deg > 
                                         MOONS_REPRO_MAX_EMISSION)
    ok_body_mask = np.logical_not(ok_body_mask_inv)
    
    # Divide the data by the lambert model in an attempt to account for
    # projected illumination
    lambert[ok_body_mask_inv] = 1e300
    adj_data = obs.data / lambert
    adj_data[ok_body_mask_inv] = 0.
    lambert[ok_body_mask_inv] = 0.

    bp_emission[ok_body_mask_inv] = 1e300
    resolution[ok_body_mask_inv] = 1e300

    # The number of pixels in the final reprojection 
    latitude_pixels = int(180. / latitude_resolution)
    longitude_pixels = int(360. / longitude_resolution)

    # Restrict the latitude and longitude ranges for some attempt at efficiency
    min_latitude_pixel = (np.floor(np.min(bp_latitude[ok_body_mask]+90) / 
                                   latitude_resolution)).astype('int')
    min_latitude_pixel = np.clip(min_latitude_pixel, 0, latitude_pixels-1)
    max_latitude_pixel = (np.ceil(np.max(bp_latitude[ok_body_mask]+90) / 
                                  latitude_resolution)).astype('int')
    max_latitude_pixel = np.clip(max_latitude_pixel, 0, latitude_pixels-1)
    num_latitude_pixel = max_latitude_pixel - min_latitude_pixel + 1

    min_longitude_pixel = (np.floor(np.min(bp_longitude[ok_body_mask]) / 
                                    longitude_resolution)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, longitude_pixels-1)
    max_longitude_pixel = (np.ceil(np.max(bp_longitude[ok_body_mask]) / 
                                   longitude_resolution)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, longitude_pixels-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    # Latitude bin numbers
    lat_bins = np.repeat(np.arange(min_latitude_pixel, max_latitude_pixel+1), 
                         num_longitude_pixel)
    # Actual latitude (deg)
    lat_bins_act = lat_bins * latitude_resolution - 90.

    # Longitude bin numbers
    long_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1), 
                        num_latitude_pixel)
    # Actual longitude (deg)
    long_bins_act = long_bins * longitude_resolution

    logger.debug('Latitude range %8.2f %8.2f', 
                 np.min(bp_latitude[ok_body_mask]), 
                 np.max(bp_latitude[ok_body_mask]))
    logger.debug('Latitude bin range %8.2f %8.2f', 
                 np.min(lat_bins_act), 
                 np.max(lat_bins_act))
    logger.debug('Longitude range %6.2f %6.2f', 
                 np.min(bp_longitude[ok_body_mask]), 
                 np.max(bp_longitude[ok_body_mask]))
    logger.debug('Longitude bin range %6.2f %6.2f', 
                 np.min(long_bins_act),
                 np.max(long_bins_act))
    logger.debug('Resolution range %7.2f %7.2f', 
                 np.min(resolution[ok_body_mask]),
                 np.max(resolution[ok_body_mask]))
    logger.debug('Data range %f %f', 
                 np.min(adj_data), 
                 np.max(adj_data))

    u_pixels, v_pixels = moons_latitude_longitude_to_pixels(
                            obs, body_name, lat_bins_act, long_bins_act,
                            lat_type=lat_type)
        
    zoom_data = ndinterp.zoom(adj_data, zoom) # XXX Default to order=3 - OK?

    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    good_u = u_pixels[goodmask]
    good_v = v_pixels[goodmask]
    
    good_lat = lat_bins[goodmask]
    good_long = long_bins[goodmask]
    
    bad_data_mask = (adj_data == 0.)
    # Replicate the mask in each zoom x zoom block
    bad_data_mask_zoom = ndinterp.zoom(bad_data_mask, zoom, order=0)
    zoom_data[bad_data_mask_zoom] = 0.
    
    interp_data = zoom_data[(good_v*zoom).astype('int'), 
                            (good_u*zoom).astype('int')]
    
    repro_img = ma.zeros((latitude_pixels, longitude_pixels), 
                         dtype=np.float32)
    repro_img.mask = True
    repro_img[good_lat,good_long] = interp_data

    good_u = good_u.astype('int')
    good_v = good_v.astype('int')
    
    repro_res = ma.zeros((latitude_pixels, longitude_pixels), 
                         dtype=np.float32)
    repro_res.mask = True
    repro_res[good_lat,good_long] = resolution[good_v,good_u]
    full_mask = ma.getmaskarray(repro_res)

    repro_phase = ma.zeros((latitude_pixels, longitude_pixels), 
                           dtype=np.float32)
    repro_phase.mask = True
    repro_phase[good_lat,good_long] = bp_phase[good_v,good_u]

    repro_emission = ma.zeros((latitude_pixels, longitude_pixels), 
                              dtype=np.float32)
    repro_emission.mask = True
    repro_emission[good_lat,good_long] = bp_emission_deg[good_v,good_u]

    repro_incidence = ma.zeros((latitude_pixels, longitude_pixels), 
                               dtype=np.float32)
    repro_incidence.mask = True
    repro_incidence[good_lat,good_long] = bp_incidence[good_v,good_u]

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
                              min_longitude_pixel:max_longitude_pixel+1], 0.)

    ret = {}
    
    ret['full_mask'] = full_mask
    ret['lat_idx_range'] = (min_latitude_pixel, max_latitude_pixel)
    ret['long_idx_range'] = (min_longitude_pixel, max_longitude_pixel)
    ret['img'] = repro_img
    ret['resolution'] = repro_res
    ret['phase'] = repro_phase
    ret['emission'] = repro_emission
    ret['incidence'] = repro_incidence
    ret['time'] = obs.midtime
    
    return ret

def moons_mosaic_init(
      latitude_resolution=MOONS_DEFAULT_REPRO_LATITUDE_RESOLUTION,
      longitude_resolution=MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Create the data structure for a moon mosaic.

    Inputs:
        latitude_resolution      The latitude resolution of the new image
                                 (deg/pix).
        longitude_resolution     The longitude resolution of the new image
                                 (deg/pix).
                                 
    Returns:
        A dictionary containing an empty mosaic

        'img'              The full mosaic image.
        'full_mask'        The valid-pixel mask (all False).
        'resolution'       The per-pixel resolution.
        'phase'            The per-pixel phase angle.
        'emission'         The per-pixel emission angle.
        'incidence'        The per-pixel incidence angle.
        'image_number'     The per-pixel image number giving the image
                           used to fill the data for each pixel.
        'time'             The per-pixel time (TDB).
    """

    latitude_pixels = int(180. / latitude_resolution)
    longitude_pixels = int(360. / longitude_resolution)
    
    ret = {}
    ret['img'] = np.zeros((latitude_pixels, longitude_pixels), 
                          dtype=np.float32)
    ret['full_mask'] = np.ones((latitude_pixels, longitude_pixels), 
                               dtype=np.bool)
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
    ret['time'] = np.zeros((latitude_pixels, longitude_pixels), 
                           dtype=np.float32)
    
    return ret

def moons_mosaic_add(mosaic_metadata, repro_metadata, image_number):
    """Add a reprojected image to an existing mosaic.
    
    For each valid pixel in the reprojected image, it is copied to the
    mosaic if it has better resolution.
    """
    mosaic_img = mosaic_metadata['img']
    mosaic_mask = mosaic_metadata['full_mask']
    
    repro_lat_idx_range = repro_metadata['lat_idx_range']
    repro_long_idx_range = repro_metadata['long_idx_range']

    repro_mask = np.ones(mosaic_img.shape, dtype=np.bool) 
    repro_mask[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
               repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['full_mask']
        
    repro_img = np.zeros(mosaic_img.shape) 
    repro_img[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
              repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['img']
    
    mosaic_res = mosaic_metadata['resolution']
    repro_res = np.zeros(mosaic_res.shape)
    repro_res[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
              repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['resolution']
    
    mosaic_phase = mosaic_metadata['phase']
    repro_phase = np.zeros(mosaic_phase.shape)
    repro_phase[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['phase']
    
    mosaic_emission = mosaic_metadata['emission']
    repro_emission = np.zeros(mosaic_emission.shape)
    repro_emission[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                   repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['emission']
    
    mosaic_incidence = mosaic_metadata['incidence']
    repro_incidence = np.zeros(mosaic_incidence.shape)
    repro_incidence[repro_lat_idx_range[0]:repro_lat_idx_range[1]+1,
                    repro_long_idx_range[0]:repro_long_idx_range[1]+1] = \
                                repro_metadata['incidence']
    
    mosaic_image_number = mosaic_metadata['image_number']
    mosaic_time = mosaic_metadata['time']
    
    # Calculate where the new resolution is better
    better_resolution_mask = repro_res < mosaic_res
    replace_mask = np.logical_and(np.logical_not(repro_mask),
                                  np.logical_or(better_resolution_mask, 
                                                mosaic_mask))

    mosaic_img[replace_mask] = repro_img[replace_mask]
    mosaic_res[replace_mask] = repro_res[replace_mask] 
    mosaic_phase[replace_mask] = repro_phase[replace_mask] 
    mosaic_emission[replace_mask] = repro_emission[replace_mask] 
    mosaic_incidence[replace_mask] = repro_incidence[replace_mask] 
    mosaic_image_number[replace_mask] = image_number 
    mosaic_mask[replace_mask] = False
