import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp
import os
from pdstable import PdsTable
import oops
import gravity
import cspice

from cb_config import *
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

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
# 
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

def _moons_create_cartographic(bp, body_name, force_spherical=True):
    if force_spherical:
        body_name_spherical = body_name + '_SPHERICAL'
    else:
        body_name_spherical = body_name
        
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
    
    center_longitude = good_row['CENTER_LONGITUDE']
    center_latitude = good_row['CENTER_LATITUDE']
    minimum_latitude = good_row['MINIMUM_LATITUDE']
    maximum_latitude = good_row['MAXIMUM_LATITUDE']
    westernmost_longitude = good_row['WESTERNMOST_LONGITUDE']
    easternmost_longitude = good_row['EASTERNMOST_LONGITUDE']
    positive_longitude_direction = good_row['POSITIVE_LONGITUDE_DIRECTION'].lower()
    # XXX POSITIVE_LONGITUDE_DIRECTION should be a string, but it's marked
    # XXX as a float. This is bogus and has been reported.
    positive_longitude_direction = positive_longitude_direction[-4:]
    line_first_pixel = good_row['LINE_FIRST_PIXEL']-1
    assert line_first_pixel == 0 # Computation below would be wrong
    line_last_pixel = good_row['LINE_LAST_PIXEL']-1
    line_projection_offset = good_row['LINE_PROJECTION_OFFSET']
    nline = line_last_pixel - line_first_pixel + 1
    sample_first_pixel = good_row['SAMPLE_FIRST_PIXEL']-1
    assert sample_first_pixel == 0 # Computation below would be wrong
    sample_last_pixel = good_row['SAMPLE_LAST_PIXEL']-1
    sample_projection_offset = good_row['SAMPLE_PROJECTION_OFFSET']
    nsamp = sample_last_pixel - sample_first_pixel + 1
    map_projection_rotation = good_row['MAP_PROJECTION_ROTATION']
    map_resolution = good_row['MAP_RESOLUTION']
    map_scale = good_row['MAP_SCALE']
    map_filename = os.path.join(COISS_3XXX_ROOT, iss_dir,
                                good_row['FILE_SPECIFICATION_NAME'])

    latitude = bp.latitude(body_name_spherical,
                           lat_type='graphic').vals.astype('float') * oops.DPR
    longitude = bp.longitude(body_name_spherical,
                             direction=positive_longitude_direction)
    longitude = longitude.vals.astype('float') * oops.DPR

    if map_filename in CARTOGRAPHIC_FILE_CACHE:
        map_data = CARTOGRAPHIC_FILE_CACHE[map_filename]
    else:
        map_data = np.fromfile(map_filename, dtype='uint8')
        read_nline = len(map_data) // nsamp
        map_data = map_data.reshape((read_nline, nsamp))
        map_data = map_data[read_nline-nline:,:]
        CARTOGRAPHIC_FILE_CACHE[map_filename] = map_data
        
    print 'CENTER LAT', latitude[latitude.shape[0]//2, latitude.shape[1]//2], 'LONG', longitude[latitude.shape[0]//2, latitude.shape[1]//2]
    print np.min(latitude), np.max(latitude)
    print np.min(longitude), np.max(longitude)
    
    import matplotlib.pyplot as plt
#    plt.imshow(latitude)
#    plt.show()
#    
#    plt.imshow(longitude)
#    plt.show()
    
    print line_projection_offset, center_latitude, map_resolution
    line = (line_projection_offset -
            (latitude - center_latitude) * map_resolution)

    print sample_projection_offset, center_longitude, map_resolution
    sample = ((sample_projection_offset -
               (longitude - center_longitude) * map_resolution) % nsamp)

    line = line.astype('int')
    sample = sample.astype('int')
    
#    plt.imshow(line)
#    plt.show()
#    plt.imshow(sample)
#    plt.show()
    
    line_mask = np.logical_or(line < line_first_pixel,
                              line > line_last_pixel)
    sample_mask = np.logical_or(sample < sample_first_pixel,
                                sample > sample_last_pixel)
    mask = np.logical_or(line_mask, sample_mask)

    line[mask] = 0
    sample[mask] = 0
    
#    plt.imshow(map_data)
#    plt.show()
    model = map_data[line, sample] 
    model[mask] = 0
    
    return model

def moons_create_model(obs, body_name, lambert=True,
                       u_min=0, u_max=10000, v_min=0, v_max=10000,
                       extend_fov=(0,0),
                       force_spherical=True, use_cartographic=True):
    logger = logging.getLogger(LOGGING_NAME+'.moons_create_model')

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
    if end_num is None:
        end_num = int(360. / latitude_resolution)-1
    return np.arange(start_num, end_num+1) * latitude_resolution - 90.

def moons_generate_longitudes(start_num=0,
                              end_num=None,
                              longitude_resolution=
                                    MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    if end_num is None:
        end_num = int(360. / longitude_resolution)-1
    return np.arange(start_num, end_num+1) * longitude_resolution

def moons_latitude_longitude_to_pixels(obs, body_name, latitude, longitude):
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    moon_surface = oops.Body.lookup(body_name).surface
    obs_event = oops.Event(obs.midtime, (Vector3.ZERO,Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = moon_surface.photon_to_event_by_coords(obs_event,
                                      (longitude*oops.RPD, latitude*oops.RPD))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals
    
def moons_reproject(obs, body_name, offset_u=0, offset_v=0,
            latitude_resolution=MOONS_DEFAULT_REPRO_LATITUDE_RESOLUTION,
            longitude_resolution=MOONS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            zoom=MOONS_DEFAULT_REPRO_ZOOM):
    logger = logging.getLogger(LOGGING_NAME+'moons_reproject')

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
    ok_body_mask_inv = np.logical_or(body_mask_inv, lambert < MOONS_REPRO_MIN_LAMBERT)
    ok_body_mask_inv = np.logical_or(ok_body_mask_inv, bp_emission_deg > MOONS_REPRO_MAX_EMISSION)
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

    logger.debug('Latitude range %8.2f %8.2f', np.min(bp_latitude[ok_body_mask]), 
                 np.max(bp_latitude[ok_body_mask]))
    logger.debug('Latitude bin range %8.2f %8.2f', np.min(lat_bins_act), 
                 np.max(lat_bins_act))
    logger.debug('Longitude range %6.2f %6.2f', np.min(bp_longitude[ok_body_mask]), 
                 np.max(bp_longitude[ok_body_mask]))
    logger.debug('Longitude bin range %6.2f %6.2f', np.min(long_bins_act),
                 np.max(long_bins_act))
    logger.debug('Resolution range %7.2f %7.2f', np.min(resolution[ok_body_mask]),
                 np.max(resolution[ok_body_mask]))
    logger.debug('Data range %f %f', np.min(adj_data), np.max(adj_data))

    u_pixels, v_pixels = moons_latitude_longitude_to_pixels(obs, body_name,
                                                            lat_bins_act,
                                                            long_bins_act)
        
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
    
    interp_data = zoom_data[(good_v*zoom).astype('int'), (good_u*zoom).astype('int')]
    
    repro_img = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_img.mask = True
    repro_img[good_lat,good_long] = interp_data

    good_u = good_u.astype('int')
    good_v = good_v.astype('int')
    
    repro_res = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_res.mask = True
    repro_res[good_lat,good_long] = resolution[good_v,good_u]
    good_lat_lon_mask = np.logical_not(ma.getmaskarray(repro_res))

    repro_phase = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_phase.mask = True
    repro_phase[good_lat,good_long] = bp_phase[good_v,good_u]

    repro_emission = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_emission.mask = True
    repro_emission[good_lat,good_long] = bp_emission_deg[good_v,good_u]

    repro_incidence = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_incidence.mask = True
    repro_incidence[good_lat,good_long] = bp_incidence[good_v,good_u]

    good_lat_mask = ma.count(repro_img, axis=1) > 0 
    good_long_mask = ma.count(repro_img, axis=0) > 0 
    
    repro_img = ma.filled(repro_img[:,good_long_mask][good_lat_mask,:], 0.)
    repro_res = ma.filled(repro_res[:,good_long_mask][good_lat_mask,:], 0.)
    repro_phase = ma.filled(repro_phase[:,good_long_mask][good_lat_mask,:], 0.)
    repro_emission = ma.filled(repro_emission[:,good_long_mask][good_lat_mask,:], 0.)
    repro_incidence = ma.filled(repro_incidence[:,good_long_mask][good_lat_mask,:], 0.)

    ret = {}
    
    ret['good_mask'] = good_lat_lon_mask
    ret['good_lat_mask'] = good_lat_mask
    ret['good_long_mask'] = good_long_mask
    ret['img'] = repro_img
    ret['resolution'] = repro_res
    ret['phase'] = repro_phase
    ret['emission'] = repro_emission
    ret['incidence'] = repro_incidence
    ret['time'] = obs.midtime
    
    return ret

def add_to_mosaic(mosaic, mosaic_resolution, repro_img, repro_resolution):
    better_resolution_mask = (repro_resolution < mosaic_resolution)
    ok_value_mask = np.logical_and(-100 < repro_img, repro_img < 100)
    new_mosaic_mask = np.logical_and(better_resolution_mask, ok_value_mask)
    
    mosaic[new_mosaic_mask] = repro_img[new_mosaic_mask]
    mosaic_resolution[new_mosaic_mask] = repro_resolution[new_mosaic_mask]

    overlay = (repro_img-np.min(repro_img)) / (np.max(repro_img)-np.min(repro_img))
    im = imgdisp.ImageDisp([mosaic], [overlay],
                           canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()
