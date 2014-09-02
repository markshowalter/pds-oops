import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import os
from pdstable import PdsTable
import oops
import gravity
import cspice

from cb_config import *
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

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
    
    new_mask = np.empty(shape)
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
    
    u_min = np.clip(u_min, -extend_fov[0], obs.data.shape[1]-extend_fov[0]-1)
    u_max = np.clip(u_max, -extend_fov[0], obs.data.shape[1]-extend_fov[0]-1)
    v_min = np.clip(v_min, -extend_fov[1], obs.data.shape[0]-extend_fov[1]-1)
    v_max = np.clip(v_max, -extend_fov[1], obs.data.shape[0]-extend_fov[1]-1)
           
    logger.debug('"%s" image size %d %d extend %d %d subrect U %d to %d '
                 'V %d to %d',
                 body_name, obs.ext_data.shape[1], obs.data.shape[0],
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
                      obs.data.shape[1]+extend_fov[0]*2))
    model[v_min+extend_fov[1]:v_max+extend_fov[1]+1,
          u_min+extend_fov[0]:u_max+extend_fov[0]+1] = restr_model
        
    return model
