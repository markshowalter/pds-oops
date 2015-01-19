###############################################################################
# cb_rings.py
#
# Routines related to rings.
#
# Exported routines:
#    rings_create_model
#    rings_sufficient_curvature
#    rings_fiducial_features
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
import matplotlib.pyplot as plt

import polymath
import oops
import cspice
from pdstable import PdsTable
from tabulation import Tabulation

from cb_config import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__


RINGS_MIN_RADIUS = oops.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.SATURN_MAIN_RINGS[1]

RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.02 * oops.RPD
RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION = 5. # KM
_RINGS_DEFAULT_REPRO_ZOOM_AMT = 5
_RINGS_DEFAULT_REPRO_ZOOM_ORDER = 3

FRING_DEFAULT_REPRO_RADIUS_INNER = 139500. - 140220.
FRING_DEFAULT_REPRO_RADIUS_OUTER = 141000. - 140220.

_RINGS_LONGITUDE_SLOP = 1e-6 # Must be smaller than any longitude or radius
_RINGS_RADIUS_SLOP = 1e-6    # resolution we will be using
_RINGS_MAX_LONGITUDE = oops.TWOPI-_RINGS_LONGITUDE_SLOP*2

# These are bodies that might cast shadows on the ring plane near equinox
_RINGS_SHADOW_BODY_LIST = ['ATLAS', 'PROMETHEUS', 'PANDORA', 'EPIMETHEUS', 
                           'JANUS', 'MIMAS', 'ENCELADUS', 'TETHYS'] # XXX

_RINGS_UVIS_OCCULTATION = 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM'

# From French et al. 1993, Geometry of the Saturn system from the 3 July 1989
# occultation of 28 SGR and Voyager observations
_RINGS_FIDUCIAL_FEATURES_1993 = [
    # A RING
    (136552.0, 0.),         # Keeler Gap OEG
    (133745.2, 0.),         # Encke Gap OEG
    (133423.5, 0.),         # Encke Gap IEG XXX
    (122052.5, 0.),         # A Ring IER
    (120245.0, 0.),         # 1.994 Rs ringlet IER
    (120076.3, 0.),         # 1.990 Rs ringlet OER
    (118968.3, 0.),         # OEG
    (118629.1, 0.),         # OEG
    (118283.9, 0.),         # OEG
    (117932.2, 0.),         # OEG
    # B RING
    (104083.4, 0.),         # OEG
    (103673.2, 0.),         # IEG
    (101549.3, 0.),         # IEG
    (101009.7, 0.),         # IEG
    ( 98287.2, 0.),         # OER
    ( 96899.6, 0.),         # IEG
    ( 95358.3, 0.),         # B Ring flat spot OEG
    ( 94444.1, 0.),         # B Ring flat spot IEG
    # C RING
    ( 90614.9, 0.),         # OER
    ( 90405.7, 0.),         # IER
    ( 89938.8, 0.),         # OER
    ( 89788.3, 0.),         # IER
    ( 89295.3, 0.),         # OER
    ( 89190.4, 0.),         # IER
    ( 88594.3, 0.),         # OER
    ( 86602.4, 0.),         # OER
    ( 86371.9, 0.),         # IER
    ( 85923.6, 0.),         # IER
    ( 85758.7, 0.),         # OER
    ( 85661.5, 0.),         # IER
    ( 84949.2, 0.),         # OER
    ( 84750.3, 0.),         # IER
    ( 82041.6, 0.),         # COR
    ( 79265.1, 0.),         # OER
    ( 79221.0, 0.),         # IER
    ( 77164.4, 0.),         # OER
    ( 76262.9, 0.),         # OER
    ( 74490.0, 0.),         # C Ring IER    
]

_RINGS_FIDUCIAL_FEATURES_PATH = os.path.join(
     SUPPORT_FILES_ROOT, '20140419toRAJ', 'ringfit_v1.8.Sa025S-RF-V4927.out')
_RINGS_FIDUCIAL_FEATURES = []

#==============================================================================
# 
# RING MODELS
#
#==============================================================================

def rings_sufficient_curvature(obs, extend_fov=None, threshold=2):
    logger = logging.getLogger(_LOGGING_NAME+'.rings_sufficient_curvature')

    if extend_fov is not None:
        set_obs_ext_bp(obs, extend_fov)
        bp = obs.ext_bp
    else:
        set_obs_bp(obs)
        bp = obs.bp

    radii = bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    longitudes = bp.ring_longitude('saturn:ring').vals.astype('float') 
    min_longitude = np.min(longitudes)
    max_longitude  = np.max(longitudes)

    logger.debug('Radii %.2f to %.2f Longitudes %.2f to %.2f',
                 min_radius, max_radius, 
                 min_longitude*oops.DPR, max_longitude*oops.DPR)
    
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.debug('No main rings in image - returning bad curvature')
        return False

    if max_longitude - min_longitude > np.pi/2.:
        # XXX This does not handle the wrap-around case! XXX
        # Seeing 90+ degress of the ring - definitely enough curvature!
        logger.debug('More than 90 degrees visible - returning enough '+
                     'curvature')
        return True

    min_radius = max(min_radius, RINGS_MIN_RADIUS)
    max_radius = min(max_radius, RINGS_MAX_RADIUS)
    
    # Find the approximate radius with the greatest span of longitudes
    
    best_len = 0
    best_radius = None
    best_longitudes = None

    radius_step = (max_radius-min_radius) / 10.
    longitude_step = (max_longitude-min_longitude) / 100.
    for radius in np.arange(min_radius, max_radius+radius_step, radius_step):
        trial_longitudes = np.arange(min_longitude, 
                                     max_longitude+longitude_step,
                                     longitude_step)
        trial_radius = np.empty(trial_longitudes.shape)
        trial_radius[:] = radius
        (new_longitudes, new_radius,
         u_pixels, v_pixels) = _rings_restrict_longitude_radius_to_obs(
                                         obs, trial_longitudes, trial_radius,
                                         extend_fov=extend_fov)
        if len(new_longitudes) > best_len:
            best_radius = radius
            best_longitudes = new_longitudes
            best_len = len(new_longitudes)
    
    assert best_len > 0    
    
    logger.debug('Optimal radius %.2f longitude range %.2f to %.2f',
                 best_radius, best_longitudes[0]*oops.DPR,
                 best_longitudes[-1]*oops.DPR)
    
    # Now for this optimal radius, find the pixel values of the minimum
    # and maximum available longitudes as well as a point halfway between.
    
    line_radius = np.empty(3)
    line_radius[:] = best_radius
    line_longitude = np.empty(3)
    line_longitude[0] = best_longitudes[0]
    line_longitude[2] = best_longitudes[-1]
    line_longitude[1] = (line_longitude[0]+line_longitude[2])/2
    
    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                      obs, line_longitude, line_radius)
    
    logger.debug('Linear pixels %.2f,%.2f / %.2f,%.2f / %.2f,%.2f',
                 u_pixels[0], v_pixels[0], u_pixels[1], v_pixels[1],
                 u_pixels[2], v_pixels[2])
    
    mid_pt_u = (u_pixels[0]+u_pixels[2])/2
    mid_pt_v = (v_pixels[0]+v_pixels[2])/2
    
    dist = np.sqrt((mid_pt_u-u_pixels[1])**2+(mid_pt_v-v_pixels[1])**2)
    
    if dist < threshold:
        logger.debug('Distance %.2f is too close for curvature', dist)
        return False
    
    logger.debug('Distance %.2f is far enough for curvature', dist)
    return True

def _rings_read_fiducial_features():
    if len(_RINGS_FIDUCIAL_FEATURES) > 0:
        return
    with open(_RINGS_FIDUCIAL_FEATURES_PATH, 'r') as fp:
        for line in fp:
            if line.startswith('Ring         A'):
                break
        else:
            assert False
        for line in fp:
            if line.startswith('Index'):
                break
            if line[9] != '*': # Circular feature?
                continue
            a = float(line[10:21])
            _RINGS_FIDUCIAL_FEATURES.append((a,0.))            
    
    _RINGS_FIDUCIAL_FEATURES.sort(key=lambda x:x[0], reverse=True)
    
def rings_fiducial_features(obs, extend_fov=None):
    logger = logging.getLogger(_LOGGING_NAME+'.rings_fiducial_features')

    _rings_read_fiducial_features()
    
    if extend_fov is not None:
        set_obs_ext_bp(obs, extend_fov)
        bp = obs.ext_bp
    else:
        set_obs_bp(obs)
        bp = obs.bp

    radii = bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    logger.debug('Radii %.2f to %.2f', min_radius, max_radius) 
    
    feature_list = []
    
    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES:
        location, resolution = fiducial_feature
        if min_radius < location < max_radius:
            feature_list.append(fiducial_feature)
    
    logger.debug('Returning %d fiducial features', len(feature_list))
    
    return feature_list


_RING_VOYAGER_IF_DATA = None
_RING_UVIS_OCC_DATA = None

def _blur_ring_radial_data(tab, resolution):
    min_blur = 10. # km - Don't blur anything closer than this
    max_blur = 100. # km
    blur_copy = 10
    radial_resolution = 1.
    domain = tab.domain()
    start_radius, end_radius = domain
    new_radius = np.arange(start_radius, end_radius+radial_resolution,
                           radial_resolution)
    data = tab(new_radius)
    blurred_data = data.copy()
    
    # Blur 2-sigma over 100 km (50 km each side)
#    _data = filt.gaussian_filter(data, blur_sigma)
#    blurred_data = np.zeros(data.shape)
#    # Now copy in the original data if near a fiducial feature
#    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES:
#        location, resolution = fiducial_feature
#        x = location - domain[0]
#        x1 = max(x-blur_copy, 0)
#        x2 = min(x+blur_copy, domain[1])
#        blurred_data[x1:x2+1] = data[x1:x2+1]

    # Compute the distance to the nearest fiducial feature
    fiducial_dist = np.zeros(new_radius.shape) + 1e38
    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES:
        location, resolution = fiducial_feature
        temp_dist = np.zeros(new_radius.shape) + location
        temp_dist = np.abs(temp_dist-new_radius)
        fiducial_dist = np.minimum(fiducial_dist, temp_dist)
    
    num_blur = int((max_blur-min_blur) / radial_resolution)
    for dist in xrange(num_blur):
        blur_amt = float(dist)
        one_blur = filt.gaussian_filter(data, blur_amt)
        replace_bool = fiducial_dist >= (dist+min_blur)
        blurred_data[replace_bool] = one_blur[replace_bool]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    plt.plot(new_radius, fiducial_dist, '-', color='black')
    plt.plot(new_radius, data, '-', color='#ff4040')
    plt.plot(new_radius, blurred_data, '-', color='black')
    plt.show()
    
    return blurred_data, start_radius, end_radius, radial_resolution

def _compute_ring_radial_data(source, resolution):
    assert source in ('voyager', 'uvis')
    
    if source == 'voyager':
        global _RING_VOYAGER_IF_DATA
        if not _RING_VOYAGER_IF_DATA:
            if_table = PdsTable(os.path.join(SUPPORT_FILES_ROOT,
                                             'IS2_P0001_V01_KM002.LBL'))
            _RING_VOYAGER_IF_DATA = Tabulation(
                   if_table.column_dict['RING_INTERCEPT_RADIUS'],
                   if_table.column_dict['I_OVER_F'])
        tab = _RING_VOYAGER_IF_DATA

    if source == 'uvis':
        global _RING_UVIS_OCC_DATA
        if not _RING_UVIS_OCC_DATA:
            occ_root = os.path.join(COUVIS_8XXX_ROOT, 'COUVIS_8001', 'DATA',
                                    'EASYDATA')
            occ_file = os.path.join(occ_root, _RINGS_UVIS_OCCULTATION+'.LBL')
            label = PdsTable(occ_file)
            data = (1.-np.e**-label.column_dict['NORMAL OPTICAL DEPTH'])
            _RING_UVIS_OCC_DATA = Tabulation(
                   label.column_dict['RING_RADIUS'],
                   data)
        tab = _RING_UVIS_OCC_DATA

    ret = _blur_ring_radial_data(tab, resolution)
    
    return ret


def rings_create_model(obs, extend_fov=(0,0), source='voyager'):
    """Create a model for the rings.
    
    If there are no rings in the image or they are entirely in shadow,
    return None.
    
    The rings model is created by interpolating from the Voyager I/F
    profile. Portions in Saturn's shadow are removed.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.rings_create_model')

    assert source in ('uvis', 'voyager')
    
    metadata = {}
    metadata['shadow_bodies'] = []
    metadata['curvature_ok'] = False
    metadata['fiducial_features'] = []
    
    set_obs_ext_bp(obs, extend_fov)

    if not rings_sufficient_curvature(obs, extend_fov):
       logger.debug('Too little curvature - no ring model produced')
       return None, metadata
   
    metadata['curvature_ok'] = True     
   
    fiducial_features = rings_fiducial_features(obs, extend_fov)
    metadata['fiducial_features'] = fiducial_features
    fiducial_features_ok = len(fiducial_features) >= 2
    metadata['fiducial_features_ok'] = fiducial_features_ok
    
    if not fiducial_features_ok:
        logger.debug('Insufficient number of fiducial features - '+
                     'no ring model produced')
        return None, metadata
    
    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    logger.debug('Radii %.2f to %.2f', min_radius, max_radius)
    
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.debug('No main rings in image - returning null model')
        return None, metadata

    radii[radii < RINGS_MIN_RADIUS] = 0
    radii[radii > RINGS_MAX_RADIUS] = 0
    
    ret = _compute_ring_radial_data(source, 0.) # XXX
    radial_data, start_radius, end_radius, radial_resolution = ret

    radii[radii < start_radius] = 0
    radii[radii > end_radius] = 0

    radial_index = np.round((radii-start_radius)/radial_resolution)
    radial_index = np.clip(radial_index, 0, radial_data.shape[0]-1)
    radial_index = radial_index.astype('int')
    model = radial_data[radial_index]
    
#    model = ma.masked_equal(model, 0.)
#    model = ma.masked_equal(model, 10000.)
#    model[model==0] = 0.001
    
    shadow_body_list = []
    
    saturn_shadow = obs.ext_bp.where_inside_shadow('saturn:ring',
                                                   'saturn').vals
    if np.any(saturn_shadow):
        logger.debug('Rings shadowed by SATURN')
        shadow_body_list.append('SATURN')
        model[saturn_shadow] = 0

    # XXX Equinox only
    # XXX There must be a way to make this more efficient in the case
    # when a moon isn't in a position to cast a shadow
    for body_name in _RINGS_SHADOW_BODY_LIST:
        shadow = obs.ext_bp.where_inside_shadow('saturn:ring',
                                                body_name).vals
        if np.any(shadow):
            logger.debug('Rings shadowed by '+body_name)
            shadow_body_list.append(body_name)
            model[shadow] = 0
    
    metadata['shadow_bodies'] = shadow_body_list
    
    if not np.any(model):
        logger.debug('Rings are entirely shadowed - returning null model')
        return None, metadata
    
    return model, metadata


#==============================================================================
# 
# RING REPROJECTION UTILITIES
#
#==============================================================================

##
# Non-F-ring-specific routines
##


def rings_generate_longitudes(longitude_start=0.,
                              longitude_end=_RINGS_MAX_LONGITUDE,
                              longitude_resolution=
                                    RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Generate a list of longitudes.
    
    The list will be on longitude_resolution boundaries and is guaranteed to
    not contain a longitude less than longitude_start or greater than
    longitude_end."""
    longitude_start = (np.ceil(longitude_start/longitude_resolution) *
                       longitude_resolution)
    longitude_end = (np.floor(longitude_end/longitude_resolution) *
                     longitude_resolution)
    return np.arange(longitude_start, longitude_end+_RINGS_LONGITUDE_SLOP,
                     longitude_resolution)

def rings_generate_radii(radius_inner, radius_outer,
                         radius_resolution=
                             RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Generate a list of radii (km)."""
    return np.arange(radius_inner, radius_outer+_RINGS_RADIUS_SLOP,
                     radius_resolution)

def _rings_restrict_longitude_radius_to_obs(obs, longitude, radius,
                                            offset=None, extend_fov=None):
    """Restrict the list of longitude and radius to those present
    in the image. Also return the U,V coordinate of each longitude,radius
    pair."""
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset
        
    if extend_fov is not None:
        set_obs_ext_bp(obs, extend_fov)
        bp = obs.ext_bp
        u_min = -obs.extend_fov[0]
        v_min = -obs.extend_fov[1]
        u_max = obs.data.shape[1]-1 + obs.extend_fov[0]
        v_max = obs.data.shape[0]-1 + obs.extend_fov[1]
    else:
        set_obs_bp(obs)
        bp = obs.bp
        u_min = 0
        v_min = 0
        u_max = obs.data.shape[1]-1
        v_max = obs.data.shape[0]-1

    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float')
    
    min_bp_radius = np.min(bp_radius)
    max_bp_radius = np.max(bp_radius)
    min_bp_longitude = np.min(bp_longitude)
    max_bp_longitude = np.max(bp_longitude)
    
    # First pass restriction so rings_longitude_radius_to_pixels will run
    # faster
    goodr = np.logical_and(radius >= min_bp_radius, radius <= max_bp_radius)
    goodl = np.logical_and(longitude >= min_bp_longitude,
                           longitude <= max_bp_longitude)
    good = np.logical_and(goodr, goodl)
    
    radius = radius[good]
    longitude = longitude[good]

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                                  obs, longitude, radius)

    u_pixels += offset_u
    v_pixels += offset_v
        
    # Catch the cases that fell outside the image boundaries
    goodumask = np.logical_and(u_pixels >= u_min, u_pixels <= u_max)
    goodvmask = np.logical_and(v_pixels >= v_min, v_pixels <= v_max)
    good = np.logical_and(goodumask, goodvmask)
    
    radius = radius[good]
    longitude = longitude[good]
    u_pixels = u_pixels[good]
    v_pixels = v_pixels[good]
    
    return longitude, radius, u_pixels, v_pixels
    
def rings_longitude_radius_to_pixels(obs, longitude, radius, corotating=None):
    """Convert longitude,radius pairs to U,V."""
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
                                                          (radius,longitude))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals

##
# F ring routines
##

FRING_ROTATING_ET = cspice.utc2et("2007-1-1")
FRING_MEAN_MOTION = 581.964 * oops.RPD # rad/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 24.2 * oops.RPD # deg
FRING_DW = 2.70025 * oops.RPD # deg/day                

def _compute_fring_longitude_shift(et): 
    return - (FRING_MEAN_MOTION * 
              ((et - FRING_ROTATING_ET) / 86400.)) % oops.TWOPI

def rings_fring_inertial_to_corotating(longitude, et):
    """Convert inertial longitude to corotating."""
    return (longitude + _compute_fring_longitude_shift(et)) % oops.TWOPI

def rings_fring_corotating_to_inertial(co_long, et):
    """Convert corotating longitude (deg) to inertial."""
    return (co_long - _compute_fring_longitude_shift(et)) % oops.TWOPI

def rings_fring_radius_at_longitude(obs, longitude):
    """Return the radius (km) of the F ring core at a given inertial longitude
    (deg)."""
    curly_w = FRING_W0 + FRING_DW*obs.midtime/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(longitude-curly_w)))

    return radius
    
def rings_fring_longitude_radius(obs, longitude_step=0.01*oops.RPD):
    """Return  a set of longitude,radius pairs for the F ring core."""
    num_longitudes = int(oops.TWOPI / longitude_step)
    longitudes = np.arange(num_longitudes) * longitude_step
    radius = rings_fring_radius_at_longitude(obs, longitudes)
    
    return longitudes, radius

def rings_fring_pixels(obs, offset=None, longitude_step=0.01*oops.RPD):
    """Return a set of U,V pairs for the F ring in an image."""
    longitude, radius = rings_fring_longitude_radius(
                                     obs,
                                     longitude_step=longitude_step)
    
    (longitude, radius,
     u_pixels, v_pixels) = _rings_restrict_longitude_radius_to_obs(
                                     obs, longitude, radius,
                                     offset=offset)
    
    return u_pixels, v_pixels


#==============================================================================
# 
# RING REPROJECTION MAIN ROUTINES
#
#==============================================================================

def rings_reproject(
            obs, offset=None,
            longitude_resolution=RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            radius_resolution=RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
            radius_inner=None,
            radius_outer=None,
            zoom_amt=_RINGS_DEFAULT_REPRO_ZOOM_AMT,
            zoom_order=_RINGS_DEFAULT_REPRO_ZOOM_ORDER,
            corotating=None,
            longitude_range=None,
            uv_range=None,
            compress_longitude=True,
            mask_fill_value=0.):
    """Reproject the rings in an image into a rectangular longitude/radius
    space.
    
    Inputs:
        obs                      The Observation.
        offset                   The offsets in (U,V) to apply to the image
                                 when computing the longitude and radius
                                 values.
        longitude_resolution     The longitude resolution of the new image
                                 (rad/pix).
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
        longitude_range          None, or a tuple (start,end) specifying the
                                 longitude limits to reproject.
        uv_range                 None, or a tuple (start_u,end_u,start_v,end_v)
                                 that defines the part of the image to be
                                 reprojected.
        compress_longitude       True to compress the returned image to contain
                                 only valid longitudes. False to return the
                                 entire range 0-2PI or as specified by
                                 longitude_range.
        mask_fill_value          What to replace masked values with. None means
                                 leave the values masked.
                                 
    Returns:
        A dictionary containing
        
        'long_mask'        The mask of longitudes from the full 2PI-radian
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
    logger = logging.getLogger(_LOGGING_NAME+'.rings_reproject')
    
    assert corotating in (None, 'F')
    
    if longitude_range is None:
        longitude_start = 0.
        longitude_end = _RINGS_MAX_LONGITUDE
    else:
        longitude_start, longitude_end = longitude_range
        
    # We need to be careful not to use obs.bp from this point forward because
    # it will disagree with our current OffsetFOV
    orig_fov = None
    if offset is not None:
        orig_fov = obs.fov
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    
    # Get all the info for each pixel
    meshgrid = None
    start_u = 0
    end_u = obs.data.shape[1]-1
    start_v = 0
    end_v = obs.data.shape[0]-1
    if uv_range is not None:
        start_u, end_u, start_v, end_v = uv_range
        meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                     origin=(start_u+.5, start_v+.5), 
                                     limit=(end_u+.5, end_v+.5), swap=True)

    bp = oops.Backplane(obs, meshgrid)
    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float') 
    bp_resolution = (bp.ring_radial_resolution('saturn:ring')
                     .vals.astype('float'))
    bp_phase = bp.phase_angle('saturn:ring').vals.astype('float')
    bp_emission = bp.emission_angle('saturn:ring').vals.astype('float') 
    bp_incidence = bp.incidence_angle('saturn:ring').vals.astype('float') 
    saturn_shadow = bp.where_inside_shadow('saturn:ring','saturn').vals
    data = obs.data.copy()
    data[saturn_shadow] = 0
    
    # The number of pixels in the final reprojection
    radius_pixels = int(np.ceil((radius_outer-radius_inner+
                                 _RINGS_RADIUS_SLOP) / radius_resolution))
    longitude_pixels = int(np.ceil((longitude_end-longitude_start+
                                    _RINGS_LONGITUDE_SLOP) /
                                   longitude_resolution))
    longitude_start_pixel = int(longitude_start / longitude_resolution)

    if corotating == 'F':
        # Convert longitude to co-rotating
        bp_longitude = rings_fring_inertial_to_corotating(bp_longitude, 
                                                          obs.midtime)
    
    # Restrict the longitude range for some attempt at efficiency.
    # This fails to be efficient if the longitude range wraps around.
    min_longitude_pixel = (np.floor(max(longitude_start, np.min(bp_longitude))/ 
                                    longitude_resolution)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, longitude_pixels-1)
    max_longitude_pixel = (np.ceil(min(longitude_end, np.max(bp_longitude)) / 
                                   longitude_resolution)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, longitude_pixels-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    # Longitude bin numbers
    long_bins = np.tile(np.arange(num_longitude_pixel), radius_pixels)
    # Actual longitude for each bin (deg)
    long_bins_act = (long_bins * longitude_resolution + 
                     min_longitude_pixel * longitude_resolution)

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
    logger.debug('Longitude range %6.2f %6.2f', 
                 np.min(bp_longitude)*oops.DPR, 
                 np.max(bp_longitude)*oops.DPR)
    logger.debug('Longitude bin range %6.2f %6.2f', 
                 np.min(long_bins_act)*oops.DPR,
                 np.max(long_bins_act)*oops.DPR)
    logger.debug('Resolution range %7.2f %7.2f', np.min(bp_resolution),
                 np.max(bp_resolution))
    logger.debug('Data range %f %f', np.min(data), np.max(data))

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                                  obs, long_bins_act,
                                                  rad_bins_act,
                                                  corotating=corotating)
    
    # Zoom the data and restrict the bins and pixels to ones actually in the
    # final reprojection.
    zoom_data = ndinterp.zoom(data, zoom_amt, order=zoom_order)

    u_zoom = (u_pixels*zoom_amt).astype('int')
    v_zoom = (v_pixels*zoom_amt).astype('int')
    
    goodumask = np.logical_and(u_pixels >= start_u, 
                               u_zoom <= (end_u+1)*zoom_amt-1)
    goodvmask = np.logical_and(v_pixels >= start_v, 
                               v_zoom <= (end_v+1)*zoom_amt-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    u_pixels = u_pixels[goodmask].astype('int') - start_u
    v_pixels = v_pixels[goodmask].astype('int') - start_v
    u_zoom = u_zoom[goodmask]
    v_zoom = v_zoom[goodmask]
    good_rad = rad_bins[goodmask]
    good_long = long_bins[goodmask] + min_longitude_pixel
    
    interp_data = zoom_data[v_zoom, u_zoom]
    
    # Create the reprojected results.
    repro_img = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_img.mask = True
    repro_img[good_rad,good_long] = interp_data

    repro_res = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_res.mask = True
    repro_res[good_rad,good_long] = bp_resolution[v_pixels,u_pixels]
    repro_mean_res = ma.mean(repro_res, axis=0)
    # Mean will mask if ALL radii are masked are a particular longitude

    # All interpolated data should be masked the same, so we might as well
    # take one we've already computed.
    good_long_mask = np.logical_not(ma.getmaskarray(repro_mean_res))

    repro_phase = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_phase.mask = True
    repro_phase[good_rad,good_long] = bp_phase[v_pixels,u_pixels]
    repro_mean_phase = ma.mean(repro_phase, axis=0)

    repro_emission = ma.zeros((radius_pixels, longitude_pixels), 
                              dtype=np.float32)
    repro_emission.mask = True
    repro_emission[good_rad,good_long] = bp_emission[v_pixels,u_pixels]
    repro_mean_emission = ma.mean(repro_emission, axis=0)

    repro_incidence = ma.zeros((radius_pixels, longitude_pixels), 
                               dtype=np.float32)
    repro_incidence.mask = True
    repro_incidence[good_rad,good_long] = bp_incidence[v_pixels,u_pixels]
    repro_mean_incidence = ma.mean(repro_incidence) # scalar

    if compress_longitude:
        repro_img = repro_img[:,good_long_mask]
        repro_res = repro_res[:,good_long_mask]
        repro_phase = repro_phase[:,good_long_mask]
        repro_emission = repro_emission[:,good_long_mask]
        repro_incidence = repro_incidence[:,good_long_mask]

        repro_mean_res = repro_mean_res[good_long_mask]
        repro_mean_phase = repro_mean_phase[good_long_mask]
        repro_mean_emission = repro_mean_emission[good_long_mask]

        assert ma.count_masked(repro_mean_res) == 0
        assert ma.count_masked(repro_mean_phase) == 0
        assert ma.count_masked(repro_mean_emission) == 0

    if mask_fill_value is not None:
        repro_img = ma.filled(repro_img, mask_fill_value)
        repro_res = ma.filled(repro_res, mask_fill_value)
        repro_phase = ma.filled(repro_phase, mask_fill_value)
        repro_emission = ma.filled(repro_emission, mask_fill_value)
        repro_incidence = ma.filled(repro_incidence, mask_fill_value)

    if orig_fov is not None:   
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
                                 (rad/pix).
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
    radius_pixels = int(np.ceil((radius_outer-radius_inner+_RINGS_RADIUS_SLOP) / 
                                radius_resolution))
    longitude_pixels = int(oops.TWOPI / longitude_resolution)
    
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
