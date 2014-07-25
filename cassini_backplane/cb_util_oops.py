import cb_logging
import logging

import oops
import solar
import numpy as np
from polymath import *

LOGGING_NAME = 'cb.' + __name__


#===============================================================================
#
# OOPS HELPERS 
#
#===============================================================================

def compute_ra_dec_limits(obs):
    """Find the RA and DEC limits of an observation.
    
    The observation is assumed to cover less than half the sky.
    If either RA or DEC wraps around, the min value will be greater than the
    max value.
    
    Returns:
        ra_min, ra_max, dec_min, dec_max (radians)
    """
    logger = logging.getLogger(LOGGING_NAME+'.ra_dec_limits')
    
    # Create a meshgrid for the four corners of the image
    corner_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                 origin=(0.5,0.5),
                 limit=(obs.data.shape[0]-0.5, obs.data.shape[1]-0.5),
                 undersample=(obs.data.shape[0]-1, obs.data.shape[1]-1))
    
    # Compute the RA and DEC for the corners
    backplane = oops.Backplane(obs, corner_meshgrid)
    ra = backplane.right_ascension()
    dec = backplane.declination()
    
    ra_min = ra.min()
    ra_max = ra.max()    
    if ra_max-ra_min > oops.PI:
        # Wrap around
        ra_min = ra[np.where(ra>np.pi)].min()
        ra_max = ra[np.where(ra<np.pi)].max()
        
    dec_min = dec.min()
    dec_max = dec.max()
    if dec_max-dec_min > oops.PI:
        # Wrap around
        dec_min = dec[np.where(dec>np.pi)].min()
        dec_max = dec[np.where(dec<np.pi)].max()

    logger.debug('RA, DEC @ 0,0 %6.4f %7.4f', 
                 ra[ 0, 0].vals, dec[ 0, 0].vals)
    logger.debug('RA, DEC @ 0,N %6.4f %7.4f',
                 ra[ 0,-1].vals, dec[ 0,-1].vals)
    logger.debug('RA, DEC @ N,0 %6.4f %7.4f',
                 ra[-1, 0].vals, dec[-1, 0].vals)
    logger.debug('RA, DEC @ N,N %6.4f %7.4f',
                 ra[-1,-1].vals, dec[-1,-1].vals)
    logger.debug('RAMIN %6.4f RAMAX %6.4f DECMIN %7.4f DECMAX %7.4f',
                 ra_min, ra_max, dec_min, dec_max)
    
    return ra_min, ra_max, dec_min, dec_max

def compute_sun_saturn_distance(obs):
    """Compute the distance from the Sun to Saturn in AU."""
    target_sun_path = oops.Path.as_waypoint("SATURN").wrt('SUN')
    sun_event = target_sun_path.event_at_time(obs.midtime)
    solar_range = sun_event.pos.norm().vals / solar.AU

    return solar_range

#===============================================================================
# 
#===============================================================================

class InventoryBody(object):
    pass

def obs_inventory(obs, bodies):
    """Return the body names that appear unobscured inside the FOV.

    Input:
        bodies      a list of the names of the body objects to be included
                    in the inventory.
        expand      an optional angle in radians by which to extend the
                    limits of the field of view. This can be used to
                    accommodate pointing uncertainties.

    Return:         list, array, or (list,array)

    Restrictions: All inventory calculations are performed at the
    observation midtime and all bodies are assumed to be spherical.
    """

    body_names = [oops.Body.as_body_name(body) for body in bodies]
    bodies  = [oops.Body.as_body(body) for body in bodies]
    nbodies = len(bodies)

    path_ids = [body.path for body in bodies]
    multipath = oops.path.MultiPath(path_ids)

    obs_event = oops.Event(obs.midtime, (Vector3.ZERO,Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = multipath.photon_to_event(obs_event)   # insert photon arrivals

    body_uv = obs.fov.uv_from_los(-obs_event.arr).vals

    centers = -obs_event.arr
    ranges = centers.norm().vals
    radii = [body.radius for body in bodies]
    radius_angles = np.arcsin(radii / ranges)

    inner_radii = [body.inner_radius for body in bodies]
    inner_angles = np.arcsin(inner_radii / ranges)

    # This array equals True for each body falling somewhere inside the FOV
    falls_inside = np.empty(nbodies, dtype='bool')
    for i in range(nbodies):
        falls_inside[i] = obs.fov.sphere_falls_inside(centers[i], radii[i])

    # This array equals True for each body completely hidden by another
    is_hidden = np.zeros(nbodies, dtype='bool')
    for i in range(nbodies):
      if not falls_inside[i]: continue

      for j in range(nbodies):
        if not falls_inside[j]: continue

        if ranges[i] < ranges[j]: continue
        if radius_angles[i] > inner_angles[j]: continue

        sep = centers[i].sep(centers[j])
        if sep < inner_angles[j] - radius_angles[i]:
            is_hidden[i] = True

    flags = falls_inside & ~is_hidden

    body_list = []
    for i in range(nbodies):
        if flags[i]:
            body = InventoryBody()
            body.body_name = body_names[i]
            body.center_uv = body_uv[i]
            body.center = centers[i]
            body.range = ranges[i]
            body.outer_radius = radii[i]
            body.inner_radius = inner_radii[i]
            u_scale = obs.fov.uv_scale.vals[0]
            v_scale = obs.fov.uv_scale.vals[1]
            u = body.center_uv[0]
            v = body.center_uv[1]
            body.u_min = np.clip(np.floor(u-radius_angles[i]/u_scale), 0,
                                 obs.data.shape[1]-1)
            body.u_max = np.clip(np.ceil(u+radius_angles[i]/u_scale), 0,
                                 obs.data.shape[1]-1)
            body.v_min = np.clip(np.floor(v-radius_angles[i]/v_scale), 0,
                                 obs.data.shape[0]-1)
            body.v_max = np.clip(np.ceil(v+radius_angles[i]/v_scale), 0,
                                 obs.data.shape[0]-1)
            body_list.append(body)

    return body_list

#===============================================================================
# 
#===============================================================================

def mask_to_array(mask, shape):
    if np.shape(mask) == shape:
        return mask
    
    new_mask = np.empty(shape)
    new_mask[:,:] = mask
    return new_mask

def XXcreate_model_one_body(obs, body_name, lambert=False,
                          u_min=None, u_max=None, v_min=None, v_max=None):
    logger = logging.getLogger(LOGGING_NAME+'.create_model_one_body')

    if u_min is None:
        u_min = 0
    if u_max is None:
        u_max = obs.data.shape[1]-1
    if v_min is None:
        v_min = 0
    if v_max is None:
        v_max = obs.data.shape[0]-1
           
    logger.debug('"%s" range U %d to %d V %d to %d',
                 body_name, u_min, u_max, v_min, v_max)
    
    # Create a Meshgrid that only covers the extent of the body
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

    # Take the full-resolution object and put it back in the right place in a
    # full-size image
    model = np.zeros(obs.data.shape)
    model[v_min:v_max+1,u_min:u_max+1] = restr_model
        
    return model
