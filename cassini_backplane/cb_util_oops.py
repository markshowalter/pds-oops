import cb_logging
import logging

import oops
import solar
import numpy as np
from polymath import *

from cb_util_image import *

LOGGING_NAME = 'cb.' + __name__


#===============================================================================
#
# OOPS HELPERS 
#
#===============================================================================

def make_corner_meshgrid(obs, extend_fov=(0,0)):
    mg = oops.Meshgrid.for_fov(obs.fov,
             origin     =(-extend_fov[0]+0.5,-extend_fov[1]+0.5),
             limit      =(obs.data.shape[1]+extend_fov[0]-0.5,
                          obs.data.shape[0]+extend_fov[1]-0.5),
             undersample=(obs.data.shape[1]+extend_fov[0]*2-1,
                          obs.data.shape[0]+extend_fov[1]*2-1),
             swap       =True)
    return mg

def make_ext_meshgrid(obs, extend_fov=(0,0)):
    mg = oops.Meshgrid.for_fov(obs.fov,
              origin=(-extend_fov[0]+.5, -extend_fov[1]+.5),
              limit =(obs.data.shape[1]+extend_fov[0]-.5,
                      obs.data.shape[0]+extend_fov[1]-.5),
              swap  =True)
    return mg

def set_obs_corner_bp(obs):
    # The non-extended FOV
    if (not hasattr(obs, 'corner_bp') or obs.corner_meshgrid is None or
        obs.corner_bp is None):
        obs.corner_meshgrid = make_corner_meshgrid(obs)
        obs.corner_bp = oops.Backplane(obs, meshgrid=obs.corner_meshgrid)

def set_obs_ext_corner_bp(obs, extend_fov):
    # The extended FOV
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_corner_bp') or obs.extend_fov is None or
        obs.ext_corner_meshgrid is None or obs.ext_corner_bp is None):
        obs.extend_fov = extend_fov
        obs.ext_corner_meshgrid = make_corner_meshgrid(obs, extend_fov)
        obs.ext_corner_bp = oops.Backplane(obs, meshgrid=obs.ext_corner_meshgrid)

def set_obs_bp(obs):
    # The non-extended FOV
    if not hasattr(obs, 'bp') or obs.bp is None:
        obs.bp = oops.Backplane(obs)

def set_obs_ext_bp(obs, extend_fov):
    # The extended FOV
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_bp') or obs.extend_fov is None or
        obs.ext_meshgrid is None or obs.ext_bp is None):
        obs.extend_fov = extend_fov
        obs.ext_meshgrid = make_ext_meshgrid(obs, extend_fov)
        obs.ext_bp = oops.Backplane(obs, meshgrid=obs.ext_meshgrid)

def set_obs_ext_data(obs, extend_fov):
    # The extended FOV
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_data') or obs.extend_fov is None or
        obs.ext_data is None):
        obs.extend_fov = extend_fov
        obs.ext_data = pad_image(obs.data, extend_fov)
        
def compute_ra_dec_limits(obs, extend_fov=(0,0)):
    """Find the RA and DEC limits of an observation.
        
    Inputs:
        obs            The observation, which isassumed to cover less than
                       half the sky. If either RA or DEC wraps around, the min
                       value will be greater than the max value.
        extend_fov     The amount of padding to add in the (U,V) dimension
       
    Returns:
        ra_min, ra_max, dec_min, dec_max (radians)
    """
    logger = logging.getLogger(LOGGING_NAME+'.ra_dec_limits')
    
    # Create a meshgrid for the four corners of the image
    corner_meshgrid = make_corner_meshgrid(obs, extend_fov)
    
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

#    logger.debug('RA, DEC @ 0,0 %6.4f %7.4f', 
#                 ra[ 0, 0].vals, dec[ 0, 0].vals)
#    logger.debug('RA, DEC @ 0,N %6.4f %7.4f',
#                 ra[ 0,-1].vals, dec[ 0,-1].vals)
#    logger.debug('RA, DEC @ N,0 %6.4f %7.4f',
#                 ra[-1, 0].vals, dec[-1, 0].vals)
#    logger.debug('RA, DEC @ N,N %6.4f %7.4f',
#                 ra[-1,-1].vals, dec[-1,-1].vals)
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

def mask_to_array(mask, shape):
    if np.shape(mask) == shape:
        return mask
    
    new_mask = np.empty(shape)
    new_mask[:,:] = mask
    return new_mask
