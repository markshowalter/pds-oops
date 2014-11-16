###############################################################################
# cb_util_oops.py
#
# Routines related to oops.
#
# Exported routines:
#    make_corner_meshgrid
#    make_ext_meshgrid
#    set_obs_corner_bp
#    set_obs_ext_corner_bp
#    set_obs_bp
#    set_obs_ext_bp
#    set_obs_ext_data    
#    compute_ra_dec_limits
#    compute_sun_saturn_distance
#    mask_to_array
###############################################################################

import cb_logging
import logging

import numpy as np

import oops

from cb_util_image import *

_LOGGING_NAME = 'cb.' + __name__


#===============================================================================
#
# MESHGRIDS AND BACKPLANES
#
#===============================================================================

def make_corner_meshgrid(obs, extend_fov=(0,0)):
    """Create a Meshgrid with points only in the four corners of the extended
    FOV."""
    mg = oops.Meshgrid.for_fov(obs.fov,
             origin     =(-extend_fov[0]+0.5,-extend_fov[1]+0.5),
             limit      =(obs.data.shape[1]+extend_fov[0]-0.5,
                          obs.data.shape[0]+extend_fov[1]-0.5),
             undersample=(obs.data.shape[1]+extend_fov[0]*2-1,
                          obs.data.shape[0]+extend_fov[1]*2-1),
             swap       =True)
    return mg

def make_ext_meshgrid(obs, extend_fov=(0,0)):
    """Create a Meshgrid for the entire extended FOV."""
    mg = oops.Meshgrid.for_fov(obs.fov,
              origin=(-extend_fov[0]+.5, -extend_fov[1]+.5),
              limit =(obs.data.shape[1]+extend_fov[0]-.5,
                      obs.data.shape[0]+extend_fov[1]-.5),
              swap  =True)
    return mg

def set_obs_corner_bp(obs, force=False):
    """Create a Backplane for the corner points of the original FOV.
    
    Sets obs.corner_meshgrid and obs.corner_bp.
    """
    if (not hasattr(obs, 'corner_bp') or obs.corner_meshgrid is None or
        obs.corner_bp is None or force):
        obs.corner_meshgrid = make_corner_meshgrid(obs)
        obs.corner_bp = oops.Backplane(obs, meshgrid=obs.corner_meshgrid)

def set_obs_ext_corner_bp(obs, extend_fov, force=False):
    """Create a Backplane for the corner points of the extended FOV.
    
    Sets obs.extend_fov, obs.ext_corner_meshgrid and obs.ext_corner_bp.
    """
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_corner_bp') or obs.extend_fov is None or
        obs.ext_corner_meshgrid is None or
        obs.ext_corner_bp is None or force):
        obs.extend_fov = extend_fov
        obs.ext_corner_meshgrid = make_corner_meshgrid(obs, extend_fov)
        obs.ext_corner_bp = oops.Backplane(obs,
                                           meshgrid=obs.ext_corner_meshgrid)

def set_obs_bp(obs, force=False):
    """Create a Backplane for the original FOV.
    
    Sets obs.bp.
    """
    if not hasattr(obs, 'bp') or obs.bp is None or force:
        obs.bp = oops.Backplane(obs)

def set_obs_ext_bp(obs, extend_fov, force=False):
    """Create a Backplane for the extended FOV.
    
    Sets obs.extend_fov, obs.ext_meshgrid, and obs.ext_bp.
    """
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_bp') or obs.extend_fov is None or
        obs.ext_meshgrid is None or obs.ext_bp is None or force):
        obs.extend_fov = extend_fov
        obs.ext_meshgrid = make_ext_meshgrid(obs, extend_fov)
        obs.ext_bp = oops.Backplane(obs, meshgrid=obs.ext_meshgrid)

def set_obs_ext_data(obs, extend_fov, force=False):
    """Pad the data and store the result.
    
    Sets obs.extend_fov, obs.ext_data.
    """
    if (not hasattr(obs, 'extend_fov') or obs.extend_fov != extend_fov or
        not hasattr(obs, 'ext_data') or obs.extend_fov is None or
        obs.ext_data is None or force):
        obs.extend_fov = extend_fov
        obs.ext_data = pad_image(obs.data, extend_fov)


#===============================================================================
#
# COMPUTATIONAL ASSISTANCE
#
#===============================================================================
        
def compute_ra_dec_limits(obs, extend_fov=(0,0)):
    """Find the RA and DEC limits of an observation.
        
    Inputs:
        obs            The observation, which is assumed to cover less than
                       half the sky. If either RA or DEC wraps around, the min
                       value will be greater than the max value.
        extend_fov     The amount of padding to add in the (U,V) dimension.
       
    Returns:
        ra_min, ra_max, dec_min, dec_max (radians)
    """
    logger = logging.getLogger(_LOGGING_NAME+'.ra_dec_limits')
    
    set_obs_ext_corner_bp(obs, extend_fov)

    ra = obs.ext_corner_bp.right_ascension()
    dec = obs.ext_corner_bp.declination()
    
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

    logger.debug('RAMIN %6.4f RAMAX %6.4f DECMIN %7.4f DECMAX %7.4f',
                 ra_min, ra_max, dec_min, dec_max)
    
    return ra_min, ra_max, dec_min, dec_max

def compute_sun_saturn_distance(obs):
    """Compute the distance from the Sun to Saturn in AU."""
    target_sun_path = oops.Path.as_waypoint("SATURN").wrt('SUN')
    sun_event = target_sun_path.event_at_time(obs.midtime)
    solar_range = sun_event.pos.norm().vals / oops.AU

    return solar_range

#==============================================================================
# 
# MISCELLANEOUS
#
#==============================================================================

def mask_to_array(mask, shape):
    """Given a mask, convert a single Boolean value to the proper-sized array."""
    if np.shape(mask) == shape:
        return mask
    
    new_mask = np.empty(shape)
    new_mask[:,:] = mask
    return new_mask
