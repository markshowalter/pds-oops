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

def make_corner_meshgrid(obs, extend_fov_u=0, extend_fov_v=0, padding=0):
    mg = oops.Meshgrid.for_fov(obs.fov,
             origin     =(-extend_fov_u-padding+0.5,-extend_fov_v-padding+0.5),
             limit      =(obs.data.shape[1]+extend_fov_u+padding-0.5,
                          obs.data.shape[0]+extend_fov_v+padding-0.5),
             undersample=(obs.data.shape[1]+extend_fov_u*2+padding*2-1,
                          obs.data.shape[0]+extend_fov_v*2+padding*2-1),
             swap       =True)
    return mg

def make_oversize_meshgrid(obs, extend_fov_u=0, extend_fov_v=0):
    mg = oops.Meshgrid.for_fov(obs.fov,
              origin=(-extend_fov_u+.5, -extend_fov_v+.5),
              limit =(obs.data.shape[1]+extend_fov_u-.5,
                      obs.data.shape[0]+extend_fov_v-.5),
              swap  =True)
    return mg
        
def compute_ra_dec_limits(obs, extend_fov_u=0, extend_fov_v=0,
                          padding=5):
    """Find the RA and DEC limits of an observation.
    
    
    Inputs:
        obs            The observation, which isassumed to cover less than
                       half the sky. If either RA or DEC wraps around, the min
                       value will be greater than the max value.
        extend_fov_u   The amount of padding to add in the U dimension
        extend_fov_v   The amount of padding to add in the V dimension
        padding        Extra padding to add on each edge just to be safe
       
    Returns:
        ra_min, ra_max, dec_min, dec_max (radians)
    """
    logger = logging.getLogger(LOGGING_NAME+'.ra_dec_limits')
    
    # Create a meshgrid for the four corners of the image
    corner_meshgrid = make_corner_meshgrid(obs, extend_fov_u, extend_fov_v,
                                           padding)
    
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
