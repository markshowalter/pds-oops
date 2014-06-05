import cb_logging
import logging

import oops
import solar
import numpy as np

LOGGING_NAME = 'cb.' + __name__


#===============================================================================
# 
# IMAGE MANIPULATION
#
#===============================================================================

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset"""
    image = np.roll(image, -offset_u, 1)
    image = np.roll(image, -offset_v, 0)

    if offset_u != 0:    
        if offset_u < 0:
            image[:,:-offset_u] = 0
        else:
            image[:,-offset_u:] = 0
    if offset_v != 0:
        if offset_v < 0:
            image[:-offset_v,:] = 0
        else:
            image[-offset_v:,:] = 0
    
    return image


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
    """Compute the distance from the Sun to Saturn in AU"""
    target_sun_path = oops.Path.as_waypoint("SATURN").wrt('SUN')
    sun_event = target_sun_path.event_at_time(obs.midtime)
    solar_range = sun_event.pos.norm().vals / solar.AU

    return solar_range
