import cb_logging
import logging

import numpy as np

from psfmodel.gaussian import GaussianPSF
from imgdisp import draw_circle
from cb_util_oops import *
from cb_util_flux import *
from cb_util_star import *

LOGGING_NAME = 'cb.' + __name__


#===============================================================================
# 
#===============================================================================

MIN_DETECTABLE_DN = {'NAC': 10, 'WAC': 200}

NAC_SIGMA = 0.54

def _star_list_for_obs(star_catalog, obs, ra_min, ra_max, dec_min, dec_max,
                       mag_min, mag_max, **kwargs):
    logger = logging.getLogger(LOGGING_NAME+'._star_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
    star_list = [x for x in 
                 star_catalog.find_stars(allow_double=True,
                                         ra_min=ra_min, ra_max=ra_max,
                                         dec_min=dec_min, dec_max=dec_max,
                                         vmag_min=mag_min, vmag_max=mag_max,
                                         **kwargs)]

    star_list.sort(key=lambda x: x.vmag)
    
    new_star_list = []
    for star in star_list:
        if star.temperature is None:
            continue        
        star.dn = compute_dn_from_star(obs, star)
        if star.dn < min_dn:
            continue
        new_star_list.append(star)
        
    star_list = new_star_list
    
    ra_dec_list = [x.ra_dec_with_pm(obs.midtime) for x in star_list]
    ra_list = [x[0] for x in ra_dec_list]
    dec_list = [x[1] for x in ra_dec_list]
    
    uv = obs.uv_from_ra_and_dec(ra_list, dec_list)
    u_list, v_list = uv.to_scalars()
    u_list = u_list.vals
    v_list = v_list.vals

    new_star_list = []
        
    for star, u, v in zip(star_list, u_list, v_list):
        if (u < 0 or u > obs.data.shape[1]-1 or
            v < 0 or v > obs.data.shape[0]-1):
            continue
        
        if u < 460: continue
        
        star.u = u
        star.v = v

        logger.debug('Star %9d U %8.3f V %8.3f DN %7.2f MAG %6.3f BMAG %6.3f '+
                     'VMAG %6.3f SCLASS %3s TEMP %6d',
                     star.unique_number, star.u, star.v, star.dn, star.vmag,
                     0 if star.johnson_mag_b is None else star.johnson_mag_b,
                     0 if star.johnson_mag_v is None else star.johnson_mag_v,
                     'XX' if star.spectral_class is None else
                             star.spectral_class,
                     0 if star.temperature is None else star.temperature)

        new_star_list.append(star)
        
    return new_star_list

def star_list_for_obs(star_catalog, obs, num_stars=30, **kwargs):
    logger = logging.getLogger(LOGGING_NAME+'.star_list_for_obs')

    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs)

    # XXX THIS COULD BE MADE MUCH MORE EFFICIENT
    # PAY ATTENTION TO WHERE RA/DEC IS POINTING - SGR? OUT OF PLANE?
    # ESTIMATE MAG NEEDED ON FIRST TRY
    
    if obs.detector == 'NAC':
        magnitude_list = [0., 14., 14.3, 14.5, 14.7, 14.9]
    else:
        magnitude_list = np.log10(np.log10(np.arange(4.,30))*6)*13
        magnitude_list[0] = 0.
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        star_list = _star_list_for_obs(star_catalog, obs,
                                       ra_min, ra_max, dec_min, dec_max,
                                       mag_min=mag_min, mag_max=mag_max,
                                        **kwargs)
        full_star_list += star_list
        
        logger.debug('Got %d stars, total %d', len(star_list),
                     len(full_star_list))
        
        if len(full_star_list) >= num_stars:
            break
        
    if len(full_star_list) > num_stars:
        full_star_list = full_star_list[:num_stars]
        
    return full_star_list

    
def star_create_model(obs, star_list, psf_size=9,
                      star_offset_u=0., star_offset_v=0.,
                      verbose=False, **kwargs):
    model = np.zeros(obs.data.shape)
    
    gausspsf = GaussianPSF(sigma=NAC_SIGMA)
    
    vmag_list = [x.vmag for x in star_list]
    star_mag_min = min(vmag_list)
    star_mag_max = max(vmag_list)
    
    margin = psf_size//2
    
    for star in star_list:
        u = star.u
        v = star.v
        u += star_offset_u
        v += star_offset_v
        if (u < margin or u >= obs.data.shape[0]-margin or
            v < margin or v >= obs.data.shape[1]-margin):
            continue

        u_int = int(u)
        v_int = int(v)
        u_frac = u-u_int
        v_frac = v-v_int
        psf_scale = 2.512**(star_mag_min - star.vmag + 1)
        psf = gausspsf.eval_rect((psf_size,psf_size),
                                 offset=(v_frac,u_frac),
                                 scale=psf_scale)
        model[v-margin:v+margin+1, u-margin:u+margin+1] += psf
        #2.512**(star_mag_min - star_mag)
        
    return model


#def _number_of_stars_per_mag_bin():
#    from starcat import UCAC4StarCatalog
#    star_catalog = UCAC4StarCatalog('t:/external/ucac4')
#
#    mag_gran = 0.01
#    mag_min = 8
#    mag_max = 16
#    
#    vmag_bins = np.zeros(mag_max/mag_grand)
#    
#    for star in star_catalog.find_stars(require_clean=True,
#                                        vmag_max=mag_max,
#                                        dec_min=0,dec_max=0.1):
#        vmag = max(star.vmag-mag_min, 0.)
#        bin = int(star.vmag / mag_gran)
#        vmag_bins[bin] += 1
        
    