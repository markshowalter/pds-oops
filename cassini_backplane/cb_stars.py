import cb_logging
import logging

import numpy as np
import numpy.ma as ma

from psfmodel.gaussian import GaussianPSF
from imgdisp import draw_circle
from starcat import UCAC4StarCatalog
from cb_config import STAR_CATALOG_FILENAME, ISS_PSF_SIGMA
from cb_util_oops import *
from cb_util_flux import *


STAR_CATALOG = UCAC4StarCatalog(STAR_CATALOG_FILENAME)

LOGGING_NAME = 'cb.' + __name__

STAR_MIN_CONFIDENCE = 0.65

#===============================================================================
# 
#===============================================================================

MIN_DETECTABLE_DN = {'NAC': 60, 'WAC': 200}

def _star_list_for_obs(obs, ra_min, ra_max, dec_min, dec_max,
                       mag_min, mag_max, margin, **kwargs):
    logger = logging.getLogger(LOGGING_NAME+'._star_list_for_obs')

    logger.debug('Mag range %7.4f to %7.4f', mag_min, mag_max)
    
    min_dn = MIN_DETECTABLE_DN[obs.detector]
    
    orig_star_list = [x for x in 
              STAR_CATALOG.find_stars(allow_double=True,
                                      ra_min=ra_min, ra_max=ra_max,
                                      dec_min=dec_min, dec_max=dec_max,
                                      vmag_min=mag_min, vmag_max=mag_max,
                                      **kwargs)]

    star_list = []
    for star in orig_star_list:
        if star.temperature is None:
            continue
        if star.spectral_class[0] == 'M':
            continue
        star.dn = compute_dn_from_star(obs, star)
        if star.dn < min_dn:
            continue
        star_list.append(star)
        
    ra_dec_list = [x.ra_dec_with_pm(obs.midtime) for x in star_list]
    ra_list = [x[0] for x in ra_dec_list]
    dec_list = [x[1] for x in ra_dec_list]
    
    uv = obs.uv_from_ra_and_dec(ra_list, dec_list)
    u_list, v_list = uv.to_scalars()
    u_list = u_list.vals
    v_list = v_list.vals

    new_star_list = []
        
    for star, u, v in zip(star_list, u_list, v_list):
        if (u < margin or u > obs.data.shape[1]-margin-1 or
            v < margin or v > obs.data.shape[0]-margin-1):
            continue
        
        star.u = u
        star.v = v

        new_star_list.append(star)

    return new_star_list

def star_list_for_obs(obs, num_stars=30, psf_size=9, **kwargs):
    """Return a list of stars in the FOV of the obs.

    Inputs:
        obs                The observation.
        num_stars          The maximum number of stars to return.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd. This is just used to restrict
                           the list of stars to ones not too close to the edge
                           of the image.
        **kwargs           Passed to find_stars to restrict the types of stars
                           returned.
                           
    Returns:
        star_list          The list of Star objects.
    """
    logger = logging.getLogger(LOGGING_NAME+'.star_list_for_obs')

    margin = psf_size//2
    
    ra_min, ra_max, dec_min, dec_max = compute_ra_dec_limits(obs)

    # XXX THIS COULD BE MADE MUCH MORE EFFICIENT
    # PAY ATTENTION TO WHERE RA/DEC IS POINTING - SGR? OUT OF PLANE?
    # ESTIMATE MAG NEEDED ON FIRST TRY
    
    magnitude_list = [0., 14., 14.3, 14.5, 14.7, 14.9] #, 15.1, 15.3, 15.5, 15.7, 15.9, 16.1, 16.3, 16.5, 16.7, 16.9]
    
    full_star_list = []
    
    for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
        star_list = _star_list_for_obs(obs,
                                       ra_min, ra_max, dec_min, dec_max,
                                       mag_min=mag_min, mag_max=mag_max,
                                       margin=margin, **kwargs)
        full_star_list += star_list
        
        logger.debug('Got %d stars, total %d', len(star_list),
                     len(full_star_list))
        
        if len(full_star_list) >= num_stars:
            break

    full_star_list.sort(key=lambda x: x.dn, reverse=True)
            
    if len(full_star_list) > num_stars:
        full_star_list = full_star_list[:num_stars]

    logger.debug('Returned star list:')
    for star in full_star_list:
        logger.debug('Star %9d U %8.3f V %8.3f DN %7.2f MAG %6.3f BMAG %6.3f '+
                     'VMAG %6.3f SCLASS %3s TEMP %6d',
                     star.unique_number, star.u, star.v, star.dn, star.vmag,
                     0 if star.johnson_mag_b is None else star.johnson_mag_b,
                     0 if star.johnson_mag_v is None else star.johnson_mag_v,
                     'XX' if star.spectral_class is None else
                             star.spectral_class,
                     0 if star.temperature is None else star.temperature)
        
    return full_star_list
    
def star_create_model(obs, star_list, psf_size=9, offset_u=0., offset_v=0.):
    """Create a model containing nothing but stars.
    
    Inputs:
        obs                The observation.
        star_list          The list of Stars.
        psf_size           The total width and height of the PSF used to model
                           a star. Must be odd.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        
    Returns:
        model              The model.
    """
    model = np.zeros(obs.data.shape)
    
    gausspsf = GaussianPSF(sigma=ISS_PSF_SIGMA[obs.detector])
    
    vmag_list = [x.vmag for x in star_list]
    star_mag_min = min(vmag_list)
    star_mag_max = max(vmag_list)
    
    margin = psf_size//2
    
    for star in star_list:
        u = star.u
        v = star.v
        u += offset_u
        v += offset_v
        if (u < margin or u >= obs.data.shape[0]-margin or
            v < margin or v >= obs.data.shape[1]-margin):
            continue

        u_int = int(u)
        v_int = int(v)
        u_frac = u-u_int
        v_frac = v-v_int
        psf = gausspsf.eval_rect((psf_size,psf_size),
                                 offset=(v_frac,u_frac),
                                 scale=star.dn)
        model[v-margin:v+margin+1, u-margin:u+margin+1] += psf
        
    return model


def star_perform_photometry(image, star_list, offset_u=0, offset_v=0,
                            confidence_sigma=0.5, max_bkgnd=100,
                            min_confidence=STAR_MIN_CONFIDENCE,
                            fit_psf=False):
    """Perform photometry on a list of stars.
    
    Inputs:
        image              The 2-D image used to do the photometry.
        star_list          The list of Star objects.
        offset_u           The amount to offset a star's position in the U,V
        offset_v           directions.
        confidence_sigma
        max_bkgnd
        min_confidence
        fit_spf
    
    Returns:
        good_stars, confidence
        
        good_stars         The number of good stars.
        confidence         The aggregate confidence.
        
        Each Star is populated with:
        
        integrated_dn
        photometry_confidence
    """
    logger = logging.getLogger(LOGGING_NAME+'.star_perform_photometry')

#    logger.debug('OFFSET U,V %d %d BOXSIZE %d to %d MINCONF %f FITPSF %d',
#                 offset_u, offset_v, min_boxsize, max_boxsize, min_confidence,
#                 fit_psf)

    if fit_psf:
        gausspsf = GaussianPSF(sigma=0.54) # XXX

    for star in star_list:
        u = int(np.round(star.u)) + offset_u
        v = int(np.round(star.v)) + offset_v

        if star.dn > 100:
            min_boxsize = 5
            max_boxsize = 7
        else:
            min_boxsize = 3
            max_boxsize = 5
        
        star.integrated_dn = -1
        star.photometry_confidence = 0.

        for boxsize in xrange(min_boxsize, max_boxsize+1, 2):
            box_halfsize = boxsize // 2

            if (u < box_halfsize or
                u > image.shape[1]-box_halfsize-1 or
                v < box_halfsize or
                v > image.shape[0]-box_halfsize-1):
                star.integrated_dn = -1
                star.photometry_confidence = 0.
                break
                
            
            if fit_psf:
                assert False # XXX
                if boxsize < 5:
                    continue
                ret = gausspsf.find_position(image, (boxsize,boxsize), (v,u), 
                                             search_limit=(1.,1.),
                                             bkgnd_degree=2,
                                             bkgnd_ignore_center=(1,1),
                                             tolerance=1e-4, num_sigma=5,
                                             bkgnd_num_sigma=5)
                if ret is None:
                    continue
                
                new_v, new_u, metadata = ret
                star.integrated_dn = metadata['scale']
            else:
                subimage = image[v-box_halfsize:v+box_halfsize+1,
                                 u-box_halfsize:u+box_halfsize+1]
                subimage = subimage.view(ma.MaskedArray)
                subimage[1:-1, 1:-1] = ma.masked # Mask out the center
                
                bkgnd = ma.mean(subimage)
                bkgnd_std = ma.std(subimage)
                # Standard deviation of the mean - How far off do we think
                # our measurement of the mean background is?
                bkgnd_std_mean = bkgnd_std / np.sqrt(ma.count(subimage))
                bkgnd_peak = ma.max(subimage)
                
                subimage.mask = ~subimage.mask # Mask out the edge
                
                integrated_dn = np.sum(subimage-bkgnd)
                peak_dn = np.max(subimage)
                peak_dn_bkgnd = np.max(subimage-bkgnd)
                
                int_dn_lo = np.sum(subimage - bkgnd - bkgnd_std_mean*confidence_sigma)
                int_dn_hi = np.sum(subimage - bkgnd + bkgnd_std_mean*confidence_sigma)
            
            confidence = 0.

            lo_ratio = 1e38
            hi_ratio = 1e38
            
            if int_dn_lo != 0 and int_dn_hi != 0:
                lo_ratio = star.dn / int_dn_lo
                if lo_ratio > 1: lo_ratio = 1. / lo_ratio
                hi_ratio = star.dn / int_dn_hi
                if hi_ratio > 1: hi_ratio = 1. / hi_ratio
            
            if integrated_dn > 0 and lo_ratio > 0 and hi_ratio > 0:
                # This eliminates stars behind bright objects like rings
                # or moons
                if bkgnd <= max_bkgnd:
                    confidence += 0.3
                if peak_dn > bkgnd_peak:
                    confidence += 0.1
                if peak_dn_bkgnd > bkgnd_std*confidence_sigma:
                    confidence += 0.1
                confidence += 0.5 * lo_ratio
#                confidence += 0.25 * hi_ratio
                
#                logger.debug('Star %9d %2s UV %4d %4d BOX %2d PREDICTED %9.3f MEASURED %9.3f CONF %5.3f',
#                             star.unique_number, star.spectral_class,
#                             u-offset_u, v-offset_v, boxsize, 
#                             star.dn, star.integrated_dn, confidence)

            print 'Star %9d %2s UV %4d %4d BOX %2d PRED DN %7.2f DNHI %9.2f DNLO %9.2f RHI %9.3f RLO %7.3f BKGND %7.2f BSIGMA %7.3f BSIGMAMEAN %7.3f PEAK %7.2f CONF %5.3f' % (
                star.unique_number, star.spectral_class,
                u-offset_u, v-offset_v, boxsize, star.dn,
                int_dn_hi, int_dn_lo, hi_ratio, lo_ratio, bkgnd, bkgnd_std, bkgnd_std_mean, peak_dn, confidence)
            
            if confidence > star.photometry_confidence:           
                star.photometry_confidence = confidence
                star.integrated_dn = integrated_dn 
    
        logger.debug('Star %9d %2s UV %4d %4d FINAL PREDICTED %9.3f MEASURED %9.3f CONF %5.3f',
                     star.unique_number, star.spectral_class,
                     u-offset_u, v-offset_v, 
                     star.dn, star.integrated_dn, star.photometry_confidence)

    good_stars = 0
    confidence_list = []
    for star in star_list:
        if star.photometry_confidence > 0:
            confidence_list.append(star.photometry_confidence)
        if star.photometry_confidence > min_confidence:
            good_stars += 1
            
    if good_stars < 2:
        confidence = 0
    else:
        confidence_list = np.array(confidence_list)
        confidence = np.mean(confidence_list)
        
#    logger.debug('OFFSET U,V %d %d Number of high confidence stars = %d', 
#                 offset_u, offset_v, good_stars)
            
    return good_stars, confidence

def star_make_good_bad_overlay(image, star_list,
                               min_confidence=STAR_MIN_CONFIDENCE):
    """Create an overlay with high and low confidence stars marked.
    
    Inputs:
        image              The 2-D image used to determine the size of the
                           overlay.
        star_list          The list of Star objects.
        min_confidence     The minimum confidence for a star to be considered
                           good.
                           
    Returns:
        overlay            The overlay.
    """
    overlay = np.zeros(data.shape+(3,))
    
    for star in star_list:
        u = int(np.round(star.u))
        v = int(np.round(star.v))
        
        if star.photometry_confidence > min_confidence:
            color = (0,1,0)
        else:
            color = (1,0,0)
        draw_circle(overlay, u, v, 1, color, 3)

    return overlay
