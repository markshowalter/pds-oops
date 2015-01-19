###############################################################################
# cb_util_flux.py
#
# Routines related to image flux calibration.
#
# Exported routines:
#    calibrate_iof_image_as_flux
#    calibrate_iof_image_as_dn
#
#    compute_dn_from_star
#
#    plot_cassini_filter_transmission
#    plot_johnson_filter_transmission
#    plot_planck_vs_solar_flux
###############################################################################

import cb_logging
import logging

import os

import numpy as np
import scipy.constants as const
import scipy.interpolate as interp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import oops

from cb_config import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__


#===============================================================================
# 
# FILTER CONVOLUTIONS
#
#===============================================================================

def _interpolate_and_convolve_2(x1, y1, x2, y2):
    """Convolve two tabulations and return the intersected interval."""
    min_x = max(np.min(x1), np.min(x2))
    max_x = min(np.max(x1), np.max(x2))
    new_x = np.arange(min_x, max_x+0.1)

    new_y1 = interp.interp1d(x1, y1)(new_x)
    new_y2 = interp.interp1d(x2, y2)(new_x)

    return new_x, new_y1*new_y2

def _interpolate_and_convolve_3(x1, y1, x2, y2, x3, y3):
    """Convolve three tabulations and return the intersected interval."""
    min_x = max(np.ceil(np.min(x1)), np.ceil(np.min(x2)), np.ceil(np.min(x3)))
    max_x = min(np.floor(np.max(x1)), np.floor(np.max(x2)), np.floor(np.max(x3)))
    new_x = np.arange(min_x, max_x+0.1)

    new_y1 = interp.interp1d(x1, y1)(new_x)
    new_y2 = interp.interp1d(x2, y2)(new_x)
    new_y3 = interp.interp1d(x3, y3)(new_x)

    return new_x, new_y1*new_y2*new_y3


#===============================================================================
# 
# CISSCAL-Related Functions
#
#===============================================================================

_CISSCAL_DETECTOR_GAIN = {'NAC': 30.27, 'WAC': 27.68}
_CISSCAL_DETECTOR_GAIN_RATIO = {'NAC': [0.135386, 0.309569, 1.0, 2.357285],
                                'WAC': [0.125446, 0.290637, 1.0, 2.360374]}
_CISSCAL_DETECTOR_SOLID_ANGLE = {'NAC': 3.6e-11, 'WAC': 3.6e-9} # Steradians
_CISSCAL_DETECTOR_OPTICS_AREA = {'NAC': 264.84, 'WAC': 29.32} # Aperture cm^2

# From the CISSCAL User Guide March 20, 2009 section 5.9
# Converting from DN to Flux:
#   1) dntoelectrons.pro
#          ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]
#   2) dividebyexpot.pro
#          We ignore shutter offset timing for our purposes
#          DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
#   3) dividebyareapixel.pro
#          SUM_FACTOR = (SAMPLES/1024) * (LINES/1024)
#          DATA = DATA * SUM_FACTOR / (SOLID_ANGLE * OPTICS_AREA)
#   4) dividebyefficiency.pro
#      This is Quantum Efficiency * Optics Transmission *
#             Filter 1 * Filter 2
#          DATA = DATA / INTEG(QE * TRANS)
#      In I/F mode:
#          FLUX = SOLAR_FLUX / (PI * DIST^2)
#          DATA = DATA / INTEG(QE * TRANS * FLUX)
# This yields a result in phot/cm^2/s/nm/ster

def _read_cisscal_calib_file(filename):
    """Read a CISSCAL calibration table."""
    logger = logging.getLogger(_LOGGING_NAME+'.read_cisscal_calib_file')
    logger.debug('Reading "%s"', filename)

    with open(filename, 'r') as fp:
        for line in fp:
            if line.startswith('\\begindata'):
                break
        else:
            assert False
        ret_list = []
        for line in fp:
            if line.startswith('\\enddata'):
                break
            fields = line.strip('\r\n').split()
            field_list = [float(x) for x in fields]
            ret_list.append(field_list)
    
    return ret_list


_CISSCAL_FILTER_TRANSMISSION_CACHE = {}

def _cisscal_filter_transmission(obs):
    """Return the (wavelengths, transmission) for the joint filters."""
    key = (obs.detector, obs.filter1, obs.filter2)
    if key not in _CISSCAL_FILTER_TRANSMISSION_CACHE:
        filter_filename = ('iss' + obs.detector.lower()[:2] + 
                           obs.filter1.lower())
        filter_filename += obs.filter2.lower() + '_systrans.tab'
        systrans_filename = os.path.join(CISSCAL_CALIB_ROOT, 'efficiency',
                                         'systrans', filter_filename)
        systrans_list = _read_cisscal_calib_file(systrans_filename)
        systrans_wl = [x[0] for x in systrans_list] # Wavelength in nm
        systrans_xmit = [x[1] for x in systrans_list]
    
        _CISSCAL_FILTER_TRANSMISSION_CACHE[key] = (systrans_wl, systrans_xmit)
    
    return _CISSCAL_FILTER_TRANSMISSION_CACHE[key]


_CISSCAL_QE_CORRECTION_CACHE = {}

def _cisscal_qe_correction(obs):
    """Return the (wavelengths, vals) for QE correction."""
    key = obs.detector
    if key not in _CISSCAL_QE_CORRECTION_CACHE:    
        qecorr_filename = os.path.join(
                               CISSCAL_CALIB_ROOT, 'correction',
                               obs.detector.lower()+'_qe_correction.tab')
        qecorr_list = _read_cisscal_calib_file(qecorr_filename)
        qecorr_wl = [x[0] for x in qecorr_list] # Wavelength in nm
        qecorr_val = [x[1] for x in qecorr_list]

        _CISSCAL_QE_CORRECTION_CACHE[key] = (qecorr_wl, qecorr_val)
    
    return _CISSCAL_QE_CORRECTION_CACHE[key]


_CISSCAL_SOLAR_FLUX_CACHE = None

def _cisscal_solar_flux():
    """Return the (wavelengths, flux) for the solar flux in phot/cm^2/s/nm at
    1 AU."""
    global _CISSCAL_SOLAR_FLUX_CACHE
    
    if _CISSCAL_SOLAR_FLUX_CACHE is None:    
        # Flux is in photons / cm^2 / s / angstrom at 1 AU
        solarflux_filename = os.path.join(CISSCAL_CALIB_ROOT, 'efficiency',
                                          'solarflux.tab')
        solarflux_list = _read_cisscal_calib_file(solarflux_filename)
        solarflux_wl = [x[0]/10. for x in solarflux_list] # Wavelength in nm
        solarflux_flux = [x[1]*10. for x in solarflux_list]
        # Flux is now in photons / cm^2 / s / nm at 1 AU
    
        _CISSCAL_SOLAR_FLUX_CACHE = (solarflux_wl, solarflux_flux)

    return _CISSCAL_SOLAR_FLUX_CACHE

#===============================================================================

def _compute_cisscal_efficiency(obs):
    """Compute the efficiency factor without solar flux."""
    # From cassimg__dividebyefficiency.pro

    logger = logging.getLogger(_LOGGING_NAME+'._compute_cisscal_efficiency')

    # Read in filter transmission
    systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)
    
    # Read in QE correction
    qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)

    min_wl = np.ceil(np.max([np.min(systrans_wl),
                             np.min(qecorr_wl)]))
    max_wl = np.floor(np.min([np.max(systrans_wl),
                              np.max(qecorr_wl)]))

    all_wl = systrans_wl + qecorr_wl
    all_wl = list(set(all_wl)) # uniq
    all_wl.sort()
    all_wl = np.array(all_wl)
    all_wl = all_wl[all_wl >= min_wl]
    all_wl = all_wl[all_wl <= max_wl]

    new_trans = interp.interp1d(systrans_wl, systrans_xmit)(all_wl)
    new_qe = interp.interp1d(qecorr_wl, qecorr_val)(all_wl)
    
    # Note the original IDL code uses 5-point Newton-Coates while
    # we only use 3-point. This really shouldn't make any difference
    # for our purposes.
    eff_fact = integrate.simps(new_trans*new_qe, all_wl)

    logger.debug('w/o solar flux wavelength range %f to %f, eff factor %f',
                 min_wl, max_wl, eff_fact)

    return eff_fact

def _compute_cisscal_solar_flux_efficiency(obs):
    """Compute the efficiency factor including solar flux."""
    # From cassimg__dividebyefficiency.pro

    logger = logging.getLogger(_LOGGING_NAME+
                               '._compute_cisscal_solar_flux_efficiency')
    
    # Read in filter transmission
    systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)
    
    # Read in QE correction
    qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)
    
    # Read in solar flux
    solarflux_wl, solarflux_flux = _cisscal_solar_flux()
    
    # Compute distance Sun-Saturn
    solar_range = compute_sun_saturn_distance(obs)

    logger.debug('Solar range = %f AU', solar_range)

    # We do the convolutions in this particular manner because it's the
    # way CISSCAL does it and we are trying to get as precise a result
    # as we can while undoing CISSCAL's computations. 
    min_wl = np.ceil(max(np.min(systrans_wl),
                         np.min(qecorr_wl),
                         np.min(solarflux_wl)))
    max_wl = np.floor(min(np.max(systrans_wl),
                          np.max(qecorr_wl),
                          np.max(solarflux_wl)))

    all_wl = systrans_wl + qecorr_wl + solarflux_wl
    all_wl = list(set(all_wl)) # uniq
    all_wl.sort()
    all_wl = np.array(all_wl)
    all_wl = all_wl[all_wl >= min_wl]
    all_wl = all_wl[all_wl <= max_wl]

    new_trans = interp.interp1d(systrans_wl, systrans_xmit)(all_wl)
    new_qe = interp.interp1d(qecorr_wl, qecorr_val)(all_wl)
    new_flux = interp.interp1d(solarflux_wl, solarflux_flux)(all_wl)
    new_flux /= (oops.PI * solar_range**2)
    # Flux is now in photons / cm^2 / s / nm at Saturn's distance
    # Dividing by pi is necessary because Solar Flux = pi F in I/F
    # thus I/F = I/ [Solar flux / pi]
    
    # Note the original IDL code uses 5-point Newton-Coates while
    # we only use 3-point. This really shouldn't make any difference
    # for our purposes.
    eff_fact = integrate.simps(new_trans*new_qe*new_flux, all_wl)
    
    logger.debug('w/solar flux wavelength range %f to %f, eff factor %f',
                 min_wl, max_wl, eff_fact)

    return eff_fact

#===============================================================================

_IOF_FLUX_CONVERSION_FACTOR_CACHE = {}

def calibrate_iof_image_as_flux(obs):
    """Convert an image currently in I/F to flux.
    
    The input observation data is in I/F.
    The output data is in phot/cm^2/s/nm/ster.
    """
    # We undo step 4 and then redo it with no stellar flux

    logger = logging.getLogger(_LOGGING_NAME+'.calibrate_iof_image_as_flux')

    key = (obs.detector, obs.filter1, obs.filter2)
    if key in _IOF_FLUX_CONVERSION_FACTOR_CACHE:
        factor = _IOF_FLUX_CONVERSION_FACTOR_CACHE[key]
        logger.debug('Calibration for %s %s %s cached; factor = %f',
                     obs.detector, obs.filter1, obs.filter2, factor)
        return obs.data * factor

    logger.debug('Calibrating %s %s %s', 
                 obs.detector, obs.filter1, obs.filter2)

    # Undo Step 4 by multiplying by system transmission
    # efficiency including solar flux
    factor = _compute_cisscal_solar_flux_efficiency(obs)

    # Redo Step 4 by dividing by system transmission efficiency
    # excluding solar flux
    factor /= _compute_cisscal_efficiency(obs)

    _IOF_FLUX_CONVERSION_FACTOR_CACHE[key] = factor

    logger.debug('Final adjustment factor = %e', factor)
            
    return obs.data * factor


_IOF_DN_CONVERSION_FACTOR_CACHE = {}

def calibrate_iof_image_as_dn(obs, data=None):
    """Convert an image currently in I/F to post-LUT raw DN.
    
    The input observation data is in I/F.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.calibrate_iof_image_as_dn')

    if data is None:
        # Can be overriden if we want to calibrate some other data block
        data = obs.data
        
    key = (obs.detector, obs.filter1, obs.filter2, obs.texp)
    if key in _IOF_DN_CONVERSION_FACTOR_CACHE:
        factor = _IOF_DN_CONVERSION_FACTOR_CACHE[key]
        logger.debug('Calibration for %s %s %s %.2f cached; factor = %f',
                     obs.detector, obs.filter1, obs.filter2, obs.texp, factor)
        return data * factor

    logger.debug('Calibrating %s %s %s', 
                 obs.detector, obs.filter1, obs.filter2)
    
    # 4) dividebyefficiency.pro
    #        FLUX = SOLAR_FLUX / (PI * DIST^2)
    #        DATA = DATA / INTEG(QE * TRANS * FLUX)
    factor = _compute_cisscal_solar_flux_efficiency(obs)
    # Image data now in photons / cm^2 / s / nm assuming no filters or QE
    # correction

    # 3) dividebyareapixel.pro
    #        SUM_FACTOR = (SAMPLES/1024) * (LINES/1024)
    #        DATA = DATA * SUM_FACTOR / (SOLID_ANGLE * OPTICS_AREA)
    # Use obs.data not data here because we want the real size of the original
    # image.
    sum_factor = obs.data.shape[0] / 1024. * obs.data.shape[1] / 1024.
    
    factor = (factor / sum_factor *
              (_CISSCAL_DETECTOR_SOLID_ANGLE[obs.detector] *
               _CISSCAL_DETECTOR_OPTICS_AREA[obs.detector]))
    # photons / s
    
    # 2) dividebyexpot.pro
    #        We ignore shutter offset timing for our purposes
    #        DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
    factor *= obs.texp # texp is already in sec
    # photons
    
    # 1) dntoelectrons.pro
    #        ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]
    factor = (factor / _CISSCAL_DETECTOR_GAIN[obs.detector] *
              _CISSCAL_DETECTOR_GAIN_RATIO[obs.detector][obs.gain_mode])

    _IOF_DN_CONVERSION_FACTOR_CACHE[key] = factor

    logger.debug('Final adjustment factor = %e', factor)
            
    return data * factor

#===============================================================================
# 
# CASSINI FILTER TRANSMISSION FUNCTIONS
#
#===============================================================================

_CASSINI_FILTER_TRANSMISSION = {} 

def _cassini_filter_transmission(detector, filter):
    """Return the (wavelengths, transmission) for the given Cassini filter."""

    logger = logging.getLogger(_LOGGING_NAME+
                               '.cassini_filter_transmission')

    if len(_CASSINI_FILTER_TRANSMISSION) == 0:
        for iss_det in ['NAC', 'WAC']:
            base_dirname = iss_det[0].lower() + '_c_trans_sum'
            filename = os.path.join(CASSINI_CALIB_ROOT, base_dirname,
                                    'all_filters.tab') 
            logger.debug('Reading "%s"', filename)
            with open(filename, 'r') as filter_fp:
                header = filter_fp.readline().strip('\r\n')
                header_fields = header.split('\t')
                assert header_fields[0] == 'WAVELENGTH (nm)'
                filter_name_list = []
                for i in xrange(1, len(header_fields)):
                    filter_name = header_fields[i]
                    # For unknown reasons the headers of the NAC and WAC files are
                    # formatted differently. They also have weird names for the
                    # polarized filters.
                    if filter_name[:7] == 'VIS POL': # NAC
                        filter_name = 'P0'
                    elif filter_name[:6] == 'IR POL': # NAC
                        filter_name = 'IRP0'
                    elif filter_name[:6] == 'IR_POL': # WAC
                        filter_name = 'IRP0'
                    elif filter_name[0].isdigit():
                        filter_name = filter_name[-3:]
                    else:
                        filter_name = filter_name[:3]
                    
                    filter_name_list.append(filter_name)
                
                for i in xrange(len(filter_name_list)):
                    key = (iss_det, filter_name_list[i])
                    _CASSINI_FILTER_TRANSMISSION[key] = ([], []) # wl, xmission
                for filter_line in filter_fp:
                    filter_line = filter_line.strip('\r\n')
                    filter_fields = filter_line.split('\t')
                    assert len(filter_fields) == len(filter_name_list)+1
                    if len(filter_fields[0]) == 0:
                        continue
                    wl = float(filter_fields[0])
                    for i in xrange(len(filter_name_list)):
                        filter_field = filter_fields[i+1].strip('\r\n')
                        if len(filter_field) > 0:
                            key = (iss_det, filter_name_list[i])
                            xmission = float(filter_field)
                            _CASSINI_FILTER_TRANSMISSION[key][0].append(wl)
                            _CASSINI_FILTER_TRANSMISSION[key][1].append(xmission)
    
        pol = _CASSINI_FILTER_TRANSMISSION[('NAC', 'P0')]
        _CASSINI_FILTER_TRANSMISSION[('NAC', 'P60')] = pol 
        _CASSINI_FILTER_TRANSMISSION[('NAC', 'P120')] = pol 
        pol = _CASSINI_FILTER_TRANSMISSION[('WAC', 'IRP0')]
        _CASSINI_FILTER_TRANSMISSION[('NAC', 'IRP90')] = pol 

    return _CASSINI_FILTER_TRANSMISSION[(detector, filter)]


def plot_cassini_filter_transmission():
    """Plot the Cassini filter transmission functions."""
    
    color_info = {
        'CL1': ('#000000', '-'),
        'CL2': ('#808080', '-'),
        'BL1': ('#4040a0', '-'),
        'BL2': ('#0000ff', '-'),
        'UV1': ('#800080', '-'),
        'UV2': ('#c000c0', '-'),
        'UV3': ('#ff00ff', '-'),
        'GRN': ('#00ff00', '-'),
        'RED': ('#ff0000', '-'),
        'VIO': ('#000040', '-'),
        'IR1': ('#ff0000', '--'),
        'IR2': ('#ff4040', '--'),
        'IR3': ('#ff8080', '--'),
        'IR4': ('#ffa0a0', '--'),
        'IR5': ('#ffc0c0', '--'),
        
        'MT1': ('#008080', ':'),
        'MT2': ('#00c0c0', ':'),
        'MT3': ('#00ffff', ':'),
        'CB1': ('#400040', ':'),
        'CB2': ('#408080', ':'),
        'CB3': ('#4080ff', ':'),

        'HAL': ('#ff8080', ':'),
        
        'P0':  ('#404040', '--'),
        'P60': ('#808080', '--'),
        'P120':('#c0c0c0', '--'),
        'IRP0':('#404080', '--'),
        'IRP90':('#4040c0', '--')
    }
    
    cassini_filter_transmission('NAC', 'CL1') # This reads all filters
    for detector in ['NAC', 'WAC']:
        fig = plt.figure()
        plt.title(detector)
        for key in sorted(_CASSINI_FILTER_TRANSMISSION.keys()):
            filter_det, filter_name = key
            if filter_det != detector:
                continue
            if len(filter_name) == 1: # Bogus single-letter filters
                continue
            wl_list, xmission_list = _CASSINI_FILTER_TRANSMISSION[key]
            plt.plot(wl_list, xmission_list, color_info[filter_name][1],
                     color=color_info[filter_name][0], label=filter_name)
        plt.legend()
    plt.show()


#===============================================================================
# 
# STANDARD PHOTOMETRIC FILTER TABLES
#
#===============================================================================

# From Bessel 1990
_JOHNSON_B_WL = np.arange(360.,561.,10)
_JOHNSON_B = np.array([
    0.000, 0.030, 0.134, 0.567, 0.920, 0.978, 1.000, 0.978, 0.935, 0.853, 0.740,
    0.640, 0.536, 0.424, 0.325, 0.235, 0.150, 0.095, 0.043, 0.009, 0.000])

_JOHNSON_V_WL = np.arange(470.,701.,10)
_JOHNSON_V = np.array([
    0.000, 0.030, 0.163, 0.458, 0.780, 0.967, 1.000, 0.973, 0.898, 0.792, 0.684,
    0.574, 0.461, 0.359, 0.270, 0.197, 0.135, 0.081, 0.045, 0.025, 0.017, 0.013,
    0.009, 0.000])

def plot_johnson_filter_transmission():
    """Plot the Johnson B and V filter transmission functions"""
    fig = plt.figure()
    plt.plot(_JOHNSON_B_WL, _JOHNSON_B, '-', color='blue', label='B')
    plt.plot(_JOHNSON_V_WL, _JOHNSON_V, '-', color='green', label='V')
    plt.legend()
    plt.show()
    

#===============================================================================
# 
# OPERATIONS ON STELLAR SPECTRA
#
#===============================================================================

def _compute_planck_curve(wavelength, T):
    """Compute the Planck spectral radiance.
    
    Wavelength is in nm. Temperature is in K.
    Result is in photons / cm^2 / s / nm / steradian.
    """
    wavelength = np.asarray(wavelength) * 1e-9
    
    return (2*const.c/
            (wavelength**4.*(np.exp(const.h*const.c/
                                    (wavelength*const.k*T))-1.))) * 1e-9

def plot_planck_vs_solar_flux():
    """Plot a scale Planck curve vs. the solar flux."""
    
    solarflux_wl, solarflux_flux = _cisscal_solar_flux()

    planck_flux = _compute_planck_curve(solarflux_wl, 5778)
    # Angular size of Sun at 1 AU
    planck_flux *= oops.PI * (0.52/2 * oops.RPD) ** 2
    # The absolute flux seems to be off by a factor of ~10,000
    # No idea why. It doesn't matter, though, since we treat
    # it as dimensionless in other parts of the code.
    #    planck_flux /= 10000

    scale_factor = np.sum(solarflux_flux) / np.sum(planck_flux)
    
    fig = plt.figure()
    plt.plot(solarflux_wl, solarflux_flux, '-', color='red', lw=2.5, 
             label='Sun')
    plt.plot(solarflux_wl, planck_flux*scale_factor, '-', lw=2.5, color='blue',
             label='Planck')
    wl, solarflux_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
                                                  solarflux_wl, solarflux_flux)
    wl, planck_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
                                               solarflux_wl, planck_flux)

    scale_factor = np.mean(solarflux_v) / np.mean(planck_v)
        
    plt.plot(wl, solarflux_v, '--', color='red', lw=1, label='Sun w/V')
    plt.plot(wl, planck_v*scale_factor, '--', color='blue', lw=1,
             label='Planck w/V')

    wl, solarflux_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
                                                  solarflux_wl, solarflux_flux)
    wl, planck_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
                                               solarflux_wl, planck_flux)

    scale_factor = np.mean(solarflux_b) / np.mean(planck_b)
        
    plt.plot(wl, solarflux_b, ':', color='red', lw=1, label='Sun w/B')
    plt.plot(wl, planck_b*scale_factor, ':', color='blue', lw=1,
             label='Planck w/B')

    plt.legend()
    plt.title('Planck vs. Solar Flux')
        
    plt.show()

def _v_magnitude_to_photon_flux(v, detector):
    """Return the V-band photon flux for a star with the given Johnson V
    magnitude.
    
    detector is the Cassini camera ('NAC' or 'WAC').
    
    Returned value is in photons / cm^2 / s
    """
    # http://www.astro.umd.edu/~ssm/ASTR620/mags.html#flux
    # From Bessel, M. S. 2005, ARA&A, 43, 293 
    # V band flux at m = 0: 3640
    # V band dlambda/lambda = 0.16

    # Jansky = 1.51e3 photons / cm^2 / s / (dlambda/lambda)
    jy = 3640. * 10**(-0.4*v)
    
    # flux in photons / cm^2 / s
    flux = jy * 1.51e3 * 0.16
    
    return flux

def _b_magnitude_to_photon_flux(b, detector):
    """Return the V-band photon flux for a star with the given Johnson B
    magnitude.
    
    detector is the Cassini camera ('NAC' or 'WAC').
    
    Returned value is in photons / cm^2 / s
    """
    # http://www.astro.umd.edu/~ssm/ASTR620/mags.html#flux
    # From Bessel, M. S. 2005, ARA&A, 43, 293 
    # B band flux at m = 0: 4260
    # B band dlambda/lambda = 0.22

    # Jansky = 1.51e3 photons / cm^2 / s / (dlambda/lambda)
    jy = 4260. * 10**(-0.4*b)
    
    # flux in photons / cm^2 / s
    flux = jy * 1.51e3 * 0.22
    
    return flux

def _compute_stellar_spectrum(obs, star):
    """Compute the stellar spectrum for a given star.
    
    Returned value is in photons / cm^2 / s
    """
    
    logger = logging.getLogger(_LOGGING_NAME+'.compute_stellar_spectrum')

    # Planck is in photons / cm^2 / s / nm / steradian
    # However, it might as well be photons / cm^2 / s / nm because we're just
    # going to scale it later
    planck_wl = np.arange(100., 1600.)
    planck = _compute_planck_curve(planck_wl, star.temperature)
    
    wl_new, planck_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
                                                   planck_wl, planck)
    # Total photons seen through V filter 
    planck_v_sum = np.sum(planck_v)
    # Predicted photons seen through V - photons / cm^2 / s
    predicted_v = _v_magnitude_to_photon_flux(star.johnson_mag_v,
                                              obs.detector)
#    logger.debug('Star %9d Temp %9.2f Predicted V-band total flux %e', 
#                 star.unique_number, star.temperature, predicted_v)
    scale_factor_v = predicted_v / planck_v_sum

    wl_new, planck_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
                                                   planck_wl, planck)
    # Total photons seen through B filter 
    planck_b_sum = np.sum(planck_b)
    # Predicted photons seen through V - photons / cm^2 / s
    predicted_b = _b_magnitude_to_photon_flux(star.johnson_mag_b,
                                              obs.detector)
#    logger.debug('Star %9d Temp %9.2f Predicted V-band total flux %e', 
#                 star.unique_number, star.temperature, predicted_v)
    scale_factor_b = predicted_b / planck_b_sum

    logger.debug('Star %9d Temp %9.2f Scale V %e Scale B %e', 
                 star.unique_number, star.temperature, scale_factor_v,
                 scale_factor_b)

    # Return is in photons / cm^2 / s
    return planck_wl, planck*scale_factor_v

def _compute_dn_from_spectrum(obs, spectrum_wl, spectrum):
    """Compute the original DN expected from a given spectrum.
    
    The spectrum is in photons / cm^2 / s / nm
    """

    logger = logging.getLogger(_LOGGING_NAME+'._compute_dn_from_spectrum')

    # Read in filter transmission
    systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)
    
    # Read in QE correction
    qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)
 
    conv_wl, conv_flux = _interpolate_and_convolve_3(
                             systrans_wl, systrans_xmit, qecorr_wl, qecorr_val,
                             spectrum_wl, spectrum)

    conv_flux_sum = integrate.simps(conv_flux, conv_wl)
    # photons / cm^2 / s

    logger.debug('Total flux through %s+%s = %e',
                 obs.filter1, obs.filter2, conv_flux_sum)

    if False: # Make True to compare flux with a CISSCAL-calibrated file
        weights_wl, weights = _interpolate_and_convolve_2(systrans_wl,
                                                          systrans_xmit,
                                                          qecorr_wl,
                                                          qecorr_val)
        assert conv_wl[0] == weights_wl[0] and conv_wl[-1] == weights_wl[-1]
    
        conv_flux_avg = conv_flux_sum / integrate.simps(weights, weights_wl)
        conv_flux_avg /= CISSCAL_DETECTOR_SOLID_ANGLE[obs.detector]
        logger.debug('Total flux through %s+%s = %e /nm/sr',
                     obs.filter1, obs.filter2, conv_flux_avg)
    
    # 3) dividebyareapixel.pro
    #        SUM_FACTOR = (SAMPLES/1024) * (LINES/1024)
    #        DATA = DATA * SUM_FACTOR / (SOLID_ANGLE * OPTICS_AREA)
    # We don't need to divide by SOLID_ANGLE because we are assuming
    # an integrated source object. 
    sum_factor = obs.data.shape[0] / 1024. * obs.data.shape[1] / 1024.
    
    data = (conv_flux_sum / sum_factor *
            _CISSCAL_DETECTOR_OPTICS_AREA[obs.detector])
    # photons / s
    
    # 2) dividebyexpot.pro
    #        We ignore shutter offset timing for our purposes
    #        DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
    electrons = data * obs.texp # texp is already in sec
    # photons
    
    # 1) dntoelectrons.pro
    #        ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]
    dn = (electrons / _CISSCAL_DETECTOR_GAIN[obs.detector]
                    * _CISSCAL_DETECTOR_GAIN_RATIO[obs.detector][obs.gain_mode])
    
    logger.debug('Returned DN = %f', dn)
    
    return dn

def compute_dn_from_star(obs, star):
    """Compute the theoretical integrated DN for a star."""
    spectrum_wl, spectrum = _compute_stellar_spectrum(obs, star)
    dn = _compute_dn_from_spectrum(obs, spectrum_wl, spectrum)

    return dn
