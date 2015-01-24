###############################################################################
# cb_config.py
#
# Global configuration parameters.  
###############################################################################

import os
import os.path

import oops

#####################
# ROOT DIRECTORIES ##
#####################

# Cassini ISS calibrated 2XXX volume.
COISS_2XXX_DERIVED_ROOT = os.environ['COISS_2XXX_DERIVED_ROOT']

# Cassini ISS non-calibrated volumes.
COISS_ROOT = os.environ['COISS_ROOT']

# CB-generated offset and backplane files.
RESULTS_ROOT = os.environ['CB_RESULTS_ROOT']

# The UCAC4 star catalog.
STAR_CATALOG_ROOT = os.environ['CB_STAR_CATALOG']

# CB support files such as Voyager ring profiles.
SUPPORT_FILES_ROOT = os.environ['CB_SUPPORT_ROOT']

# Cassini ISS moon maps.
COISS_3XXX_ROOT = os.path.join(COISS_ROOT, 'COISS_3xxx')

# Cassini UVIS.
COUVIS_8XXX_ROOT = os.path.join(COISS_ROOT, 'COUVIS_8xxx_lien_resolution')

# Contains solar flux, filter transmission convolved with quantum efficiency
CISSCAL_CALIB_ROOT = os.getenv('CISSCAL_CALIB_PATH')

# Contains filter transmission and PSF data
CASSINI_CALIB_ROOT  = os.getenv('CASSINI_CALIB_PATH')


########################
# INSTRUMENT CONSTANTS #
########################

# The maximum pointing error we allow in the (V,U) directions.
MAX_POINTING_ERROR = {'NAC': (85,75), 'WAC': (15,15)} # Pixels
#MAX_POINTING_ERROR = {'NAC': (10,10), 'WAC': (15,15)} # Pixels

# The FOV size of the ISS cameras in radians.
ISS_FOV_SIZE = {'NAC': 0.35*oops.RPD, 'WAC': 3.48*oops.RPD}

# The Gaussian sigma of the ISS camera PSFs in pixels.
ISS_PSF_SIGMA = {'NAC': 0.54, 'WAC': 0.77}


########################
# BODY CHARACTERISTICS #
########################

# These are bodies large enough to be picked up in an image.
LARGE_BODY_LIST = ['SATURN', 'PAN', 'DAPHNIS', 'ATLAS', 'PROMETHEUS',
                   'PANDORA', 'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                   'TETHYS', 'TELESTO', 'CALYPSO', 'DIONE', 'HELENE',
                   'RHEA', 'TITAN', 'HYPERION', 'IAPETUS', 'PHOEBE']

# These are bodies that shouldn't be used for navigation because they
# are "fuzzy" in some way or at least don't have a well-defined orientation.
FUZZY_BODY_LIST = ['TITAN', 'HYPERION'] # XXX


##################
# CONFIGURATIONS #
##################

STARS_DEFAULT_CONFIG = {
    # Minimum number of stars that much photometrically match for an offset
    # to be considered good.
    'min_stars': 3,

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # PSF size for modeling a star (must be odd).
    'psf_size': 9,
        
    # The default star class when none is available.
    'default_star_class': 'G0',
    
    # The minimum DN that is guaranteed to be visible in the image.
    'min_brightness_guaranteed_vis': 200.,

    # The minimum DN count for a star to be detectable. These values are pretty
    # aggressively dim - there's no guarantee a star with this brightness can
    # actually be seen.
    ('min_detectable_dn', 'NAC'): 20,
    ('min_detectable_dn', 'WAC'): 150,

    # The range of vmags to use when determining the dimmest star visible.
    'min_vmag': 5.,
    'max_vmag': 15.,
    'vmag_increment': 0.5,

    # The size of the box to analyze vs. the predicted integrated DN.
    # XXX These numbers are just made up, but seem to work.
    # More study should be done.
    'photometry_boxsize_1': (500,11),
    'photometry_boxsize_2': (100,9),
    'photometry_boxsize_default': 7,
    
    # How far a star has to be from a major body before it is no longer
    # considered to conflict.
    'star_body_conflict_margin': 3,
    
    # If star finding fails, get rid of the brightest star if it's at least
    # this bright.
    'too_bright_dn': 1000,
    
    # If star finding fails, get rid of the brightest star if it's at least
    # this much brighter than the next dimmest star.
    'too_bright_factor': None
}

BODIES_DEFAULT_CONFIG = {
    # The fraction of the width/height of a body that must be visible on either
    # side of the center in order for the curvature to be sufficient for 
    # correlation.
    'curvature_threshold_frac': 0.1,
    
    # The number of pixels of the width/height of a body that must be visible
    # on either side of the center in order for the curvature to be sufficient
    # for correlation. Both curvature_threshold_frac and 
    # curvature_threshold_pixels must be true for correlation. The _pixels
    # version is useful for the case of small moons.
    'curvature_threshold_pixels': 20,
}

RINGS_DEFAULT_CONFIG = {
    # The source for profile data - 'voyager' or 'uvis'.
    'model_source': 'uvis',
    
    # There must be at least this many fiducial features for correlation
    # to be done.
    'fiducial_feature_threshold': 3,
    
    # The number of pixels of curvature that must be present for correlation
    # to be done.
    'curvature_threshold': 2,
}

BOOTSTRAP_DEFAULT_COINFIG = {
}

