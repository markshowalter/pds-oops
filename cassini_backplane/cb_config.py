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
FUZZY_BODY_LIST = ['HYPERION', 'PHOEBE', 'TITAN'] # XXX

# These bodies can be used for bootstrapping.
BOOTSTRAP_BODY_LIST = ['DIONE', 'ENCELADUS', 'IAPETUS', 'MIMAS', 'PHOEBE',
                       'RHEA', 'TETHYS']

##################
# CONFIGURATIONS #
##################

STARS_DEFAULT_CONFIG = {
    # Minimum number of stars that must photometrically match for an offset
    # to be considered good.
    'min_stars': 3,

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # PSF size for modeling a star (must be odd). The PSF is square.
    'psf_size': 9,
        
    # The default star class when none is available in the star catalog.
    'default_star_class': 'G0',
    
    # The minimum DN that is guaranteed to be visible in the image.
    'min_brightness_guaranteed_vis': 200.,

    # The minimum DN count for a star to be detectable. These values are pretty
    # aggressively dim - there's no guarantee a star with this brightness can
    # actually be seen. But there's no point in looking at a star any dimmer.
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
    
    # How far (in pixels) a star has to be from a major body before it is no
    # longer considered to conflict.
    'star_body_conflict_margin': 3,
    
    # If star navigation fails, get rid of the brightest star if it's at least
    # this bright. None means don't apply this test.
    'too_bright_dn': 1000,
    
    # If star navigation fails, get rid of the brightest star if it's at least
    # this much brighter (factor) than the next dimmest star. None means don't
    # apply this test.
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
    # curvature_threshold_pixels must be true for correlation to be trusted.
    # The _pixels version is useful for the case of small moons.
    'curvature_threshold_pixels': 20,
    
    # The maximum incidence that can be considered a limb instead of a
    # terminator.
    'limb_incidence_threshold': 87. * oops.RPD, # cos = 0.05
    
    # The resolution in longitude and latitude (radians) for the metadata
    # latlon mask.
    'mask_lon_resolution': 1. * oops.RPD,
    'mask_lat_resolution': 1. * oops.RPD,

    # The latlon coordinate type and direction for the metadata latlon mask.
    'mask_latlon_type': 'centric',
    'mask_lon_direction': 'east',
}

RINGS_DEFAULT_CONFIG = {
    # The source for profile data - 'voyager' or 'uvis'.
    'model_source': 'uvis',
    
    # There must be at least this many fiducial features for rings to be used
    # for correlation.
    'fiducial_feature_threshold': 3,
    
    # There must be at least this number of pixels of curvature present for
    # rings to be used for correlation.
    'curvature_threshold': 2,
}

BOOTSTRAP_DEFAULT_CONFIG = {
    # The resolution in longitude and latitude (radians) for the mosaic.
    'lon_resolution': 0.1 * oops.RPD,
    'lat_resolution': 0.1 * oops.RPD,

    # The latlon coordinate type and direction for the mosaic.
    'latlon_type': 'centric',
    'lon_direction': 'east',
}

OFFSET_DEFAULT_CONFIG = {
    # A body has to be at least this many pixels in area for us to pay 
    # attention to it for bootstrapping purposes.
    'min_body_area': 9,
    
    # If there are at least this many bodies in the image, then we trust the
    # body-based model correlation result.
    'num_bodies_threshold': 3,

    # OR
        
    # If the bodies cover at least this fraction of the image, then we trust
    # the body-based model correlation result. 
    'bodies_cov_threshold': 0.0005,
    
    # If the total model covers at least this fraction of the image, then we 
    # might trust it.
    'model_cov_threshold': 0.0005,
    
    # AND
    
    # If the total model has any contents within this distance of an edge, 
    # then we might trust it.
    'model_edge_pixels': 5,
    
    # The number of pixels to search in U,V during secondary correlation.
    'secondary_corr_search_size': 15,  
    
    # If the stars-based and bodies/rings-based correlations differ by at
    # least this amount, then we don't trust the bodies/rings-based model.
    'stars_model_diff_threshold': 5,
    
    # If there are at least this many good stars, then the stars can override
    # the bodies/rings model when they differ by too many pixels.
    'stars_override_threshold': 6,

    # If secondary correlation is off by more than this amount, it fails.
    'secondary_corr_threshold': 2,
    
    # If the secondary correlation peak isn't at least this fraction of the
    # primary correlation peak, correlation fails.
    'secondary_corr_peak_threshold': 0.75,
}
