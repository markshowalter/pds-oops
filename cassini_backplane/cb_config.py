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

# Root of the cassini_backplane source directory.
CB_ROOT = os.environ['CB_ROOT']

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


####################################
# EXECUTABLES AND PYTHON PROGRAMS ##
####################################

PYTHON_EXE = os.environ['PYTHON_EXE']

CBMAIN_OFFSET_PY = os.path.join(CB_ROOT, 'cb_main_offset.py')
DISPLAY_OFFSET_METADATA_PY = os.path.join(CB_ROOT, 'utilities',
                                          'display_offset_metadata.py')
DISPLAY_MOSAIC_METADATA_PY = os.path.join(CB_ROOT, 'utilities',
                                          'display_mosaic_metadata.py')


########################
# INSTRUMENT CONSTANTS #
########################

# The maximum pointing error we allow in the (V,U) directions.
MAX_POINTING_ERROR_NAC = (85,75)  # Pixels
MAX_POINTING_ERROR_WAC = (15,15)
MAX_POINTING_ERROR = {((1024,1024), 'NAC'): MAX_POINTING_ERROR_NAC,
                      ((1024,1024), 'WAC'): MAX_POINTING_ERROR_WAC,
                      (( 512, 512), 'NAC'): (MAX_POINTING_ERROR_NAC[0]//2,
                                             MAX_POINTING_ERROR_NAC[1]//2),
                      (( 512, 512), 'WAC'): (MAX_POINTING_ERROR_WAC[0]//2,
                                             MAX_POINTING_ERROR_WAC[1]//2),
                      (( 256, 256), 'NAC'): (MAX_POINTING_ERROR_NAC[0]//4,
                                             MAX_POINTING_ERROR_NAC[1]//4),
                      (( 256, 256), 'WAC'): (MAX_POINTING_ERROR_WAC[0]//4,
                                             MAX_POINTING_ERROR_WAC[1]//4)
                     }

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
FUZZY_BODY_LIST = ['HYPERION', 'PHOEBE']#, 'TITAN'] # XXX


##################
# CONFIGURATIONS #
##################

STARS_DEFAULT_CONFIG = {
    # Allow non-integer offsets; these use astrometry to refine the mean
    # star offset.
    'allow_fractional_offsets': True,
    
    # Minimum number of stars that must photometrically match for an offset
    # to be considered good.
    'min_stars': 3,

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # Minimum PSF size for modeling a star (must be odd). The PSF is square.
    # This will be added to the smearing in each dimension to create a final
    # possibly-rectangular PSF.
    'min_psf_size': 9,
    
    # The maximum number of steps to use when smearing a PSF. This is really 
    # only a suggestion, as the number will be clipped at either extreme to
    # guarantee a good smear.
    'max_movement_steps': 50,
    
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
    # The minimum number of pixels in the bounding box surrounding the body
    # in order to bother with it.
    'min_bounding_box_area': 16,
    
    # The fraction of the width/height of a body that must be visible on either
    # side of the center in order for the curvature to be sufficient for 
    # correlation.
    # Set both 'curvature_threshold_frac' and 'curvature_threshold_pixels' to
    # eliminate the check for curvature and mark all bodies as OK. 
    'curvature_threshold_frac': 0.02,
    
    # The number of pixels of the width/height of a body that must be visible
    # on either side of the center in order for the curvature to be sufficient
    # for correlation. Both curvature_threshold_frac and 
    # curvature_threshold_pixels must be true for correlation to be trusted.
    # The _pixels version is useful for the case of small moons.
    'curvature_threshold_pixels': 20,
    
    # The maximum incidence that can be considered a limb instead of a
    # terminator.
    # Set to oops.TWOPI to eliminate the check for limbs and mark
    # all bodies as OK.
    'limb_incidence_threshold': 180*oops.RPD, # XXX 87. * oops.RPD, # cos = 0.05
    
    # Whether or not Lambert shading should be used, as opposed to just a
    # solid unshaded shape, when a cartographic reprojection is not
    # available.
    'use_lambert': True,
    
    # The resolution in longitude and latitude (radians) for the metadata
    # latlon mask.
    'mask_lon_resolution': 1. * oops.RPD,
    'mask_lat_resolution': 1. * oops.RPD,

    # The latlon coordinate type and direction for the metadata latlon mask.
    'mask_latlon_type': 'centric',
    'mask_lon_direction': 'east',
    
    # A body has to take up at least this many pixels in order to be labeled.
    'text_min_area': 9,
}

RINGS_DEFAULT_CONFIG = {
    # The source for profile data - 'voyager', 'uvis', or 'ephemeris'.
    'model_source': 'ephemeris',
    
    # There must be at least this many fiducial features for rings to be used
    # for correlation.
    'fiducial_feature_threshold': 3,
    
    # There must be at least this many pixels beyond a fiducial feature in the
    # non-extended image for it to count as being in the image for counting
    # purposes.
    'fiducial_feature_margin': 50,
    
    # The RMS error of a feature must be this many times less than the
    # coarsest resolution of the feature in the image in order for the feature
    # to be used. This makes sure that the statistical scatter of the feature
    # is blurred out during correlation.
    'fiducial_rms_gain': 2,
    
    # When manufacturing a model from an ephemeris list, each feature is
    # approximately this many pixels wide.
    'fiducial_ephemeris_width': 30,
     
    # There must be at least this number of pixels of curvature present for
    # rings to be used for correlation.
    'curvature_threshold': 5,
    
    # The minimum ring emission angle in the image must be at least this
    # many degrees away from 90 for rings to be used for correlation.
    'emission_threshold': 5., 
    
    # When making the text overlay, only label a full ringlet or gap if it's
    # at least this many pixels wide somewhere in the image.
    'text_ringlet_gap_threshold': 2,
    
    # When making the text overlay, only label a non-full ringlet or gap
    # if...
    'text_threshold': 0, # XXX
    
    # Remove the shadow of Saturn from the model
    'remove_saturn_shadow': False,
    
    # Remove the shadow of other bodies from the model
    'remove_body_shadows': False
}

BOOTSTRAP_DEFAULT_CONFIG = {
    # These bodies can be used for bootstrapping.
    # Includes the orbital period (seconds) and the maximum allowable
    # resolution (km/pix).
    'body_list': {'DIONE':     (  2.736915 * 86400, 15.),
                  'ENCELADUS': (  1.370218 * 86400, 15.),
                  'IAPETUS':   ( 79.3215   * 86400, 15.),
                  'MIMAS':     (  0.942    * 86400, 15.),
                  'PHOEBE':    (550.564636 * 86400, 15.),
                  'RHEA':      (  4.518212 * 86400, 15.),
                  'TETHYS':    (  1.887802 * 86400, 15.)
                  },
    
    # The fraction of an orbit that a moon can move and still be OK for
    # creating a mosaic.
    'orbit_frac': 0.25,
    
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
    'min_body_area': 100,
    
    # By default, each image and model is Gaussian blurred by this much
    # before correlation. This can be overridden if the rings model requests
    # additional blurring.
    'default_gaussian_blur': 0.25,
    
    # By default, the median filter looks at this many pixels.
    'median_filter_size': 11,
    
    # By default, the median filter is Gaussian blurred by this much before 
    # being subtracted from the image or model.
    'median_filter_blur': 1.2,
    
    #vvv
    # If there are at least this many bodies in the image, then we trust the
    # body-based model correlation result.
    'num_bodies_threshold': 3,

    # OR
        
    # If the bodies cover at least this fraction of the image, then we trust
    # the body-based model correlation result. 
    'bodies_cov_threshold': 0.0005,
    #^^^
    
    #vvv
    # If the total model covers at least this number of pixels the given
    # distance from an edge, then we might trust it.
    'model_cov_threshold': 25,
    'model_edge_pixels': 5,
    #^^^
    
    # The number of pixels to search in U,V during secondary correlation.
    'secondary_corr_search_size': 15,  
    
    # If the stars-based and bodies/rings-based correlations differ by at
    # least this number of pixels, then we need to choose between the stars
    # and the bodies.
    'stars_model_diff_threshold': 2,
    
    # If there are at least this many good stars, then the stars can override
    # the bodies/rings model when they differ by the above number of pixels.
    # Otherwise, the bodies/rings model overrides the stars.
    'stars_override_threshold': 6,

    # If secondary correlation is off by at least this number of pixels, it 
    # fails.
    'secondary_corr_threshold': 3,
    
    # If the secondary correlation peak isn't at least this fraction of the
    # primary correlation peak, correlation fails.
    'secondary_corr_peak_threshold': 0.75,
}
