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
CB_SOURCE_ROOT = os.environ['CB_SOURCE_ROOT']

# CB support files such as Voyager ring profiles.
CB_SUPPORT_FILES_ROOT = os.environ['CB_SUPPORT_ROOT']

# Cassini ISS non-calibrated volumes.
COISS_ROOT = os.environ['COISS_ROOT']

# Cassini ISS calibrated 2XXX volume.
COISS_2XXX_DERIVED_ROOT = os.environ['COISS_2XXX_DERIVED_ROOT']

# Cassini ISS moon maps.
COISS_3XXX_ROOT = os.path.join(COISS_ROOT, 'COISS_3xxx')

# Cassini UVIS.
COUVIS_8XXX_ROOT = os.path.join(COISS_ROOT, 'COUVIS_8xxx_lien_resolution')

# CB-generated offset and backplane files.
CB_RESULTS_ROOT = os.environ['CB_RESULTS_ROOT']

# Contains solar flux, filter transmission convolved with quantum efficiency
CISSCAL_CALIB_ROOT = os.getenv('CISSCAL_CALIB_PATH')

# Contains filter transmission and PSF data
CASSINI_CALIB_ROOT = os.getenv('CASSINI_CALIB_PATH')

# The star catalog root (must contain UCAC4 and YBSC)
STAR_CATALOG_ROOT = os.environ['CB_STAR_CATALOG_ROOT']

# The URL for files in the PDS Ring-Moon Systems Node
PDS_RINGS_VOLUMES_ROOT = 'http://pds-rings.seti.org/volumes/COISS_2xxx/'
PDS_RINGS_CALIB_ROOT = 'http://pds-rings.seti.org/derived/COISS_2xxx/'

####################################
# EXECUTABLES AND PYTHON PROGRAMS ##
####################################

PYTHON_EXE = os.environ['PYTHON_EXE']

CBMAIN_OFFSET_PY = os.path.join(CB_SOURCE_ROOT, 'cb_main_offset.py')
CBMAIN_REPROJECT_BODY_PY = os.path.join(CB_SOURCE_ROOT, 'cb_main_reproject_body.py')
CBMAIN_MOSAIC_BODY_PY = os.path.join(CB_SOURCE_ROOT, 'cb_main_mosaic_body.py')
DISPLAY_OFFSET_METADATA_PY = os.path.join(CB_SOURCE_ROOT, 'utilities',
                                          'display_offset_metadata.py')
DISPLAY_MOSAIC_METADATA_PY = os.path.join(CB_SOURCE_ROOT, 'utilities',
                                          'display_mosaic_metadata.py')


########################
# INSTRUMENT CONSTANTS #
########################

# The maximum pointing error we allow in the (V,U) directions.
MAX_POINTING_ERROR_NAC =   (110,75)  # Pixels
MAX_POINTING_ERROR_WAC =   (11,15)
MAX_POINTING_ERROR_LORRI = (40,40)
MAX_POINTING_ERROR = {((1024,1024), 'NAC'):   MAX_POINTING_ERROR_NAC,
                      ((1024,1024), 'WAC'):   MAX_POINTING_ERROR_WAC,
                      (( 512, 512), 'NAC'):   (MAX_POINTING_ERROR_NAC[0]//2,
                                               MAX_POINTING_ERROR_NAC[1]//2),
                      (( 512, 512), 'WAC'):   (MAX_POINTING_ERROR_WAC[0]//2,
                                               MAX_POINTING_ERROR_WAC[1]//2),
                      (( 256, 256), 'NAC'):   (MAX_POINTING_ERROR_NAC[0]//4,
                                               MAX_POINTING_ERROR_NAC[1]//4),
                      (( 256, 256), 'WAC'):   (MAX_POINTING_ERROR_WAC[0]//4,
                                               MAX_POINTING_ERROR_WAC[1]//4),
                      ((1024,1024), 'LORRI'): MAX_POINTING_ERROR_LORRI,
                     }

# The FOV size of the ISS cameras in radians.
FOV_SIZE = {'NAC':   0.35*oops.RPD, 
            'WAC':   3.48*oops.RPD,
            'LORRI': 0.29*oops.RPD}

# The Gaussian sigma of the ISS camera PSFs in pixels.
PSF_SIGMA = {'NAC':   0.54, 
             'WAC':   0.77,
             'LORRI': 0.5} # XXX


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
FUZZY_BODY_LIST = ['HYPERION', 'PHOEBE']

# These are bodies inside the rings that should not be used to compute
# ring occlusions.
RINGS_BODY_LIST = ['PAN', 'DAPHNIS']


##################
# CONFIGURATIONS #
##################

STARS_DEFAULT_CONFIG = {
    # True if data is already calibrated as I/F and needs to be converted back
    # to raw DN.
    'calibrated_data': True,
    
    # Allow non-integer offsets; these use astrometry to refine the mean
    # star offset.
    'allow_fractional_offsets': True,
    
    # The order of multipliers to use to gradually expand the search area.
    'search_multipliers': [0.25, 0.5, 0.75, 1.],

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # Verify offset with photometry?
    'perform_photometry': True,
    
    # If using photometry, try again at the end without using it?
    'try_without_photometry': False,
    
    # Minimum number of stars that must photometrically match for an offset
    # to be considered acceptable and the corresponding confidence.
    # Also the minimum number of stars that must match to give a confidence 
    # of 1.0.
    'min_stars_low_confidence': (3, 0.75),
    'min_stars_high_confidence': (6, 1.0),

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Minimum PSF size for modeling a star (must be odd). The PSF is square.
    # This will be added to the smearing in each dimension to create a final
    # possibly-rectangular PSF.
    'min_psf_size': 9,
    
    # The maximum number of steps to use when smearing a PSF. This is really 
    # only a suggestion, as the number will be clipped at either extreme to
    # guarantee a good smear.
    'max_movement_steps': 50,
    
    # The maximum amount of smearing to tolerate before giving up on star
    # navigation entirely. 
    # OLD: This is currently set low because the smear angles
    #      are wrong thanks to SPICE inaccuracies.
    # NEW: This is currently set high because we have access to the
    #      predicted kernels.
    'max_smear': 100,
    
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
    'too_bright_factor': None,

    # The font and font size for star labels (font filename, size)
    'font': None
}

BODIES_DEFAULT_CONFIG = {
    # The minimum number of pixels in the bounding box surrounding the body
    # in order to bother with it.
    'min_bounding_box_area': 3*3,

    # The minimum number of pixels in the bounding box surrounding the body
    # in order to compute the latlon mask for bootstrapping.
    'min_latlon_mask_area': 10*10,
    
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
    'limb_incidence_threshold': 87. * oops.RPD, # cos = 0.05
    
    # What fraction of the total visible limb needs to meet the above criterion
    # in order for the limb to be marked as OK. This is only used in the case
    # where curvature is bad.
    'limb_incidence_frac': 0.4, 
    
    # What resolution is so small that the surface features make the moon
    # non-circular when viewing the limb?
    'surface_bumpiness':
        {'SATURN': 90., # This is really a measure of atmospheric haze
         'PAN': 7., #30.,
         'DAPHNIS': 3., #9.,
         'ATLAS': 10., #41.,
         'PROMETHEUS': 75., #140.,
         'PANDORA': 52., #105.,
         'EPIMETHEUS': 21., # 131
         'JANUS': 35., # 205
         'MIMAS': 0.5,
         'ENCELADUS': 0.5,
         'TETHYS': 1.,
         'TELESTO': 34.,
         'CALYPSO': 32.,
         'DIONE': 1.,
         'HELENE': 45.,
         'RHEA': 1.2,
         'TITAN': 0.,
         'HYPERION': 0.,
         'IAPETUS': 35.,
         'PHOEBE': 0.,},
                         
    # Whether or not Lambert shading should be used, as opposed to just a
    # solid unshaded shape, when a cartographic reprojection is not
    # available.
    'use_lambert': True,
    
    # The resolution in longitude and latitude (radians) for the metadata
    # latlon mask.
    'mask_lon_resolution': 1. * oops.RPD,
    'mask_lat_resolution': 1. * oops.RPD,

    # The latlon coordinate type and direction for the metadata latlon mask
    # and sub-solar and sub-observer longitudes.
    'mask_latlon_type': 'centric',
    'mask_lon_direction': 'east',
    
    # A body has to take up at least this many pixels in order to be labeled.
    'min_text_area': 0.5,

    # The font and font size for body labels (font filename, size)
    'font': None
}

TITAN_DEFAULT_CONFIG = {
    # The altitude of the top of the atmosphere (km).
    'atmosphere_height': 700,

    # Increment in incidence and emission angles for creating photometric
    # grid. These should have a decent beat frequency (e.g. be relatively
    # prime) for best coverage.
    'incidence_min': 5. * oops.RPD,
    'incidence_max': oops.PI,
    'incidence_increment': 5. * oops.RPD,
    'emission_min': 2.5*oops.RPD,
    'emission_max': oops.HALFPI,
    'emission_increment': 5. * oops.RPD,
    'max_emission_angle': 80. * oops.RPD,
    
    # The minimum number of pixels between the two clusters for an
    # incidence/emission intersection to be used.
    'cluster_gap_threshold': 10,
    
    # The largest number of pixels that can make up a cluster.
    'cluster_max_pixels': 10,
}

RINGS_DEFAULT_CONFIG = {
    # The source for profile data - 'voyager', 'uvis', or 'ephemeris'.
    'model_source': 'ephemeris',
    
    # There must be at least this many fiducial features for rings to be used
    # for correlation.
    'fiducial_feature_threshold': 3,
    
    # The RMS error of a feature must be this many times less than the
    # coarsest resolution of the feature in the image in order for the feature
    # to be used. This makes sure that the statistical scatter of the feature
    # is blurred out during correlation.
    'fiducial_rms_gain': 2,
    
    # A full gap or ringlet must be at least this many pixels wide at some
    # place in the image to use it.
    'fiducial_min_feature_width': 3,
    
    # Assume a one-sided feature is about this wide in km. This is used to 
    # determine if the local resolution is high enough for the feature to be 
    # visible. 
    'one_sided_feature_width': 30.,
    
    # When manufacturing a model from an ephemeris list, each one-sided feature 
    # is shaded approximately this many pixels wide.
    'fiducial_ephemeris_width': 30,
     
    # There must be at least this number of pixels of curvature present for
    # rings to be used for correlation.
    'curvature_threshold': 5,
    
    # The minimum ring emission angle in the image must be at least this
    # many degrees away from 90 for rings to be used for correlation.
    'emission_threshold': 2., #5., 
    
    # Remove the shadow of Saturn from the model
    'remove_saturn_shadow': True,
    
    # Remove the shadow of other bodies from the model
    'remove_body_shadows': False,
    
    # The font and font size for ring labels (font filename, size)
    'font': None
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
    # distance from an edge, then we trust it.
    'model_cov_threshold': 25,
    'model_edge_pixels': 5,
    #^^^
    
    # The number of pixels to search in U,V during secondary correlation.
    'secondary_corr_search_size': (15,15),  
    
    # The lowest confidence to allow for models
    'lowest_confidence': 0.05,
    
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

BOOTSTRAP_DEFAULT_CONFIG = {
    # These bodies can be used for bootstrapping.
    'body_list': ['DIONE', 'ENCELADUS', 'IAPETUS', 'MIMAS', 'RHEA', 'TETHYS'],

    # The maximum phase angle that can be used to create part of a mosaic.
    'max_phase_angle': 135. * oops.RPD,
    
    # The minimum square size of a moon to be used to create part of a mosaic.
    'min_area': 128*128,

    # The size of the longitude and latitude bins used to create multiple
    # mosaics.
    'mosaic_lon_bin_size': 30. * oops.RPD,
    'mosaic_lat_bin_size': 30. * oops.RPD,
        
    # The resolution in longitude and latitude (radians) for the mosaic.
    'lon_resolution': 0.5 * oops.RPD,
    'lat_resolution': 0.5 * oops.RPD,

    # The latlon coordinate type and direction for the mosaic.
    'latlon_type': 'centric',
    'lon_direction': 'east',
    
    # The minimum fraction of of moon that is available from cartographic
    # data in order for a bootstrapped offset to be attempted.
    'min_coverage_frac': 0.25,
    
    # The maximum difference in resolution allowable between the mosaic
    # and the image to be bootstrapped.
    'max_res_factor': 10000, # XXX
}
