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
