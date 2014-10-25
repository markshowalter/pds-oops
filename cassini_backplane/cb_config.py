###############################################################################
# cb_config.py
#
# Global configuration parameters.  
###############################################################################

import oops

#####################
# ROOT DIRECTORIES ##
#####################

# The UCAC4 star catalog.
STAR_CATALOG_ROOT = 't:/external/ucac4'

# CB support files such as Voyager ring profiles.
SUPPORT_FILES_ROOT = 't:/external/cb_support'

# Cassini ISS calibrated files.
COISS_2XXX_DERIVED_ROOT = 't:/external/cassini/derived/COISS_2xxx'

# Cassini ISS moon maps.
COISS_3XXX_ROOT = 't:/external/cassini/volumes/COISS_3xxx'


########################
# INSTRUMENT CONSTANTS #
########################

# The maximum pointing error we allow in the (V,U) directions.
MAX_POINTING_ERROR = {'NAC': (85,75), 'WAC': (15,15)} # Pixels

# The FOV size of the ISS cameras in radians.
ISS_FOV_SIZE = {'NAC': 0.35*oops.RPD, 'WAC': 3.48*oops.RPD}

# The Gaussian sigma of the ISS camera PSFs in pixels.
ISS_PSF_SIGMA = {'NAC': 0.54, 'WAC': 0.77}

# The minimum DN count for a star to be detectable. These values are pretty
# aggressively dim - there's no guarantee a star with this brightness can
# actually be seen.
MIN_DETECTABLE_DN = {'NAC': 20, 'WAC': 150}


########################
# BODY CHARACTERISTICS #
########################

# These are bodies large enough to be picked up in an image.
LARGE_BODY_LIST = ['SATURN', 'ATLAS', 'PROMETHEUS', 'PANDORA',
                   'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                   'TETHYS', 'DIONE', 'RHEA', 'TITAN', 'HYPERION',
                   'IAPETUS', 'PHOEBE']

# These are bodies that shouldn't be used for navigation because they
# are "fuzzy" in some way or at least don't have a well-defined orientation.
FUZZY_BODY_LIST = ['TITAN', 'PANDORA', 'HYPERION']
