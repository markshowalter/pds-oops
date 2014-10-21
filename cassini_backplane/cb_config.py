import oops

STAR_CATALOG_ROOT = 't:/external/ucac4'
SUPPORT_FILES_ROOT = 't:/external/cb_support'

COISS_2XXX_DERIVED_ROOT = 't:/external/cassini/derived/COISS_2xxx'
COISS_3XXX_ROOT = 't:/external/cassini/volumes/COISS_3xxx'

MAX_POINTING_ERROR = {'NAC': (85,85), 'WAC': (15,15)} # Pixels

ISS_FOV_SIZE = {'NAC': 0.35*oops.RPD, 'WAC': 3.48*oops.RPD}

ISS_PSF_SIGMA = {'NAC': 0.54, 'WAC': 0.77}


# These are bodies large enough to be picked up in an image
LARGE_BODY_LIST = ['SATURN', 'ATLAS', 'PROMETHEUS', 'PANDORA',
                   'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                   'TETHYS', 'DIONE', 'RHEA', 'TITAN', 'HYPERION',
                   'IAPETUS', 'PHOEBE']

FUZZY_BODY_LIST = ['TITAN', 'PANDORA']


# These values are pretty aggressively dim - there's no guarantee a star with 
# this brightness can actually be seen.
MIN_DETECTABLE_DN = {'NAC': 20, 'WAC': 150}

