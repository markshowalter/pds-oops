import oops

STAR_CATALOG_FILENAME = 't:/external/ucac4'

COISS_2XXX_DERIVED_ROOT = 't:/external/cassini/derived/COISS_2xxx'
COISS_3XXX_ROOT = 't:/external/cassini/volumes/COISS_3xxx'

MAX_POINTING_ERROR = {'NAC': (45,45), 'WAC': (5,5)} # Pixels

ISS_FOV_SIZE = {'NAC': 0.35*oops.RPD, 'WAC': 3.48*oops.RPD}

ISS_PSF_SIGMA = {'NAC': 0.54, 'WAC': 0.77}


# These are bodies large enough to be picked up in an image
LARGE_BODY_LIST = ['SATURN', 'ATLAS', 'PROMETHEUS', 'PANDORA',
                   'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                   'TETHYS', 'DIONE', 'RHEA', 'TITAN', 'HYPERION',
                   'IAPETUS', 'PHOEBE']

FUZZY_BODY_LIST = ['SATURN', 'TITAN']
