import oops

STAR_CATALOG_FILENAME = 't:/external/ucac4'

COISS_2XXX_DERIVED_ROOT = 't:/external/cassini/derived/COISS_2xxx'
COISS_3XXX_ROOT = 't:/external/cassini/volumes/COISS_3xxx'

MAX_POINTING_ERROR = {'NAC': (40,40), 'WAC': (4,4)} # Pixels

ISS_FOV_SIZE = {'NAC': 0.35*oops.RPD, 'WAC': 3.48*oops.RPD}

ISS_PSF_SIGMA = {'NAC': 0.54, 'WAC': 0.77}

