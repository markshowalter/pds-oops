###############################################################################
# cb_test.py
#
# Offset unit tests.
###############################################################################

import cb_logging
import logging

import oops.inst.cassini.iss as iss

from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *

_LOGGING_NAME = 'cb.' + __name__

test_data = (
#    (1493446380, ('ATLAS', 'PANDORA', 'RINGS'), # UV3, rings without curvature
#     {},
#    ),
#               
#    (1503069206, ('ATLAS', 'RINGS'), # GRN, rings without curvature
#     {},
#    ),
#               
#    (1464189709, ('DIONE',), # Overexposed, high noise, star field
#     {},
#    ),
#               
#    (1466380040, ('DIONE',), # P0+UV3, overexposed, high noise, star field
#     {},
#    ),
#                
#    (1480221004, ('DIONE',), # overexposed
#     {},
#    ),
#             
#    (1554736600, ('DIONE',), #overexposed
#     {},
#    ),
#     
#    (1481736616, ('DIONE',), # UV3, fills most of image, terminator
#     {},
#    ),
#     
#    (1481738274, ('DIONE',), # quadrant, terminator
#     {},
#    ),
#     
#    (1481738469, ('DIONE',), # quadrant
#     {},
#    ),
#     
#    (1481766978, ('DIONE',), # partial, terminator
#     {},
#    ),
     
#1481767088 - DIONE, partial
#1481767211 - DIONE, moon only
#1507738278 - DIONE, moon only
#1507745867 - DIONE, moon only
#1506052003 - DIONE, P0+UV3, edge on rings
#1506058905 - DIONE, on rings

#    (1487410178, ('DIONE', 'TITAN'),
#     {},
#    ),
#     
#    (1487595561, ('DIONE', 'RHEA'),
#     {},
#    ),
#     
#    (1507685420, ('DIONE', 'RHEA'), # DIONE on top of RHEA
#     {},
#    ),
#     
#    (1507741763, ('DIONE', 'SATURN'), # DIONE on Saturn, edge on rings, GRN
#     {'allow_rings': False},
#    ),

### MIMAS

    (1492217357, ('MIMAS'),
     """MIMAS alone, large, centered, good partial limb, partial terminator.
     """,
     {},
    ),
    
    (1501645855, ('MIMAS'),
     """MIMAS alone, large, slightly off the edge but good curvature,
     good partial limb, partial terminator.
     """,
     {},
    ),
             
    (1501646143, ('MIMAS'),
     """MIMAS alone, large, significantly off the edge but good curvature,
     terminator only. Bootstrapping candidate.
     """,
     {},
    ),
             
    (1501646177, ('MIMAS'),
     """MIMAS alone, large, mostly off the edge, bad curvature, terminator
     only. Bootstrapping candidate.
     """,
     {},
    ),
             
               
             
# RING MOONS
#1751425716 - AEGAEON, G ring, star streaks
#1540685777 - DAPHNIS, rings without curvature
#1540685777 - DAPHNIS, rings
#1580178527 - DAPHNIS, BL1, rings without curvature
#1627458056 - DAPHNIS, rings without curvature, equinox shadows

# IRREGULAR MOONS
#1749722296 - ALBIORIX, streaks, star field
#1610857212 - ANTHE, high noise, star field
#1634622640 - ANTHE, ring ansa, blurry
#1651873081 - BEBHIONN, star field
#1667130344 - BERGELMIR, streaks, star field
#1667176532 - BERGELMIR, streaks, star field
#1725826526 - BESTLA, streaks, star field, 256x256
#1634895040 - BESTLA, GRN, streaks, star field, 256x256
#1487506533 - CALYPSO, high noise
#1506187742 - CALYPSO, GRN
#1506184376 - CALYPSO, IR3
#1603214971 - CALYPSO, high noise
)

logger = logging.getLogger(_LOGGING_NAME+'.main')

skip = False

for test_entry in test_data:
    (image_num, descr, attribs, offset_kwargs) = test_entry
    
    for image_filename in yield_image_filenames(image_num, image_num):
        if skip:
            skip = False
            continue
        obs = read_iss_file(image_filename)
        logger.info('Processing %s', image_filename)
        logger.info('Size %d,%d TEXP %f Filters %s %s', 
                    obs.data.shape[0], obs.data.shape[1],
                    obs.texp, obs.filter1, obs.filter2)
        metadata = master_find_offset(obs, create_overlay=True,
                                      **offset_kwargs)

        display_offset_data(obs, metadata)
        