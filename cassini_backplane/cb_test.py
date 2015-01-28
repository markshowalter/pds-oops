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

# 1662351509 - CRASHES during display - rings problem?

test_data = (


#     (1662351509, ('RHEA'),
#      "Model and stars agree",
#      {'allow_stars':True, 'allow_rings':False},
#     ), # STARS


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

#    # TERMINATOR TESTS
#    # These have Mimas alone, centered, and entirely in the image
#
#     (1493446065, ('MIMAS'),
#      """MIMAS alone, small, centered, entirely in the image, overexposed,
#      incidence 20-105.""",
#      {},
#     ), # STARS
#     
#     (1484509816, ('MIMAS'),
#      """MIMAS alone, medium, centered, entirely in the image,
#      incidence 0-130.""",
#      {},
#     ),
# 
#     (1484530421, ('MIMAS'),
#      """MIMAS alone, medium, centered, entirely in the image,
#      incidence 0-160.""",
#      {},
#     ),
#              
#     (1484535522, ('MIMAS'),
#      """MIMAS alone, medium, centered, entirely in the image,
#      incidence 0-165.""",
#      {},
#     ),
#
#    (1484573247, ('MIMAS'),
#     """MIMAS alone, medium, centered, entirely in the image,
#     incidence 50-180.""",
#     {},
#    ),
#
#     (1484580522, ('MIMAS'),
#      """MIMAS alone, small, centered, entirely in the image, slightly
#      overexposed, incidence 60-180.""",
#      {},
#     ),
# 
#     (1487445616, ('MIMAS'),
#      """MIMAS alone, small, centered, entirely in the image, slightly
#      overexposed, incidence 30-160.""",
#      {},
#     ),
#
#    # These have Mimas alone and offset
#
#     (1501640895, ('MIMAS'),
#      """MIMAS alone, large, slightly off the top edge but good curvature,
#      good limb on left, terminator on right, incidence 0-130.""",
#      {},
#     ),
# 
#     (1501645855, ('MIMAS'),
#      """MIMAS alone, large, slightly off the top left edge but good curvature,
#      good partial limb on right, terminator on left, incidence 0-120.""",
#      {},
#     ),
# 
#     (1501646143, ('MIMAS'),
#      """MIMAS alone, large, significantly off the right edge but good
#      curvature, terminator on left, incidence 30-135.
#      Bootstrapping candidate.""",
#      {},
#     ),
#              
#     (1501646177, ('MIMAS'),
#      """MIMAS alone, large, mostly off the right edge, bad curvature,
#      terminator on left, incidence 60-135. Bootstrapping candidate.""",
#      {},
#     ),
# 
# 
# 
# 
# 
# 
#     (1492217357, ('MIMAS'),
#      """MIMAS alone, large, centered, entirely in the image,
#      incidence 0-150.""",
#      {},
#     ),
    
             
# MIMAS ON TOP OF RINGS
     
     
     
### RHEA

    # TERMINATOR TESTS
    # These have Rhea alone, centered, and entirely in the image

#     (1498348607, 'N', ('RHEA'),
#      """RHEA alone, small, centered, entirely in the image,
#      incidence 0-90.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),
#
#     (1484528177, 'N', ('RHEA'),
#      """RHEA alone, medium, centered, entirely in the image,
#      incidence 0-120. Terminator on top right.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),
#
#     (1507943927, 'N', ('RHEA'),
#      """RHEA alone, small, centered, entirely in the image,
#      incidence 20-170. Terminator down the center.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),
#
#     (1509645920, 'N', ('RHEA'),
#      """RHEA alone, small, centered, entirely in the image,
#      incidence 30-175. Terminator down the center.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),
#
#     (1516373800, 'N', ('RHEA'),
#      """RHEA alone, small, centered, entirely in the image,
#      incidence 80-175. Thin crescent at bottom.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),
#
#     (1500013667, 'N', ('RHEA'),
#      """RHEA alone, small, centered, entirely in the image,
#      incidence 5-135.""",
#      {'allow_stars': False, 'allow_rings': False},
#     ),

     # These have Rhea off the side of the image
     
     (1558972182, 'N', ('RHEA'),
      """RHEA alone, large, off all sides, good curvature,
      incidence 0-120. Terminator at right but left has limb.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1484584650, 'N', ('RHEA'),
      """RHEA alone, large, off top right corner, bad curvature,
      incidence 0-60.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1484584990, 'N', ('RHEA'),
      """RHEA alone, large, off bottom left corner, bad curvature,
      incidence 45-155. Terminator on outer curve.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1484584990, 'N', ('RHEA'),
      """RHEA alone, large, off bottom right corner, good curvature,
      incidence 35-135. Terminator on top portion.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1484600538, 'N', ('RHEA'),
      """RHEA alone, large, off bottom left corner, good curvature,
      incidence 50-165. Terminator on most of top portion.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1521604316, 'N', ('RHEA'),
      """RHEA alone, large, off bottom, bad curvature,
      incidence 100-180.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1521616930, 'N', ('RHEA'),
      """RHEA alone, large, off top left corner, bad curvature,
      incidence 50-105. Terminator at top but part of curve has limb.""",
      {'allow_stars': False, 'allow_rings': False},
     ),

     (1521617131, 'N', ('RHEA'),
      """RHEA alone, large, off top, bad curvature,
      incidence 45-110. Terminator at top but part of curve has limb.""",
      {'allow_stars': False, 'allow_rings': False},
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

cb_logging.log_set_stars_level(logging.DEBUG)
cb_logging.log_set_offset_level(logging.DEBUG)
cb_logging.log_set_bodies_level(logging.DEBUG)
cb_logging.log_set_rings_level(logging.DEBUG)

skip = False

for test_entry in test_data:
    (image_num, camera, descr, attribs, offset_kwargs) = test_entry
    
    for image_filename in yield_image_filenames(image_num, image_num, camera):
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
        