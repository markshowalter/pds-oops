###############################################################################
# test_bootstrap_moon.py
#
# Test bootstrapping for moons.
###############################################################################

import cb_logging
import logging

import os
import os.path
import argparse

import oops.inst.cassini.iss as iss

from cb_bootstrap import *
from cb_offset import *
from cb_util_file import *


_LOGGING_NAME = 'cb.' + __name__


logger = logging.getLogger(_LOGGING_NAME+'.test_bootstrap_moon')

cb_logging.log_set_stars_level(logging.DEBUG)
cb_logging.log_set_offset_level(logging.DEBUG)
cb_logging.log_set_bodies_level(logging.DEBUG)
cb_logging.log_set_rings_level(logging.DEBUG)
cb_logging.log_set_bootstrap_level(logging.DEBUG)

# MIMAS

#img_start_num = 1501640419
#img_end_num = 1501640895

# Closeups
#img_start_num = 1501645855
#img_end_num = 1501646177

#img_start_num = 1501630084
#img_end_num = 1501630117

#img_start_num = 1492217357
#img_end_num = 1492217397


# RHEA

# 1490874611 bottom left no limb
# 1490874664 bottom left no limb
# 1490874718 top left good limb - star navigation differs from bootstrapping
# 1490874782 top good limb good curvature
# 1490874834 fills image
# 1490874889 bottom no limb
# 1490874954 bottom right no limb
# 1490875006 right good limb good curvature - curvature marked bad XXX
# 1490875063 top right good limb bad curvature
# img_start_num = 1490874611
# img_end_num = 1490893221


# ENCELADUS, MIMAS, RHEA 1484506476 to 1484614776

MOSAICS = [
#     ('DIONE', 1496876347, 1496883920),
#     ('DIONE', 1507733604, 1507748838),
#     ('DIONE', 1534428692, 1534507401),
#     ('DIONE', 1481738274, 1481767211),
#     ('DIONE', 1556123061, 1556135770),
#     ('DIONE', 1569802679, 1569815593),
#     ('DIONE', 1569826692, 1569839110),
#     ('DIONE', 1643286490, 1643300999),
#     ('DIONE', 1649313601, 1649331848),
#     ('DIONE', 1660410970, 1660414856),
#     ('DIONE', 1662192021, 1662202839),
#     ('DIONE', 1665971522, 1665979017),
#     ('DIONE', 1702369043, 1702393485),
#     ('DIONE', 1711604521, 1711614785),
#     ('DIONE', 1714678134, 1714693817),
# 
#     ('ENCELADUS', 1487299402, 1487302209),
#      ('ENCELADUS', 1500041648, 1500069258),
#      ('ENCELADUS', 1516151439, 1516171418),
#      ('ENCELADUS', 1597175743, 1597239766),
#      ('ENCELADUS', 1602263870, 1602294337),
#      ('ENCELADUS', 1604136974, 1604188433),
#      ('ENCELADUS', 1604151794, 1604218747),
#      ('ENCELADUS', 1637450504, 1637482321),
#      ('ENCELADUS', 1652858990, 1652867484),
#      ('ENCELADUS', 1660419699, 1660446193),
#      ('ENCELADUS', 1669795989, 1669856551),
#      ('ENCELADUS', 1671569397, 1671602206),
#      ('ENCELADUS', 1694646860, 1694652019),
#      ('ENCELADUS', 1694646860, 1694652019),
#      ('ENCELADUS', 1697700931, 1697717648),
#      ('ENCELADUS', 1702359393, 1702361420),

    ('IAPETUS', 1483151477, 1483281115),
    ('IAPETUS', 1568091469, 1568160072),
 
    ('MIMAS', 1501627117, 1501651303),
    ('MIMAS', 1521584495, 1521620702),
    ('MIMAS', 1644777693, 1644802455),
    ('MIMAS', 1717565987, 1717571685),
 
    ('PHOEBE', 1465643815, 1465700275),
 
    ('RHEA', 1499996767, 1500033234),
    ('RHEA', 1501593597, 1501622213),
    ('RHEA', 1511700120, 1511729588),
    ('RHEA', 1514076806, 1514141337),
    ('RHEA', 1516199412, 1516238311),
    ('RHEA', 1532332188, 1532369498),
    ('RHEA', 1558967673, 1558983696),
    ('RHEA', 1558967673, 1558983696),
    ('RHEA', 1567120233, 1567141726),
    ('RHEA', 1612250115, 1612279822),
    ('RHEA', 1637518901, 1637524867),
    ('RHEA', 1646247204, 1646250276),
    ('RHEA', 1654298641, 1654306484),
    ('RHEA', 1665993037, 1665998079),
    ('RHEA', 1673418114, 1673423161),
    ('RHEA', 1710072038, 1710091169),
    ('RHEA', 1734909944, 1734922104),
    ('RHEA', 1741540671, 1741579125),
 
    ('TETHYS', 1506213903, 1506222566),
    ('TETHYS', 1558900157, 1558913368),
    ('TETHYS', 1561660053, 1561674548),
    ('TETHYS', 1563651352, 1563698228),
    ('TETHYS', 1567078575, 1567099538),
    ('TETHYS', 1600993006, 1600994982),
    ('TETHYS', 1606213368, 1606215570),
    ('TETHYS', 1634166777, 1634217061),
    ('TETHYS', 1660458484, 1660474380),
    ('TETHYS', 1713136229, 1713154519),
    ('TETHYS', 1716174363, 1716189859),
    ('TETHYS', 1719609520, 1719615428),
]

FORCE_RECOMPUTE = False
DISPLAY_SINGLE_OFFSETS = False

def test_one(body_name, img_start_num, img_end_num,
             force_recompute=False):
    # Compute singular offset files where possible
    for image_path in yield_image_filenames(img_start_num, img_end_num):
        offset_path = file_img_to_offset_path(image_path)
        if not force_recompute and os.path.exists(offset_path):
            logger.info('Skipping %s', image_path)
            continue
        obs = read_iss_file(image_path)
        logger.info('Processing %s', image_path)
        logger.info('Size %d,%d TEXP %f Filters %s %s', 
                    obs.data.shape[0], obs.data.shape[1],
                    obs.texp, obs.filter1, obs.filter2)
    
        metadata = master_find_offset(
              obs, create_overlay=True,
              allow_stars=False, allow_rings=False)
        if DISPLAY_SINGLE_OFFSETS and not metadata['body_only']:
            display_offset_data(obs, metadata, show_rings=False)
        file_write_offset_metadata(image_path, metadata)

for body_name, img_start_num, img_end_num in MOSAICS:
    test_one(body_name, img_start_num, img_end_num,
             force_recompute=FORCE_RECOMPUTE)

assert False

# Find bootstrapping candidates
 
for image_path in yield_image_filenames(img_start_num, img_end_num):
#                                        restrict_list=['N1496876400', 'N1496876347', 'N1496883920']):
#                restrict_list=['N1496876347', 'W1496877602', 'N1496883920']):
    _, image_filename = os.path.split(image_path)
    offset_path = file_img_to_offset_path(image_path)
    if not os.path.exists(offset_path):
        logger.info('No offset file for %s', image_filename)
        continue
 
    metadata = file_read_offset_metadata(image_path) # XXX
     
    bootstrap_add_file(image_path, metadata, redo_bootstrapped=True)
 
bootstrap_add_file(None, None)
