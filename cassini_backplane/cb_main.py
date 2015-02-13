###############################################################################
# cb_main.py
#
# The main top-level driver for all of CB.
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


logger = logging.getLogger(_LOGGING_NAME+'.main')

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
img_start_num = 1490874611
img_end_num = 1490893221


# PHOEBE 1465643815 to 1465700275
# IAPETUS 1481738274 to 1483281115
# ENCELADUS, MIMAS, RHEA 1484506476 to 1484614776
# ENCELADUS 1487299402 to 1487415680
# DIONE 1496876347 to 1496883920
# RHEA 1499996767 to 1500033234
# ENCELADUS 1500041648 to 1500069258
# RHEA 1501593597 to 1501622213
# MIMAS 1501627117 to 1501651303
# TETHYS 1506213903 to 1506220559
# DIONE 1507733604 to 1507748838
# RHEA 1511700120 to 1511729588
# RHEA 1514076806 to 1514141337
# ENCELADUS 1516151439 to 1516171418
# RHEA 1516199412 to 1516238311
# MIMAS 1521584495 to 1521620702
# RHEA 1532332188 to 1532369498
# DIONE 1534428692 to 1534507401

force_recompute = False

# Compute singular offset files where possible

force_bootstrap = False

for image_path in yield_image_filenames(img_start_num, img_end_num):
    offset_path = file_offset_path(image_path)
    if not force_recompute and os.path.exists(offset_path):
        logger.info('Skipping %s', image_path)
        continue
    obs = read_iss_file(image_path)
    logger.info('Processing %s', image_path)
    logger.info('Size %d,%d TEXP %f Filters %s %s', 
                obs.data.shape[0], obs.data.shape[1],
                obs.texp, obs.filter1, obs.filter2)

    metadata = master_find_offset(obs, create_overlay=True,
                                  force_bootstrap_candidate=force_bootstrap,
                                  allow_stars=False, allow_rings=False)
    file_write_offset_metadata(image_path, metadata)
#    display_offset_data(obs, metadata)


# Find bootstrapping candidates

for image_path in yield_image_filenames(img_start_num, img_end_num):
    _, image_filename = os.path.split(image_path)
    offset_path = file_offset_path(image_path)
    if not os.path.exists(offset_path):
        logger.info('No offset file for %s', image_filename)
        continue

    metadata = file_read_offset_metadata(image_path, read_overlay=True) # XXX
    
    bootstrap_add_file(image_path, metadata)

bootstrap_add_file(None, None)
