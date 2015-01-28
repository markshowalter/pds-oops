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
# 1490874718 top left good limb maybe OK for curvature - let it run? XXX
# 1490874782 top good limb good curvature - curvature marked bad XXX
# 1490874834 fills image
# 1490874889 bottom no limb
# 1490874954 bottom right no limb
# 1490875006 right good limb good curvature - curvature marked bad XXX
# 1490875063 top right good limb bad curvature
img_start_num = 1490874611
img_end_num = 1490875063

force_recompute = True

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

# Find bootstrapping candidates

known_list = []
candidate_list = []

for image_path in yield_image_filenames(img_start_num, img_end_num):
    _, image_filename = os.path.split(image_path)
    offset_path = file_offset_path(image_path)
    if not os.path.exists(offset_path):
        logger.info('No offset file for %s', image_filename)
        continue

    metadata = file_read_offset_metadata(image_path)
    
    if metadata['offset'] is not None:
        known_list.append((image_path,metadata))
        logger.debug('Known offset %s', image_filename)
    elif metadata['bootstrap_candidate']:
        candidate_list.append((image_path,metadata))
        logger.debug('Candidate %s', image_filename)

for cand_path, cand_metadata in candidate_list:
    _, cand_filename = os.path.split(cand_path)
    logger.debug('Attempting bootstrap on %s', cand_filename)
    cand_midtime = cand_metadata['midtime']
    known_list.sort(key=lambda x: abs(x[1]['midtime']-cand_midtime))
    for known_path, known_metadata in known_list:
#        if not bootstrap_viable(known_path, known_metadata,
#                            cand_path, cand_metadata):
#            pass
        bootstrap(known_path, known_metadata,
#                  known_path, known_metadata)
                  cand_path, cand_metadata)
