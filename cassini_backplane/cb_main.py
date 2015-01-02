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

from cb_offset import *
from cb_util_file import *


_LOGGING_NAME = 'cb.' + __name__


logger = logging.getLogger(_LOGGING_NAME+'.main')

for image_filename in yield_image_filenames(1505974555, 1506049806):
    obs = read_iss_file(image_filename)
    logger.info('Processing %s', image_filename)
    logger.info('Size %d,%d TEXP %f Filters %s %s', 
                obs.data.shape[0], obs.data.shape[1],
                obs.texp, obs.filter1, obs.filter2)

    metadata = master_find_offset(obs, create_overlay=True)

    file_write_offset_metadata(image_filename, metadata)
    