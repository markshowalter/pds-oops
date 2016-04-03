###############################################################################
# cb_main_bootstrap.py
#
# The main top-level driver for choosing and executing bootstraps.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import csv
from datetime import datetime
import os
import subprocess
import sys
import time
import traceback

import oops.inst.cassini.iss as iss
import oops

from cb_bootstrap import *
from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file --verbose --volume COISS_2024'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Statistics',
    epilog='''Default behavior is to collect statistics on all images
              with associated offset files''')

file_add_selection_arguments(parser)

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(RESULTS_ROOT)/logs/cb_main_offset/<datetime>.log''')
LOGGING_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical', 'none']
parser.add_argument(
    '--main-logfile-level', metavar='LEVEL', default='info', 
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to the main loop logfile')
parser.add_argument(
    '--main-console-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for the main loop')
parser.add_argument(
    '--image-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for each image file; 
            defaults to 
            $(RESULTS_ROOT)/logs/<image-path>/<image_filename>.log''')
parser.add_argument(
    '--image-logfile-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--image-console-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')


# Arguments about the bootstrap process
# parser.add_argument(
#     '--bootstrap', dest='bootstrap', action='store_true', default=True,
#     help='Perform a bootstrap pass (default)')


arguments = parser.parse_args(command_list)


#===============================================================================
# 
#===============================================================================

if arguments.profile:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_bootstrap', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
# if bootstrap_config is None:
bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
        



###############################################################################
#
# BUILD DATABASE OF BOOTSTRAP FILES
#
###############################################################################

def _bootstrap_time_expired(body_name, metadata, bootstrap_config):
    known_time = None
    candidate_time = None
    # KNOWN and CANDIDATE lists are pre-sorted
    if (body_name in _BOOTSTRAP_INIT_KNOWNS and
        len(_BOOTSTRAP_INIT_KNOWNS[body_name]) > 0):
        known_time = _BOOTSTRAP_INIT_KNOWNS[body_name][0][1]['midtime'] 
    if (body_name in _BOOTSTRAP_CANDIDATES and
        len(_BOOTSTRAP_CANDIDATES[body_name]) > 0):
        candidate_time = _BOOTSTRAP_CANDIDATES[body_name][0][1]['midtime'] 
    
    if known_time is None and candidate_time is None:
        return False
    
    if known_time is None:
        min_time = candidate_time
    elif candidate_time is None:
        min_time = known_time
    else:
        min_time = min(known_time, candidate_time)
    time_diff = metadata['midtime'] - min_time
    allowed_diff = (bootstrap_config['body_list'][body_name][0] * 
                    bootstrap_config['orbit_frac'])

    return time_diff > allowed_diff
    

def check_add_one_image(image_path):
    """XXX DOCUMENT THIS"""
    image_filename = file_clean_name(image_path)

    metadata = file_read_offset_metadata(image_path, overlay=False)

    if metadata is None:
        main_logger.debug('%s - No offset file', image_filename)
        return

    if 'error' in metadata:
        main_logger.info('%s - Skipping due to offset file error', image_filename)
        return
         
#     if metadata is None:
#         # End of the file list - force everything to process
#         for body_name in _BOOTSTRAP_CANDIDATES:
#             _bootstrap_process_one_body(body_name, 
#                                         image_logfile_level,
#                                         bootstrap_config, **kwargs)
#         return

    body_name = metadata['bootstrap_body']
    if body_name is None:
        # No bootstrap body
        return

    if body_name not in bootstrap_config['body_list']:
        # Bootstrap body isn't one we handle
        return
    
    image_filename = file_clean_name(image_path)
    already_bootstrapped = ('bootstrapped' in metadata and 
                            metadata['bootstrapped'])

    if _bootstrap_time_expired(body_name, metadata, bootstrap_config):
        _bootstrap_process_one_body(body_name,
                                    image_logfile_level,
                                    bootstrap_config)

    if (metadata is not None and metadata['offset'] is not None and
        not already_bootstrapped):
        if metadata['filter1'] != 'CL1' or metadata['filter2'] != 'CL2':
            logger.info('%s - %s - Known offset for %s but not clear filter', 
                        image_filename, cspice.et2utc(metadata['midtime'], 'C', 0),
                        body_name)
            return
        if body_name not in _BOOTSTRAP_INIT_KNOWNS:
            _BOOTSTRAP_INIT_KNOWNS[body_name] = []
        _BOOTSTRAP_INIT_KNOWNS[body_name].append((image_path,metadata))
        _BOOTSTRAP_INIT_KNOWNS[body_name].sort(key=lambda x: 
                                               abs(x[1]['midtime']))
        logger.info('%s - %s - Known offset for %s', 
                    image_filename, cspice.et2utc(metadata['midtime'], 'C', 0),
                    body_name)
    elif (metadata['bootstrap_candidate'] or 
          (already_bootstrapped and redo_bootstrapped)):
        if body_name not in _BOOTSTRAP_CANDIDATES:
            _BOOTSTRAP_CANDIDATES[body_name] = []
        _BOOTSTRAP_CANDIDATES[body_name].append((image_path,metadata))
        _BOOTSTRAP_CANDIDATES[body_name].sort(key=lambda x: 
                                                    abs(x[1]['midtime']))
        logger.info('%s - %s - Candidate for %s', 
                    image_filename, cspice.et2utc(metadata['midtime'], 'C', 0),
                    body_name)
    else:
        logger.info('%s - %s - No offset and not a candidate', 
                    image_filename, cspice.et2utc(metadata['midtime'], 'C', 0))


def register_one_image(image_path, image_logfile_level):
#     if image_path is None:
#         bootstrap_add_file(None, None,
#                            image_logfile_level=image_logfile_level, 
#                            log_root='cb_main_bootstrap',
#                            redo_bootstrapped=True)
#         return
    

start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

main_logger.info('***********************************************')
main_logger.info('*** BEGINNING BUILD BOOTSTRAP DATABASE PASS ***')
main_logger.info('***********************************************')
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    check_add_one_image(image_path)

process_bootstrap_one_image(None, image_logfile_level)

end_time = time.time()
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
