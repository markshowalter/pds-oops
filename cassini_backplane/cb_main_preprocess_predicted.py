###############################################################################
# cb_main_preprocess_predicted.py
#
# The main top-level driver for scanning images and determining pointing
# information from the predicted kernels. This is necessary because the
# predicted kernels have better time derivatives, which are necessary for
# things like star streaks.
#
# This pass should be run before anything else.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import os
import sys
import time
import traceback

import oops.inst.cassini.iss as iss
import oops
import polymath

from cb_config import *
from cb_util_file import *
from cb_util_oops import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'N1595480846_1 --main-console-level debug'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for preprocessing predicted kernels',
    epilog='''Default behavior is to gather information on all images''')

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_preprocess_predicted/<datetime>.log''')
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
    '--profile', action='store_true', 
    help='Do performance profiling')


file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


###############################################################################
#
# RETRIEVE PREDICTED KERNEL INFORMATION ON ONE IMAGE
#
###############################################################################

def process_predicted_one_image(image_path):
    obs = file_read_iss_file(image_path)
    center_meshgrid = make_center_meshgrid(obs)
    center_bp = oops.Backplane(obs, meshgrid=center_meshgrid,
                               time=polymath.Scalar([obs.midtime,
                                                     obs.midtime-obs.texp/2,
                                                     obs.midtime+obs.texp/2]))
    ra = center_bp.right_ascension().mvals.astype('float')
    dec = center_bp.declination().mvals.astype('float')
    dra_dt = (ra[2]-ra[1])/obs.texp
    ddec_dt = (dec[2]-dec[1])/obs.texp
    
    pred_metadata = {}
    pred_metadata['ra_center_midtime'] = ra[0]
    pred_metadata['dec_center_midtime'] = dec[0]
    pred_metadata['dra_dt'] = dra_dt
    pred_metadata['ddec_dt'] = ddec_dt
    
    file_write_predicted_metadata(image_path, pred_metadata)
    
    main_logger.debug('%s RA %9.7f DEC %9.7f DRA %16.13f DDEC %16.13f',
                      file_clean_name(image_path),
                      ra[0], dec[0], dra_dt, ddec_dt)
    
    return True

#===============================================================================
# 
#===============================================================================

iss.initialize(ck='predicted')

if arguments.profile:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_preprocess_predicted', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               None, None)

start_time = time.time()
num_files_processed = 0
num_files_skipped = 0

main_logger.info('')
main_logger.info('**************************************************')
main_logger.info('*** BEGINNING PREPROCESS PREDICTED KERNEL PASS ***')
main_logger.info('**************************************************')
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    if process_predicted_one_image(image_path):
        num_files_processed += 1
    else:
        num_files_skipped += 1

end_time = time.time()

main_logger.info('Total files processed %d', num_files_processed)
main_logger.info('Total files skipped %d', num_files_skipped)
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
