###############################################################################
# cb_main_bootstrap.py
#
# The main top-level driver for all of CB.
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
#    command_line_str = '--first-image-num 1487299402 --last-image-num 1487302209'
    
    command_list = command_line_str.split()

## XXX Check restrict image list is included in first->last range 

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface',
    epilog='''Default behavior is to perform an offset pass on all images
              without associated offset files followed by a bootstrap pass
              on all images''')

def validate_image_name(name):
    valid = (len(name) == 13 and name[0] in 'NW' and name[11] == '_')
    if valid:
        try:
            _ = int(name[1:11])
            _ = int(name[12])
        except ValueError:
            valid = False
    if not valid:
        raise argparse.ArgumentTypeError(
             name+
             ' is not a valid image name - format must be [NW]dddddddddd_d')
    return name

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0

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

# Arguments about selecting the images to process
parser.add_argument(
    '--first-image-num', type=int, default='1', metavar='IMAGE_NUM',
    help='The starting image number')
parser.add_argument(
    '--last-image-num', type=int, default='9999999999', metavar='IMAGE_NUM',
    help='The ending image number')
nacwac_group = parser.add_mutually_exclusive_group()
nacwac_group.add_argument(
    '--nac-only', action='store_true', default=False,
    help='Only process NAC images')
nacwac_group.add_argument(
    '--wac-only', action='store_true', default=False,
    help='Only process WAC images')
parser.add_argument(
    'image_name', action='append', nargs='*', type=validate_image_name,
    help='Specific image names to process')
parser.add_argument(
    '--image-full-path', action='append',
    help='The full path for an image')
parser.add_argument(
    '--image-pds-csv', action='append',
    help=''''A CSV file downloaded from PDS that contains filespecs of images
to process''')

# Arguments about the bootstrap process
parser.add_argument(
    '--bootstrap', dest='bootstrap', action='store_true', default=True,
    help='Perform a bootstrap pass (default)')
parser.add_argument(
    '--no-bootstrap', dest='bootstrap', action='store_false',
    help='Don\'t perform a bootstrap pass')


arguments = parser.parse_args(command_list)


#===============================================================================
# 
#===============================================================================

if arguments.profile and arguments.max_subprocesses == 0:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_bootstrap', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
if arguments.image_pds_csv:
    for filename in arguments.image_pds_csv:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = csvreader.next()
            for colnum in xrange(len(header)):
                if header[colnum] == 'primaryfilespec':
                    break
            else:
                main_logger.error('Badly formatted CSV file %s', filename)
                sys.exit(-1)
            if arguments.image_name is None:
                arguments.image_name = []
                arguments.image_name.append([])
            for row in csvreader:
                filespec = row[colnum]
                filespec = filespec.replace('.IMG', '').replace('_CALIB', '')
                _, filespec = os.path.split(filespec)
                arguments.image_name[0].append(filespec)

restrict_camera = 'NW'
if arguments.nac_only:
    restrict_camera = 'N'
if arguments.wac_only:
    restrict_camera = 'W'

restrict_image_list = None
if arguments.image_name is not None and arguments.image_name != [[]]:
    restrict_image_list = arguments.image_name[0]

first_image_number = arguments.first_image_num
last_image_number = arguments.last_image_num

    
###############################################################################
#
# PERFORM BOOTSTAPPING
#
###############################################################################

def process_bootstrap_one_image(image_path, image_logfile_level):
    if image_path is None:
        bootstrap_add_file(None, None,
                           image_logfile_level=image_logfile_level, 
                           log_root='cb_main_bootstrap',
                           redo_bootstrapped=True)
        return
    
    image_filename = file_clean_name(image_path)

    metadata = file_read_offset_metadata(image_path)

    if metadata is None:
        main_logger.debug('%s - No offset file', image_filename)
        return

    if 'error' in metadata:
        main_logger.info('%s - Skipping due to offset file error', image_filename)
        return
         
    bootstrap_add_file(image_path, metadata, 
                       image_logfile_level=image_logfile_level, 
                       log_root='cb_main_bootstrap',
                       redo_bootstrapped=True)


start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

main_logger.info('')
main_logger.info('********************************')
main_logger.info('*** BEGINNING BOOTSTRAP PASS ***')
main_logger.info('********************************')
main_logger.info('')

if arguments.image_full_path:
    for image_path in arguments.image_full_path:
        process_bootstrap_one_image(image_path)
    
if first_image_number <= last_image_number:
    main_logger.info('*** Image #s %010d - %010d / Camera %s',
                     first_image_number, last_image_number,
                     restrict_camera)
    if restrict_image_list is not None:
        main_logger.info('*** Images restricted to list:')
        for filename in restrict_image_list:
            main_logger.info('        %s', filename)
    for image_path in yield_image_filenames(
            first_image_number, last_image_number,
            camera=restrict_camera, restrict_list=restrict_image_list):
        process_bootstrap_one_image(image_path, image_logfile_level)

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
