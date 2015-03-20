###############################################################################
# cb_main.py
#
# The main top-level driver for all of CB.
###############################################################################

import cb_logging
import logging

import argparse
from datetime import datetime
import os
import os.path
import subprocess
import sys
import time
import traceback

import oops.inst.cassini.iss as iss

from cb_bootstrap import *
from cb_config import *
from cb_offset import *
from cb_util_file import *

LOGGING_SUPERCRITICAL = 60

PYTHON_EXE = '/home/rfrench/Enthought/Canopy_64bit/User/bin/python'
CBMAIN_EXE = '/seti/src/pds-tools-dev/cassini_backplane/cb_main.py'

command_list = sys.argv[1:]

if len(command_list) == 0:
#     command_line_str = '--first-image-num 1481738274 --offset-redo-error --no-bootstrap --max-subprocesses 4'
#     command_line_str = '--first-image-num 1481738274 --last-image-num 1496491595 --no-offset --image-log-console-level none'
    command_line_str = '--first-image-num 1483279205 --last-image-num 1496491595 --no-offset --image-log-console-level none'

    command_list = command_line_str.split()

## XXX Check restrict image list is included in first->last range 

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface',
    epilog="""Default behavior is to perform an offset pass on all images
              without associated offset files followed by a bootstrap pass
              on all images""")

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
             " is not a valid image name - format must be [NW]dddddddddd_d")
    return name

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help="""The full path of the logfile to write for the main loop; defaults 
            to $(RESULTS_ROOT)/logs/cb_main/<datetime>.log""")
LOGGING_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical', 'none']
parser.add_argument(
    '--main-logfile-level', metavar='LEVEL', default='debug', 
    choices=LOGGING_LEVEL_CHOICES,
    help="Choose the logging level to be output to the main loop logfile")
parser.add_argument(
    '--main-log-console-level', metavar='LEVEL', default='debug',
    choices=LOGGING_LEVEL_CHOICES,
    help="Choose the logging level to be output to stdout for the main loop")
parser.add_argument(
    '--image-logfile', metavar='FILENAME',
    help="""The full path of the logfile to write for each image file; 
            defaults to 
            $(RESULTS_ROOT)/logs/<image-path>/<image_filename>.log""")
parser.add_argument(
    '--image-logfile-level', metavar='LEVEL', default='debug',
    choices=LOGGING_LEVEL_CHOICES,
    help="Choose the logging level to be output to stdout for each image")
parser.add_argument(
    '--image-log-console-level', metavar='LEVEL', default='debug',
    choices=LOGGING_LEVEL_CHOICES,
    help="Choose the logging level to be output to stdout for each image")

# Arguments about subprocesses
parser.add_argument(
    '--is-subprocess', action='store_true',
    help="Internal flag used to indicate this process was spawned by a parent")
parser.add_argument(
    '--max-subprocesses', type=int, default=0, metavar='NUM',
    help="The maximum number jobs to perform in parallel")

# Arguments about selecting the images to process
parser.add_argument(
    '--first-image-num', type=int, default='1', metavar='IMAGE_NUM',
    help="The starting image number")
parser.add_argument(
    '--last-image-num', type=int, default='9999999999', metavar='IMAGE_NUM',
    help="The ending image number")
nacwac_group = parser.add_mutually_exclusive_group()
nacwac_group.add_argument(
    '--nac-only', action='store_true', default=False,
    help="Only process NAC images")
nacwac_group.add_argument(
    '--wac-only', action='store_true', default=False,
    help="Only process WAC images")
parser.add_argument(
    'image_name', action='append', nargs='*', type=validate_image_name,
    help="Specific image names to process")
parser.add_argument(
    '--image-full-path', action='append',
    help="The full path for an image")

# Arguments about the offset process
parser.add_argument(
    '--offset', dest='offset', action='store_true', default=True,
    help="Perform an offset computation pass (default)")
parser.add_argument(
    '--no-offset', dest='offset', action='store_false',
    help="Don't perform an offset computation pass")
parser.add_argument(
    '--offset-force', action='store_true', default=False,
    help="Force offset computation even if the offset file exists")
parser.add_argument(
    '--offset-redo-error', action='store_true', default=False,
    help="""Force offset computation if the offset file exists and 
            indicates a fatal error""")

# Arguments about the bootstrap process
parser.add_argument(
    '--bootstrap', dest='bootstrap', action='store_true', default=True,
    help="Perform a bootstrap pass (default)")
parser.add_argument(
    '--no-bootstrap', dest='bootstrap', action='store_false',
    help="Don't perform a bootstrap pass")


arguments = parser.parse_args(command_list)

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

force_offset = arguments.offset_force
redo_offset_error = arguments.offset_redo_error


###############################################################################
#
# SET UP LOGGING
#
###############################################################################

def decode_log_level(s):
    if s.upper() == 'NONE':
        return LOGGING_SUPERCRITICAL
    return getattr(logging, s.upper())

def min_log_level(level1, level2):
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
                  logging.CRITICAL, LOGGING_SUPERCRITICAL]:
        if level1 == level or level2 == level:
            return level
    return LOGGING_SUPERCRITICAL

# Set up main loop logging
main_logfile_level = decode_log_level(arguments.main_logfile_level)
main_log_console_level = decode_log_level(arguments.main_log_console_level)

# Note the main loop logger is not part of the cb.* name hierarchy.
main_logger = logging.getLogger('cb_main')
main_logger.setLevel(min_log_level(main_logfile_level,
                                   main_log_console_level))

if arguments.main_logfile is not None:
    main_log_path = arguments.main_logfile
else:
    main_log_path = os.path.join(RESULTS_ROOT, 'logs')
    if not os.path.exists(main_log_path):
        os.mkdir(main_log_path)
    main_log_path = os.path.join(main_log_path, 'cb_main')
    if not os.path.exists(main_log_path):
        os.mkdir(main_log_path)
    main_log_datetime = datetime.now().isoformat()[:-7]
    main_log_path = os.path.join(main_log_path, main_log_datetime+'.log')

main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %H:%M:%S')

main_log_file_handler = logging.FileHandler(main_log_path)
main_log_file_handler.setLevel(main_logfile_level)
main_log_file_handler.setFormatter(main_formatter)
main_logger.addHandler(main_log_file_handler)

main_log_console_handler = logging.StreamHandler()
main_log_console_handler.setLevel(main_log_console_level)
main_log_console_handler.setFormatter(main_formatter)
main_logger.addHandler(main_log_console_handler)

# Set up per-image logging
_LOGGING_NAME = 'cb.' + __name__
image_logger = logging.getLogger(_LOGGING_NAME)

image_logfile_level = decode_log_level(arguments.image_logfile_level)
image_log_console_level = decode_log_level(arguments.image_log_console_level)

cb_logging.log_set_default_level(min_log_level(image_logfile_level,
                                               image_log_console_level))
cb_logging.log_set_util_flux_level(logging.CRITICAL)

cb_logging.log_remove_console_handler()
cb_logging.log_add_console_handler(image_log_console_level)

# The rest of the per-image logging is set up in the main loops


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def collect_cmd_line(image_path):
    ret = []
    ret += ['--is-subprocess']
    ret += ['--first-image-num', '1']
    ret += ['--last-image-num', '0']
    ret += ['--main-logfile-level', 'none']
    ret += ['--main-log-console-level', 'none']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-log-console-level', 'none']
    ret += ['--offset-force']
    ret += ['--no-bootstrap']
    ret += ['--image-full-path', image_path]
    
    return ret

SUBPROCESS_LIST = []

def run_and_maybe_wait(args):
    said_waiting = False
    while len(SUBPROCESS_LIST) == arguments.max_subprocesses:
        if not said_waiting:
            main_logger.debug('WAITING')
            said_waiting = True
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i].poll() is not None:
                del SUBPROCESS_LIST[i]
                break
        if len(SUBPROCESS_LIST) == arguments.max_subprocesses:
            time.sleep(1)

    main_logger.debug('SPAWNING SUBPROCESS')
        
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append(pid)

###############################################################################
#
# FIRST PASS - PERFORM INDIVIDUAL OFFSETS ON NAC/WAC IMAGES
#
###############################################################################

def process_offset_one_image(image_path):
    offset_path = file_img_to_offset_path(image_path)
    if os.path.exists(offset_path):
        if not force_offset:
            if redo_offset_error:
                offset_metadata = file_read_offset_metadata(image_path)
                if 'error' not in offset_metadata:
                    main_logger.debug(
                        'Skipping %s - offset file exists and metadata OK', 
                        image_path)
                    return
                main_logger.info(
                    'Processing %s - offset file indicates error', image_path)
            else:
                main_logger.debug('Skipping %s - offset file exists', 
                                  image_path)
                return
        main_logger.info(
          'Processing %s - offset file exists but redoing offsets', image_path)
    else:
        main_logger.info('Processing %s - no offset file', image_path)

    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_EXE] + 
                           collect_cmd_line(image_path)) 
        return

    if image_logfile_level != LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(image_path, bootstrap=True)
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this
            
        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    else:
        image_log_filehandler = None

    image_logger = logging.getLogger('cb')
    
    try:   
        obs = read_iss_file(image_path)
    except:
        main_logger.exception('File reading failed - %s', image_path)
        image_logger.exception('File reading failed - %s', image_path)
        metadata = {}
        err = 'File reading failed:\n' + traceback.format_exc() 
        metadata['error'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return

    try:
        metadata = master_find_offset(obs, create_overlay=True)
    except:
        main_logger.exception('Offset finding failed - %s', image_path)
        image_logger.exception('Offset finding failed - %s', image_path)
        metadata = {}
        err = 'Offset finding failed:\n' + traceback.format_exc() 
        metadata['error'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return
    
    try:
        file_write_offset_metadata(image_path, metadata)
    except:
        main_logger.exception('Offset file writing failed - %s', image_path)
        image_logger.exception('Offset file writing failed - %s', image_path)
        metadata = {}
        err = 'Offset file writing failed:\n' + traceback.format_exc() 
        metadata['error'] = err
        try:
            file_write_offset_metadata(image_path, metadata)
        except:
            main_logger.exception('Error offset file writing failed - %s', image_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return

    cb_logging.log_remove_file_handler(image_log_filehandler)


if not arguments.offset:
    main_logger.info('*** Skipping main offset pass')
else:
    main_logger.info('')
    main_logger.info('**********************************')
    main_logger.info('*** BEGINNING MAIN OFFSET PASS ***')
    main_logger.info('**********************************')
    main_logger.info('')
    
    if arguments.image_full_path:
        for image_path in arguments.image_full_path:
            process_offset_one_image(image_path)
        
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
            process_offset_one_image(image_path)


###############################################################################
#
# SECOND PASS - PERFORM BOOTSTAPPING
#
###############################################################################

def process_bootstrap_one_image(image_path, image_logfile_level):
    if image_path is None:
        bootstrap_add_file(None, None)
        return
    
    _, image_filename = os.path.split(image_path)
    
    offset_path = file_img_to_offset_path(image_path)
    if not os.path.exists(offset_path):
        main_logger.debug('%s - No offset file', image_filename)
        return

    metadata = file_read_offset_metadata(image_path)

    if 'error' in metadata:
        main_logger.info('%s - Skipping due to offset file error', image_filename)
        return
         
    bootstrap_add_file(image_path, metadata, 
                       image_logfile_level=image_logfile_level, 
                       log_root='cb_main',
                       redo_bootstrapped=True)


if not arguments.bootstrap:
    main_logger.info('*** Skipping bootstrap pass')
else:
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
            process_bootstrap_one_image(image_path,
                                        image_logfile_level=image_logfile_level)

    process_bootstrap_one_image(None)
