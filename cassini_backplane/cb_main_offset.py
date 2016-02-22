###############################################################################
# cb_main_offset.py
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

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
#     command_line_str = '--first-image-num 1487299402 --last-image-num 1487302209 --max-subprocesses 4'
#     command_line_str = '--first-image-num 1481738274 --last-image-num 1496491595 --force-offset --image-console-level none --max-subprocesses 4'
#     command_line_str = '--first-image-num 1637518901 --last-image-num 1665998079 --image-console-level none --max-subprocesses 4'
#N1736967486_1
#N1736967706_1
#    command_line_str = '''--force-offset --image-console-level debug --display-offset-results
#N1760870348_1'''
#    command_line_str = '--force-offset N1496877261_8 --image-console-level debug --profile'
#    command_line_str = '--first-image-num 1507717036 --last-image-num 1507748838 --main-console-level debug --max-subprocesses 1 --profile' #--max-subprocesses 4'
#    command_line_str = '--image-pds-csv t:/external/cb_support/titan-clear-151203.csv --stars-only --max-subprocesses 4'
#    command_line_str = 'N1595336241_1 --force-offset --image-console-level debug --display-offset-results' # Smear
#    command_line_str = 'N1751425716_1 --force-offset --image-console-level debug --display-offset-results' # Smear
#    command_line_str = 'N1484580522_1 --force-offset --image-console-level debug --display-offset-results'
#    command_line_str = 'N1654250545_1 --force-offset --image-console-level debug --display-offset-results' # rings closeup
#    command_line_str = 'N1477599121_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1588310978_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1600327271_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1608902918_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1624548280_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup

#    command_line_str = 'N1589083632_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # A ring edge
#    command_line_str = 'N1591063671_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge
#    command_line_str = 'N1595336241_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge
#    command_line_str = 'N1601009125_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge
#    command_line_str = 'N1625958009_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge
#    command_line_str = 'N1492060009_1 --force-offset --image-console-level debug --display-offset-results' # Bad star match
#    command_line_str = 'N1492072293_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Bad star match
#    command_line_str = 'N1493613276_1 --force-offset --image-console-level debug --display-offset-results' # A ring anti-alias
#    command_line_str = 'N1543168726_1 --force-offset --image-console-level debug --display-offset-results' # Good star match
#    command_line_str = 'N1601009320_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # High res A ring edge - only works with blurring
    command_line_str = 'N1595336719_1 --force-offset --image-console-level debug --display-offset-results' # High res A ring edge - only works with blurring
    
    command_list = command_line_str.split()

## XXX Check restrict image list is included in first->last range 

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Offsets',
    epilog='''Default behavior is to perform an offset pass on all images
              without associated offset files''')

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

# Arguments about subprocesses
parser.add_argument(
    '--is-subprocess', action='store_true',
    help='Internal flag used to indicate this process was spawned by a parent')
parser.add_argument(
    '--max-subprocesses', type=int, default=0, metavar='NUM',
    help='The maximum number jobs to perform in parallel')

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

# Arguments about the offset process
parser.add_argument(
    '--force-offset', action='store_true', default=False,
    help='Force offset computation even if the offset file exists')
parser.add_argument(
    '--offset-redo-error', action='store_true', default=False,
    help='''Force offset computation if the offset file exists and 
            indicates a fatal error''')
parser.add_argument(
    '--display-offset-results', action='store_true', default=False,
    help='Graphically display the results of the offset process')
parser.add_argument(
    '--offset-xy', type=str,
    help='Force the offset to be x,y')
parser.add_argument(
    '--stars-only', action='store_true', default=False,
    help='Navigate only using stars')
parser.add_argument(
    '--allow-stars', dest='allow_stars', action='store_true', default=True,
    help='Include stars in navigation')
parser.add_argument(
    '--no-allow-stars', dest='allow_stars', action='store_false',
    help='Do not include stars in navigation')
parser.add_argument(
    '--rings-only', action='store_true', default=False,
    help='Navigate only using rings')
parser.add_argument(
    '--allow-rings', dest='allow_rings', action='store_true', default=True,
    help='Include rings in navigation')
parser.add_argument(
    '--no-allow-rings', dest='allow_rings', action='store_false',
    help='Do not include rings in navigation')
parser.add_argument(
    '--moons-only', action='store_true', default=False,
    help='Navigate only using moons')
parser.add_argument(
    '--allow-moons', dest='allow_moons', action='store_true', default=True,
    help='Include moons in navigation')
parser.add_argument(
    '--no-allow-moons', dest='allow_moons', action='store_false',
    help='Do not include moons in navigation')
parser.add_argument(
    '--saturn-only', action='store_true', default=False,
    help='Navigate only using Saturn')
parser.add_argument(
    '--allow-saturn', dest='allow_saturn', action='store_true', default=True,
    help='Include saturn in navigation')
parser.add_argument(
    '--no-allow-saturn', dest='allow_saturn', action='store_false',
    help='Do not include saturn in navigation')


arguments = parser.parse_args(command_list)


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def offset_result_str(image_path):
    metadata = file_read_offset_metadata(image_path)
    filename = file_clean_name(image_path)
    
    ret = filename + ' - '
    if metadata is None:
        ret += 'No offset file written'
        return ret

    if 'error' in metadata:
        ret += 'ERROR: '
        error = metadata['error']
        if error.startswith('SPICE(NOFRAMECONNECT)'):
            ret += 'SPICE KERNEL MISSING DATA AT ' + error[34:53]
        else:
            ret += error 
        return ret
    
    offset = metadata['offset']
    if offset is None:
        offset_str = '  N/A  '
    else:
        offset_str = '%3d,%3d' % tuple(offset)
    star_offset_str = '  N/A  '
    if metadata['stars_metadata'] is not None:
        star_offset = metadata['stars_metadata']['offset']
        if star_offset is not None:
            star_offset_str = '%3d,%3d' % tuple(star_offset)
    model_offset = metadata['model_offset']
    if model_offset is None:
        model_offset_str = '  N/A  '
    else:
        model_offset_str = '%3d,%3d' % tuple(model_offset)
    filter1 = metadata['filter1']
    filter2 = metadata['filter2']
    the_size = '%dx%d' % tuple(metadata['image_shape'])
    the_size = '%9s' % the_size
    the_time = cspice.et2utc(metadata['midtime'], 'C', 0)
    single_body_str = None
    if metadata['body_only']:
        single_body_str = 'Filled with ' + metadata['body_only']
    if metadata['rings_only']:
        single_body_str = 'Filled with rings'
    bootstrap_str = None
    if metadata['bootstrap_candidate']:
        bootstrap_str = 'Bootstrap cand ' + metadata['bootstrap_body']
        
    ret += the_time + ' ' + ('%4s'%filter1) + '+' + ('%4s'%filter2) + ' '
    ret += the_size
    ret += ' Final ' + offset_str
    if metadata['used_objects_type'] == 'stars':
        ret += '  STAR ' + star_offset_str
    else:
        ret += '  Star ' + star_offset_str
    if metadata['used_objects_type'] == 'model':
        ret += '  MODEL ' + model_offset_str
    else:
        ret += '  Model ' + model_offset_str
    if bootstrap_str:
        ret += ' ' + bootstrap_str
    if bootstrap_str and single_body_str:
        ret += ' '
    if single_body_str:
        ret += ' ' + single_body_str
        
    return ret
    
def collect_cmd_line(image_path):
    ret = []
    ret += ['--is-subprocess']
    ret += ['--first-image-num', '1']
    ret += ['--last-image-num', '0']
    ret += ['--main-logfile-level', 'none']
    ret += ['--main-console-level', 'none']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-console-level', 'none']
    ret += ['--force-offset']
    if arguments.profile:
        ret += ['--profile']
    if not arguments.allow_stars:
        ret += ['--no-allow-stars']
    if not arguments.allow_rings:
        ret += ['--no-allow-rings']
    if not arguments.allow_moons:
        ret += ['--no-allow-moons']
    if not arguments.allow_saturn:
        ret += ['--no-allow-saturn']
    ret += ['--image-full-path', image_path]
    
    return ret

SUBPROCESS_LIST = []

def run_and_maybe_wait(args, image_path):
    said_waiting = False
    while len(SUBPROCESS_LIST) == arguments.max_subprocesses:
        if not said_waiting:
            main_logger.debug('Waiting for a free subprocess')
            said_waiting = True
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                results = offset_result_str(SUBPROCESS_LIST[i][1])
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                break
        if len(SUBPROCESS_LIST) == arguments.max_subprocesses:
            time.sleep(1)

    main_logger.debug('Spawning subprocess %s', str(args))
        
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, image_path))

def wait_for_all():
    while len(SUBPROCESS_LIST) > 0:
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                results = offset_result_str(SUBPROCESS_LIST[i][1])
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                break


###############################################################################
#
# PERFORM INDIVIDUAL OFFSETS ON NAC/WAC IMAGES
#
###############################################################################

def process_offset_one_image(image_path, allow_stars=True, allow_rings=True,
                             allow_moons=True, allow_saturn=True,
                             offset_xy=None):
    offset_path = file_img_to_offset_path(image_path)
    if os.path.exists(offset_path):
        if not force_offset:
            if redo_offset_error:
                offset_metadata = file_read_offset_metadata(image_path)
                if 'error' not in offset_metadata:
                    main_logger.debug(
                        'Skipping %s - offset file exists and metadata OK', 
                        image_path)
                    return False
                main_logger.debug(
                    'Processing %s - offset file indicates error', image_path)
            else:
                main_logger.debug('Skipping %s - offset file exists', 
                                  image_path)
                return False
        main_logger.debug(
          'Processing %s - offset file exists but redoing offsets', image_path)
    else:
        main_logger.debug('Processing %s - no offset file', image_path)

    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_OFFSET_PY] + 
                           collect_cmd_line(image_path), image_path) 
        return True

    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(image_path, bootstrap=False)
        
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
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return True

    if arguments.profile and arguments.is_subprocess:
        # Per-image profiling
        image_pr = cProfile.Profile()
        image_pr.enable()

    try:
        metadata = master_find_offset(obs, create_overlay=True,
                                      allow_stars=allow_stars,
                                      allow_rings=allow_rings,
                                      allow_moons=allow_moons,
                                      allow_saturn=allow_saturn,
                                      force_offset=offset_xy)
    except:
        main_logger.exception('Offset finding failed - %s', image_path)
        image_logger.exception('Offset finding failed - %s', image_path)
        metadata = {}
        err = 'Offset finding failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and argument.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        return True
    
    try:
        file_write_offset_metadata(image_path, metadata)
    except:
        main_logger.exception('Offset file writing failed - %s', image_path)
        image_logger.exception('Offset file writing failed - %s', image_path)
        metadata = {}
        err = 'Offset file writing failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        try:
            file_write_offset_metadata(image_path, metadata)
        except:
            main_logger.exception('Error offset file writing failed - %s', image_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and argument.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        return True

    if arguments.display_offset_results:
        display_offset_data(obs, metadata, canvas_size=None)

    results = offset_result_str(image_path)
    main_logger.info(results)

    if arguments.profile and arguments.is_subprocess:
        image_pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        image_logger.info('Profile results:\n%s', s.getvalue())

    cb_logging.log_remove_file_handler(image_log_filehandler)

    return True

#===============================================================================
# 
#===============================================================================

if arguments.profile and arguments.max_subprocesses == 0:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

if arguments.display_offset_results:
    root = tk.Tk()
    root.withdraw()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_offset', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
force_offset = arguments.force_offset
redo_offset_error = arguments.offset_redo_error

offset_xy = None

if arguments.offset_xy:
    x, y = arguments.offset_xy.split(',')
    offset_xy = (float(x), float(y))
    
if arguments.stars_only:
    arguments.allow_rings = False
    arguments.allow_moons = False
    arguments.allow_saturn = False
if arguments.rings_only:
    arguments.allow_stars = False
    arguments.allow_moons = False
    arguments.allow_saturn = False
if arguments.moons_only:
    arguments.allow_stars = False
    arguments.allow_rings = False
    arguments.allow_saturn = False
if arguments.saturn_only:
    arguments.allow_stars = False
    arguments.allow_rings = False
    arguments.allow_moons = False
    
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

    
start_time = time.time()
num_files_processed = 0
num_files_skipped = 0

main_logger.info('')
main_logger.info('**********************************')
main_logger.info('*** BEGINNING MAIN OFFSET PASS ***')
main_logger.info('**********************************')
main_logger.info('')
main_logger.info('Allow stars:  %s', str(arguments.allow_stars))
main_logger.info('Allow rings:  %s', str(arguments.allow_rings))
main_logger.info('Allow moons:  %s', str(arguments.allow_moons))
main_logger.info('Allow Saturn: %s', str(arguments.allow_saturn))
main_logger.info('Offset XY:    %s', str(offset_xy))
main_logger.info('')

if arguments.image_full_path:
    for image_path in arguments.image_full_path:
        process_offset_one_image(
                     image_path, 
                     allow_stars=arguments.allow_stars, 
                     allow_rings=arguments.allow_rings, 
                     allow_moons=arguments.allow_moons, 
                     allow_saturn=arguments.allow_saturn,
                     offset_xy=offset_xy)
    
if first_image_number <= last_image_number:
    main_logger.info('*** Image #s %010d - %010d / Camera %s',
                     first_image_number, last_image_number,
                     restrict_camera)
    main_logger.info('*** %d subprocesses', arguments.max_subprocesses)
    if restrict_image_list is not None:
        main_logger.info('*** Images restricted to list:')
        for filename in restrict_image_list:
            main_logger.info('        %s', filename)
    for image_path in yield_image_filenames(
            first_image_number, last_image_number,
            camera=restrict_camera, restrict_list=restrict_image_list):
        if process_offset_one_image(
                        image_path,
                        allow_stars=arguments.allow_stars, 
                        allow_rings=arguments.allow_rings, 
                        allow_moons=arguments.allow_moons, 
                        allow_saturn=arguments.allow_saturn,
                        offset_xy=offset_xy):
            num_files_processed += 1
        else:
            num_files_skipped += 1

wait_for_all()

end_time = time.time()

main_logger.info('Total files processed %d', num_files_processed)
main_logger.info('Total files skipped %d', num_files_skipped)
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.profile and arguments.max_subprocesses == 0:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
