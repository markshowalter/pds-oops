###############################################################################
# cb_main_botsim.py
#
# The main top-level driver for computing NAC offsets based on adjacent WAC
# images.
#
# Note that this should be run AFTER all single file offsets and also AFTER
# bootstrapping, since both of those passes will give more accurate results.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import os
import subprocess
import sys
import time
import traceback

_TKINTER_AVAILABLE = True
try:
    import Tkinter as tk
except ImportError:
    _TKINTER_AVAILABLE = False

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = ''

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for BOTSIM pairs',
    epilog='''Default behavior is to perform a BOTSIM pass on all images''')

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_botsim/<datetime>.log''')
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
            $(CB_RESULTS_ROOT)/logs/<image-path>/<image_filename>.log''')
parser.add_argument(
    '--image-logfile-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--image-console-level', metavar='LEVEL', default='warning',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')

# Arguments about subprocesses
parser.add_argument(
    '--max-subprocesses', type=int, default=0, metavar='NUM',
    help='The maximum number jobs to perform in parallel')

# Arguments about the offset process
parser.add_argument(
    '--redo-offset', action='store_true', default=False,
    help='Redo offset computation even if the BOTSIM offset file exists')
parser.add_argument(
    '--display-offset-results', action='store_true', default=False,
    help='Graphically display the results of the offset process')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def collect_cmd_line(image_path, botsim_offset):
    ret = []
    ret += ['--is-subprocess']
    ret += ['--first-image-num', '1']
    ret += ['--last-image-num', '0']
    ret += ['--main-logfile-level', 'none']
    ret += ['--main-console-level', 'none']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-console-level', 'none']
    # Stupid comma required for negative numbers 
    ret += ['--botsim-offset', ',%d,%d'%botsim_offset]
    ret += ['--force-offset']
    if arguments.profile:
        ret += ['--profile']
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
                old_image_path = SUBPROCESS_LIST[i][1]
                metadata = file_read_offset_metadata(old_image_path,
                                                     bootstrap_pref='no')
                filename = file_clean_name(old_image_path)
                results = filename + ' - ' + offset_result_str(metadata)
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
                old_image_path = SUBPROCESS_LIST[i][1]
                metadata = file_read_offset_metadata(old_image_path,
                                                     bootstrap_pref='no')
                filename = file_clean_name(old_image_path)
                results = filename + ' - ' + offset_result_str(metadata)
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                break
        time.sleep(1)

###############################################################################
#
# PERFORM INDIVIDUAL OFFSETS ON NAC/WAC IMAGES
#
###############################################################################

def process_botsim_one_image(image_path_nac, image_path_wac, redo_offset):
    offset_metadata_wac = file_read_offset_metadata(image_path_wac, 
                                                    overlay=False,
                                                    bootstrap_pref='prefer')
    if offset_metadata_wac is None:
        main_logger.info('Skipping %s - no WAC offset file', image_path_wac)
        return False

    if 'status' not in offset_metadata_wac:
        main_logger.error('Skipping %s - offset file missing status indicator',
                          image_path_wac)
        return False
    
    if offset_metadata_wac['status'] != 'ok':
        main_logger.info('Skipping %s - offset file indicates error or skipped', 
                         image_path_wac)
        return False
    
    if 'offset' not in offset_metadata_wac:
        main_logger.error('Skipping %s - WAC image not error but has no offset!',
                          image_path_wac)
        return False
    
    wac_offset = offset_metadata_wac['offset']
    
    if wac_offset is None:
        main_logger.info('Skipping %s - WAC image has failed offset', image_path_wac)
        return False
        
    botsim_offset = (wac_offset[0]*10, wac_offset[1]*10)
    
    offset_metadata_nac = file_read_offset_metadata(image_path_nac, 
                                                    overlay=False,
                                                    bootstrap_pref='prefer')
    if offset_metadata_nac is None:
        main_logger.info('Skipping %s - no NAC offset file', image_path_nac)
        return False

    nac_offset = None
    if 'offset' in offset_metadata_nac:
        nac_offset = offset_metadata_nac['offset'] 
    if nac_offset is not None:
        if ('offset_winner' in offset_metadata_nac and
            offset_metadata_nac['offset_winner'] == 'BOTSIM'):
            if not redo_offset:
                main_logger.info('Skipping %s - BOTSIM already done',
                                 image_path_nac)
                return False
        else:
            main_logger.debug('Skipping %s - NAC offset already found '+
                              '(NAC %.2f,%.2f / WAC %.2f,%.2f)', image_path_nac,
                              nac_offset[0], nac_offset[1],
                              wac_offset[0], wac_offset[1])
            if (abs(nac_offset[0]-botsim_offset[0]) > 10 or
                abs(nac_offset[1]-botsim_offset[1]) > 10):
                main_logger.warn('%s - Offsets differ by too much',
                                 image_path_nac)
            return False
            
    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_OFFSET_PY] + 
                           collect_cmd_line(image_path_nac, botsim_offset), 
                           image_path_nac) 
        return True

    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(image_path_nac, 'BOTSIM', 
                                                  bootstrap=False)
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path)
            
        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    else:
        image_log_filehandler = None

    image_logger = logging.getLogger('cb')
    
    try:   
        obs = file_read_iss_file(image_path_nac)
    except:
        main_logger.exception('File reading failed - %s', image_path_nac)
        image_logger.exception('File reading failed - %s', image_path_nac)
        metadata = {}
        err = 'File reading failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path_nac, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return True

    if arguments.profile and arguments.is_subprocess:
        # Per-image profiling
        image_pr = cProfile.Profile()
        image_pr.enable()

    try:
        metadata = master_find_offset(obs, create_overlay=True,
                                      botsim_offset=botsim_offset)
    except:
        main_logger.exception('Offset finding failed - %s', image_path_nac)
        image_logger.exception('Offset finding failed - %s', image_path_nac)
        metadata = {}
        err = 'Offset finding failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path_nac, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        return True
    
    try:
        file_write_offset_metadata(image_path_nac, metadata)
    except:
        main_logger.exception('Offset file writing failed - %s', 
                              image_path_nac)
        image_logger.exception('Offset file writing failed - %s', 
                               image_path_nac)
        metadata = {}
        err = 'Offset file writing failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        try:
            file_write_offset_metadata(image_path_nac, metadata)
        except:
            main_logger.exception('Error offset file writing failed - %s', 
                                  image_path_nac)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        return True

    png_image = offset_create_overlay_image(obs, metadata)
    file_write_png_from_image(image_path_nac, png_image)
    
    if arguments.display_offset_results:
        display_offset_data(obs, metadata, canvas_size=None)

    metadata = file_read_offset_metadata(image_path_nac,
                                         overlay=False,
                                         bootstrap_pref='no')
    filename = file_clean_name(image_path_nac)
    results = filename + ' - ' + offset_result_str(metadata)
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
    assert _TKINTER_AVAILABLE
    root = tk.Tk()
    root.withdraw()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_botsim', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
redo_offset = arguments.redo_offset

offset_xy = None

start_time = time.time()
num_files_processed = 0
num_files_skipped = 0

main_logger.info('')
main_logger.info('************************************')
main_logger.info('*** BEGINNING BOTSIM OFFSET PASS ***')
main_logger.info('************************************')
main_logger.info('')
main_logger.info('Subprocesses: %d', arguments.max_subprocesses)
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

last_nac_filename = None
last_nac_image_path = None
for image_path in file_yield_image_filenames_from_arguments(arguments):
    _, filename = os.path.split(image_path)
    if filename[0] == 'N':
        last_nac_filename = filename
        last_nac_image_path = image_path
        continue
    
    assert filename[0] == 'W'
    if (last_nac_filename is None or 
        filename[1:] != last_nac_filename[1:]):
        continue
    
    if process_botsim_one_image(last_nac_image_path, image_path, redo_offset):
        num_files_processed += 1
    else:
        num_files_skipped += 1
    last_image_path = image_path

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
