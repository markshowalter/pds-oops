import argparse
import copy
import cProfile, pstats, StringIO
import subprocess
import sys
import time

import msgpack
import msgpack_numpy
import pickle
import scipy.ndimage.interpolation as ndinterp

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_logging import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
# Cand image N1644778756_1 - MIMAS - Subsolar 230.94   1.86 / Subobs 264.44 -15.61 / Res    0.25 / GRN
# Extremes:
# sub_solar_lon sub_solar_lat 1
# Good image N1665933684_1 - MIMAS - Subsolar 310.88   5.45 / Subobs 196.96   4.00 / Res    0.61 / CLEAR / Mask 0.7224
# sub_solar_lon sub_solar_lat -1
# Good image N1644787615_1 - MIMAS - Subsolar 191.77   1.87 / Subobs 199.86  -4.21 / Res    0.43 / BL1 / Mask 0.5165
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1665933684_1 --main-console-level debug'
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1644787615_1 --main-console-level debug'

#####
# Cand image N1644781802_1 - MIMAS - Subsolar 217.47   1.86 / Subobs 238.36  -8.97 / Res    0.43 / BL1
# NEAR sub_solar_lon sub_solar_lat 1 N1644781164_1 - MIMAS - Subsolar 220.29   1.86 / Subobs 243.07  -9.89 / Res    0.20 / CLEAR  / Mask 0.2754
# New offset found U,V -5.00,0.00
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1644781164_1 --main-console-level debug --reset-mosaics'

                                                                                                                                                                                                                    # NEAR sub_solar_lon sub_solar_lat -1 N1644787173_1 - MIMAS - Subsolar 193.72   1.87 / Subobs 202.59  -4.44 / Res    0.41 / CLEAR  / Mask 0.4278
                                                                        # New offset found U,V -2.00,0.00
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1644787173_1 --main-console-level debug --reset-mosaics'

# NEAR sub_solar_lat sub_solar_lon 1 N1675156698_1 - MIMAS - Subsolar 218.12   7.10 / Subobs 230.11  -2.53 / Res    1.01 / CLEAR  / Mask 0.9196
# New offset found U,V -5.00,1.00
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1675156698_1 --main-console-level debug --reset-mosaics'

# NEAR sub_solar_lat sub_solar_lon -1 N1644781164_1 - MIMAS - Subsolar 220.29   1.86 / Subobs 243.07  -9.89 / Res    0.20 / CLEAR  / Mask 0.2754

# FAR sub_solar_lon sub_solar_lat 1 N1665933684_1 - MIMAS - Subsolar 310.88   5.45 / Subobs 196.96   4.00 / Res    0.61 / CLEAR  / Mask 0.3904
# Bootstrapping failed - no offset found
#     command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1665933684_1 --main-console-level debug --reset-mosaics'

# FAR sub_solar_lon sub_solar_lat -1 N1644787615_1 - MIMAS - Subsolar 191.77   1.87 / Subobs 199.86  -4.21 / Res    0.43 / BL1  / Mask 0.3987
# New offset found U,V -8.00,0.00
    command_line_str = 'N1644781802_1 --body-name MIMAS --seed-image N1644787615_1 --main-console-level debug --reset-mosaics'


    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description="",
    epilog="")

# Arguments about body and shadow selection
parser.add_argument(
    'cand_images', action='append', nargs='*', 
    help='Candidate images')
parser.add_argument(
    '--seed-image', action='append', nargs='*',
    help='The seed images to use')
parser.add_argument(
    '--body-name', type=str, default='',
    help='Body name to use for navigation')

# Arguments about mosaic generation
parser.add_argument(
    '--reset-mosaics', action='store_true', default=False, 
    help='''Reprocess the mosaic from scratch instead of doing an incremental 
            addition''')
parser.add_argument(
    '--no-collapse-filters', action='store_true', default=False, 
    help='''Don't collapse all filters into a single one by using photometric
            averaging''')
parser.add_argument(
    '--lat-resolution', metavar='N', type=float, default=0.5,
    help='The latitude resolution deg/pix')
parser.add_argument(
    '--lon-resolution', metavar='N', type=float, default=0.5,
    help='The longitude resolution deg/pix')
parser.add_argument(
    '--latlon-type', metavar='centric|graphic', default='centric',
    help='The latitude and longitude type (centric or graphic)')
parser.add_argument(
    '--lon-direction', metavar='east|west', default='east',
    help='The longitude direction (east or west)')

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_bootstrap_run/<datetime>.log''')
LOGGING_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical', 'none']
parser.add_argument(
    '--main-logfile-level', metavar='LEVEL', default='debug', 
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
    '--image-logfile-level', metavar='LEVEL', default='debug',
    choices=LOGGING_LEVEL_CHOICES,    
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--image-console-level', metavar='LEVEL', default='warning',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')

# Misc arguments
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')

# Arguments about subprocesses
parser.add_argument(
    '--max-subprocesses', type=int, default=1, metavar='NUM',
    help='The maximum number jobs to perform in parallel')


arguments = parser.parse_args(command_list)


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def collect_reproj_cmd_line(image_path, body_name, use_bootstrap):
    ret = []
    ret += ['--main-logfile-level', arguments.image_logfile_level]
    ret += ['--main-console-level', arguments.image_console_level]
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-console-level', arguments.image_console_level]
    ret += ['--lat-resolution', '%.3f'%arguments.lat_resolution] 
    ret += ['--lon-resolution', '%.3f'%arguments.lon_resolution] 
    ret += ['--latlon-type', arguments.latlon_type] 
    ret += ['--lon-direction', arguments.lon_direction] 

    if arguments.profile:
        ret += ['--profile']
    if use_bootstrap:
        ret += ['--use-bootstrap']
    ret += ['--force-reproject']
    ret += ['--body-name', body_name]
    ret += ['--image-full-path', image_path]
    
    return ret

SUBPROCESS_LIST = []

def run_reproj_and_maybe_wait(args, image_path):
    said_waiting = False
    while len(SUBPROCESS_LIST) == arguments.max_subprocesses:
        if not said_waiting:
            main_logger.debug('Waiting for a free subprocess')
            said_waiting = True
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                old_image_path = SUBPROCESS_LIST[i][1]
                filename = file_clean_name(old_image_path)
                results = filename + ' - REPROJ DONE'
                main_logger.debug(results)
                del SUBPROCESS_LIST[i]
                break
        if len(SUBPROCESS_LIST) == arguments.max_subprocesses:
            time.sleep(1)

    main_logger.debug('Spawning subprocess %s', str(args))
        
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, image_path))

def reproj_wait_for_all():
    while len(SUBPROCESS_LIST) > 0:
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                old_image_path = SUBPROCESS_LIST[i][1]
                filename = file_clean_name(old_image_path)
                results = filename + ' - REPROJ DONE'
                main_logger.debug(results)
                del SUBPROCESS_LIST[i]
                break
        time.sleep(1)

def run_mosaic(image_path, body_name, mosaic_root, reset_num,
               collapse_filters):
    args = []
    args += [PYTHON_EXE, CBMAIN_MOSAIC_BODY_PY]
    args += ['--main-logfile-level', arguments.image_logfile_level]
    args += ['--main-console-level', arguments.image_console_level]

    if arguments.profile:
        args += ['--profile']
    args += ['--body-name', body_name]
    args += ['--mosaic-root', mosaic_root]
    if reset_num:
        args += ['--reset-mosaic']
    if collapse_filters:
        args += ['--normalize-images']
    args += ['--lat-resolution', '%.3f'%arguments.lat_resolution] 
    args += ['--lon-resolution', '%.3f'%arguments.lon_resolution] 
    args += ['--latlon-type', arguments.latlon_type] 
    args += ['--lon-direction', arguments.lon_direction] 
    args += ['--image-full-path', image_path]
        
    pid = subprocess.Popen(args)

    while pid.poll() is None:
        time.sleep(1)
        
def run_offset(image_path, body_name, mosaic_root):
    args = []
    args += [PYTHON_EXE, CBMAIN_OFFSET_PY]
    args += ['--force-offset']
    args += ['--main-logfile-level', 'warning']
    args += ['--main-console-level', 'warning']
    args += ['--image-logfile-level', arguments.image_logfile_level]
    args += ['--image-console-level', arguments.image_console_level]

    if arguments.profile:
        args += ['--profile']
    mosaic_path = file_mosaic_path(body_name, mosaic_root)
    args += ['--body-cartographic-data', body_name+'='+mosaic_path]
    args += ['--image-full-path', image_path]
    
    pid = subprocess.Popen(args)

    while pid.poll() is None:
        time.sleep(1)
        

###############################################################################
#
# MOSAIC BUILDING
#
###############################################################################

def create_seed_reprojections(body_name, seed_list):
    main_logger.info('*** Reprojecting seed images')

    main_logger.debug('    Seed images:')
    for seed_image_name in seed_list:
        main_logger.debug('        %s', seed_image_name)
         
    did_any_repro = False
    for seed_image_path in seed_list:
        repro_path = file_img_to_reproj_body_path(
                                          seed_image_path, body_name,
                                          arguments.lat_resolution*oops.RPD,
                                          arguments.lon_resolution*oops.RPD,
                                          arguments.latlon_type,
                                          arguments.lon_direction)
        if os.path.exists(repro_path):
            continue

        did_any_repro = True
        run_reproj_and_maybe_wait([PYTHON_EXE, CBMAIN_REPROJECT_BODY_PY] + 
                                  collect_reproj_cmd_line(seed_image_path, 
                                                          body_name, False), 
                                  seed_image_path) 

    reproj_wait_for_all()

    if not did_any_repro:
        main_logger.info('All reprojections already exist - skipping')

def create_mosaic(body_name, seed_list, mosaic_root):
    main_logger.info('*** Creating seed mosaic')

    found_all = True

    mosaic_metadata = None
    if arguments.reset_mosaics:
        found_all = False
    else:
        mosaic_metadata = file_read_mosaic_metadata(body_name, mosaic_root)

    reset_num = arguments.reset_mosaics

    for seed_image_path in seed_list:   
        if mosaic_metadata and seed_image_path in mosaic_metadata['path_list']:
            continue
        
        main_logger.info('Adding to mosaic from %s', seed_image_path)
        found_all = False
        run_mosaic(seed_image_path, body_name, mosaic_root, reset_num,
                   not arguments.no_collapse_filters) 
        
        reset_num = False

    if found_all:
        main_logger.info('Mosaic already contains all seed images - skipping')

def perform_offsets(body_name, cand_list, mosaic_root):
    for cand_image_path in cand_list:
        cand_metadata = file_read_offset_metadata(cand_image_path, 
                                                  overlay=False,
                                                  bootstrap_pref='no')
        if cand_metadata is None:
            main_logger.error('%s - Normal offset file does not exist!',
                              file_clean_name(cand_image_path))
            continue
        
        bodies_metadata = cand_metadata['bodies_metadata']
        
        if body_name not in bodies_metadata:
            main_logger.error('%s - Body %s not in normal offset data',
                              file_clean_name(cand_image_path), body_name)
            continue
    
        cand_body_metadata = bodies_metadata[body_name]

#         if not np.any(cand_body_metadata['latlon_mask']):
#             main_logger.debug('Skipping %s - Empty latlon mask', 
#                               file_clean_name(cand_image_path))
#             bootstrap_metadata = copy.deepcopy(cand_metadata)
#             bootstrap_metadata['bootstrapped'] = True
#             bootstrap_metadata['bootstrap_status'] = 'Body not present in image'
#             file_write_offset_metadata(cand_image_path, bootstrap_metadata)
#             continue
    
        main_logger.debug('Performing offset on %s', 
                          file_clean_name(cand_image_path))
        
        run_offset(cand_image_path, body_name, mosaic_root)
        new_metadata = file_read_offset_metadata(cand_image_path, 
                                                 bootstrap_pref='force',
                                                 overlay=True)
        if new_metadata is None:
            main_logger.warning('Bootstrapping failed - program execution failure')
            bootstrap_metadata = copy.deepcopy(cand_body_metadata)
            bootstrap_metadata['bootstrapped'] = True
            bootstrap_metadata['bootstrap_status'] = 'Program execution failure'
            file_write_offset_metadata(cand_image_path, bootstrap_metadata, 
                                       bootstrap=True)
            continue
    
        if not new_metadata['bootstrapped']:
            main_logger.error('New offset file does not have bootstrapped flag set!')
            continue
        
        if new_metadata['offset'] is None:
            main_logger.info('Bootstrapping failed - no offset found')
            new_metadata['bootstrap_status'] = 'No offset found'
            file_write_offset_metadata(cand_image_path, new_metadata)
            continue
    
        new_metadata['bootstrap_status'] = 'Success'
        file_write_offset_metadata(cand_image_path, new_metadata)
        
        main_logger.info('New offset found U,V %.2f,%.2f', 
                         new_metadata['offset'][0], new_metadata['offset'][1])
        
    
#===============================================================================
# 
#===============================================================================

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'bootstrap_trial_run', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

seed_list = []
for seed_image_path in file_yield_image_filenames(restrict_list=
                                                  arguments.seed_image[0]):
    seed_list.append(seed_image_path)
cand_list = []
for cand_image_path in file_yield_image_filenames(restrict_list=
                                                  arguments.cand_images[0]):
    cand_list.append(cand_image_path)

create_seed_reprojections(arguments.body_name, seed_list)
create_mosaic(arguments.body_name, seed_list, 'TEST')
perform_offsets(arguments.body_name, cand_list, 'TEST')

if arguments.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
