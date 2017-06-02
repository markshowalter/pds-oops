###############################################################################
# cb_main_bootstrap_init.py
#
# The main top-level driver for creating the initial list of good and candidate
# images.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import sys
import time

import msgpack
import msgpack_numpy

import cspice
import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_util_file import *
from cb_util_misc import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Cassini Backplane Main Interface for creating lists of
                bootstrap images''',
    epilog='''Default behavior is to create lists based on the complete image
              set''')

file_add_selection_arguments(parser)

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_bootstrap_init/<datetime>.log''')
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
parser.add_argument(
    '--no-write', action='store_true',
    help="Don't write the output file")

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
               'cb_main_bootstrap_init', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               None, None)

# if bootstrap_config is None:
bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
        



###############################################################################
#
# BUILD DATABASE OF BOOTSTRAP FILES
#
###############################################################################

def check_add_one_image(image_path, last_image_path):
    image_filename = file_clean_name(image_path)

    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='no')

    if metadata is None:
        main_logger.debug('%s - No offset file', image_filename)
        return

    botsim_available = False
    last_image_filename = file_clean_name(last_image_path)
    if (last_image_filename is not None and
        last_image_filename[0] == 'N' and
        image_filename[0] == 'W' and
        image_filename[1:] == last_image_filename[1:]):
        botsim_available = True

    status = metadata['status']
    if status == 'error' or status == 'skipped':
        main_logger.debug('%s - Skipping due to offset file error', 
                          image_filename)
        return

    # For right now, we're just going to do the simple thing and handle
    # the front-most body. There might be obscure cases where this doesn't
    # work.    
    bodies_metadata = metadata['bodies_metadata']
    large_body_list = metadata['large_bodies']
    if len(large_body_list) == 0:
        main_logger.debug('%s - Skipping due to no large bodies',
                          image_filename)
        return
    
    body_name = large_body_list[0]     
    if (body_name not in bootstrap_config['body_list'] or
        body_name in FUZZY_BODY_LIST or
        body_name == 'TITAN' or
        body_name not in bodies_metadata):
        main_logger.debug('%s - Skipping due to bad body %s',
                          image_filename, body_name)
        return
    body_metadata = bodies_metadata[body_name]
    if not body_metadata['size_ok']:
        main_logger.debug('%s - Skipping due to bad body size %s',
                          image_filename, body_name)
        return
    if body_metadata['latlon_mask'] is None:
        main_logger.debug('%s - Skipping due to bad body latlon mask %s',
                          image_filename, body_name)
        return
    
    if metadata['offset'] is None or metadata['offset_winner'] == 'BOTSIM':
        check_add_one_image_candidate(image_path, image_filename, 
                                      metadata, body_metadata, 
                                      botsim_available)
    else:
        if ('description' in metadata and
            metadata['description'] != 'N/A'):
            main_logger.debug('%s - Skipping good image due to image description',
                              image_filename)
            return
        check_add_one_image_good(image_path, image_filename, 
                                 metadata, body_metadata)
    
def check_add_one_image_candidate(image_path, image_filename, 
                                  offset_metadata, body_metadata,
                                  botsim_available): 
    if not offset_metadata['bootstrap_candidate']:
        main_logger.debug('%s - No offset and not bootstrap candidate',
                          file_clean_name(image_path))
        return

    body_name = body_metadata['body_name']
    filter = simple_filter_name_metadata(offset_metadata,
                                         consolidate_pol=True)

    inventory = body_metadata['inventory']
    sub_solar_lon = body_metadata['sub_solar_lon']
    sub_solar_lat = body_metadata['sub_solar_lat']
    sub_obs_lon = body_metadata['sub_observer_lon']
    sub_obs_lat = body_metadata['sub_observer_lat']
    phase_angle = body_metadata['phase_angle']
    resolution_uv = inventory['resolution']
    resolution = (resolution_uv[0]+resolution_uv[1])/2

    main_logger.info('%s - %s - %s - CAND Subsolar %6.2f %6.2f / '+
                     'Subobs %6.2f %6.2f / Res %7.2f / %s / BOTSIM AVAIL %s',
                     image_filename, 
                     cspice.et2utc(offset_metadata['midtime'], 'C', 0),
                     body_name,
                     sub_solar_lon*oops.DPR, sub_solar_lat*oops.DPR,
                     sub_obs_lon*oops.DPR, sub_obs_lat*oops.DPR,
                     resolution, filter,
                     str(botsim_available))
    
    if body_name not in CAND_IMAGE_BY_BODY_DB:
        GOOD_IMAGE_BY_BODY_DB[body_name] = []
        CAND_IMAGE_BY_BODY_DB[body_name] = []
        
    entry = (image_path, sub_solar_lat, sub_solar_lon, sub_obs_lat, sub_obs_lon, 
             phase_angle, resolution, filter, botsim_available)

    CAND_IMAGE_BY_BODY_DB[body_name].append(entry)

        
def check_add_one_image_good(image_path, image_filename, 
                             offset_metadata, body_metadata):
    body_name = body_metadata['body_name']
    
    offset_winner = offset_metadata['offset_winner']
    if offset_winner != 'STARS' and offset_winner != 'MODEL':
        main_logger.debug('%s - Skipping due to untrusted offset winner %s', 
                          image_filename, offset_winner)
        return
    
    offset_confidence = offset_metadata['confidence']
    if offset_confidence < 0.1:
        main_logger.debug('%s - Skipping due to low confidence %.2f', 
                          image_filename, offset_confidence)
        return
    already_bootstrapped = ('bootstrapped' in offset_metadata and 
                            offset_metadata['bootstrapped'])
    if already_bootstrapped:
        main_logger.debug('%s - Already bootstrapped', image_filename)
        return

    if offset_metadata['bootstrap_candidate']:
        main_logger.error('%s - Has offset and also bootstrap candidate - '+
                          'something is very wrong!')
        return
    
    filter = simple_filter_name_metadata(offset_metadata, consolidate_pol=True)
    
    if body_metadata['in_saturn_shadow']:
        main_logger.debug('%s - %s - In Saturn\'s shadow',
                          image_filename, body_name)
        return
        
    inventory = body_metadata['inventory']
    
    sub_solar_lon = body_metadata['sub_solar_lon']
    sub_solar_lat = body_metadata['sub_solar_lat']
    sub_obs_lon = body_metadata['sub_observer_lon']
    sub_obs_lat = body_metadata['sub_observer_lat']
    phase_angle = body_metadata['phase_angle']
    resolution_uv = inventory['resolution']
    resolution = (resolution_uv[0]+resolution_uv[1])/2

    bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
    if bb_area < bootstrap_config['min_area']:
        main_logger.debug('%s - %s - Too small', 
                          image_filename, body_name)
        return            

    if (body_metadata['occulted_by'] is not None and
        len(body_metadata['occulted_by']) > 0):
        main_logger.debug('%s - %s - Occulted by %s', 
                          image_filename, body_name, 
                          str(body_metadata['occulted_by']))
        return
        
    if phase_angle > bootstrap_config['max_phase_angle']:
        main_logger.debug('%s - %s - Phase angle %.2f too large', 
                          image_filename, body_name, phase_angle*oops.DPR)
        return
    
    main_logger.info('%s - %s - %s - GOOD Subsolar %6.2f %6.2f / '+
                     'Subobs %6.2f %6.2f / Res %7.2f / %s',
                     image_filename, 
                     cspice.et2utc(offset_metadata['midtime'], 'C', 0),
                     body_name,
                     sub_solar_lon*oops.DPR, sub_solar_lat*oops.DPR,
                     sub_obs_lon*oops.DPR, sub_obs_lat*oops.DPR,
                     resolution, filter)
    
    if body_name not in GOOD_IMAGE_BY_BODY_DB:
        GOOD_IMAGE_BY_BODY_DB[body_name] = []
        CAND_IMAGE_BY_BODY_DB[body_name] = []
        
    entry = (image_path, sub_solar_lat, sub_solar_lon, sub_obs_lat, sub_obs_lon, 
             phase_angle, resolution, filter)
    
    GOOD_IMAGE_BY_BODY_DB[body_name].append(entry)


start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

GOOD_IMAGE_BY_BODY_DB = {}
CAND_IMAGE_BY_BODY_DB = {}

main_logger.info('*******************************************************')
main_logger.info('*** BEGINNING BUILD INITIAL BOOTSTRAP DATABASE PASS ***')
main_logger.info('*******************************************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')

last_image_path = None
image_path_done = False

for image_path in file_yield_image_filenames_from_arguments(arguments):
    image_path_done = False
    if last_image_path is not None:
        check_add_one_image(image_path, last_image_path)
        image_path_done = True
    last_image_path = image_path

if image_path is not None and not image_path_done:
    check_add_one_image(image_path, last_image_path)

if not arguments.no_write:
    for body_name in GOOD_IMAGE_BY_BODY_DB:
        body_path = file_bootstrap_good_image_path(body_name, make_dirs=True)
        body_fp = open(body_path, 'wb')
        body_fp.write(msgpack.packb(GOOD_IMAGE_BY_BODY_DB[body_name], 
                                    default=msgpack_numpy.encode))    
        body_fp.close()
    
        body_path = file_bootstrap_candidate_image_path(body_name, make_dirs=True)
        body_fp = open(body_path, 'wb')
        body_fp.write(msgpack.packb(CAND_IMAGE_BY_BODY_DB[body_name], 
                                    default=msgpack_numpy.encode))    
        body_fp.close()

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
