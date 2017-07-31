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

arguments = parser.parse_args(command_list)


###############################################################################
#
# BUILD DATABASE OF BOOTSTRAP FILES
#
###############################################################################

def check_add_one_image(image_path, bootstrap_config):
    image_filename = file_clean_name(image_path)

    offset_metadata = file_read_offset_metadata(image_path, overlay=False,
                                                bootstrap_pref='no')

    if offset_metadata is None:
        main_logger.debug('%s - No offset file', image_filename)
        return

    status = offset_metadata['status']
    if status != 'ok':
        main_logger.debug('%s - Skipping due to offset file error', 
                          image_filename)
        return

    # For right now, we're just going to do the simple thing and handle
    # the front-most body. There might be obscure cases where this doesn't
    # work.    
    bodies_metadata = offset_metadata['bodies_metadata']
    large_body_list = offset_metadata['large_bodies']
    if len(large_body_list) == 0:
        main_logger.debug('%s - Skipping due to no large bodies',
                          image_filename)
        return
    
    body_name = large_body_list[0]     
    if (body_name not in bootstrap_config['body_list'] or
        body_name not in bodies_metadata):
        main_logger.debug('%s - Skipping due to bad body %s',
                          image_filename, body_name)
        return

    body_metadata = bodies_metadata[body_name]
    
    if not body_metadata['size_ok']:
        main_logger.debug('%s - Skipping due to bad body size %s',
                          image_filename, body_name)
        return
    
    if 'reproj' not in body_metadata or body_metadata['reproj'] is None:
        main_logger.debug('%s - Skipping due to bad body trial reprojection %s',
                          image_filename, body_name)
        return

    body_metadata['image_path'] = image_path
    body_metadata['image_filename'] = image_filename
    body_metadata['midtime'] = offset_metadata['midtime']
    body_metadata['filter1'] = offset_metadata['filter1']
    body_metadata['filter2'] = offset_metadata['filter2']
    body_metadata['offset'] = offset_metadata['offset']
    if 'model_corr_psf_details' in body_metadata:
        body_metadata['model_corr_psf_details'] = offset_metadata[
                                                      'model_corr_psf_details']
    else:
        body_metadata['model_corr_psf_details'] = None
    body_metadata['confidence'] = offset_metadata['confidence']
    body_metadata['offset_winner'] = offset_metadata['offset_winner']

    if offset_metadata['offset'] is None:
        check_add_one_image_candidate(image_path, image_filename, 
                                      offset_metadata, body_metadata,
                                      bootstrap_config) 
    else:
        check_add_one_image_good(image_path, image_filename, 
                                 offset_metadata, body_metadata,
                                 bootstrap_config)
    
def check_add_one_image_candidate(image_path, image_filename, 
                                  offset_metadata, body_metadata,
                                  bootstrap_config):
    if not offset_metadata['bootstrap_candidate']:
        main_logger.debug('%s - No offset and not bootstrap candidate',
                          file_clean_name(image_path))
        return

    body_name = body_metadata['body_name']
    filter = simple_filter_name_metadata(offset_metadata,
                                         consolidate_pol=False)

    inventory = body_metadata['inventory']
    resolution_uv = inventory['resolution']
    resolution = (resolution_uv[0]+resolution_uv[1])/2
    sub_solar_lon = body_metadata['sub_solar_lon']
    sub_solar_lat = body_metadata['sub_solar_lat']
    sub_obs_lon = body_metadata['sub_observer_lon']
    sub_obs_lat = body_metadata['sub_observer_lat']
    phase_angle = body_metadata['phase_angle']

    main_logger.info('%s - %s - %s - CAND Subsolar %6.2f %6.2f / '+
                     'Subobs %6.2f %6.2f / Res %7.2f / %s',
                     image_filename, 
                     cspice.et2utc(offset_metadata['midtime'], 'C', 0),
                     body_name,
                     sub_solar_lon*oops.DPR, sub_solar_lat*oops.DPR,
                     sub_obs_lon*oops.DPR, sub_obs_lat*oops.DPR,
                     resolution, filter)
    
    if body_name not in CAND_IMAGE_BY_BODY_DB:
        GOOD_IMAGE_BY_BODY_DB[body_name] = []
        CAND_IMAGE_BY_BODY_DB[body_name] = []
    
    CAND_IMAGE_BY_BODY_DB[body_name].append(body_metadata)

        
def check_add_one_image_good(image_path, image_filename, 
                             offset_metadata, body_metadata,
                             bootstrap_config):
    body_name = body_metadata['body_name']
    
    offset_winner = offset_metadata['offset_winner']
    if offset_winner != 'STARS' and offset_winner != 'MODEL':
        main_logger.debug('%s - Skipping due to untrusted offset winner %s', 
                          image_filename, offset_winner)
        return
    
    offset_confidence = offset_metadata['confidence']
    if offset_confidence < bootstrap_config['min_confidence']:
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

    if ('description' in offset_metadata and
        offset_metadata['description'] != 'N/A'):
        main_logger.debug('%s - Skipping good image due to non-null '+
                          'image description', image_filename)
        return
    
    if body_metadata['in_saturn_shadow']:
        main_logger.debug('%s - %s - In Saturn\'s shadow',
                          image_filename, body_name)
        return

    if (body_metadata['occulted_by'] is not None and
        len(body_metadata['occulted_by']) > 0):
        main_logger.debug('%s - %s - Occulted by %s', 
                          image_filename, body_name, 
                          str(body_metadata['occulted_by']))
        return
        
    filter = simple_filter_name_metadata(offset_metadata, consolidate_pol=False)    
    inventory = body_metadata['inventory']
    resolution_uv = inventory['resolution']
    resolution = (resolution_uv[0]+resolution_uv[1])/2
    sub_solar_lon = body_metadata['sub_solar_lon']
    sub_solar_lat = body_metadata['sub_solar_lat']
    sub_obs_lon = body_metadata['sub_observer_lon']
    sub_obs_lat = body_metadata['sub_observer_lat']
    phase_angle = body_metadata['phase_angle']

    bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
    if bb_area < bootstrap_config['min_area']:
        main_logger.debug('%s - %s - Too small', 
                          image_filename, body_name)
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
        
    GOOD_IMAGE_BY_BODY_DB[body_name].append(body_metadata)

#===============================================================================
# 
#===============================================================================

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_bootstrap_init', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               None, None)

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

start_time = time.time()

GOOD_IMAGE_BY_BODY_DB = {}
CAND_IMAGE_BY_BODY_DB = {}

main_logger.info('*******************************************************')
main_logger.info('*** BEGINNING BUILD INITIAL BOOTSTRAP DATABASE PASS ***')
main_logger.info('*******************************************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    check_add_one_image(image_path, bootstrap_config)

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
