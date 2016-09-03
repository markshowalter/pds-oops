###############################################################################
# cb_main_mosaic_body.py
#
# The main top-level driver for adding a reprojection to a mosaic.
###############################################################################

import Tkinter as tk

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import sys

import oops.inst.cassini.iss as iss
import oops

from cb_bodies import *
from cb_config import *
from cb_gui_body_mosaic import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--body-name ENCELADUS W1652858990_1 W1652860082_1 W1652860669_1 W1652863123_1 W1652863294_1 --mosaic-root test --display-mosaic --reset-mosaic'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Adding Body Mosaics',
    epilog='''Default behavior is to do nothing''')

file_add_selection_arguments(parser)

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_mosaic_body/<datetime>.log''')
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
    '--verbose', action='store_true', 
    help='Verbose output')
parser.add_argument(
    '--display-mosaic', action='store_true', 
    help='Display the resulting reprojection')
parser.add_argument(
    '--body-name', metavar='BODY NANE',
    help='The body name to reproject in each image')
parser.add_argument(
    '--reset-mosaic', action='store_true', 
    help='Start with a fresh mosaic')
parser.add_argument(
    '--normalize-images', action='store_true', 
    help='Normalize reprojects to a mean of 1 before adding to mosaic')
parser.add_argument(
    '--mosaic-root', metavar='FILENAME', default='NONE',
    help='The filename root for the mosaic files')
parser.add_argument(
    '--lat-resolution', metavar='N', type=float, default=0.1,
    help='The latitude resolution deg/pix')
parser.add_argument(
    '--lon-resolution', metavar='N', type=float, default=0.1,
    help='The longitude resolution deg/pix')
parser.add_argument(
    '--latlon-type', metavar='centric|graphic', default='centric',
    help='The latitude and longitude type (centric or graphic)')
parser.add_argument(
    '--lon-direction', metavar='east|west', default='east',
    help='The longitude direction (east or west)')

arguments = parser.parse_args(command_list)


#===============================================================================
# 
#===============================================================================

def add_image_to_mosaic(mosaic_root, reset_mosaic, image_path):
    main_logger.info('Adding %s to mosaic', image_path)
    repro_metadata = file_read_reproj_body(image_path,
                                           arguments.body_name,
                                           arguments.lat_resolution*oops.RPD,
                                           arguments.lon_resolution*oops.RPD,
                                           arguments.latlon_type,
                                           arguments.lon_direction)

    if repro_metadata is None:
        main_logger.error('No reprojection file for %s', image_path)
        return
    
    body_name = repro_metadata['body_name']
    
    mosaic_metadata = file_read_mosaic_metadata(body_name, mosaic_root)
    if mosaic_metadata is None or reset_mosaic:
        mosaic_metadata = bodies_mosaic_init(body_name,
              lat_resolution=repro_metadata['lat_resolution'], 
              lon_resolution=repro_metadata['lon_resolution'],
              latlon_type=repro_metadata['latlon_type'],
              lon_direction=repro_metadata['lon_direction'])

    if image_path in mosaic_metadata['path_list']:
        return 
    
    if arguments.normalize_images:
        data = repro_metadata['img']
        mask = repro_metadata['full_mask']
        repro_metadata['img'] = data / np.mean(data[mask])
        
    bodies_mosaic_add(mosaic_metadata, repro_metadata) 

    file_write_mosaic_metadata(body_name, mosaic_root, mosaic_metadata,
                               reset_num=reset_mosaic)
    
    if arguments.display_mosaic:
        display_body_mosaic(mosaic_metadata, title=file_clean_name(image_path))

    
#===============================================================================
# 
#===============================================================================

if arguments.profile:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_mosaic_body', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               None, None)

start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

if arguments.display_mosaic:
    root = tk.Tk()
    root.withdraw()

body_name = arguments.body_name

if body_name is None:
    main_logger.error('No body name specified')
    sys.exit(-1)
    
main_logger.info('*******************************')
main_logger.info('*** BEGINNING ADD TO MOSAIC ***')
main_logger.info('*******************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

reset_mosaic = arguments.reset_mosaic

for image_path in file_yield_image_filenames_from_arguments(arguments):
    add_image_to_mosaic(arguments.mosaic_root, 
                        reset_mosaic, image_path)
    reset_mosaic = False

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
