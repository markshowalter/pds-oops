###############################################################################
# cb_main_reproject_body.py
#
# The main top-level driver for reprojecting a body.
###############################################################################

_TKINTER_AVAILABLE = True
try:
    import Tkinter as tk
except ImportError:
    _TKINTER_AVAILABLE = False

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
import cb_logging
from cb_util_file import *
from cb_util_image import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--body-name ENCELADUS N1637472791_1 --image-logfile-level debug'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Reprojecting Bodies',
    epilog='''Default behavior is to reproject all bodies in all images''')

file_add_selection_arguments(parser)

# Arguments about logging
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_reproject_body/<datetime>.log''')
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
            $(CB_RESULTS_ROOT)/logs/<image-path>/<image_filename>-REPROJBODY-<datetime>.log''')
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
parser.add_argument(
    '--use-bootstrap', action='store_true', 
    help='Use bootstrapped offset file')
parser.add_argument(
    '--force-reproject', action='store_true', 
    help='Force reprojection even if the reprojection file already exists')
parser.add_argument(
    '--display-reprojection', action='store_true', 
    help='Display the resulting reprojection')
parser.add_argument(
    '--body-name', metavar='BODY NANE',
    help='The body name to reproject in each image')
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

def reproject_image(image_path, body_name, bootstrap_pref):
    repro_path = file_img_to_reproj_body_path(image_path, body_name,
                                              arguments.lat_resolution*oops.RPD,
                                              arguments.lon_resolution*oops.RPD,
                                              arguments.latlon_type,
                                              arguments.lon_direction)
    if not arguments.force_reproject and os.path.exists(repro_path):
        main_logger.info('Reproject file already exists %s', image_path)
        return
    
    main_logger.info('Reprojecting %s', image_path)
    
    obs = file_read_iss_file(image_path)
    metadata = file_read_offset_metadata(image_path, 
                                         bootstrap_pref=bootstrap_pref, 
                                         overlay=False)

    if metadata is None:
        main_logger.error('%s - No offset file found',
                          file_clean_name(image_path))
        return
    
    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(image_path, 'REPROJBODY', 
                                                  bootstrap=metadata['bootstrapped'])
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this
            
        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    else:
        image_log_filehandler = None

    image_logger.info('Reprojecting %s body %s', image_path, body_name)

    image_logger.info('Taken %s / %s / Size %d x %d / TEXP %.3f / %s+%s / '+
                      'SAMPLING %s / GAIN %d',
                      cspice.et2utc(obs.midtime, 'C', 0),
                      obs.detector, obs.data.shape[1], obs.data.shape[0], obs.texp,
                      obs.filter1, obs.filter2, obs.sampling,
                      obs.gain_mode)

    image_logger.info('Offset file %s', metadata['offset_path'])

    status = metadata['status']
    if status != 'ok':
        image_logger.info('%s - Skipping due to offset file error') 
        main_logger.info('%s - Skipping due to offset file error', 
                          file_clean_name(image_path))
        repro_metadata = {'status': 'offset file error',
                          'body_name': body_name,
                          'lat_resolution': arguments.lat_resolution*oops.RPD,
                          'lon_resolution': arguments.lat_resolution*oops.RPD,
                          'latlon_type': arguments.latlon_type,
                          'lon_direction': arguments.lon_direction}
        file_write_reproj_body(image_path, repro_metadata)
        return

    data = image_interpolate_missing_stripes(obs.data)
    offset = metadata['offset']
    
    navigation_uncertainty = metadata['model_blur_amount'] # XXX Change to Sigma
    if navigation_uncertainty is None or navigation_uncertainty < 1.:
        navigation_uncertainty = 1.
    image_logger.info('Navigation uncertainty %.2f', navigation_uncertainty)
    if offset is None:
        image_logger.error('No valid offset in offset file')
        repro_metadata = {'status': 'no offset',
                          'body_name': body_name,
                          'lat_resolution': arguments.lat_resolution*oops.RPD,
                          'lon_resolution': arguments.lat_resolution*oops.RPD,
                          'latlon_type': arguments.latlon_type,
                          'lon_direction': arguments.lon_direction}
        file_write_reproj_body(image_path, repro_metadata)
    else:
        repro_metadata = bodies_reproject(
              obs, body_name, data=data, offset=offset,
              offset_path=metadata['offset_path'],
              navigation_uncertainty=navigation_uncertainty,
              lat_resolution=arguments.lat_resolution*oops.RPD, 
              lon_resolution=arguments.lon_resolution*oops.RPD,
              latlon_type=arguments.latlon_type,
              lon_direction=arguments.lon_direction,
              mask_bad_areas=True)
    
        repro_metadata['status'] = 'ok'
        
        file_write_reproj_body(image_path, repro_metadata)

        if arguments.display_reprojection:
            mosaic_metadata = bodies_mosaic_init(body_name,
                  lat_resolution=arguments.lat_resolution*oops.RPD, 
                  lon_resolution=arguments.lon_resolution*oops.RPD,
                  latlon_type=arguments.latlon_type,
                  lon_direction=arguments.lon_direction)
            bodies_mosaic_add(mosaic_metadata, repro_metadata) 
            display_body_mosaic(mosaic_metadata, title=file_clean_name(image_path))

    cb_logging.log_remove_file_handler(image_log_filehandler)

    
#===============================================================================
# 
#===============================================================================

if arguments.profile:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_reproject_body', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)

start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

if arguments.display_reprojection:
    assert _TKINTER_AVAILABLE
    root = tk.Tk()
    root.withdraw()

body_name = arguments.body_name

if arguments.use_bootstrap:
    bootstrap_pref = 'force'
else:
    bootstrap_pref = 'no'
    
if body_name is None:
    main_logger.error('No body name specified')
    sys.exit(-1)
    
main_logger.info('**********************************')
main_logger.info('*** BEGINNING REPROJECT BODIES ***')
main_logger.info('**********************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    reproject_image(image_path, body_name, bootstrap_pref)

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
