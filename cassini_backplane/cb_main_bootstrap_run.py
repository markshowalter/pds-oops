###############################################################################
# cb_main_bootstrap_run.py
#
# The main top-level driver for navigating images via bootstrapping.
###############################################################################

from cb_logging import *
import logging

import argparse
import copy
import cProfile, pstats, StringIO
import subprocess
import sys

import msgpack
import msgpack_numpy

import oops.inst.cassini.iss as iss
import oops

from cb_bootstrap import *
from cb_config import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_BL1'
    command_line_str = 'ENCELADUS --main-console-level debug'# --mosaic-root ENCELADUS_0.00_-30.00_F_F_CLEAR'
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_GRN'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Cassini Backplane Main Interface for navigating images 
                   through bootstrapping''',
    epilog='''Default behavior is to navigate all candidate images''')

# Arguments about logging
parser.add_argument(
    'body_names', action='append', nargs='*', 
    help='Specific body names to process')
parser.add_argument(
    '--mosaic-root', metavar='ROOT', 
    help='Limit processing to the given mosaic root')
parser.add_argument(
    '--reset-mosaics', action='store_true', default=False, 
    help='''Reprocess the mosaic from scratch instead of doing an incremental 
            addition''')
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
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')
parser.add_argument(
    '--verbose', action='store_true', 
    help='Verbose output')

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

def run_mosaic(image_path, body_name, mosaic_root, reset_num):
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

def get_shadow_dirs(sub_solar_lon, sub_obs_lon, sub_solar_lat, sub_obs_lat):
    if sub_solar_lon > sub_obs_lon+oops.PI:
        sub_solar_lon -= oops.TWOPI
    elif sub_solar_lon < sub_obs_lon-oops.PI:
        sub_solar_lon += oops.TWOPI
    lon_shadow_dir = sub_solar_lon < sub_obs_lon

    if sub_solar_lat > sub_obs_lat+oops.PI:
        sub_solar_lat -= oops.PI
    elif sub_solar_lat < sub_obs_lat-oops.PI:
        sub_solar_lat += oops.PI
    lat_shadow_dir = sub_solar_lat < sub_obs_lat

    return lon_shadow_dir, lat_shadow_dir

def process_body(body_name, good_image_list, cand_image_list):
    good_image_list.sort(key=lambda x: x[6]) # Sort by incr resolution
    
    for i in xrange(len(good_image_list)):
        good_entry = good_image_list[i]
        (good_image_path,
         good_sub_solar_lon, good_sub_solar_lat, 
         good_sub_obs_lon, good_sub_obs_lat, 
         good_phase_angle, good_resolution,
         good_filter) = good_entry
        good_lon_shadow_dir, good_lat_shadow_dir = get_shadow_dirs(
                                           good_sub_solar_lon, good_sub_obs_lon,
                                           good_sub_solar_lat, good_sub_obs_lat)
        good_entry = (good_image_path,
                      good_sub_solar_lon, good_sub_solar_lat, 
                      good_sub_obs_lon, good_sub_obs_lat, 
                      good_phase_angle, good_resolution,
                      good_lon_shadow_dir, good_lat_shadow_dir,
                      good_filter)
        good_image_list[i] = good_entry
        
    for i in xrange(len(cand_image_list)):
        cand_entry = cand_image_list[i]
        (cand_image_path,
         cand_sub_solar_lon, cand_sub_solar_lat, 
         cand_sub_obs_lon, cand_sub_obs_lat, 
         cand_phase_angle, cand_resolution,
         cand_filter) = cand_entry
        cand_lon_shadow_dir, cand_lat_shadow_dir = get_shadow_dirs(
                                           cand_sub_solar_lon, cand_sub_obs_lon,
                                           cand_sub_solar_lat, cand_sub_obs_lat)
        cand_entry = (cand_image_path,
                      cand_sub_solar_lon, cand_sub_solar_lat, 
                      cand_sub_obs_lon, cand_sub_obs_lat, 
                      cand_phase_angle, cand_resolution,
                      cand_lon_shadow_dir, cand_lat_shadow_dir,
                      cand_filter)
        cand_image_list[i] = cand_entry
    
    lon_bin_size = bootstrap_config['mosaic_lon_bin_size']
    lat_bin_size = bootstrap_config['mosaic_lat_bin_size']
    
    lon_bin_list = (np.arange(int(np.ceil(oops.TWOPI / lon_bin_size))) * 
                    lon_bin_size)
    lat_bin_list = (np.arange(int(np.ceil(oops.PI / lat_bin_size))) * 
                    lat_bin_size)-oops.HALFPI
    
    # Build up the seed mosaics for each bin
    bin_list = []
    
    for lon_shadow_dir in [False, True]:
        for lat_shadow_dir in [False, True]:
            for lon_bin in lon_bin_list:
                for lat_bin in lat_bin_list:
                    main_logger.debug(
                         'Analyzing bin LON %6.2f LAT %6.2f '+
                         'LONSHAD %5s LATSHAD %5s',
                         lon_bin * oops.DPR, lat_bin * oops.DPR,
                         str(lon_shadow_dir), str(lat_shadow_dir))
                    bin_good_image_list = []
                    for good_entry in good_image_list:
                        (good_image_path,
                         good_sub_solar_lon, good_sub_solar_lat, 
                         good_sub_obs_lon, good_sub_obs_lat, 
                         good_phase_angle, good_resolution,
                         good_lon_shadow_dir, good_lat_shadow_dir,
                         good_filter) = good_entry
                        if (good_lon_shadow_dir == lon_shadow_dir and
                            good_lat_shadow_dir == lat_shadow_dir and
                            lon_bin < good_sub_solar_lon <= 
                                        lon_bin+lon_bin_size and
                            lat_bin < good_sub_solar_lat <=
                                        lat_bin+lat_bin_size):
                            bin_good_image_list.append(good_entry)
                            main_logger.debug(
                                 'Keeping SEED %s - Subsolar %6.2f '+
                                 '%6.2f / Subobs %6.2f %6.2f / Res %7.2f / %s',
                                 file_clean_name(good_image_path), 
                                 good_sub_solar_lon*oops.DPR, 
                                 good_sub_solar_lat*oops.DPR,
                                 good_sub_obs_lon*oops.DPR, 
                                 good_sub_obs_lat*oops.DPR,
                                 good_resolution, good_filter)
                    bin_cand_image_list = []
                    for cand_entry in cand_image_list:
                        (cand_image_path,
                         cand_sub_solar_lon, cand_sub_solar_lat, 
                         cand_sub_obs_lon, cand_sub_obs_lat, 
                         cand_phase_angle, cand_resolution,
                         cand_lon_shadow_dir, cand_lat_shadow_dir,
                         cand_filter) = cand_entry
                        if (cand_lon_shadow_dir == lon_shadow_dir and
                            cand_lat_shadow_dir == lat_shadow_dir and
                            lon_bin < cand_sub_solar_lon <= 
                                        lon_bin+lon_bin_size and
                            lat_bin < cand_sub_solar_lat <=
                                        lat_bin+lat_bin_size):
                            bin_cand_image_list.append(cand_entry)
                            main_logger.debug(
                                 'Keeping CAND %s - Subsolar %6.2f '+
                                 '%6.2f / Subobs %6.2f %6.2f / Res %7.2f / %s',
                                 file_clean_name(cand_image_path), 
                                 cand_sub_solar_lon*oops.DPR, 
                                 cand_sub_solar_lat*oops.DPR,
                                 cand_sub_obs_lon*oops.DPR, 
                                 cand_sub_obs_lat*oops.DPR,
                                 cand_resolution, cand_filter)
                    if len(bin_good_image_list) > 0:
                        bin_list.append((bin_good_image_list, bin_cand_image_list,
                                         lon_bin, lat_bin, 
                                         lon_shadow_dir, lat_shadow_dir))

    for bin_info in bin_list:
        (bin_good_image_list, bin_cand_image_list,
         lon_bin, lat_bin,
         lon_shadow_dir, lat_shadow_dir) = bin_info
        filters = {}
        for good_entry in bin_good_image_list:
            (good_image_path,
             good_sub_solar_lon, good_sub_solar_lat, 
             good_sub_obs_lon, good_sub_obs_lat, 
             good_phase_angle, good_resolution,
             good_lon_shadow_dir, good_lat_shadow_dir,
             good_filter) = good_entry
            filters[good_filter] = True
        for filter in sorted(filters.keys()):
            new_good_image_list = [x for x in bin_good_image_list if
                                   x[9] == filter]
            new_cand_image_list = [x for x in bin_cand_image_list if
                                   x[9] == filter]
            if len(new_cand_image_list) == 0:
                main_logger.debug(
                      'Skipping bin %.2f_%.2f_%s_%s_%s because there are '+
                      'no candidate images',
                      lon_bin*oops.DPR, lat_bin*oops.DPR, 
                      str(lon_shadow_dir)[0], str(lat_shadow_dir)[0], 
                      filter)
                continue
            process_mosaic_bin(body_name, filter, new_good_image_list, 
                               lon_bin, lat_bin,
                               lon_shadow_dir, lat_shadow_dir,
                               new_cand_image_list)

def process_mosaic_bin(body_name, filter,
                       bin_good_image_list, lon_bin, lat_bin,
                       lon_shadow_dir, lat_shadow_dir,
                       cand_image_list):
    mosaic_root = '%s__%.3f_%.3f_%s_%s__%.2f_%.2f_%s_%s_%s' % (
                               body_name, 
                               arguments.lat_resolution,
                               arguments.lon_resolution,
                               arguments.latlon_type,
                               arguments.lon_direction,
                               lon_bin*oops.DPR, lat_bin*oops.DPR, 
                               str(lon_shadow_dir)[0], str(lat_shadow_dir)[0], 
                               filter)
    
    if arguments.mosaic_root and not mosaic_root.startswith(arguments.mosaic_root):
        return
    
    main_logger.info('*** Reprojecting bin %s', mosaic_root)

    main_logger.debug('    Good images:')
    for good_entry in bin_good_image_list:
        (good_image_path,
         good_sub_solar_lon, good_sub_solar_lat, 
         good_sub_obs_lon, good_sub_obs_lat, 
         good_phase_angle, good_resolution,
         good_lon_shadow_dir, good_lat_shadow_dir,
         good_filter) = good_entry
        main_logger.debug('        %s', file_clean_name(good_image_path))
         
    did_any_repro = False
    for good_entry in bin_good_image_list:
        (good_image_path,
         good_sub_solar_lon, good_sub_solar_lat, 
         good_sub_obs_lon, good_sub_obs_lat, 
         good_phase_angle, good_resolution,
         good_lon_shadow_dir, good_lat_shadow_dir,
         good_filter) = good_entry

        repro_path = file_img_to_reproj_body_path(
                                          good_image_path, body_name,
                                          arguments.lat_resolution*oops.RPD,
                                          arguments.lon_resolution*oops.RPD,
                                          arguments.latlon_type,
                                          arguments.lon_direction)
        if os.path.exists(repro_path):
            continue

        did_any_repro = True
        run_reproj_and_maybe_wait([PYTHON_EXE, CBMAIN_REPROJECT_BODY_PY] + 
                                  collect_reproj_cmd_line(good_image_path, 
                                                          body_name, False), 
                                  good_image_path) 

    reproj_wait_for_all()

    if not did_any_repro:
        main_logger.info('All reprojections already exist - skipping')
            
    main_logger.info('*** Creating seed mosaic for bin %s', mosaic_root)

    found_all = True

    mosaic_metadata = None
    if arguments.reset_mosaics:
        found_all = False
    else:
        mosaic_metadata = file_read_mosaic_metadata(body_name, mosaic_root)

    reset_num = arguments.reset_mosaics
    
    for good_entry in bin_good_image_list:
        (good_image_path,
         good_sub_solar_lon, good_sub_solar_lat, 
         good_sub_obs_lon, good_sub_obs_lat, 
         good_phase_angle, good_resolution,
         good_lon_shadow_dir, good_lat_shadow_dir,
         good_filter) = good_entry

        if mosaic_metadata and good_image_path in mosaic_metadata['path_list']:
            continue
        
        main_logger.info('Adding to mosaic from %s', good_image_path)
        found_all = False
        run_mosaic(good_image_path, body_name, mosaic_root, reset_num) 
        
        reset_num = False

    if found_all:
        main_logger.info('Mosaic already contains all seed images - skipping')

    # Make a new candidate list with all the body metadata preloaded
    
    main_logger.debug('Reading candidate image metadata')
    
    new_cand_image_list = []
    for cand_entry in cand_image_list:
        (cand_image_path,
         cand_sub_solar_lon, cand_sub_solar_lat, 
         cand_sub_obs_lon, cand_sub_obs_lat, 
         cand_phase_angle, cand_resolution,
         cand_lon_shadow_dir, cand_lat_shadow_dir,
         cand_filter) = cand_entry
        boot_metadata = file_read_offset_metadata(cand_image_path, 
                                                  overlay=False,
                                                  bootstrap_pref='force')
        if boot_metadata is not None:
            if boot_metadata['bootstrap_status'] == 'Success':
                main_logger.debug('Skipping %s - Previous bootstrap successful',
                                  file_clean_name(cand_image_path))
                body_metadata = boot_metadata['bodies_metadata'][body_name]
                if not body_metadata['in_saturn_shadow']:
                    repro_path = file_img_to_reproj_body_path(
                                              cand_image_path, body_name,
                                              arguments.lat_resolution*oops.RPD,
                                              arguments.lon_resolution*oops.RPD,
                                              arguments.latlon_type,
                                              arguments.lon_direction)
                    if not os.path.exists(repro_path):
                        main_logger.debug('   ...but has no repro file - creating')
                        run_reproj_and_maybe_wait([PYTHON_EXE, CBMAIN_REPROJECT_BODY_PY] + 
                                                  collect_reproj_cmd_line(cand_image_path, 
                                                                          body_name, True),
                                                  cand_image_path) 
                        reproj_wait_for_all()
    
                    if (not mosaic_metadata or 
                        cand_image_path not in mosaic_metadata['path_list']):
                        main_logger.debug('   ...but is not in mosaic - adding')
                        run_mosaic(cand_image_path, body_name, 
                                   mosaic_root, False) 
                continue
            elif (boot_metadata['bootstrap_status'] == 'Insufficient overlap' or
                  boot_metadata['bootstrap_status'] == 'Offset finding failed'):
                main_logger.debug('Skipping %s - Previously exhausted all '+
                                  'possibilities', 
                                  file_clean_name(cand_image_path))
                continue

        cand_metadata = file_read_offset_metadata(cand_image_path, 
                                                  overlay=False,
                                                  bootstrap_pref='no')
        if cand_metadata is None:
            main_logger.error('%s - Bootstrap offset exists but not normal offset!',
                              file_clean_name(cand_image_path))
            continue
        
        bodies_metadata = cand_metadata['bodies_metadata']
        
        if body_name not in bodies_metadata:
            main_logger.error('%s - Body %s not in normal offset data',
                              file_clean_name(cand_image_path), body_name)
            continue

        cand_body_metadata = bodies_metadata[body_name]
        total_low_res = np.sum(cand_body_metadata['latlon_mask'])
        if total_low_res == 0:
            main_logger.debug('Skipping %s - Empty latlon mask', 
                              file_clean_name(cand_image_path))
            continue

        main_logger.debug('Keeping %s', file_clean_name(cand_image_path))
        
        new_cand_entry = [cand_image_path,
                          cand_resolution,
                          cand_body_metadata, 
                          2, -1, 0] 
        new_cand_image_list.append(new_cand_entry)

    while len(new_cand_image_list) > 0:
        mosaic_metadata = file_read_mosaic_metadata(body_name, mosaic_root)
        mosaic_full_mask = mosaic_metadata['full_mask']
        mosaic_res = mosaic_metadata['resolution']
        best_entry_idx = None
        highest_res = None
        main_logger.debug('    Current candidate images:')
        for entry_idx, cand_entry in enumerate(new_cand_image_list):
            (cand_image_path,
             cand_resolution,
             cand_body_metadata, 
             cand_last_overlap_frac,
             cand_last_tried_overlap_frac,
             cand_last_mosaic_overlap_res) = cand_entry
            cand_mask = cand_body_metadata['latlon_mask'].reshape((180,360)) # XXX
            overlap_arr, mosaic_overlap_res = _bootstrap_mask_overlap(
                                                          mosaic_full_mask, 
                                                          cand_mask, mosaic_res)
            
            mosaic_res_improved_str = ' '
            if mosaic_overlap_res < cand_last_mosaic_overlap_res:
                mosaic_res_improved_str = '+'
                
            factor = (float(overlap_arr.shape[0]) * overlap_arr.shape[1] /
                      (cand_mask.shape[0] * cand_mask.shape[1]))
            # overlap_arr is in super-high-res
            # factor is the amount we have to reduce overlap_arr to have the resolution
            # of the cand latlon_mask.
            overlap_low_res = np.sum(overlap_arr) / factor
            # The total number of grid points in the image
            total_low_res = np.sum(cand_mask)
            frac_used = overlap_low_res / total_low_res

            frac_used_str = ' '
            if frac_used > cand_last_overlap_frac:
                frac_used_str = '+'
            
            cand_entry[3] = frac_used
            cand_entry[5] = mosaic_overlap_res
            
            descr = '        %-16s%6.2f%%%s %6.4f %6.4f%s' % (
                  file_clean_name(cand_image_path), 
                  frac_used*100, frac_used_str,
                  cand_resolution, 
                  mosaic_overlap_res, mosaic_res_improved_str)

            if (mosaic_overlap_res > 
                cand_resolution*bootstrap_config['max_res_factor']):
                main_logger.debug(descr+' - Resolution difference too great')
                continue
                
            if frac_used == cand_last_tried_overlap_frac:
                main_logger.debug(descr+' - Already tried this overlap') 
                continue
            main_logger.debug(descr)
            if frac_used >= bootstrap_config['min_coverage_frac']:
                if highest_res is None or highest_res < cand_resolution:
                    best_entry_idx = entry_idx
                    highest_res = cand_resolution 
        
        if best_entry_idx is None:
            main_logger.debug('No more valid candidates')
            break

        cand_entry = new_cand_image_list[best_entry_idx]
        cand_entry[4] = cand_entry[3]
        (cand_image_path,
         cand_resolution,
         cand_body_metadata,
         cand_last_overlap_frac,
         cand_last_tried_overlap_frac,
         cand_last_mosaic_overlap_res) = cand_entry
        main_logger.debug('    Choosing %-16s%6.2f%% %6.4f', 
                          file_clean_name(cand_image_path), 
                          cand_last_overlap_frac*100,
                          cand_resolution)
        if find_offset_and_update(body_name, mosaic_root, mosaic_metadata,
                                  cand_image_path, cand_body_metadata):
            del new_cand_image_list[best_entry_idx]

    for entry_idx, cand_entry in enumerate(new_cand_image_list):
        (cand_image_path,
         cand_resolution,
         cand_body_metadata,
         cand_last_overlap_frac,
         cand_last_tried_overlap_frac,
         cand_last_mosaic_overlap_res) = cand_entry
        cand_metadata = file_read_offset_metadata(cand_image_path, 
                                                  overlay=False,
                                                  bootstrap_pref='no')
        bootstrap_metadata = copy.deepcopy(cand_metadata)
        bootstrap_metadata['bootstrapped'] = True
        if cand_last_tried_overlap_frac < bootstrap_config['min_coverage_frac']:
            bootstrap_metadata['bootstrap_status'] = 'Insufficient overlap'
        else:
            bootstrap_metadata['bootstrap_status'] = 'Final offset finding failed'
        main_logger.debug('Marking %s as failed', file_clean_name(cand_image_path))
        file_write_offset_metadata(cand_image_path, bootstrap_metadata)

def _bootstrap_mask_overlap(mask1, mask2, res1):
    # mask1 must be the mosaic
    if mask2 is None:
        return np.zeros(mask1.shape)
    
    # Scale the masks along each dimension to be the size of the maximum
    scale1 = float(mask1.shape[0]) / mask2.shape[0]
    scale2 = float(mask1.shape[1]) / mask2.shape[1]
    
    if scale1 < 1. and scale2 < 1.:
        mask1 = ndinterp.zoom(mask1, (1./scale1,1./scale2), order=0)
        res1 = ndinterp.zoom(res1, (1./scale1,1./scale2), order=0)
    elif scale1 > 1. and scale2 > 1.:
        mask2 = ndinterp.zoom(mask2, (scale1,scale2), order=0)
    else:
        if scale1 < 1.:
            mask1 = ndinterp.zoom(mask1, (1./scale1,1), order=0)
            res1 = ndinterp.zoom(res1, (1./scale1,1), order=0)
        elif scale1 > 1.:
            mask2 = ndinterp.zoom(mask2, (scale1,1), order=0)
        
        if scale2 < 1.:
            mask2 = ndinterp.zoom(mask2, (1,1./scale2), order=0)
        elif scale2 > 1.:
            mask1 = ndinterp.zoom(mask1, (1,scale2), order=0)
            res1 = ndinterp.zoom(res1, (1,scale2), order=0)

    # Deal with roundoff error
    if mask1.shape != mask2.shape:
        if mask1.shape[0] < mask2.shape[0]:
            mask2 = mask2[:mask1.shape[0],:]
        elif mask1.shape[0] > mask2.shape[0]:
            mask1 = mask1[:mask2.shape[0],:]
            res1 = res1[:mask2.shape[0],:]
        if mask1.shape[1] < mask2.shape[1]:
            mask2 = mask2[:,mask1.shape[1]]
        elif mask1.shape[1] > mask2.shape[1]:
            mask1 = mask1[:,mask2.shape[1]]
            res1 = res1[:,mask2.shape[1]]
    
    intersect = np.logical_and(mask1, mask2)
    
    if not np.any(intersect):
        res = 0.
    else:
        res = np.min(res1[intersect])
        
    return intersect, res
    
def find_offset_and_update(body_name, mosaic_root, mosaic_metadata,
                           cand_path, cand_body_metadata):
    main_logger.info('Bootstrapping candidate %s - running offset', 
                     file_clean_name(cand_path))
#    print overlap_arr.shape
#    print cand_mask.shape
#    print factor
#    print overlap_low_res
#    print total_low_res
#    print frac_used    
#    plt.imshow(cand_mask)
#    plt.figure()
#    plt.imshow(mosaic_metadata['full_mask'])
#    plt.figure()
#    plt.imshow(overlap_arr)
#    plt.show()

    run_offset(cand_path, body_name, mosaic_root)
    new_metadata = file_read_offset_metadata(cand_path, bootstrap_pref='force',
                                             overlay=False)
    if new_metadata is None:
        main_logger.warning('Bootstrapping failed - program execution failure')
        bootstrap_metadata = copy.deepcopy(cand_body_metadata)
        bootstrap_metadata['bootstrapped'] = True
        bootstrap_metadata['bootstrap_status'] = 'Program execution failure'
        file_write_offset_metadata(cand_path, bootstrap_metadata, 
                                   bootstrap=True)
        return False

    if not new_metadata['bootstrapped']:
        main_logger.error('New offset file does not have bootstrapped flag set!')
        return True
    
    if new_metadata['offset'] is None:
        main_logger.info('Bootstrapping failed - no offset found')
        new_metadata['bootstrap_status'] = 'No offset found'
        file_write_offset_metadata(cand_path, new_metadata)
        return False

    new_metadata['bootstrap_status'] = 'Success'
    file_write_offset_metadata(cand_path, new_metadata)
    
    main_logger.info('New offset found U,V %.2f,%.2f', 
                     new_metadata['offset'][0], new_metadata['offset'][1])
    
    new_bodies_metadata = new_metadata['bodies_metadata']
    if new_bodies_metadata is None:
        main_logger.error('New offset file has no bodies metadata!')
        return True
    
    new_body_metadata = new_bodies_metadata[body_name]
    if new_body_metadata['in_saturn_shadow']:
        main_logger.info(
                 'Image is in Saturn\'s shadow - not adding to mosaic')
        return True

    main_logger.debug('Running reprojection')
    
    run_reproj_and_maybe_wait([PYTHON_EXE, CBMAIN_REPROJECT_BODY_PY] + 
                              collect_reproj_cmd_line(cand_path, 
                                                      body_name, True),
                              cand_path) 
    reproj_wait_for_all()

    main_logger.debug('Adding to mosaic')
    
    run_mosaic(cand_path, body_name, mosaic_root, False) 

    return True

#===============================================================================
# 
#===============================================================================

if arguments.profile:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_bootstrap_run', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

# if bootstrap_config is None:
bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

body_names = [x.upper() for x in arguments.body_names[0]]

start_time = time.time()

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

IMAGE_BY_MOON_DB = {}

main_logger.info('*******************************************')
main_logger.info('*** BEGINNING BOOTSTRAP NAVIGATION PASS ***')
main_logger.info('*******************************************')
main_logger.info('')

for body_name in bootstrap_config['body_list']:
    if len(body_names) > 0 and body_name not in body_names:
        continue
    body_path = file_bootstrap_good_image_path(body_name, make_dirs=False)
    body_fp = open(body_path, 'rb')
    good_image_list = msgpack.unpackb(body_fp.read(),
                                      object_hook=msgpack_numpy.decode)    
    body_fp.close()

    body_path = file_bootstrap_candidate_image_path(body_name, make_dirs=False)
    body_fp = open(body_path, 'rb')
    cand_image_list = msgpack.unpackb(body_fp.read(),
                                      object_hook=msgpack_numpy.decode)    
    body_fp.close()

    process_body(body_name, good_image_list, cand_image_list)
    
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
