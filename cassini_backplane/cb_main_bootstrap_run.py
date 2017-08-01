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
import time

import msgpack
import msgpack_numpy
import pickle
import scipy.ndimage.interpolation as ndinterp

import oops.inst.cassini.iss as iss
import oops

from cb_bodies import *
from cb_config import *
from cb_offset import *
from cb_util_file import *
from cb_util_misc import *

MAIN_LOG_NAME = 'cb_main_bootstrap_run'

MAXIMUM_SOLAR_DELTA = 30 * oops.RPD
MAXIMUM_OBS_DELTA = 60 * oops.RPD

command_list = sys.argv[1:]

if len(command_list) == 0:
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_BL1'
    command_line_str = 'ENCELADUS --main-console-level debug --image-console-level none'# --mosaic-root ENCELADUS_0.00_-30.00_F_F_CLEAR'
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_GRN'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Cassini Backplane Main Interface for navigating images 
                   through bootstrapping''',
    epilog='''Default behavior is to navigate all candidate images''')

# Arguments about body and shadow selection
parser.add_argument(
    'body_names', action='append', nargs='*', 
    help='Specific body names to process')
# parser.add_argument(
#     '--lon-shadow-east-only', action='store_true', default=False,
#     help='Only process image with the lat shadow direction EAST')
# parser.add_argument(
#     '--lon-shadow-west-only', action='store_true', default=False,
#     help='Only process image with the lat shadow direction WEST')
# parser.add_argument(
#     '--lat-shadow-north-only', action='store_true', default=False,
#     help='Only process image with the lon shadow direction NORTH')
# parser.add_argument(
#     '--lat-shadow-south-only', action='store_true', default=False,
#     help='Only process image with the lon shadow direction SOUTH')

# Arguments about mosaic generation
# parser.add_argument(
#     '--mosaic-root', metavar='ROOT', 
#     help='Limit processing to the given mosaic root')
# parser.add_argument(
#     '--reset-mosaics', action='store_true', default=False, 
#     help='''Reprocess the mosaic from scratch instead of doing an incremental 
#             addition''')
# parser.add_argument(
#     '--no-collapse-filters', action='store_true', default=False, 
#     help='''Don't collapse all filters into a single one by using photometric
#             averaging''')
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


# Misc arguments
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')

# Arguments about subprocesses
parser.add_argument(
    '--max-subprocesses', type=int, default=1, metavar='NUM',
    help='The maximum number jobs to perform in parallel')

log_add_arguments(parser, MAIN_LOG_NAME, 'BOOTSTRAP')

arguments = parser.parse_args(command_list)


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

        

###############################################################################
#
# MOSAIC BUILDING
#
###############################################################################


def bootstrap_mask_overlap(mask1, mask2, res1):
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

def populate_entry(body_metadata):
    inventory = body_metadata['inventory']
    resolution_uv = inventory['resolution']
    resolution = (resolution_uv[0]+resolution_uv[1])/2
    body_metadata['center_mean_resolution'] = resolution
    body_metadata['filter'] = simple_filter_name_metadata(body_metadata)
    
    reproj = body_metadata['reproj']
    latitude_pixels = int(oops.PI / reproj['lat_resolution'])
    longitude_pixels = int(oops.TWOPI / reproj['lon_resolution'])
    mask = reproj['full_mask']
    incidence = reproj['incidence']
    emission = reproj['emission']
    phase = reproj['phase']
#     plt.imshow(incidence)
#     plt.figure()
#     plt.imshow(emission)
#     plt.figure()
#     plt.imshow(phase)
#     plt.show()
    mask[phase > bootstrap_config['max_phase_angle']] = False
    mask[incidence > 70. * oops.RPD] = False
    mask[emission > 70. * oops.RPD] = False
    new_mask = np.zeros((latitude_pixels, longitude_pixels))
    lat_idx_range = reproj['lat_idx_range']
    lon_idx_range = reproj['lon_idx_range']
    new_mask[lat_idx_range[0]:lat_idx_range[1]+1,
             lon_idx_range[0]:lon_idx_range[1]+1] = mask
    reproj['full_mask'] = new_mask

def entry_str(entry):
    ret = ('%s - %s - Subsolar %6.2f %6.2f / '+
           'Subobs %6.2f %6.2f / Res %7.2f / %s') % (
          entry['image_filename'], 
          entry['body_name'],
          entry['sub_solar_lon']*oops.DPR, 
          entry['sub_solar_lat']*oops.DPR,
          entry['sub_observer_lon']*oops.DPR, 
          entry['sub_observer_lat']*oops.DPR,
          entry['center_mean_resolution'], 
          entry['filter'])

    return ret

def process_body(body_name, bootstrap_config):
    body_path = file_bootstrap_good_image_path(body_name, make_dirs=False)
    body_fp = open(body_path, 'rb')
    good_image_list_all = msgpack.unpackb(body_fp.read(),
                                          object_hook=msgpack_numpy.decode)    
    body_fp.close()

    for good_entry in good_image_list_all:
        populate_entry(good_entry)

    body_path = file_bootstrap_candidate_image_path(body_name, make_dirs=False)
    body_fp = open(body_path, 'rb')
    cand_image_list_all = msgpack.unpackb(body_fp.read(),
                                          object_hook=msgpack_numpy.decode)    
    body_fp.close()

    print '# Good', len(good_image_list_all)
    print '# Cand', len(cand_image_list_all)

    for cand_entry in cand_image_list_all:
        populate_entry(cand_entry)

    for cand_entry in cand_image_list_all:
        print 'Cand image '+entry_str(cand_entry)
        if not np.any(cand_entry['reproj']['full_mask']):
            continue
        for good_entry in good_image_list_all:
            dist = np.sqrt(
               ((cand_entry['sub_solar_lat']-good_entry['sub_solar_lat'])**2+
                (cand_entry['sub_solar_lon']-good_entry['sub_solar_lon'])**2+
                (cand_entry['sub_observer_lat']-good_entry['sub_observer_lat'])**2+
                (cand_entry['sub_observer_lon']-good_entry['sub_observer_lon'])**2))
            good_entry['dist'] = dist

#         good_image_list_all.sort(key=lambda x: x['dist'])

        for good_entry in good_image_list_all:
            joint_mask = (np.logical_and(cand_entry['reproj']['full_mask'],
                                         good_entry['reproj']['full_mask']))
#             print joint_mask.shape, np.sum(joint_mask), np.sum(cand_entry['reproj']['full_mask'])
            overlap = float(np.sum(joint_mask)) / np.sum(cand_entry['reproj']['full_mask'])
            good_entry['overlap'] = overlap

        good_image_list_all.sort(key=lambda x: x['center_mean_resolution'])

        for good_entry in good_image_list_all:
            overlap = good_entry['overlap']
            if overlap == 0:
                continue
            print '  Good image '+entry_str(good_entry) + (' / Mask %.4f'%(overlap))
        
        entries_to_try = []
        for good_entry in good_image_list_all:
            overlap = good_entry['overlap']
            if overlap < bootstrap_config['min_coverage_frac']:
                continue
            if (good_entry['center_mean_resolution'] >
                  cand_entry['center_mean_resolution'] *
                  bootstrap_config['max_res_factor']):
                continue
            entries_to_try.append(good_entry)
            
        try_offsets(cand_entry, entries_to_try, body_name,
                    bootstrap_config) 

def try_offsets(cand_entry, entries_to_try, body_name,
                bootstrap_config):
    print
    print 'Candidate: '+entry_str(cand_entry)
    cand_image_path = cand_entry['image_path']
    obs = file_read_iss_file(cand_image_path)

    for try_entry in entries_to_try:
        reproj_metadata = file_read_reproj_body(
                                    try_entry['image_path'],
                                    body_name, 
                                    arguments.lat_resolution*oops.RPD,
                                    arguments.lon_resolution*oops.RPD,
                                    arguments.latlon_type,
                                    arguments.lon_direction)
        mosaic_metadata = bodies_mosaic_init(
                 reproj_metadata['body_name'],
                 lat_resolution=reproj_metadata['lat_resolution'],
                 lon_resolution=reproj_metadata['lon_resolution'],
                 latlon_type=reproj_metadata['latlon_type'],
                 lon_direction=reproj_metadata['lon_direction'])
        mosaic_metadata['full_path'] = None

        bodies_mosaic_add(mosaic_metadata, reproj_metadata) 

        cart_data = {body_name: mosaic_metadata}
        
        metadata = master_find_offset(
                              obs, create_overlay=True,
                              allow_stars=False,
                              allow_rings=False,
                              allow_moons=True,
                              allow_saturn=False,
                              bodies_cartographic_data=cart_data,
                              add_bootstrap_info=False,
                              bootstrapped=True)

        png_image = offset_create_overlay_image(
                            obs, metadata,
                            interpolate_missing_stripes=True)
        png_path = '/tmp/'+cand_entry['image_filename']+'-'+try_entry['image_filename']+'.png'
        file_write_png_path(png_path, png_image)

        offset = metadata['offset']
        print try_entry['image_filename']+':',
        if offset is None:
            print 'NO OFFSET'
        else:
            psf_corr = metadata['corr_psf_details']
            print '%4d +/- %.2f  %4d +/- %.2f' % (offset[0], 0 if (psf_corr is None or psf_corr['sigma_x'] is None) else psf_corr['sigma_x'],
                                                offset[1], 0 if (psf_corr is None or psf_corr['sigma_y'] is None) else psf_corr['sigma_y']),
            print ' / BLUR %.2f' % metadata['model_blur_amount'],
            print ' / SS %7.2f %7.2f / SO %7.2f %7.2f' % (
                  (try_entry['sub_solar_lon']-cand_entry['sub_solar_lon'])*oops.DPR,
                  (try_entry['sub_solar_lat']-cand_entry['sub_solar_lat'])*oops.DPR,
                  (try_entry['sub_observer_lon']-cand_entry['sub_observer_lon'])*oops.DPR,
                  (try_entry['sub_observer_lat']-cand_entry['sub_observer_lat'])*oops.DPR),
            print ' / RES %5.2f / %-8s / Overlap %.4f' % (
                   try_entry['center_mean_resolution']/cand_entry['center_mean_resolution'],
                   try_entry['filter'], try_entry['overlap'])

            
#===============================================================================
# 
#===============================================================================

iss.initialize(planets=(6,))

if arguments.profile:
    pr = cProfile.Profile()
    pr.enable()

main_logger, image_logger = log_setup_main_logging(
               'cb_main_bootstrap_run', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

body_names = [x.upper() for x in arguments.body_names[0]]

if len(body_names) == 0:
    body_names = bootstrap_config['body_list']
    
body_names.sort()

start_time = time.time()

main_logger.info('*********************************************')
main_logger.info('*** BEGINNING BOOTSTRAP NAVIGATION PASS 1 ***')
main_logger.info('*** (Using original seed images)          ***')
main_logger.info('*********************************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
main_logger.info('Processing bodies: %s', str(body_names))
    
for body_name in body_names:
    process_body(body_name, bootstrap_config)
    
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
