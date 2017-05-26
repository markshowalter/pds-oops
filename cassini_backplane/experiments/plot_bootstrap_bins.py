import argparse
import cProfile, pstats, StringIO
import matplotlib.pyplot as plt
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

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description=''' ''',
    epilog=''' ''')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


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

    return False, False # XXX
    return lon_shadow_dir, lat_shadow_dir


def check_add_one_image(image_path):
    image_filename = file_clean_name(image_path)

    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='no')

    if metadata is None:
        return

    # Fix up the metadata for old files - eventually this should
    # be removed! XXX
    if 'error' in metadata:
        metadata['status'] = 'error'
        metadata['status_detail1'] = metadata['error']
        metadata['status_detail2'] = metadata['error_traceback']
    elif 'status' not in metadata:
        metadata['status'] = 'ok'

    status = metadata['status']
    if status == 'error' or status == 'skipped':
        return
    
    if metadata['offset'] is None or metadata['offset_winner'] == 'BOTSIM':
        check_add_one_image_candidate(image_path, image_filename, metadata)
    else:
        check_add_one_image_good(image_path, image_filename, metadata)
    
def check_add_one_image_candidate(image_path, image_filename, metadata):
    if not metadata['bootstrap_candidate']:
        return

    filter = simple_filter_name_metadata(metadata, consolidate_pol=True)
    
    bodies_metadata = metadata['bodies_metadata']
    
    for body_name in metadata['large_bodies']:
        if (body_name not in bootstrap_config['body_list'] or
            body_name in FUZZY_BODY_LIST or
            body_name == 'TITAN'):
            continue
        if body_name not in bodies_metadata:
            continue
        body_metadata = bodies_metadata[body_name]
        if not body_metadata['size_ok']:
            continue

        inventory = body_metadata['inventory']
        sub_solar_lon = body_metadata['sub_solar_lon']
        sub_solar_lat = body_metadata['sub_solar_lat']
        sub_obs_lon = body_metadata['sub_observer_lon']
        sub_obs_lat = body_metadata['sub_observer_lat']
        phase_angle = body_metadata['phase_angle']
        resolution_uv = inventory['resolution']
        resolution = (resolution_uv[0]+resolution_uv[1])/2

        if body_name not in CAND_IMAGE_BY_BODY_DB:
            GOOD_IMAGE_BY_BODY_DB[body_name] = []
            CAND_IMAGE_BY_BODY_DB[body_name] = []

        lon_shadow_dir, lat_shadow_dir = get_shadow_dirs(
                                           sub_solar_lon, sub_obs_lon,
                                           sub_solar_lat, sub_obs_lat)
            
        entry = (image_path, sub_solar_lon, sub_solar_lat, sub_obs_lon, sub_obs_lat, 
                 phase_angle, resolution, filter, lon_shadow_dir, lat_shadow_dir)

        CAND_IMAGE_BY_BODY_DB[body_name].append(entry)

        
def check_add_one_image_good(image_path, image_filename, metadata):
    offset_winner = metadata['offset_winner']
    if offset_winner != 'STARS' and offset_winner != 'MODEL':
        return
    
    offset_confidence = metadata['confidence']
    if offset_confidence < 0.35:
        return
    already_bootstrapped = ('bootstrapped' in metadata and 
                            metadata['bootstrapped'])
    if already_bootstrapped:
        main_logger.debug('%s - Already bootstrapped', image_filename)
        return

    if metadata['bootstrap_candidate']:
        return
    
    filter = simple_filter_name_metadata(metadata, consolidate_pol=True)
    
    bodies_metadata = metadata['bodies_metadata']
        
    for body_name in metadata['large_bodies']:
        if body_name not in bootstrap_config['body_list']:
            # Bootstrap body isn't one we handle
            continue
        if body_name not in bodies_metadata:
            continue
        body_metadata = bodies_metadata[body_name]
        if body_metadata['in_saturn_shadow']:
            continue
        
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
            continue            

        if (body_metadata['occulted_by'] is not None and
            len(body_metadata['occulted_by']) > 0):
            continue
            
        if phase_angle > bootstrap_config['max_phase_angle']:
            continue
        
        if body_name not in GOOD_IMAGE_BY_BODY_DB:
            GOOD_IMAGE_BY_BODY_DB[body_name] = []
            CAND_IMAGE_BY_BODY_DB[body_name] = []

        lon_shadow_dir, lat_shadow_dir = get_shadow_dirs(
                                           sub_solar_lon, sub_obs_lon,
                                           sub_solar_lat, sub_obs_lat)
                
        entry = (image_path, sub_solar_lon, sub_solar_lat, sub_obs_lon, sub_obs_lat, 
                 phase_angle, resolution, filter, lon_shadow_dir, lat_shadow_dir)
        
        GOOD_IMAGE_BY_BODY_DB[body_name].append(entry)

def plot_body(body_name, good_image_list, cand_image_list):
    print body_name, 'GOOD IMAGES', len(good_image_list),
    print 'CAND IMAGES', len(cand_image_list)
    for lon_shadow_dir in [False, True]:
        for lat_shadow_dir in [False, True]:
            good_sun_lon_list = np.array([x[1]*oops.DPR for x in good_image_list 
                  if get_shadow_dirs(x[1], x[4], x[2], x[5])[0] == lon_shadow_dir and
                     get_shadow_dirs(x[1], x[4], x[2], x[5])[1] == lat_shadow_dir])
            good_sun_lat_list = np.array([x[2]*oops.DPR for x in good_image_list 
                  if get_shadow_dirs(x[1], x[4], x[2], x[5])[0] == lon_shadow_dir and
                     get_shadow_dirs(x[1], x[4], x[2], x[5])[1] == lat_shadow_dir])
            cand_sun_lon_list = np.array([x[1]*oops.DPR for x in cand_image_list 
                  if get_shadow_dirs(x[1], x[4], x[2], x[5])[0] == lon_shadow_dir and
                     get_shadow_dirs(x[1], x[4], x[2], x[5])[1] == lat_shadow_dir])
            cand_sun_lat_list = np.array([x[2]*oops.DPR for x in cand_image_list 
                  if get_shadow_dirs(x[1], x[4], x[2], x[5])[0] == lon_shadow_dir and
                     get_shadow_dirs(x[1], x[4], x[2], x[5])[1] == lat_shadow_dir])
            if len(good_sun_lon_list) == 0 or len(cand_sun_lon_list) == 0:
                continue
            plt.figure()
            plt.title('%s Lon Same=%s Lat Same=%s' % (body_name, str(lon_shadow_dir), str(lat_shadow_dir)))
            plt.plot(good_sun_lon_list, good_sun_lat_list,
                     'o', mec='green', mfc='none', ms=6)
            plt.plot(cand_sun_lon_list, cand_sun_lat_list,
                     'o', mec='red', mfc='none', ms=9)
            plt.xlim(0,360.)
            plt.ylim(-30.,30)
            plt.xlabel('Sub-solar longitude')
            plt.ylabel('Sub-solar latitude)')

        plt.show()
        
GOOD_IMAGE_BY_BODY_DB = {}
CAND_IMAGE_BY_BODY_DB = {}

#for image_path in file_yield_image_filenames_from_arguments(arguments):
#    check_add_one_image(image_path)

for body_name in bootstrap_config['body_list']:
    body_path = file_bootstrap_good_image_path(body_name, make_dirs=False)
    if not os.path.exists(body_path):
        continue
    body_fp = open(body_path, 'rb')
    good_image_list = msgpack.unpackb(body_fp.read(),
                                      object_hook=msgpack_numpy.decode)    
    body_fp.close()

    body_path = file_bootstrap_candidate_image_path(body_name, make_dirs=False)
    if not os.path.exists(body_path):
        continue
    body_fp = open(body_path, 'rb')
    cand_image_list = msgpack.unpackb(body_fp.read(),
                                      object_hook=msgpack_numpy.decode)    
    body_fp.close()

    plot_body(body_name, good_image_list, cand_image_list)
