import argparse
import copy
import cProfile, pstats, StringIO
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy
import pickle
import scipy.ndimage.interpolation as ndinterp

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_util_file import *
from cb_util_misc import *

MAXIMUM_SOLAR_DELTA = 30 * oops.RPD
MAXIMUM_OBS_DELTA = 60 * oops.RPD

command_list = sys.argv[1:]

if len(command_list) == 0:
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_BL1'
    command_line_str = 'MIMAS'
#    command_line_str = 'ENCELADUS --mosaic-root ENCELADUS_0.00_-30.00_F_F_GRN'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description="",
    epilog="")

# Arguments about body and shadow selection
parser.add_argument(
    'body_names', action='append', nargs='*', 
    help='Specific body names to process')
parser.add_argument(
    '--lon-shadow-east-only', action='store_true', default=False,
    help='Only process image with the lat shadow direction EAST')
parser.add_argument(
    '--lon-shadow-west-only', action='store_true', default=False,
    help='Only process image with the lat shadow direction WEST')
parser.add_argument(
    '--lat-shadow-north-only', action='store_true', default=False,
    help='Only process image with the lon shadow direction NORTH')
parser.add_argument(
    '--lat-shadow-south-only', action='store_true', default=False,
    help='Only process image with the lon shadow direction SOUTH')

arguments = parser.parse_args(command_list)



def get_shadow_dirs(body_metadata):
    sub_solar_lat = body_metadata['sub_solar_lat']
    sub_solar_lon = body_metadata['sub_solar_lon']
    sub_obs_lat = body_metadata['sub_observer_lat']
    sub_obs_lon = body_metadata['sub_observer_lon']
    
    if sub_solar_lat > sub_obs_lat+oops.PI:
        sub_solar_lat -= oops.PI
    elif sub_solar_lat < sub_obs_lat-oops.PI:
        sub_solar_lat += oops.PI
    lat_shadow_dir = sub_solar_lat < sub_obs_lat

    if sub_solar_lon > sub_obs_lon+oops.PI:
        sub_solar_lon -= oops.TWOPI
    elif sub_solar_lon < sub_obs_lon-oops.PI:
        sub_solar_lon += oops.TWOPI
    lon_shadow_dir = sub_solar_lon < sub_obs_lon

    return lat_shadow_dir, lon_shadow_dir

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

def process_body(body_name, lat_shadow_dir, lon_shadow_dir):
    ns, ew = file_bootstrap_shadow_to_str(lat_shadow_dir, lon_shadow_dir)

    body_path = file_bootstrap_good_image_path(body_name, make_dirs=False)
    body_fp = open(body_path, 'rb')
    good_image_list_all = msgpack.unpackb(body_fp.read(),
                                          object_hook=msgpack_numpy.decode)    
    body_fp.close()

    for good_entry in good_image_list_all:
#         if 'N1665933684' in good_entry:
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
        if 'UV' in cand_entry['filter']:
            continue
        if 'IR' in cand_entry['filter']:
            continue
        if 'MT' in cand_entry['filter']:
            continue

        if 'N1644781802_1' not in cand_entry['image_filename']:
            continue
        
        res_near_list = []
        res_far_list = []
        for opt, keep, sign in (
                          ('sub_solar_lon', 'sub_solar_lat', 1),
                          ('sub_solar_lon', 'sub_solar_lat', -1),
                          ('sub_solar_lat', 'sub_solar_lon', 1),
                          ('sub_solar_lat', 'sub_solar_lon', -1)):
            best_far_entry = None
            best_far_value = -1e38
            best_far_overlap = None
            best_near_entry = None
            best_near_value = 1e38
            best_near_overlap = None
            for good_entry in good_image_list_all:
                if 'UV' in good_entry['filter']:
                    continue
                if 'IR' in good_entry['filter']:
                    continue
                if 'MT' in good_entry['filter']:
                    continue
                if abs(cand_entry[keep]-good_entry[keep])*oops.DPR > 5:
                    continue
                if good_entry['center_mean_resolution']/cand_entry['center_mean_resolution'] > 3:
                    continue
                joint_mask = (np.logical_and(cand_entry['reproj']['full_mask'],
                                             good_entry['reproj']['full_mask']))
                if np.sum(joint_mask) == 0:
                    continue
                overlap = float(np.sum(joint_mask)) / np.sum(cand_entry['reproj']['full_mask'])
                if overlap < 0.1:
                    continue
                val = (good_entry[opt]-cand_entry[opt]) * sign
                if val*oops.DPR > 10:
                    if val > best_far_value:
                        best_far_value = val
                        best_far_entry = good_entry
                        best_far_overlap = overlap
                if val > 0 and val < best_near_value:
                    best_near_value = val
                    best_near_entry = good_entry
                    best_near_overlap = overlap
#             if best_near_entry is not None:
#                 res_near_list.append((opt, keep, sign, best_near_entry, best_near_overlap))
            if best_far_entry is not None:
                res_far_list.append((opt, keep, sign, best_far_entry, best_far_overlap))

        if len(res_far_list) > 1:
            print 'Cand image '+entry_str(cand_entry)        
            for opt, keep, sign, best_entry, best_overlap in res_near_list:    
                print 'NEAR', opt, sign, entry_str(best_entry),
                print (' / Mask %.4f'%(best_overlap))
            for opt, keep, sign, best_entry, best_overlap in res_far_list:    
                print 'FAR', opt, sign, entry_str(best_entry),
                print(' / Mask %.4f'%(best_overlap))
            
        for good_entry in good_image_list_all:
            dist = np.sqrt(
               ((cand_entry['sub_solar_lat']-good_entry['sub_solar_lat'])**2+
                (cand_entry['sub_solar_lon']-good_entry['sub_solar_lon'])**2+
                (cand_entry['sub_observer_lat']-good_entry['sub_observer_lat'])**2+
                (cand_entry['sub_observer_lon']-good_entry['sub_observer_lon'])**2))
            good_entry['dist'] = dist

        good_image_list_all.sort(key=lambda x: x['dist'])

        for good_entry in good_image_list_all:
            joint_mask = (np.logical_and(cand_entry['reproj']['full_mask'],
                                         good_entry['reproj']['full_mask']))
            overlap = float(np.sum(joint_mask)) / np.sum(cand_entry['reproj']['full_mask'])
            if overlap == 0:
                continue
#             print 'Good image '+entry_str(good_entry) + (' / Mask %.4f'%(overlap))
        
        print
            
#===============================================================================
# 
#===============================================================================

assert not (arguments.lon_shadow_east_only and arguments.lon_shadow_west_only)
assert not (arguments.lat_shadow_north_only and arguments.lat_shadow_south_only)

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG

body_names = [x.upper() for x in arguments.body_names[0]]

if len(body_names) == 0:
    body_names = bootstrap_config['body_list']
    
body_names.sort()
    
for body_name in body_names:
    for lat_shadow_dir in [False,True]:
        for lon_shadow_dir in [False,True]:
            if not lat_shadow_dir and arguments.lat_shadow_north_only:
                continue
            if lat_shadow_dir and arguments.lat_shadow_south_only:
                continue
            if not lon_shadow_dir and arguments.lon_shadow_west_only:
                continue
            if lon_shadow_dir and arguments.lon_shadow_east_only:
                continue
            process_body(body_name, lat_shadow_dir, lon_shadow_dir)
