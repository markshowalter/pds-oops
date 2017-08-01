import argparse
import cProfile, pstats, StringIO
import matplotlib.pyplot as plt
import sys
import time

import pickle

import cspice
import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_util_file import *
from cb_util_misc import *

class GoodEntry(object):
    pass

class CandEntry(object):
    pass

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


def get_shadow_dirs(sub_solar_lat, sub_obs_lat, sub_solar_lon, sub_obs_lon):
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

def plot_body(body_name, lat_shadow_dir, lon_shadow_dir):
    ns, ew = file_bootstrap_shadow_to_str(lat_shadow_dir, lon_shadow_dir)
    status_path = file_bootstrap_status_image_path(body_name, 
                                                   lat_shadow_dir,
                                                   lon_shadow_dir,
                                                   make_dirs=False)
    status_fp = open(status_path, 'rb')
    good_entry_list = pickle.load(status_fp)
    cand_entry_list = pickle.load(status_fp)
    status_fp.close()
    
    good_entry_masks = np.zeros(good_entry_list[0].latlon_mask.shape)
    cand_entry_masks = np.zeros(cand_entry_list[0].latlon_mask.shape)
    
    for good_entry in good_entry_list:
        good_entry_masks = good_entry_masks+good_entry.latlon_mask
            
    for cand_entry in cand_entry_list:
        cand_entry_masks = cand_entry_masks+cand_entry.latlon_mask

    color_mask = np.zeros((cand_entry_masks.shape[0], 
                           cand_entry_masks.shape[1], 3))
    color_mask[:,:,0] = cand_entry_masks
    color_mask[:,:,1] = good_entry_masks
    color_mask[:,:,0] /= np.max(color_mask[:,:,0])
    color_mask[:,:,1] /= np.max(color_mask[:,:,1])
    color_mask[good_entry_masks==0,2] = 1
    
    plt.figure()
    plt.title('%s Lon Same=%s Lat Same=%s' % (body_name, str(lon_shadow_dir), str(lat_shadow_dir)))
    plt.imshow(color_mask)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude)')

    plt.show()
        
for body_name in bootstrap_config['body_list']:
    for lat_shadow_dir in [False,True]:
        for lon_shadow_dir in [False,True]:
            plot_body(body_name, lat_shadow_dir, lon_shadow_dir)
            
