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

def plot_body(body_name, good_image_list, cand_image_list):
    for lon_shadow_dir in [False, True]:
        for lat_shadow_dir in [False, True]:
            good_sun_lat_list = np.array([x['sub_solar_lat']*oops.DPR 
                                          for x in good_image_list])
            good_sun_lon_list = np.array([x['sub_solar_lon']*oops.DPR 
                                          for x in good_image_list])
            good_obs_lat_list = np.array([x['sub_observer_lat']*oops.DPR 
                                          for x in good_image_list])
            good_obs_lon_list = np.array([x['sub_observer_lon']*oops.DPR 
                                          for x in good_image_list])
            cand_sun_lat_list = np.array([x['sub_solar_lat']*oops.DPR 
                                          for x in cand_image_list])
            cand_sun_lon_list = np.array([x['sub_solar_lon']*oops.DPR 
                                          for x in cand_image_list])
            cand_obs_lat_list = np.array([x['sub_observer_lat']*oops.DPR 
                                          for x in cand_image_list])
            cand_obs_lon_list = np.array([x['sub_observer_lon']*oops.DPR 
                                          for x in cand_image_list])

            if len(good_obs_lon_list) == 0 or len(cand_obs_lon_list) == 0:
                continue
            
            plt.figure()
            plt.title('%s Lon Same=%s Lat Same=%s' % (body_name, 
                                                      str(lon_shadow_dir), 
                                                      str(lat_shadow_dir)))
            plt.plot(good_sun_lon_list, good_sun_lat_list,
                     'o', mec='green', mfc='none', ms=6)
            plt.plot(cand_sun_lon_list, cand_sun_lat_list,
                     'o', mec='red', mfc='none', ms=9, alpha=0.2)
            plt.xlim(0,360.)
            plt.ylim(-30.,30)
            plt.xlabel('Sub-obs longitude')
            plt.ylabel('Sub-obs latitude)')

    plt.show()
        
GOOD_IMAGE_BY_BODY_DB = {}
CAND_IMAGE_BY_BODY_DB = {}

for body_name in ['MIMAS']:#bootstrap_config['body_list']:
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
