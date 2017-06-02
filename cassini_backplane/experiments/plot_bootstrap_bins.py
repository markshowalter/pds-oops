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
    print body_name, 'GOOD IMAGES', len(good_image_list),
    print 'CAND IMAGES', len(cand_image_list)
    image_list = [x[0] for x in cand_image_list if not x[8]]
#     for image in image_list:
#         print file_clean_name(image)
    good_image_list = [(x[0], x[2], x[1], x[4], x[3], x[5], x[6], x[7])
                       for x in good_image_list]
    cand_image_list = [(x[0], x[2], x[1], x[4], x[3], x[5], x[6], x[7], x[8])
                       for x in cand_image_list]
    for lon_shadow_dir in [False, True]:
        for lat_shadow_dir in [False, True]:
            # XXX Temporary flip of lat/lon
            good_entries = [x for x in good_image_list
              if get_shadow_dirs(x[1], x[3], x[2], x[4])[0] == 
                   lat_shadow_dir and
                 get_shadow_dirs(x[1], x[3], x[2], x[4])[1] == 
                   lon_shadow_dir]
            caand_entries = [x for x in cand_image_list
              if get_shadow_dirs(x[1], x[3], x[2], x[4])[0] == 
                   lat_shadow_dir and
                 get_shadow_dirs(x[1], x[3], x[2], x[4])[1] == 
                   lon_shadow_dir]
            good_sun_lat_list = np.array([x[1]*oops.DPR for x in good_entries])
            good_sun_lon_list = np.array([x[2]*oops.DPR for x in good_entries])
            good_obs_lat_list = np.array([x[3]*oops.DPR for x in good_entries])
            good_obs_lon_list = np.array([x[4]*oops.DPR for x in good_entries])
            cand_sun_lat_list = np.array([x[1]*oops.DPR for x in cand_entries])
            cand_sun_lon_list = np.array([x[2]*oops.DPR for x in cand_entries])
            cand_obs_lat_list = np.array([x[3]*oops.DPR for x in cand_entries])
            cand_obs_lon_list = np.array([x[4]*oops.DPR for x in cand_entries])

            if len(good_obs_lon_list) == 0 or len(cand_obs_lon_list) == 0:
                continue
            
            plt.figure()
            plt.title('%s Lon Same=%s Lat Same=%s' % (body_name, str(lon_shadow_dir), str(lat_shadow_dir)))
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
