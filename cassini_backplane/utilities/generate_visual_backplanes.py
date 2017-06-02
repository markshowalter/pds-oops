###############################################################################
# generate_visual_backplanes.py
#
# Generate backplanes for the given images.
###############################################################################


import argparse
import os
import matplotlib.pyplot as plt

import oops

from cb_config import *
from cb_util_file import *


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'N1649556643_1 N1454732688_1 N1512448422_1'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Generate visual backplanes.''')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


###############################################################################
#
# 
#
###############################################################################

def save_image(data, filename):
    print 'Saving image'
    plt.imsave(filename+'_image.png', data**.5, cmap=plt.cm.gray)
    
def save_backplane(data, filename, name, cmap=plt.cm.rainbow):
    data = data.mvals

    if 'longitude' in name:
        cmap = plt.cm.hsv
        
    print 'Saving', name
    plt.imsave(filename+'_'+name+'.png', data, cmap=cmap)
    
for image_path in file_yield_image_filenames_from_arguments(arguments):
    filename = file_clean_name(image_path)
    print filename
    obs = file_read_iss_file(image_path)
    save_image(obs.data, filename)
    offset_metadata = file_read_offset_metadata(image_path, overlay=False)
    offset = offset_metadata['offset']
    if offset is None:
        offset = (0,0)
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    bp = oops.Backplane(obs)
    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    large_bodies_by_range = [(x, large_body_dict[x]) for x in large_body_dict]
    large_bodies_by_range.sort(key=lambda x: -x[1]['range'])
    for body_bp_name, body_bp_func in [
        ('longitude', bp.longitude),
        ('latitude', bp.latitude)]:
        total_data = None
        for range, inv in large_bodies_by_range:
            body_name = inv['name']
            print body_bp_name, body_name
            u_size = inv['u_pixel_size']
            v_size = inv['u_pixel_size']
            area = u_size*v_size
            if area < 100:
                continue
            data = body_bp_func(body_name)
            if total_data is None:
                total_data = data
            else:
                mask = np.logical_not(data.mvals.mask)
                total_data[mask] = data[mask]
        if total_data is not None:
            save_backplane(total_data, filename, body_bp_name)
    data = bp.ring_radius('saturn_main_rings')
    save_backplane(data, filename, 'ring_radius')
    data = bp.ring_longitude('saturn_main_rings')
    save_backplane(data, filename, 'ring_longitude')
