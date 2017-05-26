# Issues:

import argparse
import ring_util
import os
import os.path
import numpy as np
import sys
import cspice
import subprocess
import colorsys
from imgdisp import ImageDisp, FloatEntry
from PIL import Image
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy
from cb_rings import *

command_list = sys.argv[1:]

if len(command_list) == 0:
#     command_line_str = '--all-obsid --display-radius-inner 140020 --display-radius-outer 140420 --radius-polar-base-factor 1.5'
    command_line_str = '--ring-type BRING_MOUNTAINS ISS_115RI_AZSCANLIT001_PRIME --radius-polar-base-factor 2.5 --image-size 4096'# --blackpoint 0 --whitepoint 0.0071 --gamma 0.8'
#     command_line_str = '--ring-type BRING_MOUNTAINS --all-obsid --radius-polar-base-factor 2.5'
    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

##
## Options for mosaic creation
##
parser.add_argument(
    '--display-radius-inner', type=float, default=1e38)
parser.add_argument(
    '--display-radius-outer', type=float, default=1e38)
parser.add_argument(
    '--radius-polar-base-factor', type=float, default=1.0)
parser.add_argument(
    '--image-size', type=int, default=1024)
parser.add_argument(
    '--whitepoint', type=float, default=1e38)
parser.add_argument(
    '--blackpoint', type=float, default=1e38)
parser.add_argument(
    '--gamma', type=float, default=1e38)

parser.add_argument(
    '--display', action='store_true', default=False)

ring_util.ring_add_parser_options(parser)

arguments = parser.parse_args(command_list)

ring_util.ring_init(arguments)

#####################################################################################
#
# CREATE ONE POLAR PROJECTION MOSAIC
#
#####################################################################################

def polar_project_mosaic(obsid):
    (data_path, metadata_path) = ring_util.mosaic_paths(arguments, obsid)
    
    img = np.load(data_path+'.npy')
    
    mosaic_metadata_fp = open(metadata_path, 'rb')
    (metadata, 
     obsid_list,
     image_name_list,
     image_path_list,
     repro_path_list) = msgpack.unpackb(mosaic_metadata_fp.read(), 
                                        object_hook=msgpack_numpy.decode)
    metadata['img'] = img
    mosaic_metadata_fp.close()

    image_size = arguments.image_size
    radius_polar_base_factor = arguments.radius_polar_base_factor

    image_center = image_size // 2

    polar_img = np.zeros((image_size, image_size), dtype='float32')
 
    radius_inner = arguments.ring_radius + arguments.radius_inner_delta
    radius_outer = arguments.ring_radius + arguments.radius_outer_delta
    display_radius_inner = arguments.display_radius_inner
    if display_radius_inner == 1e38:
        display_radius_inner = radius_inner
    display_radius_outer = arguments.display_radius_outer
    if display_radius_outer == 1e38:
        display_radius_outer = radius_outer

    display_radius_min = (display_radius_outer-
                           (display_radius_outer-radius_inner)*
                           radius_polar_base_factor)
    display_radius_total = display_radius_outer-display_radius_min
    
    num_rad = img.shape[0]
    num_long = img.shape[1]
    
    # A grid with (y,x) for each final image pixel
    pixel_num = np.arange(image_size)
    polar_pixel_y, polar_pixel_x = np.meshgrid(pixel_num, pixel_num)
    
    # A grid with (rad,long) for each final image pixel
    offset_polar_pixel_x = polar_pixel_x-image_center
    offset_polar_pixel_y = polar_pixel_y-image_center
    polar_r = np.hypot(offset_polar_pixel_x, 
                       offset_polar_pixel_y)
    polar_r = polar_r*display_radius_total/image_center + display_radius_min
    polar_l = np.arctan2(offset_polar_pixel_y, 
                         offset_polar_pixel_x) % (2*np.pi)

    # A grid with (source rad pixel, source long pixel) for each final 
    # image pixel     
    img_x = polar_l*num_long/(2*np.pi)
    img_y = (polar_r-radius_inner)*num_rad/(radius_outer-radius_inner)
    img_x = np.round(img_x).astype(int)
    img_y = np.round(img_y).astype(int)

    mask = np.where((img_x >= 0) & (img_x < num_long) &
                    (img_y >= 0) & (img_y < num_rad))
    polar_pixel_x = polar_pixel_x[mask]
    polar_pixel_y = polar_pixel_y[mask]
    img_x = img_x[mask]
    img_y = img_y[mask]
    polar_img[polar_pixel_y, polar_pixel_x] = img[img_y, img_x]

    if arguments.blackpoint != 1e38:
        blackpoint = arguments.blackpoint
    else:
        blackpoint = max(np.min(polar_img), 0)
    whitepoint_ignore_frac = 0.995
    if arguments.whitepoint != 1e38:
        whitepoint = arguments.whitepoint
    else:
        img_sorted = sorted(list(polar_img.flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
    if arguments.gamma != 1e38:
        gamma = arguments.gamma
    else:
        gamma = 0.5
    
    # The +0 forces a copy - necessary for PIL
    scaled_polar = np.cast['int8'](ImageDisp.scale_image(polar_img, blackpoint,
                                                          whitepoint, gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_polar.shape[1], scaled_polar.shape[0]),
                               scaled_polar, 'raw', 'L', 0, 1)
    png_path = ring_util.png_path(arguments, obsid, 'polar', make_dirs=True)
    pil_img.save(png_path, 'PNG')
    
    if arguments.display:
        plt.imshow(polar_img)
        plt.show()
    
#     sp.misc.imsave(options.output_path + obs_id + options.format, final_img)    


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

cur_obsid = None

for obsid, image_name, image_path in ring_util.ring_enumerate_files(arguments):
    repro_path = ring_util.repro_path(arguments, image_path, image_name)
    
    if cur_obsid != obsid:
        cur_obsid = obsid
        polar_project_mosaic(obsid)
