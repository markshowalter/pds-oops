###############################################################################
# cb_main_copy_png.py
#
# Copy the selected image png files to a common destination directory.
###############################################################################

from cb_logging import *
import logging

import argparse
import os
import shutil

from cb_util_file import *


command_list = sys.argv[1:]

if len(command_list) == 0:
    assert False
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Copy image png files to a common destination directory''')

parser.add_argument(
    '--copy-png', action='store_true', default=False,
    help='Copy PNG files')
parser.add_argument(
    '--copy-offset', action='store_true', default=False,
    help='Copy OFFSET files')
parser.add_argument(
    '--copy-overlay', action='store_true', default=False,
    help='Copy OFFSET files')
parser.add_argument(
    '--copy-all', action='store_true', default=False,
    help='Copy all metadata files')
parser.add_argument(
    '--dest-dir', default='.',
    help='The root destination directory for all metadata files')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


###############################################################################
#
# 
#
###############################################################################

if arguments.copy_png or arguments.copy_all:
    dest_dir = os.path.join(arguments.dest_dir, 'png')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
if arguments.copy_offset or arguments.copy_all:
    dest_dir = os.path.join(arguments.dest_dir, 'offsets')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
if arguments.copy_overlay or arguments.copy_all:
    dest_dir = os.path.join(arguments.dest_dir, 'overlays')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

for image_path in file_yield_image_filenames_from_arguments(arguments):
    filename = file_clean_name(image_path)
    print filename
    if arguments.copy_png or arguments.copy_all:
        src_path = file_img_to_png_path(image_path)
        dest_path = os.path.join(arguments.dest_dir, 'png', filename+'.png')
        shutil.copyfile(src_path, dest_path)
    if arguments.copy_offset or arguments.copy_all:
        src_path = file_img_to_offset_path(image_path)
        dest_path = os.path.join(arguments.dest_dir, 'offsets', filename+'-OFFSET.dat')
        shutil.copyfile(src_path, dest_path)
    if arguments.copy_overlay or arguments.copy_all:
        src_path = file_img_to_overlay_path(image_path)
        dest_path = os.path.join(arguments.dest_dir, 'overlays', filename+'-OVERLAY.dat')
        shutil.copyfile(src_path, dest_path)
