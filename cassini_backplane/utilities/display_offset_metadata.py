###############################################################################
# display_offset_metadata.py 
###############################################################################

import Tkinter as tk
import tkFileDialog

import argparse
import os.path
import sys

from cb_config import *
from cb_gui_offset_data import *
from cb_util_file import *

results_dir = os.path.join(CB_RESULTS_ROOT, 'offsets')

command_list = sys.argv[1:]

parser = argparse.ArgumentParser(
    description='Cassini Backplane Offset Metadata GUI')

parser.add_argument(
    '--canvas-size', type=str, metavar='X,Y',
    help='Force the canvas size to be X,Y')
parser.add_argument(
    '--interpolate-missing-stripes', action='store_true', 
    help='Interpolate missing data stripes')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

canvas_size = None
if arguments.canvas_size is not None:
    x, y = arguments.canvas_size.split(',')
    canvas_size = (int(x.replace('"','')), int(y.replace('"','')))
    
root = tk.Tk()
root.withdraw()

if len(arguments.image_name[0]) > 0:
    for image_path in file_yield_image_filenames_from_arguments(arguments):
        metadata = file_read_offset_metadata(image_path)
        obs = file_read_iss_file(image_path)
        display_offset_data(obs, metadata,
                            canvas_size=canvas_size,
                            interpolate_missing_stripes=
                                arguments.interpolate_missing_stripes)
else:
    file_initialdir = results_dir
    
    while True:
        offset_path = tkFileDialog.askopenfilename(
                   initialdir=file_initialdir,
                   title='Choose an offset metadata file')
        if offset_path == '' or offset_path == ():
            break
    
        metadata = file_read_offset_metadata_path(offset_path)

        img_path = file_offset_to_img_path(offset_path)
        obs = file_read_iss_file(img_path)
        
        display_offset_data(obs, metadata, 
                            canvas_size=canvas_size,
                            interpolate_missing_stripes=
                                 arguments.interpolate_missing_stripes)
