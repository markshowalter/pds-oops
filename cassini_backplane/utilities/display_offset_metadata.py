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

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

root = tk.Tk()
root.withdraw()

if len(command_list) > 0:
    for image_path in file_yield_image_filenames_from_arguments(arguments):
        metadata = file_read_offset_metadata(image_path)
        obs = file_read_iss_file(image_path)
        display_offset_data(obs, metadata)
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
        
        display_offset_data(obs, metadata)
