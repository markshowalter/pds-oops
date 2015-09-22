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

results_dir = os.path.join(RESULTS_ROOT, 'offsets')

command_list = sys.argv[1:]

parser = argparse.ArgumentParser(
    description='Cassini Backplane Offset Metadata GUI')

parser.add_argument(
    'image_path', nargs='?',
    help="Specific image path to display")

arguments = parser.parse_args(command_list)

root = tk.Tk()
root.withdraw()

if arguments.image_path is not None:
    img_path = arguments.image_path
    metadata = file_read_offset_metadata(img_path)
    obs = read_iss_file(img_path)
    display_offset_data(obs, metadata)
else:
    file_initialdir = results_dir
    
    while True:
        offset_path = tkFileDialog.askopenfilename(
                   initialdir=file_initialdir,
                   title='Choose an offset metadata file')
        if offset_path == '' or offset_path == ():
            break
    
        file_initialdir, _ = os.path.split(offset_path)
        img_path = file_offset_to_img_path(offset_path)
        
        metadata = file_read_offset_metadata(img_path)
        obs = read_iss_file(img_path)
        display_offset_data(obs, metadata)
