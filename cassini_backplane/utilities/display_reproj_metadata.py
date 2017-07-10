###############################################################################
# display_reproj_metadata.py 
###############################################################################

import Tkinter as tk
import tkFileDialog

import argparse
import os.path
import sys

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_body_mosaic import *
from cb_util_file import *

iss.initialize(planets=(6,))

results_dir = os.path.join(CB_RESULTS_ROOT, 'reproj')

command_list = sys.argv[1:]

parser = argparse.ArgumentParser(
    description='Cassini Backplane Reprojected Body Metadata GUI')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

root = tk.Tk()
root.withdraw()

if len(command_list) > 0:
    for image_path in file_yield_image_filenames_from_arguments(arguments):
        metadata = file_read_reproj_body_path(image_path)
        display_body_reproj_metadata(metadata)
else:
    file_initialdir = results_dir
    
    while True:
        offset_path = tkFileDialog.askopenfilename(
                   initialdir=file_initialdir,
                   title='Choose an offset metadata file')
        if offset_path == '' or offset_path == ():
            break
    
        metadata = file_read_reproj_body_path(offset_path)

        display_body_reproj_metadata(metadata)
