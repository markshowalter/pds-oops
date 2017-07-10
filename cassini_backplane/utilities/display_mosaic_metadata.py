###############################################################################
# display_mosaic_metadata.py
###############################################################################

import Tkinter as tk
import tkFileDialog

import os

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_body_mosaic import *
from cb_util_file import *

iss.initialize(planets=(6,))

mosaic_dir = os.path.join(CB_RESULTS_ROOT, 'mosaics')

root = tk.Tk()
root.withdraw()

while True:
    path = tkFileDialog.askopenfilename(initialdir=mosaic_dir, 
                                        title='Choose a mosaic metadata file')
    if path == '' or path == ():
        break
    metadata = file_read_mosaic_metadata_path(path)
    _, filename = os.path.split(path)
    title = filename + ' / ' + metadata['body_name']
    display_body_mosaic_metadata(metadata, title=title)
