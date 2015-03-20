###############################################################################
# display_offset_metadata.py 
###############################################################################

import Tkinter as tk
import tkFileDialog

import os.path

from cb_config import *
from cb_gui_offset_data import *
from cb_util_file import *

results_dir = os.path.join(RESULTS_ROOT, 'offsets')

root = tk.Tk()
root.withdraw()

file_initialdir = results_dir

while True:
    toplevel = Toplevel()
    toplevel.withdraw()
    offset_path = tkFileDialog.askopenfilename(
               parent=toplevel, initialdir=file_initialdir,
               title='Choose an offset metadata file')
    if offset_path == '':
        break

    file_initialdir, _ = os.path.split(offset_path)
    img_path = file_offset_to_img_path(offset_path)
    
    metadata = file_read_offset_metadata(img_path)
    obs = read_iss_file(img_path)
    display_offset_data(obs, metadata)
