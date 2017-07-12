###############################################################################
# display_mosaic_metadata.py
###############################################################################

import Tkinter as tk
import tkFileDialog

from PIL import Image

import os

from cb_config import *
from cb_gui_body_mosaic import *
from cb_util_file import *

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

    fn = '/home/rfrench/'+filename+'.png'
    img = metadata['img'][::-1,:]
    
    gamma = 0.85
    blackpoint = None
    whitepoint = None
    whitepoint_ignore_frac = 0.95
    
    # Contrast stretch the main image
    if blackpoint is None:
        blackpoint = np.min(img)

    if whitepoint is None:
        img_sorted = sorted(list(img[img!=0].flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
    greyscale_img = ImageDisp.scale_image(img,
                                          blackpoint,
                                          whitepoint,
                                          gamma)

    greyscale_img = np.cast['uint8'](greyscale_img)

    im = Image.fromarray(greyscale_img)
    im.save(fn)
