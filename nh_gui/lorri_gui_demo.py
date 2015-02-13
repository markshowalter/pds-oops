###############################################################################
# lorri_gui_demo.py
###############################################################################

import os
import numpy as np

from imgdisp import ImageDisp
import Tkinter as tk

import oops
import oops.inst.nh.lorri as lorri
from psfmodel.gaussian import GaussianPSF

# This routine is called for any mouse move event on any image.
# It updates the visible metadata fields.
def _callback_mousemove(x, y, metadata):
    # X,Y are in image coordinates, not pixel coordinates. For example if the
    # image is overzoomed, then X,Y could be fractional. These coordinates
    # are also adjusted for reflection and rotation.
    
    x = int(x)
    y = int(y)

    ra = metadata['ra'][y,x] * oops.DPR
            
    if ra.masked():
        val = 'N/A'
    else:
        val = '%7.3f' % ra.vals
    metadata['label_ra'].config(text=val)

    dec = metadata['dec'][y,x] * oops.DPR

    if dec.masked():
        val = 'N/A'
    else:
        val = '%7.3f' % dec.vals
    metadata['label_dec'].config(text=val)


# This routine is called on a left button click. It performs Gaussian
# astrometry and displays the resulting X,Y center.
def _callback_button1down(x, y, metadata):
    x = int(x)
    y = int(y)

    psf = GaussianPSF()
    ret_y, ret_x, ret_metadata = psf.find_position(
                                           metadata['img'], (7,7), (y,x))

    metadata['label_centroidx'].config(text='%7.3f'%ret_x)
    metadata['label_centroidy'].config(text='%7.3f'%ret_y)
    

def display_one_image(obs, data=None, overlay=None, title=''):
    """Display a single LORRI image but don't actually start the GUI.
    
    Inputs:
    
    obs        The Snapshot Observation.
    data       The image data to display. If None, use the data stored in obs.
    overlay    A color overlay to display. If None, don't display an overlay.
    title      The title of the window.
    """
    toplevel = tk.Toplevel()
    toplevel.title(title)
    
    # Compute all the data to display
        
    if data is None:
        data = obs.data
        
    bp = oops.Backplane(obs)
    bp_ra = bp.right_ascension()
    bp_dec = bp.declination()

    metadata = {}

    metadata['img'] = data
    metadata['ra'] = bp_ra
    metadata['dec'] = bp_dec

    # Create the GUI object

    imgdisp = ImageDisp([data], overlay, canvas_size=(512,512), 
                        parent=toplevel,
                        allow_enlarge=True)

    callback_mousemove_func = (lambda x, y, metadata=metadata:
                               _callback_mousemove(x, y, metadata))
    imgdisp.bind_mousemove(0, callback_mousemove_func)

    callback_b1down_func = (lambda x, y, metadata=metadata:
                            _callback_button1down(x, y, metadata))
    imgdisp.bind_b1press(0, callback_b1down_func)

    # Lay out the text labels and data fields for the metadata we want to
    # display
    
    gridrow = 0
    gridcolumn = 0

    label_width = 10
    val_width = 7

    addon_control_frame = imgdisp.addon_control_frame

    label = tk.Label(addon_control_frame, text='RA:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=tk.W)
    label_ra = tk.Label(addon_control_frame, text='', 
                        anchor='e', width=val_width)
    label_ra.grid(row=gridrow, column=gridcolumn+1, sticky=tk.W)
    metadata['label_ra'] = label_ra
    gridrow += 1
    
    label = tk.Label(addon_control_frame, text='DEC:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=tk.W)
    label_dec = tk.Label(addon_control_frame, text='', 
                         anchor='e', width=val_width)
    label_dec.grid(row=gridrow, column=gridcolumn+1, sticky=tk.W)
    metadata['label_dec'] = label_dec
    gridrow += 1

    label = tk.Label(addon_control_frame, text='Centroid X:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=tk.W)
    label_cx = tk.Label(addon_control_frame, text='', 
                        anchor='e', width=val_width)
    label_cx.grid(row=gridrow, column=gridcolumn+1, sticky=tk.W)
    metadata['label_centroidx'] = label_cx
    gridrow += 1

    label = tk.Label(addon_control_frame, text='Centroid Y:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=tk.W)
    label_cy = tk.Label(addon_control_frame, text='', 
                        anchor='e', width=val_width)
    label_cy.grid(row=gridrow, column=gridcolumn+1, sticky=tk.W)
    metadata['label_centroidy'] = label_cy
    gridrow += 1


#===============================================================================
# The main code 
#===============================================================================

master_root = tk.Tk()
master_root.withdraw() # Get rid of the default window

test_data_dir = os.environ["OOPS_TEST_DATA_PATH"]

lorri_fn1 = os.path.join(test_data_dir, 'nh', 'LORRI',
                         'LOR_0034969199_0X630_SCI_1.FIT')

obs = lorri.from_file(lorri_fn1)

# Inverse video version
inv_data = np.max(obs.data)-obs.data

# Overlay based on incidence angle of Europa
bp = oops.Backplane(obs)
bp_incidence = bp.incidence_angle('europa')
inc_min = np.min(bp_incidence.mvals)
inc_max = np.max(bp_incidence.mvals)
bp_incidence = (bp_incidence-inc_min) / (inc_max-inc_min)
bp_incidence = bp_incidence.mvals.filled(0)
overlay = np.zeros(obs.data.shape + (3,))
overlay[:,:,0] = bp_incidence

display_one_image(obs, title='Positive Image')
display_one_image(obs, data=inv_data, title='Negative Image')
display_one_image(obs, overlay=overlay, title='Overlay Image')
    
tk.mainloop()
