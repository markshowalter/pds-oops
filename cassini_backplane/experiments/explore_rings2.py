'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import pickle
import os
import os.path
import sys
import numpy as np
import cspice
from imgdisp import ImageDisp, FloatEntry, draw_line
from Tkinter import *
from PIL import Image
import oops.inst.cassini.iss as iss
from cb_offset import *
from cb_rings import *
import cProfile, pstats, StringIO

class OffData(object):
    """Offset and Reprojection data."""
    def __init__(self):
        self.obsid = None
        self.image_name = None
        self.image_path = None
        self.obs = None

        self.offset_path = None
        self.the_offset = None
        self.manual_offset = None
        self.off_metadata = None
        
        self.repro_path = None
        self.repro_img = None

        self.repro_longitudes = None
        self.repro_phase_angles = None
        self.repro_incidence_angles = None
        self.repro_emission_angles = None
        self.repro_resolutions = None

class OffDispData(object):
    def __init__(self):
        self.obs = None
        self.toplevel = None
        self.imdisp_offset = None
        self.entry_x_offset = None
        self.entry_y_offset = None
        self.off_longitudes = None
        self.off_radii = None
        self.label_off_inertial_longitude = None
        self.label_off_radius = None
        self.last_xy = None

#####################################################################################
#
# FIND THE POINTING OFFSET
#
#####################################################################################
    
#
# The primary entrance for finding pointing offset
#

def offset_one_image(offdata):
    # Recompute the automatic offset
    obs = iss.from_file(offdata.image_path)
    offdata.obs = obs
    try:
        offset_u, offset_v, offdata.off_metadata = master_find_offset(obs,
                                                 create_overlay=True,
                                                 star_overlay_box_width=5,
                                                 star_overlay_box_thickness=2,
                                                 allow_stars=False)                                                                             
    except:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
        print 'EXCEPTION:'
        print sys.exc_info()
        raise
        offset_u = None
        offset_v = None
        offdata.off_metadata = {}
    if offset_u is None:
        offdata.the_offset = None
    else:
        offdata.the_offset = (offset_u, offset_v)
    if offset_u is None:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    if offset_u is not None:
        print 'FOUND %6.2f, %6.2f' % (offdata.the_offset[0], offdata.the_offset[1])



#####################################################################################
#
# DISPLAY ONE IMAGE ALLOWING MANUAL CHANGING OF THE OFFSET
#
#####################################################################################

# The callback for mouse move events on the offset image
def callback_offset(x, y, offdata, offdispdata):
    if offdata.manual_offset is not None:
        x -= offdata.manual_offset[0]
        y -= offdata.manual_offset[1]
    elif offdata.the_offset is not None:
        x -= offdata.the_offset[0]
        y -= offdata.the_offset[1]
    if (x < 0 or x > offdata.obs.data.shape[1]-1 or
        y < 0 or y > offdata.obs.data.shape[0]-1):
        return
    
    if offdispdata.off_longitudes is not None:
        offdispdata.label_off_inertial_longitude.config(text=('%7.3f'%offdispdata.off_longitudes[y,x]))
    if offdispdata.off_radii is not None:
        offdispdata.label_off_radius.config(text=('%7.3f'%offdispdata.off_radii[y,x]))

# "Manual from auto" button pressed 
def command_man_from_auto(offdata, offdispdata):
    offdata.manual_offset = offdata.the_offset
    offdispdata.entry_x_offset.delete(0, END)
    offdispdata.entry_y_offset.delete(0, END)
    if offdata.manual_offset is not None:
        offdispdata.entry_x_offset.insert(0, '%6.2f'%offdata.the_offset[0])
        offdispdata.entry_y_offset.insert(0, '%6.2f'%offdata.the_offset[1])
    draw_offset_overlay(offdata, offdispdata)
    
def command_man_from_cassini(offdata, offdispdata):
    offdata.manual_offset = (0.,0.)
    offdispdata.entry_x_offset.delete(0, END)
    offdispdata.entry_y_offset.delete(0, END)
    if offdata.manual_offset is not None:
        offdispdata.entry_x_offset.insert(0, '%6.2f'%offdata.manual_offset[0])
        offdispdata.entry_y_offset.insert(0, '%6.2f'%offdata.manual_offset[1])
    draw_offset_overlay(offdata, offdispdata)

# <Enter> key pressed in a manual offset text entry box
def command_enter_offset(event, offdata, offdispdata):
    if offdispdata.entry_x_offset.get() == "" or offdispdata.entry_y_offset.get() == "":
        offdata.manual_offset = None
    else:
        offdata.manual_offset = (float(offdispdata.entry_x_offset.get()),
                                    float(offdispdata.entry_y_offset.get()))
    draw_offset_overlay(offdata, offdispdata)

# "Recalculate offset" button pressed
def command_recalc_offset(offdata, offdispdata):
    offset_one_image(offdata, False, False, True, save_results=False)
    offdata.manual_offset = None
    offdispdata.entry_x_offset.delete(0, END)
    offdispdata.entry_y_offset.delete(0, END)
    if offdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None' 
        auto_y_text = 'Auto Y Offset: None' 
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offdata.the_offset[1]
        
    offdispdata.auto_x_label.config(text=auto_x_text)
    offdispdata.auto_y_label.config(text=auto_y_text)

def callback_b1press(x, y, offdispdata):
    if offdispdata.off_longitudes is not None and offdispdata.off_radii is not None:
        longitude = offdispdata.off_longitudes[y,x]
        radius = offdispdata.off_radii[y,x]
        ring_x = np.cos(longitude * oops.RPD) * radius
        ring_y = np.sin(longitude * oops.RPD) * radius
        print 'X %4d Y %d LONG %7.3f RADIUS %7.3f RX %.1f RY %.1f'%(x,y,longitude,radius,ring_x,ring_y)
        if offdispdata.last_xy is not None:
            print 'DIST', np.sqrt((ring_x-offdispdata.last_xy[0])**2+
                                  (ring_y-offdispdata.last_xy[1])**2)
        offdispdata.last_xy = (ring_x, ring_y)
        

# Setup the offset window with no data
def setup_offset_window(offdata, offdispdata):
    if offdata.the_offset is not None:
        offdata.obs.fov = oops.fov.OffsetFOV(offdata.obs.fov, uv_offset=offdata.the_offset)
    set_obs_bp(offdata.obs)
    
    offdispdata.off_radii = offdata.obs.bp.ring_radius('saturn:ring').vals.astype('float')
    offdispdata.off_longitudes = offdata.obs.bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
    
    offdispdata.toplevel = Tk()
    offdispdata.toplevel.title(offdata.obsid + ' / ' + offdata.image_name)
    frame_toplevel = Frame(offdispdata.toplevel)

    offset_overlay = offdata.off_metadata['overlay'].copy()

    # The original image and overlaid ring curves
    offdispdata.imdisp_offset = ImageDisp([offdata.obs.data], [offset_overlay],
                                          parent=frame_toplevel, canvas_size=(512,512),
                                             allow_enlarge=True, auto_update=True)

    callback_b1press_command = lambda x, y, offdispdata=offdispdata: callback_b1press(x, y, offdispdata)
    offdispdata.imdisp_offset.bind_b1press(0, callback_b1press_command)

    ###############################################
    # The control/data pane of the original image #
    ###############################################
    
    img_addon_control_frame = offdispdata.imdisp_offset.addon_control_frame
    
    gridrow = 0
    gridcolumn = 0

    if offdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None' 
        auto_y_text = 'Auto Y Offset: None' 
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offdata.the_offset[1]
        
    offdispdata.auto_x_label = Label(img_addon_control_frame, text=auto_x_text)
    offdispdata.auto_x_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    offdispdata.auto_y_label = Label(img_addon_control_frame, text=auto_y_text)
    offdispdata.auto_y_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    # X offset and Y offset entry boxes
    # We should really use variables for the Entry boxes, but for some reason they don't work
    label = Label(img_addon_control_frame, text='X Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    
    offdispdata.entry_x_offset = FloatEntry(img_addon_control_frame)
    offdispdata.entry_x_offset.delete(0, END)
    if offdata.manual_offset is not None:
        offdispdata.entry_x_offset.insert(0, '%6.2f'%offdata.manual_offset[0])
    offdispdata.entry_x_offset.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(img_addon_control_frame, text='Y Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    
    offdispdata.entry_y_offset = FloatEntry(img_addon_control_frame)
    offdispdata.entry_y_offset.delete(0, END)
    if offdata.manual_offset is not None:
        offdispdata.entry_y_offset.insert(0, '%6.2f'%offdata.manual_offset[1])
    offdispdata.entry_y_offset.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    enter_offset_command = (lambda x, offdata=offdata, offdispdata=offdispdata:
                            command_enter_offset(x, offdata, offdispdata))    
    offdispdata.entry_x_offset.bind('<Return>', enter_offset_command)
    offdispdata.entry_y_offset.bind('<Return>', enter_offset_command)

    # Set manual to automatic
    button_man_from_auto_command = (lambda offdata=offdata, offdispdata=offdispdata:
                                    command_man_from_auto(offdata, offdispdata))
    button_man_from_auto = Button(img_addon_control_frame, text='Set Manual from Auto',
                                  command=button_man_from_auto_command)
    button_man_from_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    #Set manual to Cassini
    button_man_from_cassini_command = (lambda offdata=offdata, offdispdata=offdispdata:
                                    command_man_from_cassini(offdata, offdispdata))
    button_man_cassini_auto = Button(img_addon_control_frame, text='Set Manual from Cassini',
                                  command=button_man_from_cassini_command)
    button_man_cassini_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Recalculate auto offset
    button_recalc_offset_command = (lambda offdata=offdata, offdispdata=offdispdata:
                                    command_recalc_offset(offdata, offdispdata))
    button_recalc_offset = Button(img_addon_control_frame, text='Recalculate Offset',
                                  command=button_recalc_offset_command)
    button_recalc_offset.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Display for longitude and radius
    label = Label(img_addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_inertial_longitude = Label(img_addon_control_frame, text='')
    offdispdata.label_off_inertial_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_radius = Label(img_addon_control_frame, text='')
    offdispdata.label_off_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    callback_offset_command = lambda x, y, offdata=offdata, offdispdata=offdispdata: callback_offset(x, y, offdata, offdispdata)
    offdispdata.imdisp_offset.bind_mousemove(0, callback_offset_command)

    frame_toplevel.pack()

# Display the original image
def display_offset(offdata, offdispdata):
    # The original image
    
    if offdata.obs is None:
        offdata.obs = iss.from_file(offdata.image_path)

    img_max_y = offdata.obs.data.shape[1]-1

    setup_offset_window(offdata, offdispdata)

    mainloop()


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

offdispdata = OffDispData()

def process(image_path):
    obsid = 'XXX'
    
    offdata = OffData()
    offdata.obsid = obsid
    _, offdata.image_name = os.path.split(image_path)
    offdata.image_path = image_path
    
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()

    # Pointing offset
    offset_one_image(offdata)
    
    # Display offset
    display_offset(offdata, offdispdata)
    
    del offdata
    offdata = None
    
    if do_profile:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        print s.getvalue()
        assert False

do_profile = False
        
process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054\data/1617917998_1618066143/W1617920879_1_CALIB.IMG')

#process(r'T:\external\cassini\derived\COISS_2xxx/COISS_2056/data/1626245310_1626407985/N1626333579_1_CALIB.IMG')
