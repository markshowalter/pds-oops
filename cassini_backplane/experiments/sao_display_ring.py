'''
Created on Sep 19, 2011

@author: rfrench
'''

import os
import os.path
import numpy as np
from imgdisp import ImageDisp, FloatEntry, draw_line
from Tkinter import *

class OffData(object):
    """Offset and Reprojection data."""
    def __init__(self):
        self.obsid = None
        self.image_name = None
        self.image_path = None
        self.data = None

class OffDispData(object):
    def __init__(self):
        self.data = None
        self.toplevel = None
        self.imdisp_offset = None
        self.off_longitudes = None
        self.off_radii = None
        self.label_off_inertial_longitude = None
        self.label_off_radius = None
        self.label_off_resolution = None
        self.label_off_emission = None
        self.label_off_incidence = None
        self.label_off_phase = None
        self.last_xy = None



#####################################################################################
#
# DISPLAY ONE IMAGE
#
#####################################################################################

# The callback for mouse move events on the offset image
def callback_offset(x, y, offdata, offdispdata):
    if offdispdata.off_longitudes is not None:
        offdispdata.label_off_inertial_longitude.config(text=('%7.3f'%offdispdata.off_longitudes[y,x]))
    if offdispdata.off_radii is not None:
        offdispdata.label_off_radius.config(text=('%7.3f'%offdispdata.off_radii[y,x]))
    if offdispdata.off_resolution is not None:
        offdispdata.label_off_resolution.config(text=('%7.3f'%offdispdata.off_resolution[y,x]))
    if offdispdata.off_emission is not None:
        offdispdata.label_off_emission.config(text=('%7.3f'%offdispdata.off_emission[y,x]))
    if offdispdata.off_incidence is not None:
        offdispdata.label_off_incidence.config(text=('%7.3f'%offdispdata.off_incidence[y,x]))
    if offdispdata.off_phase is not None:
        offdispdata.label_off_phase.config(text=('%7.3f'%offdispdata.off_phase[y,x]))

def callback_b1press(x, y, offdispdata):
    if offdispdata.off_longitudes is not None and offdispdata.off_radii is not None:
        longitude = offdispdata.off_longitudes[y,x]
        radius = offdispdata.off_radii[y,x]
        ring_x = np.cos(longitude * np.pi/180) * radius
        ring_y = np.sin(longitude * np.pi/180) * radius
        print 'X %4d Y %d LONG %7.3f RADIUS %7.3f RX %.1f RY %.1f'%(x,y,longitude,radius,ring_x,ring_y)
        if offdispdata.last_xy is not None:
            print 'DIST', np.sqrt((ring_x-offdispdata.last_xy[0])**2+
                                  (ring_y-offdispdata.last_xy[1])**2)
            print 'ANGLE', longitude-np.arctan2(ring_y-offdispdata.last_xy[1],
                                      ring_x-offdispdata.last_xy[0]) * 180./np.pi
        offdispdata.last_xy = (ring_x, ring_y)
        

# Setup the offset window with no data
def setup_offset_window(offdata, offdispdata):
    npres = np.load(offdata.image_path+'.npz')
    offdata.data = npres['data']
    offdispdata.off_radii = npres['radii']
    offdispdata.off_longitudes = npres['longitudes']
    offdispdata.off_resolution = npres['resolution']
    offdispdata.off_incidence = npres['incidence']
    offdispdata.off_emission = npres['emission']
    offdispdata.off_phase = npres['phase']
    
    offdispdata.toplevel = Tk()
    offdispdata.toplevel.title(offdata.image_name)
    frame_toplevel = Frame(offdispdata.toplevel)

    # The original image and overlaid ring curves
    offdispdata.imdisp_offset = ImageDisp([offdata.data],
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

    label = Label(img_addon_control_frame, text='Radial Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_resolution = Label(img_addon_control_frame, text='')
    offdispdata.label_off_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_incidence = Label(img_addon_control_frame, text='')
    offdispdata.label_off_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_emission= Label(img_addon_control_frame, text='')
    offdispdata.label_off_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_phase = Label(img_addon_control_frame, text='')
    offdispdata.label_off_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    callback_offset_command = lambda x, y, offdata=offdata, offdispdata=offdispdata: callback_offset(x, y, offdata, offdispdata)
    offdispdata.imdisp_offset.bind_mousemove(0, callback_offset_command)

    frame_toplevel.pack()

# Display the original image
def display_offset(offdata, offdispdata):
    # The original image
    
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
    
    # Display offset
    display_offset(offdata, offdispdata)
    
#process(r'j:/Temp/N1624900314_1_CALIB.IMG-repro')
#process(r'C:\Rob\Dropbox\Shared-Tom-Woodrow\N1622394375_1_CALIB.IMG-repro')

###### TOM

#process(r'j:/Temp/N1621959213_1_CALIB.IMG-repro')

#process(r'j:/Temp/N1627295896_1_CALIB.IMG-repro') 
#process(r'j:/Temp/N1617919678_1_CALIB.IMG-repro')

###### LORI

#process(r'j:/Temp/N1613101588_1_CALIB.IMG')
#process(r'j:/Temp/N1613101631_1_CALIB.IMG')
#process(r'j:/Temp/N1613405325_1_CALIB.IMG')
#process(r'j:/Temp/N1613405368_1_CALIB.IMG')
#process(r'j:/Temp/N1613977923_1_CALIB.IMG')
#process(r'j:/Temp/N1613977956_1_CALIB.IMG')
#process(r'j:/Temp/N1614551168_1_CALIB.IMG')
#process(r'j:/Temp/N1614551201_1_CALIB.IMG')
#process(r'j:/Temp/N1618072263_1_CALIB.IMG')
#process(r'j:/Temp/N1618072306_1_CALIB.IMG')
#process(r'j:/Temp/N1619668649_1_CALIB.IMG')
#process(r'j:/Temp/N1619668692_1_CALIB.IMG')
#process(r'j:/Temp/N1613101631_1_CALIB.IMG')
#process(r'j:/Temp/N1612537044_1_CALIB.IMG')
#process(r'j:/Temp/N1612537087_1_CALIB.IMG')
#process(r'j:/Temp/N1619963567_1_CALIB.IMG')
#process(r'j:/Temp/N1619963610_1_CALIB.IMG')
#process(r'j:/Temp/N1620555021_1_CALIB.IMG')
#process(r'j:/Temp/N1620555054_1_CALIB.IMG')
#process(r'j:/Temp/N1621003584_1_CALIB.IMG')
#process(r'j:/Temp/N1621003617_1_CALIB.IMG')
#process(r'j:/Temp/N1621841220_1_CALIB.IMG')
#process(r'j:/Temp/N1621841536_1_CALIB.IMG')
#process(r'j:/Temp/N1621841856_1_CALIB.IMG')
#process(r'j:/Temp/N1621842176_1_CALIB.IMG')
#process(r'j:/Temp/N1621842496_1_CALIB.IMG')
#process(r'j:/Temp/N1621842816_1_CALIB.IMG')
#process(r'j:/Temp/N1621843136_1_CALIB.IMG')
#process(r'j:/Temp/N1621843456_1_CALIB.IMG')
#process(r'j:/Temp/N1621843776_1_CALIB.IMG')
#process(r'j:/Temp/N1621844096_1_CALIB.IMG')
#process(r'j:/Temp/N1621844416_1_CALIB.IMG')
#process(r'j:/Temp/N1621844736_1_CALIB.IMG')
#process(r'j:/Temp/N1621845056_1_CALIB.IMG')
#process(r'j:/Temp/N1621845376_1_CALIB.IMG')
#process(r'j:/Temp/N1621845705_1_CALIB.IMG')
#process(r'j:/Temp/N1621846016_1_CALIB.IMG')
#process(r'j:/Temp/N1621846336_1_CALIB.IMG')
#process(r'j:/Temp/N1621846656_1_CALIB.IMG')
#process(r'j:/Temp/N1621846976_1_CALIB.IMG')
#process(r'j:/Temp/N1621847296_1_CALIB.IMG')
process(r'j:/Temp/N1621847616_1_CALIB.IMG')
#process(r'j:/Temp/N1621847936_1_CALIB.IMG')
#process(r'j:/Temp/N1621848256_1_CALIB.IMG')
#process(r'j:/Temp/N1621848576_1_CALIB.IMG')
#process(r'j:/Temp/N1621848896_1_CALIB.IMG')
#process(r'j:/Temp/N1621849216_1_CALIB.IMG')
#process(r'j:/Temp/N1621849536_1_CALIB.IMG')
#process(r'j:/Temp/N1621849856_1_CALIB.IMG')
#process(r'j:/Temp/N1621850176_1_CALIB.IMG')
#process(r'j:/Temp/N1621850497_1_CALIB.IMG')
#process(r'j:/Temp/N1622138672_1_CALIB.IMG')
#process(r'j:/Temp/N1622138715_1_CALIB.IMG')
#process(r'j:/Temp/N1623757093_1_CALIB.IMG')
#process(r'j:/Temp/N1623757136_1_CALIB.IMG')
process(r'j:/Temp/N1624883466_1_CALIB.IMG')
#process(r'j:/Temp/N1624883509_1_CALIB.IMG')
