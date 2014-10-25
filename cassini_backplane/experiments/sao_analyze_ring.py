'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
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
import cb_correlate
from cb_util_image import *

INTERACTIVE = True

LONGITUDE_RESOLUTION = 0.005
RADIUS_RESOLUTION = 5

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

class OffDispData(object):
    def __init__(self):
        self.obs = None
        self.toplevel = None
        self.imdisp_offset = None
        self.entry_x_offset = None
        self.entry_y_offset = None
        self.off_longitudes = None
        self.off_radii = None
        self.off_emission = None
        self.off_incidence = None
        self.off_phase = None
        self.label_off_inertial_longitude = None
        self.label_off_radius = None
        self.label_off_resolution = None
        self.label_off_emission = None
        self.label_off_incidence = None
        self.label_off_phase = None
        self.last_xy = None

#####################################################################################
#
# FIND THE POINTING OFFSET
#
#####################################################################################
    
#
# The primary entrance for finding pointing offset
#

def offset_one_image(offdata, offset_u, offset_v):
    # Recompute the automatic offset
    obs = iss.from_file(offdata.image_path)
    offdata.obs = obs
    offdata.off_metadata = {}
    if offset_u is None:
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
        if 'used_objects_type' in offdata.off_metadata:
            print offdata.off_metadata['used_objects_type'], offdata.off_metadata['model_overrides_stars']



#####################################################################################
#
# DISPLAY ONE IMAGE ALLOWING MANUAL CHANGING OF THE OFFSET
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

# Draw the offset curves
def draw_offset_overlay(offrepdata, offrepdispdata):
    # Blue - 0,0 offset
    # Yellow - auto offset
    # Green - manual offset
    try:
        offset_overlay = offrepdata.off_metadata['overlay'].copy()
        if offset_overlay.shape[:2] != offrepdata.obs.data.shape:
            # Correct for the expanded size of ext_data
            diff_y = (offset_overlay.shape[0]-offrepdata.obs.data.shape[0])/2
            diff_x = (offset_overlay.shape[1]-offrepdata.obs.data.shape[1])/2
            offset_overlay = offset_overlay[diff_y:diff_y+offrepdata.obs.data.shape[0],
                                            diff_x:diff_x+offrepdata.obs.data.shape[1],:]
    except:
        offset_overlay = np.zeros((offrepdata.obs.data.shape + (3,)))
    if offrepdata.manual_offset is not None:
        if offrepdata.the_offset is None:
            x_diff = int(-offrepdata.manual_offset[0]) 
            y_diff = int(-offrepdata.manual_offset[1])
        else: 
            x_diff = int(offrepdata.the_offset[0] - offrepdata.manual_offset[0])
            y_diff = int(offrepdata.the_offset[1] - offrepdata.manual_offset[1])
        offset_overlay = shift_image(offset_overlay, x_diff, y_diff)
    
    offrepdispdata.imdisp_offset.set_overlay(0, offset_overlay)
    offrepdispdata.imdisp_offset.pack(side=LEFT)

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
def setup_offset_window(offdata, offdispdata, reproject):
    if not reproject:
        if offdata.the_offset is not None:
            offdata.obs.fov = oops.fov.OffsetFOV(offdata.obs.fov, uv_offset=offdata.the_offset)
        set_obs_bp(offdata.obs)
        
        offdispdata.off_radii = offdata.obs.bp.ring_radius('saturn:ring').vals.astype('float')
        offdispdata.off_longitudes = offdata.obs.bp.ring_longitude('saturn:ring', reference='aries').vals.astype('float') * oops.DPR
        offdispdata.off_resolution = offdata.obs.bp.ring_radial_resolution('saturn:ring').vals.astype('float')
        offdispdata.off_incidence = offdata.obs.bp.incidence_angle('saturn:ring').vals.astype('float') * oops.DPR
        offdispdata.off_emission = offdata.obs.bp.emission_angle('saturn:ring').vals.astype('float') * oops.DPR
        offdispdata.off_phase = offdata.obs.bp.phase_angle('saturn:ring').vals.astype('float') * oops.DPR

    else:
#    ret['long_mask'] = good_long_mask
#    ret['img'] = repro_mosaic
#    ret['mean_resolution'] = repro_mean_res
#    ret['mean_phase'] = repro_mean_phase
#    ret['mean_emission'] = repro_mean_emission
#    ret['mean_incidence'] = repro_mean_incidence
#    ret['time'] = obs.midtime

        if offdata.the_offset is None:
            return
        
        ret = rings_reproject(offdata.obs, offset_u=offdata.the_offset[0], offset_v=offdata.the_offset[1],
                      longitude_resolution=LONGITUDE_RESOLUTION,
                      radius_resolution=RADIUS_RESOLUTION,
                      radius_inner=135000.,
                      radius_outer=138000.)
        offdata.obs.data = ret['img']
        radii = rings_generate_radii(135000.,138000.,radius_resolution=RADIUS_RESOLUTION)
        offdispdata.off_radii = np.zeros(offdata.obs.data.shape)
        offdispdata.off_radii[:,:] = radii[:,np.newaxis]
        longitudes = rings_generate_longitudes(longitude_resolution=LONGITUDE_RESOLUTION)
        offdispdata.off_longitudes = np.zeros(offdata.obs.data.shape)
        offdispdata.off_longitudes[:,:] = longitudes[ret['long_mask']]
        offdispdata.off_resolution = ret['resolution']
        offdispdata.off_incidence = ret['incidence']
        offdispdata.off_emission = ret['emission']
        offdispdata.off_phase = ret['phase']
        
    if reproject:
        filename = 'c:/Temp/'+offdata.image_name+'-repro'
    else:
        filename = 'c:/Temp/'+offdata.image_name
    np.savez(filename, 
             data=offdata.obs.data,
             radii=offdispdata.off_radii,
             longitudes=offdispdata.off_longitudes,
             resolution=offdispdata.off_resolution,
             incidence=offdispdata.off_incidence,
             emission=offdispdata.off_emission,
             phase=offdispdata.off_phase)

    if not INTERACTIVE:
        return
    
    offdispdata.toplevel = Tk()
    offdispdata.toplevel.title(offdata.obsid + ' / ' + offdata.image_name)
    frame_toplevel = Frame(offdispdata.toplevel)

    if reproject:
        offset_overlay = None
    else:
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

    label = Label(img_addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_incidence = Label(img_addon_control_frame, text='')
    offdispdata.label_off_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Radial Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_resolution = Label(img_addon_control_frame, text='')
    offdispdata.label_off_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offdispdata.label_off_emission = Label(img_addon_control_frame, text='')
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

    mainloop()

# Display the original image
def display_offset(offdata, offdispdata, reproject):
    # The original image
    
    if offdata.obs is None:
        offdata.obs = iss.from_file(offdata.image_path)

    setup_offset_window(offdata, offdispdata, reproject)


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

offdispdata = OffDispData()

def process(image_path, reproject=False, offset_u=None, offset_v=None):
    print image_path
    
    obsid = 'XXX'
    
    offdata = OffData()
    offdata.obsid = obsid
    _, offdata.image_name = os.path.split(image_path)
    offdata.image_path = image_path
    
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()

    # Pointing offset
    offset_one_image(offdata, offset_u, offset_v)
    
    # Display offset
    display_offset(offdata, offdispdata, reproject=reproject)
    
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

##### TOM

#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624900314_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624901034_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624901574_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621847296_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622394375_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621851000_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621957143_1621968573/N1621958853_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621957143_1621968573/N1621959213_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622711732_1623166344/N1623166278_1_CALIB.IMG',reproject=True,offset_u=-70,offset_v=21)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623167085_1_CALIB.IMG',reproject=True,offset_u=-90,offset_v=29)#,offset_u=-30,offset_v=55)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623168213_1_CALIB.IMG',reproject=True,offset_u=-120,offset_v=36)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623169053_1_CALIB.IMG',reproject=True,offset_u=-118,offset_v=64)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623170247_1_CALIB.IMG',reproject=True,offset_u=-200,offset_v=49)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624654802_1624836470/N1624728470_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624654802_1624836470/N1624729501_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624903734_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1626159850_1626244971/N1626216495_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1625995143_1626159520/N1625998036_1_CALIB.IMG',reproject=True,offset_u=5,offset_v=10)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627319215_1627453306/N1627448056_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627319215_1627453306/N1627452306_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627453556_1627522774/N1627453806_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627453556_1627522774/N1627456806_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627453556_1627522774/N1627465056_1_CALIB.IMG',reproject=True)
# process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1630191052_1630290194/N1630270653_1_CALIB.IMG',reproject=True)

#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632170323_1632227130/N1632214421_1_CALIB.IMG',reproject=False)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632170323_1632227130/N1632214600_1_CALIB.IMG',reproject=False)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632170323_1632227130/N1632214779_1_CALIB.IMG',reproject=False)
process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632170323_1632227130/N1632214958_1_CALIB.IMG',reproject=False)
process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632170323_1632227130/N1632215137_1_CALIB.IMG',reproject=False)




#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622549816_1622632159/N1622592788_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2052/data/1609906195_1610170084/N1610086205_1_CALIB.IMG')
#process(r't:/clumps/data/ISS_075RF_FMOVIE002_VIMS/N1593945662_1_CALIB.IMG')
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1630191052_1630290194/N1630270620_1_CALIB.IMG',reproject=True)
#process(r'T:\external\cassini\derived\COISS_2xxx\COISS_2054\data/1617917998_1618066143/W1617920879_1_CALIB.IMG')

#process(r'T:\external\cassini\derived\COISS_2xxx/COISS_2056/data/1626245310_1626407985/N1626333579_1_CALIB.IMG')
