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
from cb_util_file import *
from cb_util_image import *

INTERACTIVE = False

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

def offset_one_image(offdata, **kwargs):
    # Recompute the automatic offset
    obs = read_iss_file(offdata.image_path)
    offdata.obs = obs
    offdata.off_metadata = {}
    offset_u = None
    offset_v = None
    if 'offset_u' in kwargs:
        offset_u = kwargs['offset_u']
        offset_v = kwargs['offset_v']
    if offset_u is None:
        try:
            rings_config = RINGS_DEFAULT_CONFIG.copy()
            rings_config['fiducial_feature_threshold'] = 0

            offdata.off_metadata = master_find_offset(obs,
                                                     create_overlay=True,
                                                     rings_config=rings_config)                                                                             
            offset = offdata.off_metadata['offset'] 
            if offset is not None:
                offset_u, offset_v = offset
        except:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
            print 'EXCEPTION:'
            print sys.exc_info()
#            raise
    
    if offset_u is None:
        offdata.the_offset = None
    else:
        offdata.the_offset = (offset_u, offset_v)
    if offset_u is None:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    if offset_u is not None:
        print 'FOUND %6.2f, %6.2f' % (offdata.the_offset[0], offdata.the_offset[1])
        if 'used_objects_type' in offdata.off_metadata:
            print 'Model type:', offdata.off_metadata['used_objects_type'],
            print '  Model overrides stars:', offdata.off_metadata['model_overrides_stars']



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
            print 'ANGLE', np.arctan2(ring_y-offdispdata.last_xy[1],
                                      ring_x-offdispdata.last_xy[0])
        offdispdata.last_xy = (ring_x, ring_y)
        

# Setup the offset window with no data
def setup_offset_window(offdata, offdispdata, reproject, radius_inner=135000.,
                        radius_outer=138000.,**kwargs):
    if not reproject:
        if offdata.the_offset is not None:
            offdata.obs.fov = oops.fov.OffsetFOV(offdata.obs.fov, uv_offset=offdata.the_offset)
        set_obs_bp(offdata.obs)
        
        offdispdata.off_radii = offdata.obs.bp.ring_radius('saturn:ring').vals.astype('float')
        offdispdata.off_longitudes = offdata.obs.bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
        offdispdata.off_resolution = offdata.obs.bp.ring_radial_resolution('saturn:ring').vals.astype('float')
        offdispdata.off_incidence = offdata.obs.bp.incidence_angle('saturn:ring').vals.astype('float') * oops.DPR
        offdispdata.off_emission = offdata.obs.bp.emission_angle('saturn:ring').vals.astype('float') * oops.DPR
        offdispdata.off_phase = offdata.obs.bp.phase_angle('saturn:ring').vals.astype('float') * oops.DPR

        last_x = None
        last_y = None
        for pt_num in ['1', '2', '3', '4']:
            if 'x'+pt_num in kwargs:
                x = kwargs['x'+pt_num]
                y = kwargs['y'+pt_num]
                longitude = offdispdata.off_longitudes[y,x]
                radius = offdispdata.off_radii[y,x]
                ring_x = np.cos(longitude * oops.RPD) * radius
                ring_y = np.sin(longitude * oops.RPD) * radius
                print 'X%s %4d Y%s %d LONG %7.3f RADIUS %7.3f RX %.1f RY %.1f'%(pt_num, x, pt_num, y,longitude,radius,ring_x,ring_y),
                if last_x is not None:
                    print 'DIST %.1f' % (np.sqrt((ring_x-last_x)**2+(ring_y-last_y)**2))
                    last_x = None
                    last_y = None
                else:
                    print
                    last_x = ring_x
                    last_y = ring_y
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
                      radius_inner=radius_inner,
                      radius_outer=radius_outer)
        offdata.obs.data = ret['img']
        radii = rings_generate_radii(radius_inner,radius_outer,radius_resolution=RADIUS_RESOLUTION)
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
        filename = 'j:/Temp/'+offdata.image_name+'-repro'
    else:
        filename = 'j:/Temp/'+offdata.image_name
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
    
    if reproject:
        offset_overlay = None
    else:
        offset_overlay = offdata.off_metadata['overlay'].copy()

    # The original image and overlaid ring curves
    offdispdata.imdisp_offset = ImageDisp([offdata.obs.data], [offset_overlay],
                                          title=offdata.obsid + ' / ' + offdata.image_name,
                                          canvas_size=(512,512),
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

    offdispdata.imdisp_offset.pack()

    mainloop()

# Display the original image
def display_offset(offdata, offdispdata, reproject, **kwargs):
    # The original image
    
    if offdata.obs is None:
        offdata.obs = iss.from_file(offdata.image_path)

    setup_offset_window(offdata, offdispdata, reproject, **kwargs)


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

offdispdata = OffDispData()

def process(image_path, reproject=False, **kwargs):
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
    offset_one_image(offdata, **kwargs)
    
    # Display offset
    display_offset(offdata, offdispdata, reproject=reproject, **kwargs)
    
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

Tk().withdraw()

#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621851000_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1626245310_1626407985/N1626326915_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627319215_1627453306/N1627448306_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627319215_1627453306/N1627451806_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1613001873_1613171522/N1613101588_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1613291015_1613598197/N1613405325_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1613598819_1613977956/N1613977923_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1614457968_1614561850/N1614551168_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1618067253_1618407253/N1618072263_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619427338_1619724488/N1619668649_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619945739_1620034486/N1619963567_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620380865_1620646547/N1620555021_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620914692_1621017875/N1621003584_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621841220_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621847296_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621652147_1621937939/N1621850497_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622043391_1622198245/N1622138672_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622549816_1622632159/N1622592755_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623667817_1623919770/N1623757093_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624883466_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627319215_1627453306/N1627409979_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/W1622272936_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622545936_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1617049939_1617119192/W1617111781_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1617049939_1617119192/N1617112673_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2053/data/1617049939_1617119192/W1617112673_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617661596_1617917758/N1617836777_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617917998_1618066143/N1617918238_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617917998_1618066143/N1617919199_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617917998_1618066143/W1617919199_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617917998_1618066143/N1618005008_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1617917998_1618066143/W1618005008_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619427338_1619724488/N1619669599_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619427338_1619724488/W1619669599_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619427338_1619724488/N1619670961_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619427338_1619724488/W1619670961_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619725413_1619833779/N1619789123_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1619725413_1619833779/N1619790275_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620380865_1620646547/N1620595706_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620646742_1620671507/N1620663122_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620671702_1620911619/N1620674432_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1620671702_1620911619/N1620678332_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2054/data/1621957143_1621968573/N1621959123_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622043391_1622198245/N1622141708_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622198904_1622272726/W1622224838_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622198904_1622272726/N1622233144_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622396010_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622396730_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622396910_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622397810_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622546735_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622272893_1622549559/N1622548535_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622711732_1623166344/N1623166278_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623174432_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/W1623174432_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/W1623175932_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623175932_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623178145_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/W1623181896_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623166377_1623224391/N1623181896_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623224496_1623283102/W1623249847_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623224496_1623283102/N1623249847_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623224496_1623283102/W1623252547_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623224496_1623283102/N1623252547_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1623283200_1623345100/N1623338546_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624039158_1624239287/N1624153495_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624654802_1624836470/N1624729156_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624654802_1624836470/N1624731250_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624894914_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624903554_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/W1624905683_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1625995143_1626159520/N1625999575_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295298_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295382_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295466_1_CALIB.IMG')
#process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295729_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295812_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295896_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627295980_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627296064_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627296148_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627296232_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2056/data/1627217931_1627301149/N1627296316_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629144588_1629174249/N1629145473_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629342794_1629355579/N1629351859_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629342794_1629355579/N1629353816_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629342794_1629355579/N1629354694_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629428390_1629453020/N1629449450_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1629538820_1629635391/N1629557510_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2057/data/1632306929_1632459671/N1632450428_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2060/data/1641588245_1641624294/N1641603464_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2060/data/1641624586_1641842861/N1641629842_1_CALIB.IMG')
process(r't:\external\cassini\derived\COISS_2xxx\COISS_2060/data/1641624586_1641842861/N1641794334_1_CALIB.IMG')
