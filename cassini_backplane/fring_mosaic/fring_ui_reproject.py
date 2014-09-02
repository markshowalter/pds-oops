'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import fring_util
import pickle
import os
import os.path
import numpy as np
import sys
import cspice
import scipy.ndimage.interpolation as interp
from imgdisp import ImageDisp, FloatEntry, ScrolledList
from Tkinter import *
from PIL import Image
import oops.inst.cassini.iss as iss
from cb_offset import *

python_filename = sys.argv[0]

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                'ISS_029RF_FMOVIE002_VIMS',
                '--allow-exception',
#                '--recompute-reproject',
#                '--recompute-auto-offset',
                '--no-reproject',
                '--display-offset-reproject',
                 '--verbose']
#    cmd_line = ['-a', '--verbose']
    pass

parser = OptionParser() 

#
# For each of (offset, reprojection), the default behavior is to check the timestamps
# on the input file and the output file and recompute if the output file is out of date.
# Several options change this behavior:
#   --no-xxx: Don't recompute no matter what; this may leave you without an output file at all
#   --no-update: Don't recompute if the output file exists, but do compute if the output file doesn't exist at all
#   --recompute-xxx: Force recompute even if the output file exists and is current
#


##
## General options
##
parser.add_option('--allow-exception', dest='allow_exception',
                  action='store_true', default=False,
                  help="Allow exceptions to be thrown")

##
## Options for finding the pointing offset
##
parser.add_option('--no-auto-offset', dest='no_auto_offset',
                  action='store_true', default=False,
                  help="Don't compute the automatic offset even if we don't have one")
parser.add_option('--no-update-auto-offset', dest='no_update_auto_offset',
                  action='store_true', default=False,
                  help="Don't compute the automatic offset unless we don't have one")
parser.add_option('--recompute-auto-offset', dest='recompute_auto_offset',
                  action='store_true', default=False,
                  help='Recompute the automatic offset even if we already have one that is current')
parser.add_option('--display-offset-reproject', dest='display_offset_reproject',
                  action='store_true', default=False,
                  help='Display the offset and reprojection and allow manual change')
parser.add_option('--display-invalid-offset', dest='display_invalid_offset',
                  action='store_true', default=False,
                  help='Display the offset and reprojection and allow manual change only for images that have bad automatic offsets')
parser.add_option('--display-invalid-reproject', dest='display_invalid_reproject',
                  action='store_true', default=False,
                  help='Display the offset and reprojection and allow manual change only for images that have a bad reprojection')

##
## Options for reprojection
##
parser.add_option('--no-reproject', dest='no_reproject',
                  action='store_true', default=False,
                  help="Don't compute the reprojection even if we don't have one")
parser.add_option('--no-update-reproject', dest='no_update_reproject',
                  action='store_true', default=False,
                  help="Don't compute the reprojection unless if we don't have one")
parser.add_option('--recompute-reproject', dest='recompute_reproject',
                  action='store_true', default=False,
                  help='Recompute the reprojection even if we already have one that is current')

fring_util.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

class OffRepDispData:
    def __init__(self):
        self.toplevel = None
        self.imdisp_offset = None
        self.imdisp_repro = None
        self.entry_x_offset = None
        self.entry_y_offset = None
        self.repro_overlay = None
        self.repro_longitudes = None
        self.repro_phase_angle = None
        self.repro_incidence_angle = None
        self.repro_emission_angle = None
        self.repro_resolutions = None
        

#####################################################################################
#
# FIND THE POINTING OFFSET
#
#####################################################################################
    
#
# The primary entrance for finding pointing offset
#

def offset_one_image(offrepdata, option_no, option_no_update, option_recompute, save_results=True):
    # Input file: image_path (<IMAGE>_CALIB.IMG)
    # Output file: offset_path(<IMAGE>_CALIB.IMG.FOFFSET)

    if options.verbose:
        print '** Find offset', offrepdata.obsid, '/', offrepdata.image_name, '-',
    
    if option_no:  # Just don't do anything - we hope you know what you're doing!
        if options.verbose:
            print 'Ignored because of --no-auto_offset'
        return
        
    if os.path.exists(offrepdata.offset_path):
        if option_no_update:
            if options.verbose:
                print 'Ignored because offset file already exists'
            return # Offset file already exists, don't update
        # Save the manual offset!
        trash1, offrepdata.manual_offset, trash2 = fring_util.read_offset(offrepdata.offset_path)
        time_offset = os.stat(offrepdata.offset_path).st_mtime
    else:
        time_offset = 0
        
    time_image = os.stat(offrepdata.image_path).st_mtime
    if time_offset >= time_image and not option_recompute:
        # The offset file exists and is more recent than the image, and we're not forcing a recompute
        if options.verbose:
            print 'Ignored because offset file is up to date'
        return

    # Recompute the automatic offset
    obs = iss.from_file(offrepdata.image_path)
    if options.allow_exception:
        offset_u, offset_v, offrepdata.off_metadata = master_find_offset(obs, create_overlay=True)
    else:
        try:
            offset_u, offset_v, offrepdata.off_metadata = master_find_offset(obs, create_overlay=True)
        except Exception as exc:
            if options.verbose:
                print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
                print exc
            offrepdata.offset_u = None
            offrepdata.offset_v = None
            offrepdata.off_metadata = {}
    offrepdata.the_offset = (offset_u, offset_v)
    if offset_u is None:
        if options.verbose:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    if offset_u is not None and options.verbose:
        print 'FOUND %6.2f, %6.2f' % (offrepdata.the_offset[0], offrepdata.the_offset[1])
    if save_results:
        fring_util.write_offset(offrepdata.offset_path, offrepdata.the_offset, offrepdata.manual_offset,
                              offrepdata.off_metadata)


#####################################################################################
#
# REPROJECT ONE IMAGE
#
#####################################################################################

def reproject_one_image(offrepdata, option_no, option_no_update, option_recompute):
    # Input file: offset_path (<IMAGE>_CALIB.IMG.OFFSET)
    # Output file: repro_path (<IMAGE>_<RES_DATA>_REPRO.IMG)

    if options.verbose:
        print '** Reproject', offrepdata.obsid, '/', offrepdata.image_name, '-',
    
    if option_no:  # Just don't do anything
        if options.verbose:
            print 'Ignored because of --no-reproject'
        return

    if os.path.exists(offrepdata.repro_path):
        if option_no_update:
            if options.verbose:
                print 'Ignored because repro file already exists'
            return # Repro file already exists, don't update
        time_repro = os.stat(offrepdata.repro_path).st_mtime
    else:
        time_repro = 0
        
    time_offset = os.stat(offrepdata.offset_path).st_mtime
    if time_repro >= time_offset and not option_recompute:
        # The repro file exists and is more recent than the image, and we're not forcing a recompute
        if options.verbose:
            print 'Ignored because repro file is up to date'
        return
    
    (offrepdata.the_offset, offrepdata.manual_offset,
     offrepdata.off_metadata) = fring_util.read_offset(offrepdata.offset_path)
    
    if offrepdata.the_offset is None:
        if options.verbose:
            print 'OFFSET IS INVALID - ABORTING'
        return

    assert False # XXX        
    offrepdata.vicar_data = vicar.VicarImage.FromFile(offrepdata.image_path)
    offrepdata.ringim = fitreproj.VicarToRingImage(offrepdata.vicar_data)
#    if offrepdata.manual_offset is not None: XXX
#        offrepdata.ringim.SetOffset(offrepdata.manual_offset)
#    else:
#        offrepdata.ringim.SetOffset(offrepdata.the_offset)

    if options.allow_exception:
        offrepdata.repro_img = fitreproj.Reproject(offrepdata.vicar_data, offrepdata.ringim,
                                                   options.radius_start, options.radius_end,
                                                   options.radius_resolution, options.longitude_resolution)
        # Save new vicar image
        offrepdata.vicar_data.ToFile(offrepdata.repro_path)
    else:
        try:
            offrepdata.repro_img = fitreproj.Reproject(offrepdata.vicar_data, offrepdata.ringim,
                                                       options.radius_start, options.radius_end,
                                                       options.radius_resolution, options.longitude_resolution)
            # Save new vicar image
            offrepdata.vicar_data.ToFile(offrepdata.repro_path)
        except Exception as exc:
            if options.verbose:
                print 'REPROJECTION FAILED'
                print exc
            offrepdata.vicar_data = None  # vicar_data was mutated to be the repro image - we want the original
            return None
    
    if options.verbose:
        print 'OK'
    
    offrepdata.vicar_data = None  # vicar_data was mutated to be the repro image - we want the original
    return

#####################################################################################
#
# DISPLAY ONE IMAGE AND ITS REPROJECTION ALLOWING MANUAL CHANGING OF THE OFFSET
#
#####################################################################################

def draw_repro_overlay(offrepdata, offrepdispdata):
    return # XXX
#    if offrepdata.repro_img is None:
#        return
#    repro_overlay = np.zeros((offrepdata.repro_img.shape[0], offrepdata.repro_img.shape[1], 3))
#    y = repro_overlay.shape[0]-1-mosaic.RadiusToIndex(140220., options.radius_start, options.radius_resolution)
#    if 0 <= y < repro_overlay.shape[0]:
#        repro_overlay[y, :, 0] = 1
#    
#    offrepdispdata.imdisp_repro.set_overlay(0, repro_overlay)
    
# Draw the offset curves
def draw_offset_overlay(offrepdata, offrepdispdata):
    # Blue - 0,0 offset
    # Yellow - auto offset
    # Green - manual offset
    # Red - manual offset backgrounds
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
   
#    # Find where the ring is with the original Cassini offset
#    offrepdata.ringim.SetOffset((0,0))
#    fring = fitreproj.MakeFRing(FRINGS[offrepdata.fring_number], offrepdata.ringim.et)
#    rays = offrepdata.ringim.RingToRays(fring)
#    pixels = offrepdata.ringim.RaysToPixels(rays)
#    restrict = np.where(offrepdata.ringim.isInside(pixels[1]))
#    p = (pixels[1][restrict] + 0.5).astype("int")
#    fring_util.draw_lines(offset_overlay, (0,0,1), p) # Blue
#
#    # Find where the ring is with the auto offset
#    if offrepdata.the_offset is not None:
#        offrepdata.ringim.SetOffset(offrepdata.the_offset)
#        rays = offrepdata.ringim.RingToRays(fring)
#        pixels = offrepdata.ringim.RaysToPixels(rays)
#        for i in range(len(FRINGS[offrepdata.fring_number])):
#            restrict = np.where(offrepdata.ringim.isInside(pixels[i]))
#            p = (pixels[i][restrict] + 0.5).astype("int")
#            if i == 1: # F ring core
#                color = (1,1,0) # Yellow
#            else: # Background
#                color = (1,0,0) # Red
#            fring_util.draw_lines(offset_overlay, color, p)
#    
#    # Find where the ring (and background) is with the current manual offset
#    if offrepdata.manual_offset is not None:
#        offrepdata.ringim.SetOffset(offrepdata.manual_offset)
#        rays = offrepdata.ringim.RingToRays(fring)
#        pixels = offrepdata.ringim.RaysToPixels(rays)
#        restrict = np.where(offrepdata.ringim.isInside(pixels[1]))
#        p = (pixels[1][restrict] + 0.5).astype("int")
#        fring_util.draw_lines(offset_overlay, (0,.8,0), p) # Green
    offrepdispdata.imdisp_offset.set_overlay(0, offset_overlay)
    offrepdispdata.imdisp_offset.pack(side=LEFT)

# "Manual from auto" button pressed 
def command_man_from_auto(offrepdata, offrepdispdata):
    offrepdata.manual_offset = offrepdata.the_offset
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0, '%6.2f'%offrepdata.the_offset[0])
        offrepdispdata.entry_y_offset.insert(0, '%6.2f'%offrepdata.the_offset[1])
    draw_offset_overlay(offrepdata, offrepdispdata)
    
def command_man_from_cassini(offrepdata, offrepdispdata):
    offrepdata.manual_offset = (0.,0.)
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0, '%6.2f'%offrepdata.manual_offset[0])
        offrepdispdata.entry_y_offset.insert(0, '%6.2f'%offrepdata.manual_offset[1])
    draw_offset_overlay(offrepdata, offrepdispdata)

# <Enter> key pressed in a manual offset text entry box
def command_enter_offset(event, offrepdata, offrepdispdata):
    if offrepdispdata.entry_x_offset.get() == "" or offrepdispdata.entry_y_offset.get() == "":
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
    draw_offset_overlay(offrepdata, offrepdispdata)

# "Recalculate offset" button pressed
def command_recalc_offset(offrepdata, offrepdispdata):
    offset_one_image(offrepdata, False, False, True, save_results=False)
    offrepdata.manual_offset = None
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None' 
        auto_y_text = 'Auto Y Offset: None' 
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offrepdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offrepdata.the_offset[1]
        
    offrepdispdata.auto_x_label.config(text=auto_x_text)
    offrepdispdata.auto_y_label.config(text=auto_y_text)
    draw_offset_overlay(offrepdata, offrepdispdata)
    refresh_repro_img(offrepdata, offrepdispdata)

# "Refresh reprojection" button pressed
def refresh_repro_img(offrepdata, offrepdispdata):
    assert False # XXX
    offrepdata.repro_img = fitreproj.Reproject(offrepdata.vicar_data, offrepdata.ringim,
                                             options.radius_start, options.radius_end,
                                             options.radius_resolution, options.longitude_resolution)
    offrepdata.repro_img = offrepdata.repro_img[::-1,:] # Flip it upside down for display - Saturn at bottom
    offrepdispdata.repro_longitudes = np.array([float(x) for x in offrepdata.vicar_data['LONGITUDES_SAVED']])
    offrepdispdata.repro_phase_angle = float(offrepdata.vicar_data['PHASE_ANGLE'])
    offrepdispdata.repro_incidence_angle = float(offrepdata.vicar_data['INCIDENCE_ANGLE'])
    offrepdispdata.repro_emission_angle = float(offrepdata.vicar_data['EMISSION_ANGLE'])
    offrepdispdata.repro_resolutions = np.array([float(x) for x in offrepdata.vicar_data['RADIAL_RESOLUTION']])

    # Have to recompute the overlay because the repro_img might have changed size
    repro_overlay = np.zeros((offrepdata.repro_img.shape[0], offrepdata.repro_img.shape[1], 3))
    fring_140220_y = offrepdata.repro_img.shape[0]-1-int((140220-options.radius_start)/(options.radius_end-options.radius_start)*
                                                       offrepdata.repro_img.shape[0])
    fring_util.draw_line(repro_overlay, (1,0,0), 0, fring_140220_y,
                       offrepdata.repro_img.shape[1]-1, fring_140220_y)

    offrepdispdata.imdisp_repro.update_image_data([offrepdata.repro_img], [repro_overlay])

# "Commit changes" button pressed
def command_commit_changes(offrepdata, offrepdispdata):
    assert False # XXX
    if offrepdispdata.entry_x_offset.get() == "" or offrepdispdata.entry_y_offset.get() == "":
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
    fring_util.write_offset(offrepdata.offset_path, offrepdata.the_offset, offrepdata.manual_offset,
                          offrepdata.off_metadata)
    offrepdata.vicar_data.ToFile(offrepdata.repro_path)

# Setup the offset/reproject window with no data
def setup_offset_reproject_window(offrepdata, offrepdispdata):
    offrepdispdata.toplevel = Tk()
    offrepdispdata.toplevel.title(offrepdata.obsid + ' / ' + offrepdata.image_name)
    frame_toplevel = Frame(offrepdispdata.toplevel)
    
    # The original image and overlaid ring curves
    offrepdispdata.imdisp_offset = ImageDisp([offrepdata.obs.data], parent=frame_toplevel, canvas_size=(512,512),
                                             allow_enlarge=True, auto_update=True)

    # The reprojected image
    if offrepdata.repro_img is None:
        offrepdispdata.imdisp_repro = ImageDisp([np.zeros((1024,1024))], parent=frame_toplevel,
                                                canvas_size=(512,512), overlay_list=[np.zeros((1024,1024,3))],
                                                allow_enlarge=True, auto_update=True)
    else:
        offrepdispdata.imdisp_repro = ImageDisp([offrepdata.repro_img], parent=frame_toplevel,
                                                canvas_size=(512,512), overlay_list=[offrepdispdata.repro_overlay],
                                                allow_enlarge=True, auto_update=True)
    
    ###############################################
    # The control/data pane of the original image #
    ###############################################
    
    img_addon_control_frame = offrepdispdata.imdisp_offset.addon_control_frame
    
    gridrow = 0
    gridcolumn = 0

    if offrepdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None' 
        auto_y_text = 'Auto Y Offset: None' 
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offrepdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offrepdata.the_offset[1]
        
    offrepdispdata.auto_x_label = Label(img_addon_control_frame, text=auto_x_text)
    offrepdispdata.auto_x_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    offrepdispdata.auto_y_label = Label(img_addon_control_frame, text=auto_y_text)
    offrepdispdata.auto_y_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    # X offset and Y offset entry boxes
    # We should really use variables for the Entry boxes, but for some reason they don't work
    label = Label(img_addon_control_frame, text='X Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    
    offrepdispdata.entry_x_offset = FloatEntry(img_addon_control_frame)
    offrepdispdata.entry_x_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0, '%6.2f'%offrepdata.manual_offset[0])
    offrepdispdata.entry_x_offset.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(img_addon_control_frame, text='Y Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    
    offrepdispdata.entry_y_offset = FloatEntry(img_addon_control_frame)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_y_offset.insert(0, '%6.2f'%offrepdata.manual_offset[1])
    offrepdispdata.entry_y_offset.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    enter_offset_command = (lambda x, offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                            command_enter_offset(x, offrepdata, offrepdispdata))    
    offrepdispdata.entry_x_offset.bind('<Return>', enter_offset_command)
    offrepdispdata.entry_y_offset.bind('<Return>', enter_offset_command)

    # Set manual to automatic
    button_man_from_auto_command = (lambda offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                                    command_man_from_auto(offrepdata, offrepdispdata))
    button_man_from_auto = Button(img_addon_control_frame, text='Set Manual from Auto',
                                  command=button_man_from_auto_command)
    button_man_from_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    #Set manual to Cassini
    button_man_from_cassini_command = (lambda offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                                    command_man_from_cassini(offrepdata, offrepdispdata))
    button_man_cassini_auto = Button(img_addon_control_frame, text='Set Manual from Cassini',
                                  command=button_man_from_cassini_command)
    button_man_cassini_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Recalculate auto offset
    button_recalc_offset_command = (lambda offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                                    command_recalc_offset(offrepdata, offrepdispdata))
    button_recalc_offset = Button(img_addon_control_frame, text='Recalculate Offset',
                                  command=button_recalc_offset_command)
    button_recalc_offset.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Refresh reprojection buttons
    button_refresh_command = (lambda offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                              refresh_repro_img(offrepdata, offrepdispdata))
    button_refresh = Button(img_addon_control_frame, text='Refresh Reprojection',
                            command=button_refresh_command)
    button_refresh.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Commit results button - saves new offset and reprojection
    button_commit_changes_command = (lambda offrepdata=offrepdata, offrepdispdata=offrepdispdata:
                                     command_commit_changes(offrepdata, offrepdispdata))
    button_commit_changes = Button(img_addon_control_frame, text='Commit Changes',
                                   command=button_commit_changes_command)
    button_commit_changes.grid(row=gridrow, column=gridcolumn+1)

    ##################################################
    # The control/data pane of the reprojected image #
    ##################################################

    repro_addon_control_frame = offrepdispdata.imdisp_repro.addon_control_frame

    gridrow = 0
    gridcolumn = 0

    label = Label(repro_addon_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_date = Label(repro_addon_control_frame, text=cspice.et2utc(offrepdata.obs.midtime, 'C', 0))
    offrepdispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(repro_addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_inertial_longitude = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_inertial_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_corot_longitude = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_corot_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_radius = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_phase = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(repro_addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_incidence = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(repro_addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_emission = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_resolution = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    offrepdispdata.imdisp_repro.pack(side=LEFT)

    callback_repro_command = lambda x, y, offrepdata=offrepdata, offrepdispdata=offrepdispdata: callback_repro(x, y, offrepdata, offrepdispdata)
    offrepdispdata.imdisp_repro.bind_mousemove(0, callback_repro_command)
    
    frame_toplevel.pack()

    draw_offset_overlay(offrepdata, offrepdispdata)
    draw_repro_overlay(offrepdata, offrepdispdata)
    
# Display the original and reproject images (if any)
def display_offset_reproject(offrepdata, offrepdispdata, option_invalid_offset,
                             option_invalid_reproject):
    if options.verbose:
        print '** Display', offrepdata.obsid, '/', offrepdata.image_name
    if offrepdata.the_offset is None:
        (offrepdata.the_offset, offrepdata.manual_offset,
         offrepdata.off_metadata) = fring_util.read_offset(offrepdata.offset_path)

    if option_invalid_offset and (offrepdata.the_offset is not None or offrepdata.manual_offset is not None):
        if options.verbose:
            print 'Skipping because not invalid'
        return

    # The original image and overlaid ring curves
    
    if offrepdata.obs is None:
        offrepdata.obs = iss.from_file(offrepdata.image_path)
    try:
        offrepdata.fring_overlay = None #XXX
    except RuntimeError as exc:
        if options.verbose:
            print "CAN'T DISPLAY - PROBABLY SPICE ERROR"
            print exc
        return
        
#    if offrepdata.manual_offset is None: XXX
#        offrepdata.ringim.SetOffset(offrepdata.the_offset)
#    else:
#        offrepdata.ringim.SetOffset(offrepdata.manual_offset)
    img_max_y = offrepdata.obs.data.shape[1]-1

    # The reprojected image XXX
#    try:
#        repro_vicar_data = vicar.VicarImage.FromFile(offrepdata.repro_path)
#        repro_ringim = fitreproj.VicarToRingImage(repro_vicar_data)
#        repro_ringim.SetOffset((0,0))
#        offrepdata.repro_img = repro_vicar_data.Get2dArray()[::-1,:] # Flip it upside down for display - Saturn at bottom
#     
#        offrepdispdata.repro_overlay = np.zeros((offrepdata.repro_img.shape[0], offrepdata.repro_img.shape[1], 3))
#        fring_140220_y = offrepdata.repro_img.shape[0]-1-int((140220-options.radius_start)/
#                                                           (options.radius_end-options.radius_start)*
#                                                           offrepdata.repro_img.shape[0])
#        fring_util.draw_line(offrepdispdata.repro_overlay, (1,0,0), 0, fring_140220_y,
#                           offrepdata.repro_img.shape[1]-1, fring_140220_y)
#        offrepdispdata.repro_longitudes = np.array([float(x) for x in repro_vicar_data['LONGITUDES_SAVED']])
#        offrepdispdata.repro_phase_angle = float(repro_vicar_data['PHASE_ANGLE'])
#        offrepdispdata.repro_incidence_angle = float(repro_vicar_data['INCIDENCE_ANGLE'])
#        offrepdispdata.repro_emission_angle = float(repro_vicar_data['EMISSION_ANGLE'])
#        offrepdispdata.repro_resolutions = np.array([float(x) for x in repro_vicar_data['RADIAL_RESOLUTION']])
#    except IOError: # File not found
#        pass
    
    setup_offset_reproject_window(offrepdata, offrepdispdata)
    
    mainloop()

# The callback for mouse move events on the reprojected image
def callback_repro(x, y, offrepdata, offrepdispdata):
    if offrepdispdata.repro_longitudes is None:
        return
    if x < 0 or x >= len(offrepdispdata.repro_longitudes): return
    offrepdispdata.label_inertial_longitude.config(text=('%7.3f'%offrepdispdata.repro_longitudes[x]))
    offrepdispdata.label_corot_longitude.config(text=('%7.3f'%fring_util.InertialToCorotating(offrepdispdata.repro_longitudes[x],
                                                                                        offrepdata.ringim.et)))
    radius = mosaic.IndexToRadius(offrepdata.repro_img.shape[0]-1-y, options.radius_resolution)
    offrepdispdata.label_radius.config(text = '%7.3f'%radius)
    offrepdispdata.label_phase.config(text=('%7.3f'%offrepdispdata.repro_phase_angle))
    offrepdispdata.label_incidence.config(text=('%7.3f'%offrepdispdata.repro_incidence_angle))
    offrepdispdata.label_emission.config(text=('%7.3f'%offrepdispdata.repro_emission_angle))
    offrepdispdata.label_resolution.config(text=('%7.3f'%offrepdispdata.repro_resolutions[x]))


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################


offrepdispdata = OffRepDispData()

cur_obsid = None
obsid_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for obsid, image_name, image_path in fring_util.enumerate_files(options, args, '_CALIB.IMG'):
    offrepdata = fring_util.OffRepData()
    offrepdata.obsid = obsid
    offrepdata.image_name = image_name
    offrepdata.image_path = image_path
    
    offrepdata.offset_path = fring_util.offset_path(options, image_path, image_name)
    offrepdata.repro_path = fring_util.repro_path(options, image_path, image_name)

    # Pointing offset
    offset_one_image(offrepdata, options.no_auto_offset, options.no_update_auto_offset,
                     options.recompute_auto_offset)
    
    # Reprojection
    reproject_one_image(offrepdata, options.no_reproject, options.no_update_reproject,
                        options.recompute_reproject)

    # Display offset and reprojection
    if options.display_offset_reproject or options.display_invalid_offset or options.display_invalid_reproject:
        display_offset_reproject(offrepdata, offrepdispdata, options.display_invalid_offset,
                                 options.display_invalid_reproject)
    
    del offrepdata
    offrepdata = None
    
