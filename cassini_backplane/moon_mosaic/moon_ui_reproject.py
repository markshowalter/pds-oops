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
import numpy.ma as ma
import subprocess
import time
import cspice
from imgdisp import ImageDisp, FloatEntry, draw_line
from Tkinter import *
from PIL import Image
import oops.inst.cassini.iss as iss
import moon_util
from cb_offset import *
import cProfile, pstats, StringIO

#oops.LOGGING.all(True)

python_dir = os.path.split(sys.argv[0])[0]
python_reproject_program = os.path.join(python_dir, moon_util.PYTHON_MOON_REPROJECT)

python_filename = sys.argv[0]

cmd_line = sys.argv[1:]


if len(cmd_line) == 0:
    cmd_line = [
#                '-a',
#                '--max-subprocesses', '3',

'MIMAS',
                '--allow-exception',
#                '--no-auto-offset',
#                 '--recompute-auto-offset',
                 '--recompute-reproject',
#                 '--no-reproject',
                '--display-offset-reproject',
#                '--profile',
#                '--no-update-auto-offset',
#                '--no-update-reproject',
                 '--verbose']

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
                  action='store_true', default=True, # XXX
                  help="Allow exceptions to be thrown")
parser.add_option('--profile', dest='profile',
                  action='store_true', default=False,
                  help="Do performance profiling")
parser.add_option('--start-body_name', dest='start_body_name',
                  default='', help='The first body_name to process')
parser.add_option('--max-subprocesses', dest='max_subprocesses',
                  type='int', default=0,
                  help="Fork a subprocess for each file")


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
parser.add_option('--no-allow-stars', dest='no_allow_stars',
                   action='store_true', default=False,
                   help="Don't allow stars during auto offset")
parser.add_option('--no-allow-moons', dest='no_allow_moons',
                   action='store_true', default=False,
                   help="Don't allow moons during auto offset")

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

moon_util.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

assert not (options.display_offset_reproject and options.max_subprocesses)

class OffRepDispData:
    def __init__(self):
        self.obs = None
        self.toplevel = None
        self.imdisp_offset = None
        self.entry_x_offset = None
        self.entry_y_offset = None
        self.off_longitudes = None
        self.off_latitudes = None
        self.label_off_longitude = None
        self.label_off_latitude = None
        self.imdisp_repro = None
        self.repro_overlay = None
        self.label_longitude = None
        self.label_latitude = None
        self.repro_longitudes = None
        self.repro_phase_angles = None
        self.repro_incidence_angles = None
        self.repro_emission_angles = None
        self.repro_resolutions = None
        

#####################################################################################
#
# RUN IN A SUBPROCESS
#
#####################################################################################

def collect_cmd_line():
    ret += ['--latitude_resolution', str(options.latitude_resolution)]
    ret += ['--longitude_resolution', str(options.longitude_resolution)]
    if options.verbose:
        ret += ['--verbose']
    if options.no_auto_offset:
        ret += ['--no-auto-offset']
    if options.no_update_auto_offset:
        ret += ['--no-update-auto-offset']
    if options.recompute_auto_offset:
        ret += ['--recompute-auto-offset']
    if options.no_reproject:
        ret += ['--no-reproject']
    if options.no_update_reproject:
        ret += ['--no-update-reproject']
    if options.recompute_reproject:
        ret += ['--recompute-reproject']
        
    return ret

def run_and_maybe_wait(args):
    said_waiting = False
    while len(subprocess_list) == options.max_subprocesses:
        if options.verbose and not said_waiting:
            print 'WAITING'
            said_waiting = True
        for i in xrange(len(subprocess_list)):
            if subprocess_list[i].poll() is not None:
                del subprocess_list[i]
                break
        if len(subprocess_list) == options.max_subprocesses:
            time.sleep(1)

    if options.verbose:
        print 'SPAWNING SUBPROCESS'
        
    pid = subprocess.Popen(args)
    subprocess_list.append(pid)
                    

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
        print '** Find offset', offrepdata.body_name, '/', offrepdata.image_name, '-',
    
    if option_no:  # Just don't do anything - we hope you know what you're doing!
        if options.verbose:
            print 'Ignored because of --no-auto_offset'
        return
        
    if os.path.exists(offrepdata.offset_path+'.pickle'):
        if option_no_update:
            if options.verbose:
                print 'Ignored because offset file already exists'
            return # Offset file already exists, don't update
        # Save the manual offset!
        trash1, offrepdata.manual_offset, trash2 = moon_util.read_offset(offrepdata.offset_path)
        time_offset = os.stat(offrepdata.offset_path+'.pickle').st_mtime
    else:
        time_offset = 0
        
    time_image = os.stat(offrepdata.image_path).st_mtime
    if time_offset >= time_image and not option_recompute:
        # The offset file exists and is more recent than the image, and we're not forcing a recompute
        if options.verbose:
            print 'Ignored because offset file is up to date'
        return

    if options.max_subprocesses:
        if options.verbose:
            print 'QUEUEING SUBPROCESS'
        offrepdata.subprocess_run = True
        return
    
    # Recompute the automatic offset
    obs = iss.from_file(offrepdata.image_path)
    offrepdata.obs = obs
    try:
        offset_u, offset_v, offrepdata.off_metadata = master_find_offset(obs,
                                                 allow_stars=not options.no_allow_stars,
                                                 allow_moons=not options.no_allow_moons,
                                                 create_overlay=True,
                                                 star_overlay_box_width=5,
                                                 star_overlay_box_thickness=2)                                                                             
    except:
        if options.verbose:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
        print 'EXCEPTION:'
        print sys.exc_info()
        if options.allow_exception:
            raise
        offset_u = None
        offset_v = None
        offrepdata.off_metadata = {}
    if offset_u is None:
        offrepdata.the_offset = None
    else:
        offrepdata.the_offset = (offset_u, offset_v)
    if offset_u is None:
        if options.verbose:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    if offset_u is not None and options.verbose:
        print 'FOUND %6.2f, %6.2f' % (offrepdata.the_offset[0], offrepdata.the_offset[1])
    if save_results:
        moon_util.write_offset(offrepdata.offset_path, offrepdata.the_offset, offrepdata.manual_offset,
                              offrepdata.off_metadata)


#####################################################################################
#
# REPROJECT ONE IMAGE
#
#####################################################################################

def _update_offrepdata_repro(offrepdata, metadata):
    if metadata is None:
        offrepdata.repro_good_mask = None
        offrepdata.repro_good_lat_mask = None
        offrepdata.repro_good_long_mask = None
        offrepdata.repro_img = None
        offrepdata.repro_longitudes = None
        offrepdata.repro_resolutions = None
        offrepdata.repro_phase_angles = None
        offrepdata.repro_emission_angles = None
        offrepdata.repro_incidence_angles = None
        offrepdata.repro_time = None
    else:
        offrepdata.repro_good_mask = metadata['good_mask']
        offrepdata.repro_good_lat_mask = metadata['good_lat_mask']
        offrepdata.repro_good_long_mask = metadata['good_long_mask']
        offrepdata.repro_img = metadata['img']
        offrepdata.repro_resolutions = metadata['resolution']
        offrepdata.repro_phase_angles = metadata['phase']
        offrepdata.repro_emission_angles = metadata['emission']
        offrepdata.repro_incidence_angles = metadata['incidence']
        offrepdata.repro_time = metadata['time']
        
        full_latitudes = moons_generate_latitudes(latitude_resolution=options.latitude_resolution)
        offrepdata.repro_latitudes = full_latitudes[offrepdata.repro_good_lat_mask]
        full_longitudes = moons_generate_longitudes(longitude_resolution=options.longitude_resolution)
        offrepdata.repro_longitudes = full_longitudes[offrepdata.repro_good_long_mask]

def _write_repro_data(offrepdata):
    metadata = None
    if offrepdata.repro_img is not None:
        metadata = {}
        metadata['img'] = offrepdata.repro_img
        metadata['good_lat_mask'] = offrepdata.repro_good_lat_mask
        metadata['good_long_mask'] = offrepdata.repro_good_long_mask
        metadata['good_mask'] = offrepdata.repro_good_mask
        metadata['resolution'] = offrepdata.repro_resolutions
        metadata['phase'] = offrepdata.repro_phase_angles
        metadata['emission'] = offrepdata.repro_emission_angles
        metadata['incidence'] = offrepdata.repro_incidence_angles
        metadata['time'] = offrepdata.repro_time
    
    moon_util.write_repro(offrepdata.repro_path, metadata)
    
def _reproject_one_image(offrepdata):
    if offrepdata.obs is None:
        offrepdata.obs = iss.from_file(offrepdata.image_path)

    obs = offrepdata.obs

    offset_u = None
    offset_v = None
    
    if offrepdata.manual_offset is not None:
        offset_u, offset_v = offrepdata.manual_offset
    elif offrepdata.the_offset is not None:
        offset_u, offset_v = offrepdata.the_offset
    else:
        print 'NO OFFSET - REPROJECTION FAILED'
        _update_offrepdata_repro(offrepdata, None)
        return
    
    try:
        ret = moons_reproject(obs, offrepdata.body_name,
                              offset_u, offset_v,
                              options.latitude_resolution,
                              options.longitude_resolution)
    except:
        if options.verbose:
            print 'REPROJECTION FAILED'
        print 'EXCEPTION:'
        print sys.exc_info()
        if options.allow_exception:
            raise
        ret = None

    _update_offrepdata_repro(offrepdata, ret)

def reproject_one_image(offrepdata, option_no, option_no_update, option_recompute):
    # Input file: offset_path (<IMAGE>_CALIB.IMG.OFFSET)
    # Output file: repro_path (<IMAGE>_<RES_DATA>_REPRO.IMG)

    if options.verbose:
        print '** Reproject', offrepdata.body_name, '/', offrepdata.image_name, '-',
    
    if offrepdata.subprocess_run:
        if options.verbose:
            print 'LETTING SUBPROCESS HANDLE IT'
        return
    
    if option_no:  # Just don't do anything
        if options.verbose:
            print 'Ignored because of --no-reproject'
        return

    if os.path.exists(offrepdata.repro_path+'.pickle'):
        if option_no_update:
            if options.verbose:
                print 'Ignored because repro file already exists'
            return # Repro file already exists, don't update
        time_repro = os.stat(offrepdata.repro_path+'.pickle').st_mtime
    else:
        time_repro = 0
    
    if not os.path.exists(offrepdata.offset_path+'.pickle'):
        if options.verbose:
            print 'NO OFFSET FILE - ABORTING'
        return
    
    time_offset = os.stat(offrepdata.offset_path+'.pickle').st_mtime
    if time_repro >= time_offset and not option_recompute:
        # The repro file exists and is more recent than the image, and we're not forcing a recompute
        if options.verbose:
            print 'Ignored because repro file is up to date'
        return

    (offrepdata.the_offset, offrepdata.manual_offset,
     offrepdata.off_metadata) = moon_util.read_offset(offrepdata.offset_path)
    
    if offrepdata.the_offset is None and offrepdata.manual_offset is None:
        if options.verbose:
            print 'OFFSET IS INVALID - ABORTING'
        return

    if options.max_subprocesses:
        if options.verbose:
            print 'QUEUEING SUBPROCESS'
        offrepdata.subprocess_run = True
        return
    
    _reproject_one_image(offrepdata)
    
    _write_repro_data(offrepdata)
    
    if options.verbose:
        print 'OK'


#####################################################################################
#
# DISPLAY ONE IMAGE AND ITS REPROJECTION ALLOWING MANUAL CHANGING OF THE OFFSET
#
#####################################################################################

def draw_repro_overlay(offrepdata, offrepdispdata):
    if offrepdata.repro_img is None:
        return
    repro_overlay = np.zeros(offrepdata.repro_img.shape + (3,))
#    y = int(float(options.radius_outer)/(options.radius_outer-options.radius_inner)*
#            offrepdata.repro_img.shape[0])
#    if 0 <= y < repro_overlay.shape[0]:
#        repro_overlay[y, :, 0] = 1
    
    offrepdispdata.imdisp_repro.set_overlay(0, repro_overlay)

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset."""
    if offset_u == 0 and offset_v == 0:
        return image
    
    image = np.roll(image, -offset_u, 1)
    image = np.roll(image, -offset_v, 0)

    if offset_u != 0:    
        if offset_u < 0:
            image[:,:-offset_u] = 0
        else:
            image[:,-offset_u:] = 0
    if offset_v != 0:
        if offset_v < 0:
            image[:-offset_v,:] = 0
        else:
            image[-offset_v:,:] = 0
    
    return image

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
    
#XXX    x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs) # No offset - blue
#    x_pixels = x_pixels.astype('int')
#    y_pixels = y_pixels.astype('int')
#    offset_overlay[y_pixels, x_pixels, 2] = 255
#
#    if offrepdata.the_offset is not None:
#        # Auto offset - red
#        x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs, 
#                                        offset_u=offrepdata.the_offset[0],
#                                        offset_v=offrepdata.the_offset[1])
#        x_pixels = x_pixels.astype('int')
#        y_pixels = y_pixels.astype('int')
#        offset_overlay[y_pixels, x_pixels, 0] = 255
#
#    if offrepdata.manual_offset is not None:
#        # Auto offset - green
#        x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs, 
#                                        offset_u=offrepdata.manual_offset[0],
#                                        offset_v=offrepdata.manual_offset[1])
#        x_pixels = x_pixels.astype('int')
#        y_pixels = y_pixels.astype('int')
#        offset_overlay[y_pixels, x_pixels, 1] = 255

    offrepdispdata.imdisp_offset.set_overlay(0, offset_overlay)
    offrepdispdata.imdisp_offset.pack(side=LEFT)

# The callback for mouse move events on the offset image
def callback_offset(x, y, offrepdata, offrepdispdata):
    if offrepdata.manual_offset is not None:
        x -= offrepdata.manual_offset[0]
        y -= offrepdata.manual_offset[1]
    elif offrepdata.the_offset is not None:
        x -= offrepdata.the_offset[0]
        y -= offrepdata.the_offset[1]
    if (x < 0 or x > offrepdata.obs.data.shape[1]-1 or
        y < 0 or y > offrepdata.obs.data.shape[0]-1):
        return
    
    if offrepdispdata.off_longitudes is not None:
        if offrepdispdata.off_longitudes[y,x] == ma.masked:
            offrepdispdata.label_off_longitude.config(text=('N/A'))
        else:
            offrepdispdata.label_off_longitude.config(text=('%7.3f'%offrepdispdata.off_longitudes.vals[y,x]))
    if offrepdispdata.off_latitudes is not None:
        if offrepdispdata.off_latitudes[y,x] == ma.masked:
            offrepdispdata.label_off_latitude.config(text=('N/A'))
        else:
            offrepdispdata.label_off_latitude.config(text=('%7.3f'%offrepdispdata.off_latitudes.vals[y,x]))


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
#    refresh_repro_img(offrepdata, offrepdispdata)

# "Refresh reprojection" button pressed
def refresh_repro_img(offrepdata, offrepdispdata):
    _reproject_one_image(offrepdata)

    offrepdispdata.repro_latitudes = offrepdata.repro_latitudes
    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_resolutions = offrepdata.repro_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angles = offrepdata.repro_incidence_angles

#    temp_img = None
#    if offrepdata.repro_img is not None:
#        temp_img = offrepdata.repro_img[::1,:] # Flip it upside down for display - Saturn at bottom XXX
    if offrepdata.repro_img is None:
        offrepdispdata.imdisp_repro.update_image_data([np.zeros((1024,1024))], [None])
    else:
        offrepdispdata.imdisp_repro.update_image_data([offrepdata.repro_img], [None])
    draw_repro_overlay(offrepdata, offrepdispdata)

# "Commit changes" button pressed
def command_commit_changes(offrepdata, offrepdispdata):
    if offrepdispdata.entry_x_offset.get() == "" or offrepdispdata.entry_y_offset.get() == "":
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
    moon_util.write_offset(offrepdata.offset_path, offrepdata.the_offset, offrepdata.manual_offset,
                          offrepdata.off_metadata)
    _write_repro_data(offrepdata)

# Setup the offset/reproject window with no data
def setup_offset_reproject_window(offrepdata, offrepdispdata):
    set_obs_bp(offrepdata.obs)
    
    offrepdispdata.off_latitudes = offrepdata.obs.bp.latitude(offrepdata.body_name, lat_type='centric') * oops.DPR
    offrepdispdata.off_longitudes = offrepdata.obs.bp.longitude(offrepdata.body_name, direction='east') * oops.DPR
    
    offrepdispdata.toplevel = Tk()
    offrepdispdata.toplevel.title(offrepdata.body_name + ' / ' + offrepdata.image_name)
    frame_toplevel = Frame(offrepdispdata.toplevel)
    
    # The original image 
    offrepdispdata.imdisp_offset = ImageDisp([offrepdata.obs.data], parent=frame_toplevel, canvas_size=(512,512),
                                             allow_enlarge=True, auto_update=True)

    # The reprojected image
    if offrepdata.repro_img is None:
        offrepdispdata.imdisp_repro = ImageDisp([np.zeros((1024,1024))], parent=frame_toplevel,
                                                canvas_size=(512,512), overlay_list=[np.zeros((1024,1024,3))],
                                                allow_enlarge=True, auto_update=True,
                                                flip_y=True)
    else:
        offrepdispdata.imdisp_repro = ImageDisp([offrepdata.repro_img], parent=frame_toplevel,
                                                canvas_size=(512,512), overlay_list=[offrepdispdata.repro_overlay],
                                                allow_enlarge=True, auto_update=True,
                                                flip_y=True)
    
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
    gridrow += 1
    
    # Display for longitude and latitude
    label = Label(img_addon_control_frame, text='Latitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_latitude = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_latitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Longitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_longitude = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    callback_offset_command = lambda x, y, offrepdata=offrepdata, offrepdispdata=offrepdispdata: callback_offset(x, y, offrepdata, offrepdispdata)
    offrepdispdata.imdisp_offset.bind_mousemove(0, callback_offset_command)


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
    
    label = Label(repro_addon_control_frame, text='Latitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_latitude = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_latitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Longitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_longitude = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
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
                             option_invalid_reproject, do_mainloop=True):
    if options.verbose:
        print '** Display', offrepdata.body_name, '/', offrepdata.image_name
    if offrepdata.off_metadata is None:
        (offrepdata.the_offset, offrepdata.manual_offset,
         offrepdata.off_metadata) = moon_util.read_offset(offrepdata.offset_path)

    if option_invalid_offset and (offrepdata.the_offset is not None or offrepdata.manual_offset is not None):
        if options.verbose:
            print 'Skipping because not invalid'
        return

    # The original image
    
    if offrepdata.obs is None:
        offrepdata.obs = iss.from_file(offrepdata.image_path)

    img_max_y = offrepdata.obs.data.shape[1]-1

    if offrepdata.repro_img is None:
        ret = moon_util.read_repro(offrepdata.repro_path)
        _update_offrepdata_repro(offrepdata, ret)
 
    if offrepdata.repro_img is not None:
        offrepdispdata.repro_overlay = np.zeros(offrepdata.repro_img.shape + (3,))
    else:
        offrepdispdata.repro_overlay = None
        
    offrepdispdata.repro_latitudes = offrepdata.repro_latitudes
    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_resolutions = offrepdata.repro_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angles = offrepdata.repro_incidence_angles
    
    setup_offset_reproject_window(offrepdata, offrepdispdata)
    draw_repro_overlay(offrepdata, offrepdispdata)

    if do_mainloop:
        mainloop()

# The callback for mouse move events on the reprojected image
def callback_repro(x, y, offrepdata, offrepdispdata):
    if offrepdispdata.repro_longitudes is None:
        return

    offrepdispdata.label_latitude.config(text=('%7.3f'%offrepdispdata.repro_latitudes[y]))
    offrepdispdata.label_longitude.config(text=('%7.3f'%offrepdispdata.repro_longitudes[x]))
    offrepdispdata.label_resolution.config(text=('%7.3f'%offrepdispdata.repro_resolutions[y,x]))
    offrepdispdata.label_phase.config(text=('%7.3f'%offrepdispdata.repro_phase_angles[y,x]))
    offrepdispdata.label_emission.config(text=('%7.3f'%offrepdispdata.repro_emission_angles[y,x]))
    offrepdispdata.label_incidence.config(text=('%7.3f'%offrepdispdata.repro_incidence_angles[y,x]))


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

subprocess_list = []

offrepdispdata = OffRepDispData()

found_body_name = False
cur_body_name = None
body_name_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for body_name, image_name, image_path in moon_util.enumerate_files(options, args, '_CALIB.IMG'):
#    if body_name == 'ISS_006RI_LPHRLFMOV001_PRIME':
#        continue
    if options.start_body_name != '' and not found_body_name:
        if body_name != options.start_body_name:
            continue
        found_body_name = True
            
    offrepdata = moon_util.OffRepData()
    offrepdata.body_name = body_name
    offrepdata.image_name = image_name
    offrepdata.image_path = image_path
    
    offrepdata.offset_path = moon_util.offset_path(options, image_path, image_name)
    offrepdata.repro_path = moon_util.repro_path(options, image_path, image_name)

    offrepdata.subprocess_run = False

    if options.profile:
        pr = cProfile.Profile()
        pr.enable()

    # Pointing offset
    offset_one_image(offrepdata, options.no_auto_offset, options.no_update_auto_offset,
                     options.recompute_auto_offset)
    
    # Reprojection
    reproject_one_image(offrepdata, options.no_reproject, options.no_update_reproject,
                        options.recompute_reproject)

    if options.max_subprocesses and offrepdata.subprocess_run:
        run_and_maybe_wait([moon_util.PYTHON_EXE, python_reproject_program] + 
                           collect_cmd_line() + 
                           [offrepdata.body_name+'/'+offrepdata.image_name])
    
    # Display offset and reprojection
    if options.display_offset_reproject or options.display_invalid_offset or options.display_invalid_reproject:
        display_offset_reproject(offrepdata, offrepdispdata, options.display_invalid_offset,
                                 options.display_invalid_reproject, do_mainloop=not options.profile)
    
    del offrepdata
    offrepdata = None
    
    if options.profile:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        print s.getvalue()
        assert False
        
