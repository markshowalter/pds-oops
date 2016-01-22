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
import subprocess
import time
import cspice
from imgdisp import ImageDisp, FloatEntry, draw_line
from Tkinter import *
from PIL import Image
import fring_util
from cb_logging import *
from cb_util_file import *
from cb_util_image import *
from cb_offset import *
from cb_rings import *
import cProfile, pstats, StringIO
import traceback

#oops.LOGGING.all(True)

python_dir = os.path.split(sys.argv[0])[0]
python_reproject_program = os.path.join(python_dir, fring_util.PYTHON_RING_REPROJECT)

python_filename = sys.argv[0]

cmd_line = sys.argv[1:]


if len(cmd_line) == 0:
    cmd_line = [
                '--recompute-auto-offset',
#                 '--image-logfile-level', 'debug',
                '--image-log-console-level', 'warning',
#                 '-a',
#                 '--max-subprocesses', '3',
#                 '--start-obsid', 'ISS_007RI_AZSCNLOPH001_PRIME',
                '--start-obsid', 'ISS_115RF_FMOVIEEQX001_PRIME',
#                 'ISS_000RI_SATSRCHAP001_PRIME',
#                 'ISS_007RI_LPHRLFMOV001_PRIME/N1493646036_2',
#                 'ISS_030RF_FMOVIE001_VIMS',
#                '--start-obsid', 'ISS_036RF_FMOVIE001_VIMS',
#                '--start-obsid', 'ISS_085RF_FMOVIE003_PRIME_1',
#                '--start-obsid', 'ISS_106RF_FMOVIE002_PRIME',
#                '--start-obsid', 'ISS_041RF_FMOVIE001_VIMS',
#                'ISS_041RF_FMOVIE002_VIMS/N1552810957_1', # Fails - stars in F Ring
#                'ISS_036RF_FMOVIE001_VIMS/N1545563750_1', # Fails - offset too large
#                'ISS_036RF_FMOVIE001_VIMS',
#                'ISS_041RF_FMOVIE002_VIMS',
#                'ISS_106RF_FMOVIE002_PRIME',
#                'ISS_132RI_FMOVIE001_VIMS',

#'ISS_080RF_FMOVIE005_PRIME',
#'ISS_098RI_TMAPN30LP001_CIRS/W1608705204_1',
#'ISS_072RI_SPKHRLPDF001_PRIME',
#'ISS_105RI_TDIFS20HP001_CIRS',
#'ISS_105RI_TMAPN45LP001_CIRS',
#'ISS_006RI_LPHRLFMOV001_PRIME',
#'ISS_081RI_FMOVIE106_VIMS',
#'ISS_091RI_APOMOSL109_VIMS',
#'ISS_096RF_FMOVIE004_PRIME',
#'ISS_109RI_TDIFS20HP001_CIRS',
#'ISS_112RF_FMOVIE002_PRIME',
#'ISS_132RI_FMOVIE001_VIMS',
#'ISS_115RF_FMOVIEEQX001_PRIME',
#'ISS_080RF_FMOVIE005_PRIME/N1597390953_1',

#'ISS_051RI_LPMRDFMOV001_PRIME',
#'ISS_059RF_FMOVIE002_VIMS',

#'ISS_000RI_SATSRCHAP001_PRIME',
#'ISS_00ARI_SPKMOVPER001_PRIME/N1479252612_1',
#'ISS_00ARI_SPKMOVPER001_PRIME/N1479201492_1',
#'ISS_00ARI_SPKMOVPER001_PRIME/N1479246132_1',
#'ISS_006RI_LPHRLFMOV001_PRIME/N1492102189_1',
#'--no-allow-stars',
#'--no-allow-moons',

#'ISS_029RF_FMOVIE002_VIMS',
#'ISS_081RI_FMOVIE106_VIMS',
#'ISS_075RB_BMOVIE4001_VIMS',
#'ISS_096RF_FMOVIE004_PRIME',
#'ISS_112RF_FMOVIE002_PRIME',
#'ISS_111RF_FMOVIE002_PRIME',
#'ISS_115RF_FMOVIEEQX001_PRIME',

#'ISS_043RF_FMOVIE001_VIMS/N1555595413_1',
#'ISS_043RF_FMOVIE001_VIMS/N1555588813_1',
#'ISS_081RI_FMOVIE106_VIMS/N1597902245_1',
#'ISS_081RI_FMOVIE106_VIMS/N1597904975_1',
#'ISS_112RF_FMOVIE002_PRIME/N1623341484_1',
#'ISS_096RF_FMOVIE004_PRIME/N1607637941_1',
#'ISS_096RF_FMOVIE004_PRIME/N1607657381_1',
#'ISS_094RF_FMOVIE001_PRIME/N1606005062_1',
#'ISS_051RI_LPMRDFMOV001_PRIME/N1571435347_1',

                '--allow-exception',
#                '--no-auto-offset',
#                 '--recompute-auto-offset',
#                 '--recompute-reproject',
#                 '--no-reproject',
#                 '--display-offset-reproject',
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
parser.add_option('--start-obsid', dest='start_obsid',
                  default='', help='The first obsid to process')
parser.add_option('--max-subprocesses', dest='max_subprocesses',
                  type='int', default=0,
                  help="Fork a subprocess for each file")
parser.add_option('--image-logfile-level', dest='image_logfile_level',
                  default='info',
                  help='Logging level for the individual logfiles')
parser.add_option('--image-log-console-level', dest='image_log_console_level',
                  default='info',
                  help='Logging level for the console')

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

fring_util.add_parser_options(parser)

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
        self.off_radii = None
        self.label_off_inertial_longitude = None
        self.label_off_corot_longitude = None
        self.label_off_radius = None
        self.imdisp_repro = None
        self.repro_overlay = None
        self.label_inertial_longitude = None
        self.label_corot_longitude = None
        self.label_radius = None
        self.repro_longitudes = None
        self.repro_phase_angles = None
        self.repro_incidence_angle = None
        self.repro_emission_angles = None
        self.repro_resolutions = None
        

#####################################################################################
#
# RUN IN A SUBPROCESS
#
#####################################################################################

def collect_cmd_line():
    ret = ['--radius_inner', str(options.radius_inner)]
    ret += ['--radius_outer', str(options.radius_outer)]
    ret += ['--radius_resolution', str(options.radius_resolution)]
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
    ret += ['--image-logfile-level', options.image_logfile_level]
    ret += ['--image-log-console-level', options.image_log_console_level]
        
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
# LOGGING
#
#####################################################################################

def setup_image_logging(offrepdata):
    if offrepdata.image_log_filehandler is not None: # Already set up
        return
    
    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        image_log_path = file_img_to_log_path(offrepdata.image_path, bootstrap=False)
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this
            
        offrepdata.image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    else:
        offrepdata.image_log_filehandler = None


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
        metadata = file_read_offset_metadata(offrepdata.image_path)
        if 'manual_offset' in metadata:
            offrepdata.manual_offset = metadata['manual_offset']
        time_offset = os.stat(offrepdata.offset_path).st_mtime
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

    setup_image_logging(offrepdata)
    
    # Recompute the automatic offset
    obs = read_iss_file(offrepdata.image_path)
    offrepdata.obs = obs
    
    rings_config = RINGS_DEFAULT_CONFIG.copy()
    rings_config['fiducial_feature_threshold'] = 1 # XXX
    rings_config['fiducial_feature_margin'] = 30 # XXX
    rings_config['fiducial_ephemeris_width'] = 10 # XXX
    
    try:
        offrepdata.off_metadata = master_find_offset(obs,
                         allow_stars=not options.no_allow_stars,
                         allow_moons=not options.no_allow_moons,
                         create_overlay=True,
                         stars_overlay_box_width=5,
                         stars_overlay_box_thickness=2,
                         rings_config=rings_config)
    except:
        if options.verbose:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
        print 'EXCEPTION:'
        print sys.exc_info()
        err = 'Offset finding failed:\n' + traceback.format_exc() 
        offrepdata.off_metadata = {}
        offrepdata.off_metadata['error'] = str(sys.exc_value)
        offrepdata.off_metadata['error_traceback'] = err
        if options.allow_exception:
            raise
    if ('offset' in offrepdata.off_metadata and 
        offrepdata.off_metadata['offset'] is not None):
        offrepdata.the_offset = offrepdata.off_metadata['offset']
        if options.verbose:
            print 'FOUND %6.2f, %6.2f' % (offrepdata.the_offset[0], offrepdata.the_offset[1])
    else:
        offrepdata.the_offset = None
        if options.verbose:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    if offrepdata.manual_offset:
        offrepdata.off_metadata['manual_offset'] = offrepdata.manual_offset
        
    if save_results:
        file_write_offset_metadata(offrepdata.image_path, offrepdata.off_metadata)


#####################################################################################
#
# REPROJECT ONE IMAGE
#
#####################################################################################

def _update_offrepdata_repro(offrepdata, metadata):
    if metadata is None:
        offrepdata.repro_long_mask = None
        offrepdata.repro_img = None
        offrepdata.repro_longitudes = None
        offrepdata.repro_resolutions = None
        offrepdata.repro_phase_angles = None
        offrepdata.repro_emission_angles = None
        offrepdata.repro_incidence_angle = None
        offrepdata.repro_time = None
    else:
        offrepdata.repro_long_mask = metadata['long_mask']
        offrepdata.repro_img = metadata['img']
        offrepdata.repro_resolutions = metadata['mean_resolution']
        offrepdata.repro_phase_angles = metadata['mean_phase']
        offrepdata.repro_emission_angles = metadata['mean_emission']
        offrepdata.repro_incidence_angle = metadata['mean_incidence']
        offrepdata.repro_time = metadata['time']
        
        full_longitudes = rings_generate_longitudes(longitude_resolution=options.longitude_resolution*oops.RPD)
        offrepdata.repro_longitudes = full_longitudes[offrepdata.repro_long_mask]

def _write_repro_data(offrepdata):
    metadata = None
    if offrepdata.repro_img is not None:
        metadata = {}
        metadata['img'] = offrepdata.repro_img
        metadata['long_mask'] = offrepdata.repro_long_mask
        metadata['mean_resolution'] = offrepdata.repro_resolutions
        metadata['mean_phase'] = offrepdata.repro_phase_angles
        metadata['mean_emission'] = offrepdata.repro_emission_angles
        metadata['mean_incidence'] = offrepdata.repro_incidence_angle
        metadata['time'] = offrepdata.repro_time
    
    fring_util.write_repro(offrepdata.repro_path, metadata)
    
def _reproject_one_image(offrepdata):
    if offrepdata.obs is None:
        offrepdata.obs = iss.from_file(offrepdata.image_path)

    obs = offrepdata.obs

    offset = None
    
    if offrepdata.manual_offset is not None:
        offset = offrepdata.manual_offset
    elif offrepdata.the_offset is not None:
        offset = offrepdata.the_offset
    else:
        print 'NO OFFSET - REPROJECTION FAILED'
        _update_offrepdata_repro(offrepdata, None)
        return
    
    try:
        ret = rings_reproject(obs, offset=offset,
                              longitude_resolution=options.longitude_resolution*oops.RPD,
                              radius_resolution=options.radius_resolution,
                              radius_range=(options.radius_inner,options.radius_outer),
                              corotating='F')
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
        print '** Reproject', offrepdata.obsid, '/', offrepdata.image_name, '-',
    
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
    
    if not os.path.exists(offrepdata.offset_path):
        if options.verbose:
            print 'NO OFFSET FILE - ABORTING'
        return
    
    time_offset = os.stat(offrepdata.offset_path).st_mtime
    if time_repro >= time_offset and not option_recompute:
        # The repro file exists and is more recent than the image, and we're not forcing a recompute
        if options.verbose:
            print 'Ignored because repro file is up to date'
        return

    offrepdata.off_metadata = file_read_offset_metadata(offrepdata.image_path)
    if offrepdata.off_metadata['offset'] is None:
        offrepdata.the_offset = None
    else:
        offrepdata.the_offset = offrepdata.off_metadata['offset']
    if not 'manual_offset' in offrepdata.off_metadata: 
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = offrepdata.off_metadata['manual_offset']
    
    if offrepdata.the_offset is None and offrepdata.manual_offset is None:
        if options.verbose:
            print 'OFFSET IS INVALID - ABORTING'
        return

    if options.max_subprocesses:
        if options.verbose:
            print 'QUEUEING SUBPROCESS'
        offrepdata.subprocess_run = True
        return

    setup_image_logging(offrepdata)
    
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
    y = int(float(options.radius_outer)/(options.radius_outer-options.radius_inner)*
            offrepdata.repro_img.shape[0])
    if 0 <= y < repro_overlay.shape[0]:
        repro_overlay[y, :, 0] = 1
    
    offrepdispdata.imdisp_repro.set_overlay(0, repro_overlay)

# Draw the offset curves
def draw_offset_overlay(offrepdata, offrepdispdata):
    # Blue - 0,0 offset
    # Red - auto offset
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
 
    x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs) # No offset - blue
    x_pixels = x_pixels.astype('int')
    y_pixels = y_pixels.astype('int')
    offset_overlay[y_pixels, x_pixels, 2] = 255

    if offrepdata.the_offset is not None:
        # Auto offset - red
        x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs, 
                                        offset=offrepdata.the_offset)
        x_pixels = x_pixels.astype('int')
        y_pixels = y_pixels.astype('int')
        offset_overlay[y_pixels, x_pixels, 0] = 255

    if offrepdata.manual_offset is not None:
        # Manual offset - green
        x_pixels, y_pixels = rings_fring_pixels(offrepdata.obs, 
                                        offset=offrepdata.manual_offset)
        x_pixels = x_pixels.astype('int')
        y_pixels = y_pixels.astype('int')
        offset_overlay[y_pixels, x_pixels, 1] = 255

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
        offrepdispdata.label_off_corot_longitude.config(text=('%7.3f'%(offrepdispdata.off_longitudes[y,x]*oops.DPR)))
        offrepdispdata.label_off_inertial_longitude.config(text=('%7.3f'%(rings_fring_corotating_to_inertial(offrepdispdata.off_longitudes[y,x],
                                                                                                            offrepdata.obs.midtime)*oops.DPR)))
    if offrepdispdata.off_radii is not None:
        offrepdispdata.label_off_radius.config(text=('%7.3f'%offrepdispdata.off_radii[y,x]))


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

    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_resolutions = offrepdata.repro_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angle = offrepdata.repro_incidence_angle

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
        offrepdata.off_metadata['manual_offset'] = None 
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
        offrepdata.off_metadata['manual_offset'] = offrepdata.manual_offset 
    file_write_offset_metadata(offrepdata.image_path, offrepdata.off_metadata)
    _write_repro_data(offrepdata)

# Setup the offset/reproject window with no data
def setup_offset_reproject_window(offrepdata, offrepdispdata):
    set_obs_bp(offrepdata.obs)
    
    offrepdispdata.off_radii = offrepdata.obs.bp.ring_radius('saturn:ring').vals.astype('float')
    offrepdispdata.off_longitudes = offrepdata.obs.bp.ring_longitude('saturn:ring').vals.astype('float')
    offrepdispdata.off_longitudes = rings_fring_inertial_to_corotating(offrepdispdata.off_longitudes,
                                                                       offrepdata.obs.midtime)
    
    offrepdispdata.toplevel = Tk()
    offrepdispdata.toplevel.title(offrepdata.obsid + ' / ' + offrepdata.image_name)
    frame_toplevel = Frame(offrepdispdata.toplevel)
    
    # The original image and overlaid ring curves
    offrepdispdata.imdisp_offset = ImageDisp([offrepdata.obs.data], canvas_size=(512,512),
                                             allow_enlarge=True, auto_update=True,
                                             parent=frame_toplevel)
#    offrepdispdata.imdisp_offset.set_image_params(0., 0.00121, 0.5) # XXX - N1557046172_1
    

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
    
    # Display for longitude and radius
    label = Label(img_addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_inertial_longitude = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_inertial_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_corot_longitude = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_corot_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_radius = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
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
                             option_invalid_reproject, do_mainloop=True):
    if options.verbose:
        print '** Display', offrepdata.obsid, '/', offrepdata.image_name
    if offrepdata.off_metadata is None:
        offrepdata.off_metadata = file_read_offset_metadata(offrepdata.image_path)
        if offrepdata.off_metadata is None:
            offrepdata.the_offset = None
            offrepdata.manual_offset = None
        else:
            if offrepdata.off_metadata['offset'] is None:
                offrepdata.the_offset = None
            else:
                offrepdata.the_offset = offrepdata.off_metadata['offset']
            if 'manual_offset' not in offrepdata.off_metadata:
                offrepdata.manual_offset = None
            else:
                offrepdata.manual_offset = offrepdata.off_metadata['manual_offset']

    if (option_invalid_offset and 
        (offrepdata.the_offset is not None or offrepdata.manual_offset is not None)):
        if options.verbose:
            print 'Skipping because not invalid'
        return

    # The original image
    
    if offrepdata.obs is None:
        offrepdata.obs = iss.from_file(offrepdata.image_path)

    img_max_y = offrepdata.obs.data.shape[1]-1

    if offrepdata.repro_img is None:
        ret = fring_util.read_repro(offrepdata.repro_path)
        _update_offrepdata_repro(offrepdata, ret)
 
    if offrepdata.repro_img is not None:
        offrepdispdata.repro_overlay = np.zeros(offrepdata.repro_img.shape + (3,))
    else:
        offrepdispdata.repro_overlay = None
        
    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_resolutions = offrepdata.repro_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angle = offrepdata.repro_incidence_angle
    
    setup_offset_reproject_window(offrepdata, offrepdispdata)
    draw_repro_overlay(offrepdata, offrepdispdata)

    if do_mainloop:
        mainloop()

# The callback for mouse move events on the reprojected image
def callback_repro(x, y, offrepdata, offrepdispdata):
    if offrepdispdata.repro_longitudes is None:
        return

    offrepdispdata.label_corot_longitude.config(text=('%7.3f'%(offrepdispdata.repro_longitudes[x]*oops.DPR)))
    offrepdispdata.label_inertial_longitude.config(text=('%7.3f'%(rings_fring_corotating_to_inertial(offrepdispdata.repro_longitudes[x],
                                                                                                    offrepdata.obs.midtime)*oops.DPR)))
    
    radius = y*options.radius_resolution+options.radius_inner
    offrepdispdata.label_radius.config(text = '%7.3f'%radius)
    offrepdispdata.label_resolution.config(text=('%7.3f'%offrepdispdata.repro_resolutions[x]))
    offrepdispdata.label_phase.config(text=('%7.3f'%offrepdispdata.repro_phase_angles[x]))
    offrepdispdata.label_emission.config(text=('%7.3f'%offrepdispdata.repro_emission_angles[x]))
    offrepdispdata.label_incidence.config(text=('%7.3f'%offrepdispdata.repro_incidence_angle))


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

# Set up per-image logging
_LOGGING_NAME = 'cb.' + __name__
image_logger = logging.getLogger(_LOGGING_NAME)

image_logfile_level = log_decode_level(options.image_logfile_level)
image_log_console_level = log_decode_level(options.image_log_console_level)

cb_logging.log_set_default_level(log_min_level(image_logfile_level,
                                               image_log_console_level))
cb_logging.log_set_util_flux_level(logging.CRITICAL)

cb_logging.log_remove_console_handler()
cb_logging.log_add_console_handler(image_log_console_level)

subprocess_list = []

offrepdispdata = OffRepDispData()

found_obsid = False
cur_obsid = None
obsid_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for obsid, image_name, image_path in fring_util.enumerate_files(options, args):
#    if obsid == 'ISS_006RI_LPHRLFMOV001_PRIME':
#        continue
    if options.start_obsid != '' and not found_obsid:
        if obsid != options.start_obsid:
            continue
        found_obsid = True
            
    offrepdata = fring_util.OffRepData()
    offrepdata.obsid = obsid
    offrepdata.image_name = image_name
    offrepdata.image_path = image_path
    
    offrepdata.offset_path = file_img_to_offset_path(image_path)
    offrepdata.repro_path = fring_util.repro_path(options, image_path, image_name)

    offrepdata.subprocess_run = False

    if options.profile:
        pr = cProfile.Profile()
        pr.enable()

    offrepdata.image_log_filehander = None
    
    # Pointing offset
    offset_one_image(offrepdata, options.no_auto_offset, options.no_update_auto_offset,
                     options.recompute_auto_offset)
    
    # Reprojection
    reproject_one_image(offrepdata, options.no_reproject, options.no_update_reproject,
                        options.recompute_reproject)

    cb_logging.log_remove_file_handler(offrepdata.image_log_filehandler)

    if options.max_subprocesses and offrepdata.subprocess_run:
        run_and_maybe_wait([fring_util.PYTHON_EXE, python_reproject_program] + 
                           collect_cmd_line() + 
                           [offrepdata.obsid+'/'+offrepdata.image_name])
    
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
        
while len(subprocess_list):
    for i in xrange(len(subprocess_list)):
        if subprocess_list[i].poll() is not None:
            del subprocess_list[i]
            break
    if len(subprocess_list):
        time.sleep(1)
