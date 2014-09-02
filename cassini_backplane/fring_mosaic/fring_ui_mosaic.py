'''
Created on Sep 19, 2011

@author: rfrench
'''

###
### BUGS:
### - Need a better way to specify X and Y offset with sliders
### - Every time you reproject the image, it appends more metadata onto vicar_data and saves it all
###   to the output file
###

from optparse import OptionParser
import ringutil
import mosaic
import ringimage
import pickle
import os
import os.path
import fitreproj
import vicar
import numpy as np
import sys
import cspice
import subprocess
import scipy.ndimage.interpolation as interp
import colorsys
from imgdisp import ImageDisp, FloatEntry
from Tkinter import *
from PIL import Image

#Tk().withdraw()

python_dir = os.path.split(sys.argv[0])[0]
python_reproject_program = os.path.join(python_dir, ringutil.PYTHON_RING_REPROJECT)

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--verbose', 'ISS_081RI_SPKMVLFLP001_PRIME', '--ignore-bad-obsids']
#    cmd_line = ['--verbose', '--display-mosaic', 'ISS_029RF_FMOVIE001_VIMS']
#    cmd_line = ['--verbose', 'test_059RF', '--display-mosaic']

parser = OptionParser()

#
# The default behavior is to check the timestamps
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
## Options for mosaic creation
##
parser.add_option('--no-mosaic', dest='no_mosaic',
                  action='store_true', default=False,
                  help="Don't compute the mosaic even if we don't have one")
parser.add_option('--no-update-mosaic', dest='no_update_mosaic',
                  action='store_true', default=False,
                  help="Don't compute the mosaic unless we don't have one")
parser.add_option('--recompute-mosaic', dest='recompute_mosaic',
                  action='store_true', default=False,
                  help="Recompute the mosaic even if we already have one that is current")
parser.add_option('--display-mosaic', dest='display_mosaic',
                  action='store_true', default=False,
                  help='Display the mosaic')

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


class MosaicDispData:
    def __init__(self):
        self.toplevel = None
        self.imdisp_offset = None
        self.imdisp_repro = None
    
#####################################################################################
#
# MAKE A MOSAIC
#
#####################################################################################

def make_mosaic(mosaicdata, option_no, option_no_update, option_recompute):
    # Input files: image_path_list (includes repro suffix)
    # Output files:
    #  mosaic_data_filename (the basic 2-D array)
    #  mosaic_metadata_filename (obsid_list, image_name_list, image_path_list)
    #  large_png_filename (full size mosaic graphic)
    #  small_png_filename (reduced size mosaic graphic)
    (mosaicdata.data_path, mosaicdata.metadata_path,
     mosaicdata.large_png_path, mosaicdata.small_png_path) = ringutil.mosaic_paths(options, mosaicdata.obsid)
    
    if options.verbose:
        print 'Make_mosaic:', mosaicdata.obsid
        
    if option_no:  # Just don't do anything
        if options.verbose:
            print 'Not doing anything because of --no-mosaic'
        return 

    if not option_recompute:
        if (os.path.exists(mosaicdata.data_path+'.npy') and
            os.path.exists(mosaicdata.metadata_path) and
            os.path.exists(mosaicdata.large_png_path) and
            os.path.exists(mosaicdata.small_png_path)):
            if option_no_update:
                if options.verbose:
                    print 'Not doing anything because output files already exist and --no-update-mosaic'
                return  # Mosaic file already exists, don't update
    
            # Find the latest repro time
            max_repro_mtime = 0
            for repro_path in mosaicdata.repro_path_list:    
                time_repro = os.stat(repro_path).st_mtime
                max_repro_mtime = max(max_repro_mtime, time_repro)
        
            if (os.stat(mosaicdata.data_path+'.npy').st_mtime > max_repro_mtime and
                os.stat(mosaicdata.metadata_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.large_png_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.small_png_path).st_mtime > max_repro_mtime):
                # The mosaic file exists and is more recent than the reprojected images, and we're not forcing a recompute
                if options.verbose:
                    print 'Not doing anything because output files already exist and are current'
                return
    
    print 'Making mosaic for', mosaicdata.obsid
    
    result = mosaic.MakeMosaic(mosaicdata.repro_path_list, options.radius_start, options.radius_end,
                               options.radius_resolution, options.longitude_resolution)
    (mosaicdata.img, mosaicdata.longitudes, mosaicdata.resolutions,
     mosaicdata.image_numbers, mosaicdata.ETs, 
     mosaicdata.emission_angles, mosaicdata.incidence_angles,
     mosaicdata.phase_angles) = result

    # Save mosaic image array in binary
    np.save(mosaicdata.data_path, mosaicdata.img)
    
    # Save metadata
    metadata = result[1:] # Everything except img
    mosaic_metadata_fp = open(mosaicdata.metadata_path, 'wb')
    pickle.dump(metadata, mosaic_metadata_fp)
    pickle.dump(mosaicdata.obsid_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.image_name_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.image_path_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.repro_path_list, mosaic_metadata_fp)
    mosaic_metadata_fp.close()
    
    blackpoint = max(np.min(mosaicdata.img), 0)
    whitepoint = np.max(mosaicdata.img)
    gamma = 0.5
    # The +0 forces a copy - necessary for PIL
    scaled_mosaic = np.cast['int8'](ImageDisp.ScaleImage(mosaicdata.img, blackpoint,
                                                         whitepoint, gamma))[::-1,:]+0
    img = Image.frombuffer('L', (scaled_mosaic.shape[1], scaled_mosaic.shape[0]),
                           scaled_mosaic, 'raw', 'L', 0, 1)
    img.save(mosaicdata.large_png_path, 'PNG')
    reduced_scaled_mosaic = scaled_mosaic[:,::options.mosaic_reduction_factor]+0
    del scaled_mosaic
    scaled_mosaic = None
    img = Image.frombuffer('L', (reduced_scaled_mosaic.shape[1], reduced_scaled_mosaic.shape[0]),
                           reduced_scaled_mosaic, 'raw', 'L', 0, 1)
    img.save(mosaicdata.small_png_path, 'PNG')
    del reduced_scaled_mosaic
    reduced_scaled_mosaic = None


#####################################################################################
#
# DISPLAY ONE MOSAIC
#
#####################################################################################

mosaicdispdata = MosaicDispData()

def command_refresh_color(mosaicdata, mosaicdispdata):
    color_sel = mosaicdispdata.var_color_by.get()
    
    if color_sel == 'none':
        mosaicdispdata.imdisp.set_overlay(0, None)
        return
    
    minval = None
    maxval = None
    
    if color_sel == 'relresolution':
        valsrc = mosaicdata.resolutions
    elif color_sel == 'relphase':
        valsrc = mosaicdata.phase_angles
    elif color_sel == 'absphase':
        valsrc = mosaicdata.phase_angles
        minval = 0.
        maxval = 180.
    elif color_sel == 'relemission':
        valsrc = mosaicdata.emission_angles
    elif color_sel == 'absemission':
        valsrc = mosaicdata.emission_angles
        minval = 0.
        maxval = 180.
    
    if minval is None:
        minval = np.min(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])
        maxval = np.max(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])
    
    print minval, maxval
    
    color_data = np.zeros((mosaicdata.longitudes.shape[0], 3))

    for col in range(len(mosaicdata.longitudes)):
        if mosaicdata.longitudes[col] >= 0.:
            color = colorsys.hsv_to_rgb((1-(valsrc[col]-minval)/(maxval-minval))*.66, 1, 1)
            color_data[col,:] = color

    mosaicdispdata.imdisp.set_color_column(0, color_data)
    
def setup_mosaic_window(mosaicdata, mosaicdispdata):
    mosaicdispdata.toplevel = Tk()
    mosaicdispdata.toplevel.title(mosaicdata.obsid)
    frame_toplevel = Frame(mosaicdispdata.toplevel)

    mosaicdispdata.imdisp = ImageDisp([mosaicdata.img], canvas_size=(1024,512),
                                      parent=frame_toplevel, flip_y=True, one_zoom=False)

    #############################################
    # The control/data pane of the mosaic image #
    #############################################

    gridrow = 0
    gridcolumn = 0
    
    addon_control_frame = mosaicdispdata.imdisp.addon_control_frame
    
    label = Label(addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    # We make this one fixed-width so that the color-control column stays in one place
    mosaicdispdata.label_inertial_longitude = Label(addon_control_frame, text='', anchor='w', width=28)
    mosaicdispdata.label_inertial_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    # We make this one fixed-width so that the color-control column stays in one place
    mosaicdispdata.label_longitude = Label(addon_control_frame, text='', anchor='w', width=28)
    mosaicdispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_radius = Label(addon_control_frame, text='')
    mosaicdispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_phase = Label(addon_control_frame, text='')
    mosaicdispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_incidence = Label(addon_control_frame, text='')
    mosaicdispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_emission = Label(addon_control_frame, text='')
    mosaicdispdata.label_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Image:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_image = Label(addon_control_frame, text='')
    mosaicdispdata.label_image.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='OBSID:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_obsid = Label(addon_control_frame, text='')
    mosaicdispdata.label_obsid.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_date = Label(addon_control_frame, text='')
    mosaicdispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    gridrow = 0
    gridcolumn = 2

    mosaicdispdata.var_color_by = StringVar()
    refresh_color = lambda: command_refresh_color(mosaicdata, mosaicdispdata)
    
    label = Label(addon_control_frame, text='Color by:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='None', variable=mosaicdispdata.var_color_by,
                value='none', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Resolution', variable=mosaicdispdata.var_color_by,
                value='relresolution', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Phase', variable=mosaicdispdata.var_color_by,
                value='absphase', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Phase', variable=mosaicdispdata.var_color_by,
                value='relphase', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Emission', variable=mosaicdispdata.var_color_by,
                value='absemission', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Emission', variable=mosaicdispdata.var_color_by,
                value='relemission', command=refresh_color).grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.var_color_by.set('none')
    
    callback_mosaic_move_command = lambda x, y, mosaicdata=mosaicdata: callback_move_mosaic(x, y, mosaicdata)
    mosaicdispdata.imdisp.bind_mousemove(0, callback_mosaic_move_command)
    
    callback_mosaic_b1press_command = lambda x, y, mosaicdata=mosaicdata: callback_b1press_mosaic(x, y, mosaicdata)
    mosaicdispdata.imdisp.bind_b1press(0, callback_mosaic_b1press_command)

    mosaicdispdata.imdisp.pack(side=LEFT)
    
    frame_toplevel.pack()
    
    
def display_mosaic(mosaicdata, mosaicdispdata):
    if mosaicdata.img == None:
        (mosaicdata.data_path, mosaicdata.metadata_path,
         mosaicdata.large_png_path, mosaicdata.small_png_path) = ringutil.mosaic_paths(options, mosaicdata.obsid)
        
        mosaicdata.img = np.load(mosaicdata.data_path+'.npy')
        
        mosaic_metadata_fp = open(mosaicdata.metadata_path, 'rb')
        metadata = pickle.load(mosaic_metadata_fp)
        (mosaicdata.longitudes, mosaicdata.resolutions,
         mosaicdata.image_numbers, mosaicdata.ETs, 
         mosaicdata.emission_angles, mosaicdata.incidence_angles,
         mosaicdata.phase_angles) = metadata
        mosaicdata.obsid_list = pickle.load(mosaic_metadata_fp)
        mosaicdata.image_name_list = pickle.load(mosaic_metadata_fp)
        mosaicdata.image_path_list = pickle.load(mosaic_metadata_fp)
        mosaicdata.repro_path_list = pickle.load(mosaic_metadata_fp)
        mosaic_metadata_fp.close()

    setup_mosaic_window(mosaicdata, mosaicdispdata)
    
    mainloop()

# The callback for mouse move events on the mosaic image
def callback_move_mosaic(x, y, mosaicdata):
    if x < 0: return
    if mosaicdata.longitudes[x] < 0:  # Invalid longitude
        mosaicdispdata.label_inertial_longitude.config(text='')
        mosaicdispdata.label_longitude.config(text='')
        mosaicdispdata.label_phase.config(text='')
        mosaicdispdata.label_incidence.config(text='')
        mosaicdispdata.label_emission.config(text='')
        mosaicdispdata.label_resolution.config(text='')
        mosaicdispdata.label_image.config(text='')
        mosaicdispdata.label_obsid.config(text='')
        mosaicdispdata.label_date.config(text='')
    else:
        mosaicdispdata.label_inertial_longitude.config(text=('%7.3f'%ringutil.CorotatingToInertial(mosaicdata.longitudes[x],
                                                                                                   mosaicdata.ETs[x])))
        mosaicdispdata.label_longitude.config(text=('%7.3f'%mosaicdata.longitudes[x]))
        radius = mosaic.IndexToRadius(y, options.radius_resolution)
        mosaicdispdata.label_radius.config(text = '%7.3f'%radius)
        mosaicdispdata.label_phase.config(text=('%7.3f'%mosaicdata.phase_angles[x]))
        mosaicdispdata.label_incidence.config(text=('%7.3f'%mosaicdata.incidence_angles[x]))
        mosaicdispdata.label_emission.config(text=('%7.3f'%mosaicdata.emission_angles[x]))
        mosaicdispdata.label_resolution.config(text=('%7.3f'%mosaicdata.resolutions[x]))
        mosaicdispdata.label_image.config(text=mosaicdata.image_name_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_obsid.config(text=mosaicdata.obsid_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_date.config(text=cspice.et2utc(mosaicdata.ETs[x], 'C', 0))

# The command for Mosaic button press - rerun offset/reproject
def callback_b1press_mosaic(x, y, mosaicdata):
    if x < 0: return
    if mosaicdata.longitudes[x] < 0:  # Invalid longitude - nothing to do
        return
    image_number = mosaicdata.image_numbers[x]
    subprocess.Popen([ringutil.PYTHON_EXE, python_reproject_program, '--display-offset-reproject', 
                      mosaicdata.obsid_list[image_number] + '/' + mosaicdata.image_name_list[image_number]])

#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

# Each entry in the list is a tuple of obsid_list, image_name_list, image_path_list, repro_path_list
mosaic_list = []

cur_obsid = None
obsid_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for obsid, image_name, image_path in ringutil.enumerate_files(options, args, '_CALIB.IMG'):
    repro_path = ringutil.repro_path(options, image_path, image_name)
    
    if cur_obsid == None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(obsid_list) != 0:
            if options.verbose:
                print 'Adding obsid', obsid_list[0]
            mosaic_list.append((obsid_list, image_name_list, image_path_list, repro_path_list))
        obsid_list = []
        image_name_list = []
        image_path_list = []
        repro_path_list = []
        cur_obsid = obsid
    if os.path.exists(repro_path):
        obsid_list.append(obsid)
        image_name_list.append(image_name)
        image_path_list.append(image_path)
        repro_path_list.append(repro_path)
    
# Final mosaic
if len(obsid_list) != 0:
    if options.verbose:
        print 'Adding obsid', obsid_list[0]
    mosaic_list.append((obsid_list, image_name_list, image_path_list, repro_path_list))
    obsid_list = []
    image_name_list = []
    image_path_list = []
    repro_path_list = []

for mosaic_info in mosaic_list:
    mosaicdata = ringutil.MosaicData()
    (mosaicdata.obsid_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
     mosaicdata.repro_path_list) = mosaic_info
    mosaicdata.obsid = mosaicdata.obsid_list[0]
    make_mosaic(mosaicdata, options.no_mosaic, options.no_update_mosaic,
                options.recompute_mosaic) 

if options.display_mosaic:
    for mosaic_info in mosaic_list:
        mosaicdata = ringutil.MosaicData()
        (mosaicdata.obsid_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
         mosaicdata.repro_path_list) = mosaic_info
        mosaicdata.obsid = mosaicdata.obsid_list[0]
        display_mosaic(mosaicdata, mosaicdispdata) 
