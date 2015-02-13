'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import moon_util
import pickle
import os
import os.path
import numpy as np
import sys
import cspice
import subprocess
import scipy.ndimage.interpolation as interp
import colorsys
from imgdisp import ImageDisp, FloatEntry
from Tkinter import *
from PIL import Image
from cb_moons import *

python_dir = os.path.split(sys.argv[0])[0]
python_reproject_program = os.path.join(python_dir, moon_util.PYTHON_MOON_REPROJECT)

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--verbose',
                'MIMAS',
                '--recompute-mosaic',
#                'ISS_041RF_FMOVIE002_VIMS',
#                'ISS_106RF_FMOVIE002_PRIME',
#                'ISS_132RI_FMOVIE001_VIMS',
#                'ISS_029RF_FMOVIE002_VIMS',
                '--display-mosaic']

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

moon_util.add_parser_options(parser)

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

def _update_mosaicdata(mosaicdata, metadata):
    mosaicdata.img = metadata['img']
    mosaicdata.full_mask = metadata['full_mask']
    mosaicdata.resolutions = metadata['resolution']
    mosaicdata.image_numbers = metadata['image_number']
    mosaicdata.ETs = metadata['time'] 
    mosaicdata.emission_angles = metadata['emission']
    mosaicdata.incidence_angle = metadata['incidence']
    mosaicdata.phase_angles = metadata['phase']
    full_latitudes = moons_generate_latitudes(latitude_resolution=options.latitude_resolution)
    mosaicdata.latitudes = full_latitudes
    full_longitudes = moons_generate_longitudes(longitude_resolution=options.longitude_resolution)
    mosaicdata.longitudes = full_longitudes

def make_mosaic(mosaicdata, option_no, option_no_update, option_recompute):
    # Input files: image_path_list (includes repro suffix)
    # Output files:
    #  mosaic_data_filename (the basic 2-D array)
    #  mosaic_metadata_filename (body_name_list, image_name_list, image_path_list)
    #  large_png_filename (full size mosaic graphic)
    #  small_png_filename (reduced size mosaic graphic)
    (mosaicdata.data_path, 
     mosaicdata.png_path) = moon_util.mosaic_paths(options, mosaicdata.body_name)
    
    if options.verbose:
        print 'Make_mosaic:', mosaicdata.body_name
        
    if option_no:  # Just don't do anything
        if options.verbose:
            print 'Not doing anything because of --no-mosaic'
        return 

    if not option_recompute:
        if (os.path.exists(mosaicdata.data_path+'.npy') and
            os.path.exists(mosaicdata.metadata_path) and
            os.path.exists(mosaicdata.png_path)):
            if option_no_update:
                if options.verbose:
                    print 'Not doing anything because output files already exist and --no-update-mosaic'
                return  # Mosaic file already exists, don't update
    
            # Find the latest repro time
            max_repro_mtime = 0
            for repro_path in mosaicdata.repro_path_list:    
                time_repro = os.stat(repro_path+'.pickle').st_mtime
                max_repro_mtime = max(max_repro_mtime, time_repro)
        
            if (os.stat(mosaicdata.data_path+'.npy').st_mtime > max_repro_mtime and
                os.stat(mosaicdata.metadata_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.png_path).st_mtime > max_repro_mtime):
                # The mosaic file exists and is more recent than the reprojected images, and we're not forcing a recompute
                if options.verbose:
                    print 'Not doing anything because output files already exist and are current'
                return
    
    print 'Making mosaic for', mosaicdata.body_name
    
    mosaic_metadata = moons_mosaic_init(options.latitude_resolution,
                               options.longitude_resolution)
    for i, repro_path in enumerate(mosaicdata.repro_path_list):
        repro_metadata = moon_util.read_repro(repro_path)
        if repro_metadata is not None:
            if options.verbose:
                print 'Adding mosaic data for', repro_path
            moons_mosaic_add(mosaic_metadata, repro_metadata, i)
#            if i == 2:
#                break
            
    _update_mosaicdata(mosaicdata, mosaic_metadata)

    # Save mosaic image array in binary
#    np.savez(mosaicdata.data_path, 
#             img=mosaicdata.img)
#             full_mask = mosaic_metadata['full_mask'],
#             resolution = mosaic_metadata['resolution'],
#             image_number = mosaic_metadata['image_number'],
#             time = mosaic_metadata['time'], 
#             emission = mosaic_metadata['emission'],
#             incidence = mosaic_metadata['incidence'],
#             phase = mosaic_metadata['phase'])
#             body_name_list = mosaicdata.body_name_list,
#             image_name_list = mosaicdata.image_name_list,
#             image_path_list = mosaicdata.image_path_list,
#             repro_path_list = mosaicdata.repro_path_list)

#XXX    blackpoint = max(np.min(mosaicdata.img), 0)
#    whitepoint = np.max(mosaicdata.img)
#    img = mosaicdata.img
#    
#    blackpoint = 0. # XXX
#    whitepoint = whitepoint / 3. # XXX
#    img = img[75:326,::15].copy()
#    
#    gamma = 0.5
#    # The +0 forces a copy - necessary for PIL
#    scaled_mosaic = np.cast['int8'](ImageDisp.scale_image(img, blackpoint,
#                                                         whitepoint, gamma))[::-1,:]+0
#    img = Image.frombuffer('L', (scaled_mosaic.shape[1], scaled_mosaic.shape[0]),
#                           scaled_mosaic, 'raw', 'L', 0, 1)
#    img.save(mosaicdata.png_path, 'PNG')


#####################################################################################
#
# DISPLAY ONE MOSAIC
#
#####################################################################################

mosaicdispdata = MosaicDispData()

def command_refresh_color(mosaicdata, mosaicdispdata):
    color_sel = mosaicdispdata.var_color_by.get()
    
    if color_sel == 'none':
        mosaicdispdata.imdisp.set_color_column(0, None)
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
    mosaicdispdata.toplevel.title(mosaicdata.body_name)
    frame_toplevel = Frame(mosaicdispdata.toplevel)

    mosaicdispdata.imdisp = ImageDisp([mosaicdata.img], canvas_size=(1024,512),
                                      parent=frame_toplevel, flip_y=True, one_zoom=False)

    #############################################
    # The control/data pane of the mosaic image #
    #############################################

    gridrow = 0
    gridcolumn = 0
    
    addon_control_frame = mosaicdispdata.imdisp.addon_control_frame
    
    label = Label(addon_control_frame, text='Latitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_latitude= Label(addon_control_frame, text='')
    mosaicdispdata.label_latitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Longitude:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_longitude = Label(addon_control_frame, text='', anchor='w')
    mosaicdispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
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
    
    label = Label(addon_control_frame, text='Body:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_body_name = Label(addon_control_frame, text='')
    mosaicdispdata.label_body_name.grid(row=gridrow, column=gridcolumn+1, sticky=W)
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
    if mosaicdata.img is None:
        (mosaicdata.data_path, mosaicdata.png_path) = moon_util.mosaic_paths(options, mosaicdata.body_name)
        
        metadata = dict(np.load(mosaicdata.data_path+'.npz'))
        
#        mosaicdata.body_name_list = metadata.pop('body_name_list')
#        mosaicdata.image_name_list = metadata.pop('image_name_list')
#        mosaicdata.image_path_list = metadata.pop('image_path_list')
#        mosaicdata.repro_path_list = metadata.pop('repro_path_list')

        _update_mosaicdata(mosaicdata, metadata)

    setup_mosaic_window(mosaicdata, mosaicdispdata)
    
    mainloop()

# The callback for mouse move events on the mosaic image
def callback_move_mosaic(x, y, mosaicdata):
    x = int(x)
    if x < 0: return
    y = int(y)
    if y < 0: return
    if mosaicdata.full_mask[y,x]:  # Invalid pixel
        mosaicdispdata.label_longitude.config(text='')
        mosaicdispdata.label_phase.config(text='')
        mosaicdispdata.label_incidence.config(text='')
        mosaicdispdata.label_emission.config(text='')
        mosaicdispdata.label_resolution.config(text='')
        mosaicdispdata.label_image.config(text='')
        mosaicdispdata.label_body_name.config(text='')
        mosaicdispdata.label_date.config(text='')
    else:
        mosaicdispdata.label_longitude.config(text=('%7.3f'%mosaicdata.longitudes[x]))
        mosaicdispdata.label_latitude.config(text=('%7.3f'%mosaicdata.latitudes[y]))
        mosaicdispdata.label_phase.config(text=('%7.3f'%mosaicdata.phase_angles[y,x]))
        mosaicdispdata.label_incidence.config(text=('%7.3f'%mosaicdata.incidence_angle[y,x]))
        mosaicdispdata.label_emission.config(text=('%7.3f'%mosaicdata.emission_angles[y,x]))
        mosaicdispdata.label_resolution.config(text=('%7.3f'%mosaicdata.resolutions[y,x]))
        mosaicdispdata.label_image.config(text=mosaicdata.image_name_list[mosaicdata.image_numbers[y,x]])
        mosaicdispdata.label_body_name.config(text=mosaicdata.body_name_list[mosaicdata.image_numbers[y,x]])
        mosaicdispdata.label_date.config(text=cspice.et2utc(float(mosaicdata.ETs[y,x]), 'C', 0))
    
# The command for Mosaic button press - rerun offset/reproject
def callback_b1press_mosaic(x, y, mosaicdata):
    x = int(x)
    if x < 0: return
    y = int(y)
    if y < 0: return
    if mosaicdata.full_mask[y,x]:
        return
    image_number = mosaicdata.image_numbers[y,x]
    subprocess.Popen([moon_util.PYTHON_EXE, python_reproject_program, '--display-offset-reproject', 
                      mosaicdata.body_name_list[image_number] + '/' + mosaicdata.image_name_list[image_number]])

#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

# Each entry in the list is a tuple of body_name_list, image_name_list, image_path_list, repro_path_list
mosaic_list = []

cur_body_name = None
body_name_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for body_name, image_name, image_path in moon_util.enumerate_files(options, args, '_CALIB.IMG'):
    repro_path = moon_util.repro_path(options, image_path, image_name)
    
    if cur_body_name == None:
        cur_body_name = body_name
    if cur_body_name != body_name:
        if len(body_name_list) != 0:
            if options.verbose:
                print 'Adding body_name', body_name_list[0]
            mosaic_list.append((body_name_list, image_name_list, image_path_list, repro_path_list))
        body_name_list = []
        image_name_list = []
        image_path_list = []
        repro_path_list = []
        cur_body_name = body_name
    if os.path.exists(repro_path+'.pickle'):
        body_name_list.append(body_name)
        image_name_list.append(image_name)
        image_path_list.append(image_path)
        repro_path_list.append(repro_path)
    
# Final mosaic
if len(body_name_list) != 0:
    if options.verbose:
        print 'Adding body_name', body_name_list[0]
    mosaic_list.append((body_name_list, image_name_list, image_path_list, repro_path_list))
    body_name_list = []
    image_name_list = []
    image_path_list = []
    repro_path_list = []

for mosaic_info in mosaic_list:
    mosaicdata = moon_util.MosaicData()
    (mosaicdata.body_name_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
     mosaicdata.repro_path_list) = mosaic_info
    mosaicdata.body_name = mosaicdata.body_name_list[0]
    make_mosaic(mosaicdata, options.no_mosaic, options.no_update_mosaic,
                options.recompute_mosaic) 

    if options.display_mosaic:
        for mosaic_info in mosaic_list:
            (mosaicdata.body_name_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
             mosaicdata.repro_path_list) = mosaic_info
            mosaicdata.body_name = mosaicdata.body_name_list[0]
            display_mosaic(mosaicdata, mosaicdispdata) 
