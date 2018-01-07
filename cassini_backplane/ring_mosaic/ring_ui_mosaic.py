# Issues:

import argparse
import ring_util
import os
import os.path
import numpy as np
import sys
import cspice
import subprocess
import colorsys
from imgdisp import ImageDisp, FloatEntry
from Tkinter import *
from PIL import Image
import oops
import msgpack
import msgpack_numpy
from cb_rings import *

command_list = sys.argv[1:]

if len(command_list) == 0:
#     command_line_str = '--verbose ISS_029RF_FMOVIE001_VIMS --display-mosaic'
    command_line_str = '--verbose --ring-type BRING_MOUNTAINS --all-obsid'
#     command_line_str = '--verbose --ring-type FMOVIE --all-obsid'
                 
    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

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
parser.add_argument('--allow-exception', dest='allow_exception',
                  action='store_true', default=False,
                  help="Allow exceptions to be thrown")

##
## Options for mosaic creation
##
parser.add_argument('--no-mosaic', dest='no_mosaic',
                  action='store_true', default=False,
                  help="Don't compute the mosaic even if we don't have one")
parser.add_argument('--no-update-mosaic', dest='no_update_mosaic',
                  action='store_true', default=False,
                  help="Don't compute the mosaic unless we don't have one")
parser.add_argument('--recompute-mosaic', dest='recompute_mosaic',
                  action='store_true', default=False,
                  help="Recompute the mosaic even if we already have one that is current")
parser.add_argument('--display-mosaic', dest='display_mosaic',
                  action='store_true', default=False,
                  help='Display the mosaic')

ring_util.ring_add_parser_options(parser)

arguments = parser.parse_args(command_list)

ring_util.ring_init(arguments)

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
    mosaicdata.long_mask = metadata['long_mask']
    mosaicdata.resolutions = metadata['mean_resolution']
    mosaicdata.image_numbers = metadata['image_number']
    mosaicdata.ETs = metadata['time'] 
    mosaicdata.emission_angles = metadata['mean_emission']
    mosaicdata.incidence_angle = metadata['mean_incidence']
    mosaicdata.phase_angles = metadata['mean_phase']
    full_longitudes = rings_generate_longitudes(longitude_resolution=arguments.longitude_resolution*oops.RPD)
    full_longitudes[np.logical_not(mosaicdata.long_mask)] = -10
    mosaicdata.longitudes = full_longitudes

def make_mosaic(mosaicdata, option_no, option_no_update, option_recompute):
    # Input files: image_path_list (includes repro suffix)
    # Output files:
    #  mosaic_data_filename (the basic 2-D array)
    #  mosaic_metadata_filename (obsid_list, image_name_list, image_path_list)
    #  large_png_filename (full size mosaic graphic)
    (mosaicdata.data_path, 
     mosaicdata.metadata_path) = ring_util.mosaic_paths(
                                            arguments, mosaicdata.obsid,
                                            make_dirs=True)
    mosaicdata.full_png_path = ring_util.png_path(arguments, mosaicdata.obsid,
                                       'full', make_dirs=True)
    mosaicdata.small_png_path = ring_util.png_path(arguments, mosaicdata.obsid,
                                        'small', make_dirs=True)
         
    if arguments.verbose:
        print 'Make_mosaic:', mosaicdata.obsid
        
    if option_no:  # Just don't do anything
        if arguments.verbose:
            print 'Not doing anything because of --no-mosaic'
        return 

    if not option_recompute:
        if (os.path.exists(mosaicdata.data_path+'.npy') and
            os.path.exists(mosaicdata.metadata_path) and
            os.path.exists(mosaicdata.full_png_path) and
            os.path.exists(mosaicdata.small_png_path)):
            if option_no_update:
                if arguments.verbose:
                    print 'Not doing anything because output files already exist and --no-update-mosaic'
                return  # Mosaic file already exists, don't update
    
            # Find the latest repro time
            max_repro_mtime = 0
            for repro_path in mosaicdata.repro_path_list:    
                time_repro = os.stat(repro_path).st_mtime
                max_repro_mtime = max(max_repro_mtime, time_repro)
        
            if (os.stat(mosaicdata.data_path+'.npy').st_mtime > max_repro_mtime and
                os.stat(mosaicdata.metadata_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.full_png_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.small_png_path).st_mtime > max_repro_mtime):
                # The mosaic file exists and is more recent than the reprojected images, and we're not forcing a recompute
                if arguments.verbose:
                    print 'Not doing anything because output files already exist and are current'
                return
    
    print 'Making mosaic for', mosaicdata.obsid
    
    mosaic_metadata = rings_mosaic_init((arguments.ring_radius+arguments.radius_inner_delta, 
                                         arguments.ring_radius+arguments.radius_outer_delta),
                                        arguments.longitude_resolution * oops.RPD,
                                        arguments.radius_resolution)

    for i, repro_path in enumerate(mosaicdata.repro_path_list):
        repro_metadata = ring_util.read_repro(repro_path)
        if repro_metadata is not None:
            if arguments.verbose:
                print 'Adding mosaic data for', repro_path
            rings_mosaic_add(mosaic_metadata, repro_metadata, i)

    _update_mosaicdata(mosaicdata, mosaic_metadata)

    # Save mosaic image array in binary
    np.save(mosaicdata.data_path, mosaicdata.img)
    
    # Save metadata
    metadata = mosaic_metadata.copy() # Everything except img
    del metadata['img']
    mosaic_metadata_fp = open(mosaicdata.metadata_path, 'wb')
    mosaic_metadata_fp.write(msgpack.packb(
              (metadata, 
               mosaicdata.obsid_list,
               mosaicdata.image_name_list,
               mosaicdata.image_path_list,
               mosaicdata.repro_path_list),
                                  default=msgpack_numpy.encode))
    mosaic_metadata_fp.close()
    
    blackpoint = max(np.min(mosaicdata.img), 0)
    whitepoint_ignore_frac = 0.995
    img_sorted = sorted(list(mosaicdata.img.flatten()))
    whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                        whitepoint_ignore_frac),
                                    0, len(img_sorted)-1)]
#     whitepoint = np.max(mosaicdata.img)
    gamma = 0.5
    img = mosaicdata.img
    
#     blackpoint = 0. # XXX
#     whitepoint = whitepoint / 3. # XXX
#     img = img[75:326,::15].copy() # XXX
    

    # For poster DPS 2014 ISS_044RF_FMOVIE001_VIMS
#     blackpoint = 0.
#     whitepoint = 0.05977
#     gamma = 0.65
#     img = img[90:310,:].copy()
    
    # 
    # The +0 forces a copy - necessary for PIL
    scaled_mosaic = np.cast['int8'](ImageDisp.scale_image(img, blackpoint,
                                                          whitepoint, gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_mosaic.shape[1], scaled_mosaic.shape[0]),
                           scaled_mosaic, 'raw', 'L', 0, 1)

    pil_img.save(mosaicdata.full_png_path, 'PNG')

    # Reduced mosaic for easier viewing
    scale = max(img.shape[1] // 1920, 1)
    scaled_mosaic = np.cast['int8'](ImageDisp.scale_image(img[:,::scale], blackpoint,
                                                         whitepoint, gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_mosaic.shape[1], scaled_mosaic.shape[0]),
                           scaled_mosaic, 'raw', 'L', 0, 1)

    pil_img.save(mosaicdata.small_png_path, 'PNG')
    
    print 'Mosaic saved'

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
    if mosaicdata.img is None:
        (mosaicdata.data_path, 
         mosaicdata.metadata_path) = ring_util.mosaic_paths(
                                            arguments, mosaicdata.obsid,
                                            make_dirs=True)
        
        img = np.load(mosaicdata.data_path+'.npy')
        
        mosaic_metadata_fp = open(mosaicdata.metadata_path, 'rb')
        (metadata, 
         mosaicdata.obsid_list,
         mosaicdata.image_name_list,
         mosaicdata.image_path_list,
         mosaicdata.repro_path_list) = msgpack.unpackb(mosaic_metadata_fp.read(), 
                                   object_hook=msgpack_numpy.decode)
        metadata['img'] = img
        _update_mosaicdata(mosaicdata, metadata)
        mosaic_metadata_fp.close()

    setup_mosaic_window(mosaicdata, mosaicdispdata)
    
    mainloop()

# The callback for mouse move events on the mosaic image
def callback_move_mosaic(x, y, mosaicdata):
    x = int(x)
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
        mosaicdispdata.label_inertial_longitude.config(text=('%7.3f'%(rings_fring_corotating_to_inertial(mosaicdata.longitudes[x],
                                                                                                   mosaicdata.ETs[x])*oops.DPR)))
        mosaicdispdata.label_longitude.config(text=('%7.3f'%(mosaicdata.longitudes[x]*oops.DPR)))
#        radius = mosaic.IndexToRadius(y, arguments.radius_resolution)
#        mosaicdispdata.label_radius.config(text = '%7.3f'%radius)
        mosaicdispdata.label_phase.config(text=('%7.3f'%(mosaicdata.phase_angles[x]*oops.DPR)))
        mosaicdispdata.label_incidence.config(text=('%7.3f'%(mosaicdata.incidence_angle*oops.DPR)))
        mosaicdispdata.label_emission.config(text=('%7.3f'%(mosaicdata.emission_angles[x]*oops.DPR)))
        mosaicdispdata.label_resolution.config(text=('%7.3f'%mosaicdata.resolutions[x]))
        mosaicdispdata.label_image.config(text=mosaicdata.image_name_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_obsid.config(text=mosaicdata.obsid_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_date.config(text=cspice.et2utc(float(mosaicdata.ETs[x]), 'C', 0))
    
    y = int(y)
    if y < 0: return
    radius = y*arguments.radius_resolution+arguments.ring_radius+arguments.radius_inner_delta
    mosaicdispdata.label_radius.config(text = '%7.3f'%radius)

# The command for Mosaic button press - rerun offset/reproject
def callback_b1press_mosaic(x, y, mosaicdata):
    if x < 0: return
    x = int(x)
    if mosaicdata.longitudes[x] < 0:  # Invalid longitude - nothing to do
        return
    image_number = mosaicdata.image_numbers[x]
    subprocess.Popen([ring_util.PYTHON_EXE, ring_util.RING_REPROJECT_PY, 
                      '--display-offset-reproject',
                      '--no-auto-offset',
                      '--no-reproject', 
                      mosaicdata.obsid_list[image_number] + '/' + 
                      mosaicdata.image_name_list[image_number]] +
                     ring_util.ring_basic_cmd_line(arguments))


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
for obsid, image_name, image_path in ring_util.ring_enumerate_files(arguments):
    repro_path = ring_util.repro_path(arguments, image_path, image_name)
    
    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(obsid_list) != 0:
            if arguments.verbose:
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
    if arguments.verbose:
        print 'Adding obsid', obsid_list[0]
    mosaic_list.append((obsid_list, image_name_list, image_path_list, repro_path_list))
    obsid_list = []
    image_name_list = []
    image_path_list = []
    repro_path_list = []

for mosaic_info in mosaic_list:
    mosaicdata = ring_util.MosaicData()
    (mosaicdata.obsid_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
     mosaicdata.repro_path_list) = mosaic_info
    mosaicdata.obsid = mosaicdata.obsid_list[0]
    make_mosaic(mosaicdata, arguments.no_mosaic, arguments.no_update_mosaic,
                arguments.recompute_mosaic) 

if arguments.display_mosaic:
    for mosaic_info in mosaic_list:
        mosaicdata = ring_util.MosaicData()
        (mosaicdata.obsid_list, mosaicdata.image_name_list, mosaicdata.image_path_list,
         mosaicdata.repro_path_list) = mosaic_info
        mosaicdata.obsid = mosaicdata.obsid_list[0]
        display_mosaic(mosaicdata, mosaicdispdata) 