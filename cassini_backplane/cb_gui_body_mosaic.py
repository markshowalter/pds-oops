###############################################################################
# cb_gui_body_mosaic.py
#
# Interactively display body mosaic metadata.
#
# Exported routines:
#    display_body_mosaic_data
###############################################################################

import cb_logging
import logging

import os
import numpy as np
import colorsys

from imgdisp import *
import Tkinter as tk

import oops

from cb_bodies import *
from cb_config import *
from cb_util_oops import *

def _command_refresh_color(metadata):
    color_sel = metadata['var_color_by'].get()
    imgdisp = metadata['imgdisp']
    full_mask = metadata['full_mask']

    if color_sel == 'none':
        imgdisp.set_overlay(0, None)
        return
    
    minval = None
    maxval = None
    
    if color_sel == 'imagenum':
        valsrc = metadata['image_number'].astype('float')
    elif color_sel == 'reltime':
        valsrc = metadata['time']
    elif color_sel == 'relresolution':
        valsrc = metadata['resolution']
    elif color_sel == 'relphase':
        valsrc = metadata['phase']
    elif color_sel == 'absphase':
        valsrc = metadata['phase']
        minval = 0.
        maxval = oops.PI
    elif color_sel == 'relemission':
        valsrc = metadata['emission']
    elif color_sel == 'absemission':
        valsrc = metadata['emission']
        minval = 0.
        maxval = oops.HALFPI
    elif color_sel == 'relincidence':
        valsrc = metadata['incidence']
    elif color_sel == 'absincidence':
        valsrc = metadata['incidence']
        minval = 0.
        maxval = oops.HALFPI
    
    if minval is None:
        minval = np.min(valsrc[full_mask])
        maxval = np.max(valsrc[full_mask])
    
    hsv = np.empty(valsrc.shape + (3,))
    hsv[:,:,0] = (1-(valsrc-minval)/(maxval-minval))*.66
    hsv[:,:,1] = 1
    hsv[:,:,2] = 1
    overlay = hsv_to_rgb(hsv)
    overlay[np.logical_not(full_mask), :] = 0

    imgdisp.set_overlay(0, overlay)

        
def _callback_mousemove(x, y, metadata):
    x = int(x)
    y = int(y)

    if (x < 0 or x >= metadata['img'].shape[1] or
        y < 0 or y >= metadata['img'].shape[0]):
        return

    metadata['label_longitude'].config(text=
                       ('%7.3f'%(metadata['longitude'][x] * oops.DPR)))
    metadata['label_latitude'].config(text=
                       ('%7.3f'%(metadata['latitude'][y] * oops.DPR)))
    
    full_mask = metadata['full_mask']
    if not full_mask[y,x]:  # Invalid pixel
        metadata['label_phase'].config(text='')
        metadata['label_incidence'].config(text='')
        metadata['label_emission'].config(text='')
        metadata['label_resolution'].config(text='')
        metadata['label_image_num'].config(text='')
        metadata['label_image'].config(text='')
        metadata['label_date'].config(text='')
    else:
        metadata['label_phase'].config(text=
                           ('%7.3f'%(metadata['phase'][y,x] * oops.DPR)))
        metadata['label_incidence'].config(text=
                           ('%7.3f'%(metadata['incidence'][y,x] * oops.DPR)))
        metadata['label_emission'].config(text=
                           ('%7.3f'%(metadata['emission'][y,x] * oops.DPR)))
        metadata['label_resolution'].config(text=
                           ('%7.3f'%metadata['resolution'][y,x]))
        metadata['label_image'].config(text=
               metadata['filename_list'][metadata['image_number'][y,x]][:13])
        metadata['label_image_num'].config(text=
                           ('%d'%metadata['image_number'][y,x]))
        metadata['label_date'].config(text=
                           cspice.et2utc(float(metadata['time'][y,x]), 'C', 0))
        
def display_body_mosaic(metadata, toplevel=None):
    metadata = metadata.copy() # Don't mutate the one given to us

    metadata['latitude'] = bodies_generate_latitudes(latitude_resolution=
                                    metadata['lat_resolution'])
    metadata['longitude'] = bodies_generate_longitudes(longitude_resolution=
                                    metadata['lon_resolution'])

    if toplevel is None:
        toplevel = Tk()
    toplevel.title(metadata['body_name'])

    imgdisp = ImageDisp([metadata['img']], canvas_size=(1024,400),
                        parent=toplevel, toplevel=toplevel,
                        allow_enlarge=True,
                        flip_y=True, one_zoom=False, auto_update=True)

    metadata['imgdisp'] = imgdisp
    
    gridrow = 0
    gridcolumn = 0

    label_width = 12
    val_width = 6
    val2_width = 18
    
    addon_control_frame = imgdisp.addon_control_frame

    label = Label(addon_control_frame, text='Image #:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_image_num = Label(addon_control_frame, text='',
                            anchor='e', width=val_width)
    label_image_num.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_image_num'] = label_image_num

    label = Label(addon_control_frame, text='', 
                  anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+2, sticky=W)

    label = Label(addon_control_frame, text='Image Name:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label_image = Label(addon_control_frame, text='',
                        anchor='e', width=val2_width)
    label_image.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_image'] = label_image
    
    label = Label(addon_control_frame, text='', 
                  anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+5, sticky=W)

    gridrow += 1
    
    label = Label(addon_control_frame, 
                  text='Latitude '+metadata['latlon_type'][0].upper()+':',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_latitude = Label(addon_control_frame, text='', 
                           anchor='e', width=val_width)
    label_latitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_latitude'] = label_latitude

    label = Label(addon_control_frame, text='Date:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label_date = Label(addon_control_frame, text='',
                       anchor='e', width=val2_width)
    label_date.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_date'] = label_date
    gridrow += 1
    
    label = Label(addon_control_frame, 
                  text='Longitude '+metadata['latlon_type'][0].upper()+'/'+
                       metadata['lon_direction'][0].upper()+':',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_longitude = Label(addon_control_frame, text='', 
                            anchor='e', width=val_width)
    label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_longitude'] = label_longitude
    gridrow += 1

    label = Label(addon_control_frame, text='Phase:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_phase = Label(addon_control_frame, text='',
                        anchor='e', width=val_width)
    label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_phase'] = label_phase
    gridrow += 1
    
    label = Label(addon_control_frame, text='Incidence:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_incidence = Label(addon_control_frame, text='',
                            anchor='e', width=val_width)
    label_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_incidence'] = label_incidence
    gridrow += 1

    label = Label(addon_control_frame, text='Emission:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_emission = Label(addon_control_frame, text='',
                           anchor='e', width=val_width)
    label_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_emission'] = label_emission
    gridrow += 1

    label = Label(addon_control_frame, text='Resolution:',
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_resolution = Label(addon_control_frame, text='',
                             anchor='e', width=val_width)
    label_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_resolution'] = label_resolution
    gridrow += 1
    
    gridrow = 0
    gridcolumn = 6

    var_color_by = StringVar()
    metadata['var_color_by'] = var_color_by
    refresh_color_func = lambda: _command_refresh_color(metadata)
    
    label = Label(addon_control_frame, text='Color by:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='None', 
                variable=var_color_by,
                value='none', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Image Number', 
                variable=var_color_by,
                value='imagenum', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Time', 
                variable=var_color_by,
                value='reltime', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Resolution', 
                variable=var_color_by,
                value='relresolution', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Phase', 
                variable=var_color_by,
                value='absphase', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Phase', 
                variable=var_color_by,
                value='relphase', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Emission', 
                variable=var_color_by,
                value='absemission', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Emission', 
                variable=var_color_by,
                value='relemission', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Incidence', 
                variable=var_color_by,
                value='absincidence', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Incidence', 
                variable=var_color_by,
                value='relincidence', command=refresh_color_func).grid(row=gridrow, column=gridcolumn, sticky=W)
    
    var_color_by.set('none')

    callback_mousemove_func = (lambda x, y, metadata=metadata:
                               _callback_mousemove(x, y, metadata))
    imgdisp.bind_mousemove(0, callback_mousemove_func)

    imgdisp.pack(side=LEFT)
    
    tk.mainloop()
    