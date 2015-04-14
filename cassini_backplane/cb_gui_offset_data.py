###############################################################################
# cb_gui_offset_data.py
#
# Interactively display offset metadata.
#
# Exported routines:
#    display_offset_data
###############################################################################

import cb_logging
import logging

import os
import numpy as np
import polymath

from imgdisp import *
import Tkinter as tk

import oops

from cb_config import *
from cb_util_oops import *

def _callback_mousemove(x, y, metadata):
    x = int(x)
    y = int(y)

    x += metadata['ext_u']
    y += metadata['ext_v']

    if (x < 0 or x >= metadata['ext_data'].shape[1] or
        y < 0 or y >= metadata['ext_data'].shape[0]):
        return
        
    if 'ring_longitude' in metadata:
        (metadata['label_ring_longitude'].config(text=
                               ('%7.3f'%metadata['ring_longitude'][y,x])))
        (metadata['label_ring_radius'].config(text=
                               ('%8d'%metadata['ring_radius'][y,x])))
        (metadata['label_ring_resolution'].config(text=
                               ('%7.3f'%metadata['ring_resolution'][y,x])))
        (metadata['label_ring_emission'].config(text=
                               ('%7.3f'%metadata['ring_emission'][y,x])))
        (metadata['label_ring_incidence'].config(text=
                               ('%7.3f'%metadata['ring_incidence'][y,x])))
        (metadata['label_ring_phase'].config(text=
                               ('%7.3f'%metadata['ring_phase'][y,x])))

    if 'label_body_name_longitude' in metadata:
        metadata['label_body_name_longitude'].config(text=
                           ('Body '+metadata['longitude_type']+' Long:'))
        metadata['label_body_name_latitude'].config(text=
                           ('Body '+metadata['latitude_type']+' Lat:'))
        (metadata['label_body_name_resolution'].config(text=
                                                       ('Body Resolution:')))
        metadata['label_body_name_phase'].config(text=('Body Phase:'))
        metadata['label_body_name_emission'].config(text=('Body Emission:'))
        metadata['label_body_name_incidence'].config(text=('Body Incidence:'))
    
        metadata['label_body_longitude'].config(text=('  N/A  '))
        metadata['label_body_latitude'].config(text=('  N/A  '))
        metadata['label_body_resolution'].config(text=('  N/A  '))
        metadata['label_body_phase'].config(text=('  N/A  '))
        metadata['label_body_emission'].config(text=('  N/A  '))
        metadata['label_body_incidence'].config(text=('  N/A  '))

        large_body_dict = metadata['bodies']
        for body_name in sorted(large_body_dict.keys()):
            val = metadata[body_name+'_longitude'][y,x]
            if not val.masked():
                metadata['label_body_name_longitude'].config(text=
                  (body_name[:5]+' '+metadata['longitude_type']+' Long:'))
                metadata['label_body_longitude'].config(text=
                                         ('%7.3f'%val.vals))
            val = metadata[body_name+'_latitude'][y,x]
            if not val.masked():
                metadata['label_body_name_latitude'].config(text=
                  (body_name[:5]+' '+metadata['latitude_type']+' Lat:'))
                metadata['label_body_latitude'].config(text=
                                         ('%7.3f'%val.vals))
            val = metadata[body_name+'_resolution'][y,x]
            if not val.masked():
                metadata['label_body_name_resolution'].config(text=
                                     (body_name[:5]+' Resolution:'))
                metadata['label_body_resolution'].config(text=
                                     ('%7.3f'%val.vals))
            val = metadata[body_name+'_phase'][y,x]
            if not val.masked():
                metadata['label_body_name_phase'].config(text=
                                         (body_name[:5]+' Phase:'))
                metadata['label_body_phase'].config(text=
                                         ('%7.3f'%val.vals))
            val = metadata[body_name+'_emission'][y,x]
            if not val.masked():
                metadata['label_body_name_emission'].config(text=
                                         (body_name[:5]+' Emission:'))
                metadata['label_body_emission'].config(text=
                                         ('%7.3f'%val.vals))
            val = metadata[body_name+'_incidence'][y,x]
            if not val.masked():
                metadata['label_body_name_incidence'].config(text=
                                         (body_name[:5]+' Incidence:'))
                metadata['label_body_incidence'].config(text=
                                         ('%7.3f'%val.vals))
        
def display_offset_data(obs, metadata, show_rings=True, show_bodies=True,
                        latlon_type='centric', lon_direction='east'):
    metadata = metadata.copy() # Don't mutate the one given to us
    
    offset = metadata['offset']
    metadata['img'] = obs.data
    
    if 'ext_data' not in metadata:
        ext_data = obs.data
        metadata['ext_data'] = ext_data
        overlay = metadata['overlay']
    else:
        ext_data = metadata['ext_data']
        overlay = metadata['ext_overlay']
    stars_metadata = metadata['stars_metadata']
    
    ext_u = (ext_data.shape[1]-obs.data.shape[1])/2
    ext_v = (ext_data.shape[0]-obs.data.shape[0])/2
    metadata['ext_u'] = ext_u
    metadata['ext_v'] = ext_v

    orig_fov = obs.fov
    if offset is not None:
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)

    set_obs_ext_bp(obs, (ext_u, ext_v))
    
    if show_rings:
        bp_ring_radius = obs.ext_bp.ring_radius('saturn:ring')
        if not np.all(bp_ring_radius.mask):
            bp_ring_radius = bp_ring_radius.vals.astype('float')
            metadata['ring_radius'] = bp_ring_radius
            metadata['ring_longitude'] = (
                obs.ext_bp.ring_longitude('saturn:ring').vals.astype('float'))
            metadata['ring_resolution'] = (
                obs.ext_bp.ring_radial_resolution('saturn:ring').vals.
                                                            astype('float'))
            metadata['ring_phase'] = (
                obs.ext_bp.phase_angle('saturn:ring').vals.astype('float') * 
                                                                oops.DPR)
            metadata['ring_emission'] = (
                obs.ext_bp.emission_angle('saturn:ring').vals.astype('float') *
                                                                oops.DPR)
            metadata['ring_incidence'] = (
                obs.ext_bp.incidence_angle('saturn:ring').vals.astype('float')*
                                                                oops.DPR) 

    if show_bodies:
        large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
        metadata['bodies'] = large_body_dict
    
        large_bodies_by_range = [(x, large_body_dict[x]) 
                                 for x in large_body_dict]
        large_bodies_by_range.sort(key=lambda x: x[1]['range'])

        # Mask used to handle bodies being in front of and blocking other bodies
        mask = None
        
        # Start with the closest body and work into the distance
        for body_name, inv in large_bodies_by_range:
            body = large_body_dict[body_name]
            u_min = body['u_min_unclipped']-1
            u_max = body['u_max_unclipped']+1
            v_min = body['v_min_unclipped']-1
            v_max = body['v_max_unclipped']+1
            u_min = np.clip(u_min, -ext_u, obs.data.shape[1]+ext_u-1)
            u_max = np.clip(u_max, -ext_u, obs.data.shape[1]+ext_u-1)
            v_min = np.clip(v_min, -ext_v, obs.data.shape[0]+ext_v-1)
            v_max = np.clip(v_max, -ext_v, obs.data.shape[0]+ext_v-1)
            # Things break if the moon is only a single pixel wide or tall
            if u_min == u_max and u_min == obs.data.shape[1]+ext_u-1:
                u_min -= 1
            if u_min == u_max and u_min == -ext_u:
                u_max += 1
            if v_min == v_max and v_min == obs.data.shape[0]+ext_v-1:
                v_min -= 1
            if v_min == v_max and v_min == -ext_v:
                v_max += 1

            meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                             origin=(u_min+.5, v_min+.5),
                                             limit =(u_max+.5, v_max+.5),
                                             swap  =True)
        
            restr_bp = oops.Backplane(obs, meshgrid=meshgrid)
    
            bp_latitude = (restr_bp.latitude(body_name, lat_type=latlon_type) * 
                           oops.DPR)
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_latitude
    
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_latitude'] = model
            bp_longitude = restr_bp.longitude(body_name, 
                                              direction=lon_direction,
                                              lon_type=latlon_type) * oops.DPR
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_longitude    
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_longitude'] = model
            bp_phase = restr_bp.phase_angle(body_name) * oops.DPR
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_phase
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_phase'] = model
            bp_emission = restr_bp.emission_angle(body_name)

            bp_resolution = restr_bp.center_resolution(body_name) # Single scalar
            bp_resolution = bp_resolution / bp_emission.cos()
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_resolution
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_resolution'] = model

            bp_emission *= oops.DPR
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_emission
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_emission'] = model
            bp_incidence = restr_bp.incidence_angle(body_name) * oops.DPR
            model = (polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).
                     mask_where_eq(0))
            model[v_min+ext_v:v_max+ext_v+1,
                  u_min+ext_u:u_max+ext_u+1] = bp_incidence
            orig_model_mask = model.mask
            if mask is not None:
                model = model.mask_where(mask)
            metadata[body_name+'_incidence'] = model
            if mask is None:
                mask = ~orig_model_mask
            else:
                mask = ~orig_model_mask | mask 

    toplevel = tk.Toplevel()
    path1, fn1 = os.path.split(obs.full_path)
    path2, fn2 = os.path.split(path1)
    path3, fn3 = os.path.split(path2)
    path4, fn4 = os.path.split(path3)
    if fn3 == 'data':
        title = fn4 + '/' + fn2 + '/' + fn1
    else:
        title = fn4 + '/' + fn3 + '/' + fn2 + '/' + fn1
    toplevel.title(title)

    imgdisp = ImageDisp([ext_data], [overlay], canvas_size=(1024,512), 
                        parent=toplevel,
                        allow_enlarge=True,
                        auto_update=True, origin=(ext_u,ext_v))

    gridrow = 0
    gridcolumn = 0

    label_width = 15
    val_width = 9

    addon_control_frame = imgdisp.addon_control_frame

    label = tk.Label(addon_control_frame, text='Filter 1:',
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky='w')
    label = tk.Label(addon_control_frame, text=obs.filter1,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')

    label = tk.Label(addon_control_frame, text='', 
                     anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    label = tk.Label(addon_control_frame, text='Filter 2:',
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
    label = tk.Label(addon_control_frame, text=obs.filter2,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    
    label = tk.Label(addon_control_frame, text='Texp:',
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
    label = tk.Label(addon_control_frame, text=('%7.3f'%obs.texp),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    gridrow += 1
    
    label = tk.Label(addon_control_frame, text='Bootstrap Cand:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky='w')
    label = tk.Label(addon_control_frame, 
                     text=str(metadata['bootstrap_candidate']), 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')

    label = tk.Label(addon_control_frame, text='Bootstrapped:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
    label = tk.Label(addon_control_frame, 
                     text=str(metadata['bootstrapped']), 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')

    label = tk.Label(addon_control_frame, text='Bootstrap Mosaic:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
    mosaic_path = metadata['bootstrap_mosaic_path']
    if mosaic_path is not None:
        _, mosaic_path = os.path.split(mosaic_path)
    label = tk.Label(addon_control_frame,
                     text=mosaic_path,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    gridrow += 1
    
    label = tk.Label(addon_control_frame, text='Final Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky='w')
    if metadata['offset'] is None:
        offset = 'None'
    else:
        offset = str(metadata['offset'][0])+', '+str(metadata['offset'][1])
    label = tk.Label(addon_control_frame, text=offset, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')

    label = tk.Label(addon_control_frame, text='Star Offset:', 
                  anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
    if stars_metadata:
        if stars_metadata['offset'] is None:
            offset = 'None'
        else:
            offset = (str(stars_metadata['offset'][0])+', '+
                      str(stars_metadata['offset'][1]))
    else:
        offset = 'N/A'
    label = tk.Label(addon_control_frame, text=offset,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')

    label = tk.Label(addon_control_frame, text='', 
                     anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    if metadata['model_offset'] is None:
        offset = 'None'
    else:
        offset = (str(metadata['model_offset'][0])+', '+
                  str(metadata['model_offset'][1]))
    label = tk.Label(addon_control_frame, text='Model Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
    label = tk.Label(addon_control_frame, text=offset, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    gridrow += 1

    label = tk.Label(addon_control_frame, text='Used Objects:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky='w')
    label = tk.Label(addon_control_frame, 
                     text=str(metadata['used_objects_type']), 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')

    contents_str = ''
    for s in metadata['model_contents']:
        contents_str += s[0].upper()
        
    label = tk.Label(addon_control_frame, text='Model Contents:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
    label = tk.Label(addon_control_frame, text=contents_str,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')

    label = tk.Label(addon_control_frame, text='Model Overrides:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
    label = tk.Label(addon_control_frame, 
                     text=str(metadata['model_overrides_stars']),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')

    gridrow += 1

    label = tk.Label(addon_control_frame, text='Body Only:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky='w')
    label = tk.Label(addon_control_frame, text=str(metadata['body_only']),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')

    label = tk.Label(addon_control_frame, text='Rings Only:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
    label = tk.Label(addon_control_frame, 
                     text=str(metadata['rings_only']),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')

    shadow_bodies = ''
    
    if 'rings_shadow_bodies' in metadata:
        for body_name in metadata['rings_shadow_bodies']:
            shadow_bodies += body_name.upper()[:2] + ' '
        
    label = tk.Label(addon_control_frame, text='Ring Shadowed By:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
    label = tk.Label(addon_control_frame, text=shadow_bodies,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')

    gridrow += 1

    if show_rings:
        label = tk.Label(addon_control_frame, text='Ring Long:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn, sticky='w')
        label_ring_longitude = tk.Label(addon_control_frame, text='', 
                                        anchor='e', width=val_width)
        label_ring_longitude.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_ring_longitude'] = label_ring_longitude
    
        label = tk.Label(addon_control_frame, text='Ring Radius:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
        label_ring_radius = tk.Label(addon_control_frame, text='', 
                                     anchor='e', width=val_width)
        label_ring_radius.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_ring_radius'] = label_ring_radius
    
        label = tk.Label(addon_control_frame, text='Ring Radial Res:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
        label_ring_resolution = tk.Label(addon_control_frame, text='', 
                                         anchor='e', width=val_width)
        label_ring_resolution.grid(row=gridrow, column=gridcolumn+7,
                                   sticky='w')
        metadata['label_ring_resolution'] = label_ring_resolution
        
        gridrow += 1
    
        label = tk.Label(addon_control_frame, text='Ring Phase:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn, sticky='w')
        label_ring_phase = tk.Label(addon_control_frame, text='', 
                                    anchor='e', width=val_width)
        label_ring_phase.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_ring_phase'] = label_ring_phase
    
        label = tk.Label(addon_control_frame, text='Ring Emission:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
        label_ring_emission = tk.Label(addon_control_frame, text='', 
                                       anchor='e', width=val_width)
        label_ring_emission.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_ring_emission'] = label_ring_emission
    
        label = tk.Label(addon_control_frame, text='Ring Incidence:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
        label_ring_incidence = tk.Label(addon_control_frame, text='', 
                                        anchor='e', width=val_width)
        label_ring_incidence.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        metadata['label_ring_incidence'] = label_ring_incidence

        gridrow += 1

    if show_bodies:
        metadata['latitude_type'] = latlon_type[0].upper()
        label = tk.Label(addon_control_frame, 
                         text='Body Latitude '+metadata['latitude_type']+':', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn, sticky='w')
        metadata['label_body_name_latitude'] = label
        label_latitude = tk.Label(addon_control_frame, text='', 
                                  anchor='e', width=val_width)
        label_latitude.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_body_latitude'] = label_latitude
    
        metadata['longitude_type'] = (latlon_type[0].upper() + '/' +
                                      lon_direction[0].upper())
        label = tk.Label(addon_control_frame, 
                         text='Body Longitude '+metadata['longitude_type']+':', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
        metadata['label_body_name_longitude'] = label
        label_longitude = tk.Label(addon_control_frame, text='', 
                                   anchor='e', width=val_width)
        label_longitude.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_body_longitude'] = label_longitude
    
        label = tk.Label(addon_control_frame, text='Body Resolution:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
        metadata['label_body_name_resolution'] = label
        label_resolution = tk.Label(addon_control_frame, text='', 
                                    anchor='e', width=val_width)
        label_resolution.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        metadata['label_body_resolution'] = label_resolution
        
        gridrow += 1
    
        label = tk.Label(addon_control_frame, text='Body Phase:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn, sticky='w')
        metadata['label_body_name_phase'] = label
        label_phase = tk.Label(addon_control_frame, text='', 
                               anchor='e', width=val_width)
        label_phase.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_body_phase'] = label_phase
    
        label = tk.Label(addon_control_frame, text='Body Emission:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+3, sticky='w')
        metadata['label_body_name_emission'] = label
        label_emission = tk.Label(addon_control_frame, text='', 
                                  anchor='e', width=val_width)
        label_emission.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_body_emission'] = label_emission
    
        label = tk.Label(addon_control_frame, text='Body Incidence:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+6, sticky='w')
        metadata['label_body_name_incidence'] = label
        label_incidence = tk.Label(addon_control_frame, text='', 
                                   anchor='e', width=val_width)
        label_incidence.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        metadata['label_body_incidence'] = label_incidence

        gridrow += 1

    callback_mousemove_func = (lambda x, y, metadata=metadata:
                               _callback_mousemove(x, y, metadata))
    imgdisp.bind_mousemove(0, callback_mousemove_func)

    imgdisp.pack()
    
    tk.mainloop()

    obs.fov = orig_fov
    