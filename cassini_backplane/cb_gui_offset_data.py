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

_TKINTER_AVAILABLE = True
try:
    from imgdisp import *
    import Tkinter as tk
    import ttk
except ImportError:
    _TKINTER_AVAILABLE = False
    
import cspice
import oops

from cb_config import *
from cb_util_image import *
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
                               ('%12.3f'%metadata['ring_radius'][y,x])))
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
                        latlon_type='centric', lon_direction='east',
                        canvas_size=None,
                        interpolate_missing_stripes=False):
    assert _TKINTER_AVAILABLE

    metadata = metadata.copy() # Don't mutate the one given to us
    
    offset = metadata['offset']
    confidence = metadata['confidence']
    metadata['img'] = obs.data
    
    if 'ext_data' not in metadata:
        ext_data = obs.data
        metadata['ext_data'] = ext_data
        if 'overlay' in metadata:
            overlay = metadata['overlay']
        else:
            overlay = None
    else:
        ext_data = metadata['ext_data']
        overlay = metadata['ext_overlay']
    stars_metadata = metadata['stars_metadata']
    
    if interpolate_missing_stripes:
        ext_data = image_interpolate_missing_stripes(ext_data)
        
    ext_u = (ext_data.shape[1]-obs.data.shape[1])/2
    ext_v = (ext_data.shape[0]-obs.data.shape[0])/2
    metadata['ext_u'] = ext_u
    metadata['ext_v'] = ext_v

    orig_fov = obs.fov
    if offset is not None:
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)

    # Beware - this now has the offset FOV!
    set_obs_ext_bp(obs, (ext_u, ext_v), force=True)
    
    if show_rings:
        bp_ring_radius = obs.ext_bp.ring_radius('saturn:ring')
        if not np.all(bp_ring_radius.mask):
            bp_ring_radius = bp_ring_radius.mvals.astype('float')
            metadata['ring_radius'] = bp_ring_radius
            metadata['ring_longitude'] = (
                obs.ext_bp.ring_longitude('saturn:ring').mvals.astype('float') *
                                                                oops.DPR)
            metadata['ring_resolution'] = (
                obs.ext_bp.ring_radial_resolution('saturn:ring').mvals.
                                                            astype('float'))
            metadata['ring_phase'] = (
                obs.ext_bp.phase_angle('saturn:ring').mvals.astype('float') * 
                                                                oops.DPR)
            metadata['ring_emission'] = (
                obs.ext_bp.emission_angle('saturn:ring').mvals.astype('float') *
                                                                oops.DPR)
            metadata['ring_incidence'] = (
                obs.ext_bp.incidence_angle('saturn:ring').mvals.astype('float')*
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

    path1, fn1 = os.path.split(obs.full_path)
    path2, fn2 = os.path.split(path1)
    path3, fn3 = os.path.split(path2)
    path4, fn4 = os.path.split(path3)
    if fn3 == 'data':
        title = fn4 + '/' + fn2 + '/' + fn1
    else:
        title = fn4 + '/' + fn3 + '/' + fn2 + '/' + fn1
    
    if metadata['bootstrapped']:
        title += ' (BOOTSTRAPPED)'
        
    title += '    ' + cspice.et2utc(obs.midtime, 'C', 0)

    if canvas_size is None:
        canvas_size = (max(obs.data.shape[1], 1024),
                       min(obs.data.shape[0], 768))
    imgdisp = ImageDisp([ext_data], [overlay], canvas_size=canvas_size, 
                        title=title,
                        allow_enlarge=True,
                        auto_update=True, origin=(ext_u,ext_v))

    gridrow = 0
    gridcolumn = 0

    label_width = 17
    val_width = 17

    addon_control_frame = imgdisp.addon_control_frame

    label = ttk.Label(addon_control_frame, text='', 
                     anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+0, sticky='w')

    label = ttk.Label(addon_control_frame, text='Filters:',
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    label = ttk.Label(addon_control_frame, text=obs.filter1+'+'+obs.filter2,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    label = ttk.Label(addon_control_frame, text='', 
                     anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+3, sticky='w')

    label = ttk.Label(addon_control_frame, text='Texp:',
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, text=('%7.3f'%obs.texp),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    label = ttk.Label(addon_control_frame, text='', 
                     anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+6, sticky='w')

    gridrow += 1

    bootstrap_str = 'No'
    if metadata['bootstrap_candidate']:
        bootstrap_str = 'Yes'
        if (metadata['large_bodies'] is not None and
            len(metadata['large_bodies']) > 0):
            bootstrap_str += ' (' + metadata['large_bodies'][0][:2].capitalize()+')'

    label = ttk.Label(addon_control_frame, text='Bootstrap Cand:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    label = ttk.Label(addon_control_frame, 
                     text=bootstrap_str, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    label = ttk.Label(addon_control_frame, text='Bootstrapped:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, 
                     text=str(metadata['bootstrap_status']), 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

#    label = ttk.Label(addon_control_frame, text='Bootstrap Mosaic:', 
#                     anchor='w', width=label_width)
#    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
#
#    mosaic_path = metadata['bootstrap_mosaic_path']
#    if mosaic_path is None:
#        mosaic_path = 'N/A'
#    else:
#        _, mosaic_path = os.path.split(mosaic_path)
#    label = ttk.Label(addon_control_frame,
#                     text=mosaic_path,
#                     anchor='e', width=val_width)
#    label.grid(row=gridrow, column=gridcolumn+8, sticky='w')

    gridrow += 1
    
    sep = ttk.Separator(addon_control_frame)
    sep.grid(row=gridrow, column=1, columnspan=8, sticky='ew')
    gridrow += 1
    
    label = ttk.Label(addon_control_frame, text='Final Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    if offset is None:
        offset_str = 'None'
    else:
        offset_str = '%.2f, %.2f (%.2f)' % (offset[0], offset[1], confidence)
    label = ttk.Label(addon_control_frame, text=offset_str, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    label = ttk.Label(addon_control_frame, text='Offset Winner:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, 
                     text=str(metadata['offset_winner']), 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    contents_str = ''
    for s in metadata['model_contents']:
        contents_str += s[0].upper()
        
    label = ttk.Label(addon_control_frame, text='Model Contents:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    label = ttk.Label(addon_control_frame, text=contents_str,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+8, sticky='w')
    gridrow += 1

    label = ttk.Label(addon_control_frame, text='Star Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    if stars_metadata is not None:
        if metadata['stars_offset'] is None:
            offset = 'None'
        else:
            offset = '%.2f, %.2f (%.2f)' % (metadata['stars_offset'][0],
                                            metadata['stars_offset'][1],
                                            metadata['stars_confidence'])
    else:
        offset = 'N/A'
    label = ttk.Label(addon_control_frame, text=offset,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    if metadata['model_offset'] is None:
        offset = 'None'
    else:
        offset = '%d, %d (%.2f)' % (metadata['model_offset'][0],
                                    metadata['model_offset'][1],
                                    metadata['model_confidence'])
    label = ttk.Label(addon_control_frame, text='Model Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, text=offset, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    if metadata['titan_metadata'] is None:
        offset = 'N/A'
    elif metadata['titan_offset'] is None:
        offset = 'None'
    else:
        offset = '%d, %d (%.2f)' % (metadata['titan_offset'][0],
                                    metadata['titan_offset'][1],
                                    metadata['titan_confidence'])
    label = ttk.Label(addon_control_frame, text='Titan Offset:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    label = ttk.Label(addon_control_frame, text=offset, 
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+8, sticky='w')
    gridrow += 1

    label = ttk.Label(addon_control_frame, text='Body Only:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    label = ttk.Label(addon_control_frame, text=str(metadata['body_only']),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    label = ttk.Label(addon_control_frame, text='Rings Only:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, 
                     text=str(metadata['rings_only']),
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    rings_metadata = metadata['rings_metadata']
    
    shadow_bodies = 'N/A'
    
    if rings_metadata and 'shadow_bodies' in rings_metadata:
        shadow_bodies = ''
        for body_name in rings_metadata['shadow_bodies']:
            shadow_bodies += body_name.upper()[:2] + ' '

    label = ttk.Label(addon_control_frame, text='Ring Shadowed By:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    label = ttk.Label(addon_control_frame, text=shadow_bodies,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+8, sticky='w')

    gridrow += 1

    sep = ttk.Separator(addon_control_frame)
    sep.grid(row=gridrow, column=1, columnspan=8, sticky='ew')
    gridrow += 1
    
    fiducial_features = 'N/A'
    
    num_features = 'N/A'
    if rings_metadata is not None:
        num_features = rings_metadata['num_good_fiducial_features']
        fiducial_features = str(num_features)
        if rings_metadata['fiducial_features_ok']:
            fiducial_features += ' (OK)'
        else:
            fiducial_features += ' (NOT OK)'

    label = ttk.Label(addon_control_frame, text='Ring # Features:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    label = ttk.Label(addon_control_frame, text=fiducial_features,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')

    fiducial_blur = 'N/A'
    
    if rings_metadata and 'fiducial_blur' in rings_metadata:
        if rings_metadata['fiducial_blur'] is None:
            fiducial_blur = 'None'
        else:
            fiducial_blur = ('%.3f' % rings_metadata['fiducial_blur'])
        
    label = ttk.Label(addon_control_frame, text='Ring Blur:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
    label = ttk.Label(addon_control_frame, text=fiducial_blur,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+5, sticky='w')

    curvature_ok = 'N/A'
            
    if rings_metadata and 'curvature_ok' in rings_metadata:
        curvature_ok = str(rings_metadata['curvature_ok'])
        
    label = ttk.Label(addon_control_frame, text='Ring Curvature OK:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
    label = ttk.Label(addon_control_frame, text=curvature_ok,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+8, sticky='w')
        
    gridrow += 1

    emission_ok = 'N/A'
            
    if rings_metadata and 'emission_ok' in rings_metadata:
        emission_ok = str(rings_metadata['emission_ok'])
        
    label = ttk.Label(addon_control_frame, text='Ring Emission OK:', 
                     anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
    label = ttk.Label(addon_control_frame, text=emission_ok,
                     anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+2, sticky='w')
        
    gridrow += 1

    if show_rings:
        label = ttk.Label(addon_control_frame, text='Ring Long:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        label_ring_longitude = ttk.Label(addon_control_frame, text='', 
                                        anchor='e', width=val_width)
        label_ring_longitude.grid(row=gridrow, column=gridcolumn+2, sticky='w')
        metadata['label_ring_longitude'] = label_ring_longitude
    
        label = ttk.Label(addon_control_frame, text='Ring Radius:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        label_ring_radius = ttk.Label(addon_control_frame, text='', 
                                     anchor='e', width=val_width)
        label_ring_radius.grid(row=gridrow, column=gridcolumn+5, sticky='w')
        metadata['label_ring_radius'] = label_ring_radius
    
        label = ttk.Label(addon_control_frame, text='Ring Radial Res:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        label_ring_resolution = ttk.Label(addon_control_frame, text='', 
                                         anchor='e', width=val_width)
        label_ring_resolution.grid(row=gridrow, column=gridcolumn+8,
                                   sticky='w')
        metadata['label_ring_resolution'] = label_ring_resolution
        
        gridrow += 1
    
        label = ttk.Label(addon_control_frame, text='Ring Phase:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        label_ring_phase = ttk.Label(addon_control_frame, text='', 
                                    anchor='e', width=val_width)
        label_ring_phase.grid(row=gridrow, column=gridcolumn+2, sticky='w')
        metadata['label_ring_phase'] = label_ring_phase
    
        label = ttk.Label(addon_control_frame, text='Ring Emission:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        label_ring_emission = ttk.Label(addon_control_frame, text='', 
                                       anchor='e', width=val_width)
        label_ring_emission.grid(row=gridrow, column=gridcolumn+5, sticky='w')
        metadata['label_ring_emission'] = label_ring_emission
    
        label = ttk.Label(addon_control_frame, text='Ring Incidence:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        label_ring_incidence = ttk.Label(addon_control_frame, text='', 
                                        anchor='e', width=val_width)
        label_ring_incidence.grid(row=gridrow, column=gridcolumn+8, sticky='w')
        metadata['label_ring_incidence'] = label_ring_incidence

        gridrow += 1

    if show_bodies:
        sep = ttk.Separator(addon_control_frame)
        sep.grid(row=gridrow, column=1, columnspan=8, sticky='ew')
        gridrow += 1
    
        metadata['latitude_type'] = latlon_type[0].upper()
        label = ttk.Label(addon_control_frame, 
                         text='Body Latitude '+metadata['latitude_type']+':', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_body_name_latitude'] = label
        label_latitude = ttk.Label(addon_control_frame, text='', 
                                  anchor='e', width=val_width)
        label_latitude.grid(row=gridrow, column=gridcolumn+2, sticky='w')
        metadata['label_body_latitude'] = label_latitude
    
        metadata['longitude_type'] = (latlon_type[0].upper() + '/' +
                                      lon_direction[0].upper())
        label = ttk.Label(addon_control_frame, 
                         text='Body Longitude '+metadata['longitude_type']+':', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_body_name_longitude'] = label
        label_longitude = ttk.Label(addon_control_frame, text='', 
                                   anchor='e', width=val_width)
        label_longitude.grid(row=gridrow, column=gridcolumn+5, sticky='w')
        metadata['label_body_longitude'] = label_longitude
    
        label = ttk.Label(addon_control_frame, text='Body Resolution:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        metadata['label_body_name_resolution'] = label
        label_resolution = ttk.Label(addon_control_frame, text='', 
                                    anchor='e', width=val_width)
        label_resolution.grid(row=gridrow, column=gridcolumn+8, sticky='w')
        metadata['label_body_resolution'] = label_resolution
        
        gridrow += 1
    
        label = ttk.Label(addon_control_frame, text='Body Phase:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+1, sticky='w')
        metadata['label_body_name_phase'] = label
        label_phase = ttk.Label(addon_control_frame, text='', 
                               anchor='e', width=val_width)
        label_phase.grid(row=gridrow, column=gridcolumn+2, sticky='w')
        metadata['label_body_phase'] = label_phase
    
        label = ttk.Label(addon_control_frame, text='Body Emission:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+4, sticky='w')
        metadata['label_body_name_emission'] = label
        label_emission = ttk.Label(addon_control_frame, text='', 
                                  anchor='e', width=val_width)
        label_emission.grid(row=gridrow, column=gridcolumn+5, sticky='w')
        metadata['label_body_emission'] = label_emission
    
        label = ttk.Label(addon_control_frame, text='Body Incidence:', 
                         anchor='w', width=label_width)
        label.grid(row=gridrow, column=gridcolumn+7, sticky='w')
        metadata['label_body_name_incidence'] = label
        label_incidence = ttk.Label(addon_control_frame, text='', 
                                   anchor='e', width=val_width)
        label_incidence.grid(row=gridrow, column=gridcolumn+8, sticky='w')
        metadata['label_body_incidence'] = label_incidence

        gridrow += 1

    callback_mousemove_func = (lambda x, y, metadata=metadata:
                               _callback_mousemove(x, y, metadata))
    imgdisp.bind_mousemove(0, callback_mousemove_func)

    tk.mainloop()

    # Restore the original FOV
    obs.fov = orig_fov
    set_obs_ext_bp(obs, (ext_u, ext_v), force=True)
