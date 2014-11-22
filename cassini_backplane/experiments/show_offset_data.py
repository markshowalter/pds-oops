'''
Created on Apr 11, 2014

@author: rfrench
'''

import os
import random
import numpy as np
import numpy.ma as ma
import polymath
import oops.inst.cassini.iss as iss
import cspice
from imgdisp import *
import Tkinter as tk
from cb_config import *
from cb_offset import *
from cb_util_oops import *


def callback_mousemove(x, y, metadata):
    x = int(x)
    y = int(y)
    
    if (x < 0 or x >= metadata['img'].shape[1] or
        y < 0 or y >= metadata['img'].shape[0]):
        return
    
    if 'ring_longitude' in metadata:
        metadata['label_ring_longitude'].config(text=('%7.3f'%metadata['ring_longitude'][y,x]))
    if 'ring_radius' in metadata:
        metadata['label_ring_radius'].config(text=('%8d'%metadata['ring_radius'][y,x]))
    if 'ring_resolution' in metadata:
        metadata['label_ring_resolution'].config(text=('%7.3f'%metadata['ring_resolution'][y,x]))
    if 'ring_emission' in metadata:
        metadata['label_ring_emission'].config(text=('%7.3f'%metadata['ring_emission'][y,x]))
    if 'ring_incidence' in metadata:
        metadata['label_ring_incidence'].config(text=('%7.3f'%metadata['ring_incidence'][y,x]))
    if 'ring_phase' in metadata:
        metadata['label_ring_phase'].config(text=('%7.3f'%metadata['ring_phase'][y,x]))

    metadata['label_body_name_longitude'].config(text=('Moon Longitude:'))
    metadata['label_body_name_latitude'].config(text=('Moon Latitude:'))
    metadata['label_body_name_resolution'].config(text=('Moon Resolution:'))
    metadata['label_body_name_phase'].config(text=('Moon Phase:'))
    metadata['label_body_name_emission'].config(text=('Moon Emission:'))
    metadata['label_body_name_incidence'].config(text=('Moon Incidence:'))

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
            metadata['label_body_name_longitude'].config(text=(body_name[:5]+' Longitude:'))
            metadata['label_body_longitude'].config(text=('%7.3f'%val.vals))
        val = metadata[body_name+'_latitude'][y,x]
        if not val.masked():
            metadata['label_body_name_latitude'].config(text=(body_name[:5]+' Latitude:'))
            metadata['label_body_latitude'].config(text=('%7.3f'%val.vals))
            val = metadata[body_name+'_resolution']
            if not val.masked():
                metadata['label_body_name_resolution'].config(text=(body_name[:5]+' Resolution:'))
                metadata['label_body_resolution'].config(text=('%7.3f'%val.vals))
        val = metadata[body_name+'_phase'][y,x]
        if not val.masked():
            metadata['label_body_name_phase'].config(text=(body_name[:5]+' Phase:'))
            metadata['label_body_phase'].config(text=('%7.3f'%val.vals))
        val = metadata[body_name+'_emission'][y,x]
        if not val.masked():
            metadata['label_body_name_emission'].config(text=(body_name[:5]+' Emission:'))
            metadata['label_body_emission'].config(text=('%7.3f'%val.vals))
        val = metadata[body_name+'_incidence'][y,x]
        if not val.masked():
            metadata['label_body_name_incidence'].config(text=(body_name[:5]+' Incidence:'))
            metadata['label_body_incidence'].config(text=('%7.3f'%val.vals))
        
def process_image(filename, interactive=True, **kwargs):
    print filename
    
    obs = iss.from_file(filename)
    print filename
    print 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', 
    print obs.filter1, obs.filter2

    offset_u, offset_v, metadata = master_find_offset(obs, create_overlay=True,
                                                      **kwargs)

    metadata['img'] = obs.data
    
    ext_data = metadata['ext_data']
    overlay = metadata['ext_overlay']
    
    if ext_data is None:
        print 'WEIRD FAILURE - NO EXT_DATA'
        return
    
    if not interactive:
        return
    
    ext_u = (ext_data.shape[1]-obs.data.shape[1])/2
    ext_v = (ext_data.shape[0]-obs.data.shape[0])/2
    metadata['ext_u'] = ext_u
    metadata['ext_v'] = ext_v

    if offset_u is not None:
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(offset_u,offset_v))

    set_obs_bp(obs)
    
    metadata['ring_radius'] = obs.bp.ring_radius('saturn:ring').vals.astype('float')
    metadata['ring_longitude'] = obs.bp.ring_longitude('saturn:ring').vals.astype('float')
    metadata['ring_resolution'] = obs.bp.ring_radial_resolution('saturn:ring').vals.astype('float')
    metadata['ring_phase'] = obs.bp.phase_angle('saturn:ring').vals.astype('float') * oops.DPR
    metadata['ring_emission'] = obs.bp.emission_angle('saturn:ring').vals.astype('float') * oops.DPR
    metadata['ring_incidence'] = obs.bp.incidence_angle('saturn:ring').vals.astype('float') * oops.DPR 

    large_body_dict = obs.inventory(LARGE_BODY_LIST, return_type='full')
    metadata['bodies'] = large_body_dict

    for body_name in large_body_dict:
        body = large_body_dict[body_name]
        u_min = body['u_min']-1
        u_max = body['u_max']+1
        v_min = body['v_min']-1
        v_max = body['v_max']+1
        u_min = np.clip(u_min, -ext_u, obs.data.shape[1]+ext_u-1)
        u_max = np.clip(u_max, -ext_u, obs.data.shape[1]+ext_u-1)
        v_min = np.clip(v_min, -ext_v, obs.data.shape[0]+ext_v-1)
        v_max = np.clip(v_max, -ext_v, obs.data.shape[0]+ext_v-1)
        
        meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(u_min+.5, v_min+.5),
                                         limit =(u_max+.5, v_max+.5),
                                         swap  =True)
    
        restr_bp = oops.Backplane(obs, meshgrid=meshgrid)

        bp_latitude = restr_bp.latitude(body_name, lat_type='graphic') * oops.DPR
        model = polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).mask_where_eq(0)
        model[v_min+ext_v:v_max+ext_v+1,
              u_min+ext_u:u_max+ext_u+1] = bp_latitude
        metadata[body_name+'_latitude'] = model
        bp_longitude = restr_bp.longitude(body_name, direction='west') * oops.DPR
        model = polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).mask_where_eq(0)
        model[v_min+ext_v:v_max+ext_v+1,
              u_min+ext_u:u_max+ext_u+1] = bp_longitude    
        metadata[body_name+'_longitude'] = model
        bp_resolution = restr_bp.center_resolution(body_name) # Scalar
        metadata[body_name+'_resolution'] = bp_resolution
        bp_phase = restr_bp.phase_angle(body_name) * oops.DPR
        model = polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).mask_where_eq(0)
        model[v_min+ext_v:v_max+ext_v+1,
              u_min+ext_u:u_max+ext_u+1] = bp_phase
        metadata[body_name+'_phase'] = model
        bp_emission = restr_bp.emission_angle(body_name) * oops.DPR
        model = polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).mask_where_eq(0)
        model[v_min+ext_v:v_max+ext_v+1,
              u_min+ext_u:u_max+ext_u+1] = bp_emission
        metadata[body_name+'_emission'] = model
        bp_incidence = restr_bp.incidence_angle(body_name) * oops.DPR
        model = polymath.Scalar.as_scalar(np.zeros(ext_data.shape)).mask_where_eq(0)
        model[v_min+ext_v:v_max+ext_v+1,
              u_min+ext_u:u_max+ext_u+1] = bp_incidence
        metadata[body_name+'_incidence'] = model

    toplevel = Tk()
    toplevel.title(filename)
    frame_toplevel = Frame(toplevel)

#    overlay[:,:,0] = (ext_data-np.min(ext_data)) / (np.max(ext_data)-np.min(ext_data)) * 255.
    
    imgdisp = ImageDisp([ext_data], [overlay], canvas_size=(1024,768), 
                        parent=frame_toplevel, allow_enlarge=True,
                        auto_update=True, origin=(ext_u,ext_v))

    gridrow = 0
    gridcolumn = 0

    label_width = 14
    val_width = 9

    label = Label(imgdisp.addon_control_frame, text='Final Offset:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['final_offset_u'])+', '+str(metadata['final_offset_v']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='', anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+2, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='Star Offset:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['star_offset_u'])+', '+str(metadata['star_offset_v']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='', anchor='w', width=3)
    label.grid(row=gridrow, column=gridcolumn+5, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='Model Offset:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['model_offset_u'])+', '+str(metadata['model_offset_v']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky=W)
    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Used Objects:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['used_objects_type'].upper()),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky=W)

    contents_str = ''
    for s in metadata['model_contents']:
        contents_str += s[0].upper()
        
    label = Label(imgdisp.addon_control_frame, text='Model Contents:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=contents_str,
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='Model Overrides:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['model_overrides_stars']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky=W)

    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Body Only:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['body_only']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+1, sticky=W)

    label = Label(imgdisp.addon_control_frame, text='Rings Only:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=str(metadata['rings_only']),
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+4, sticky=W)

    shadow_bodies = ''
    
    for body_name in metadata['rings_shadow_bodies']:
        shadow_bodies += body_name.upper()[:2] + ' '
        
    label = Label(imgdisp.addon_control_frame, text='Rings Shadows:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    label = Label(imgdisp.addon_control_frame, text=shadow_bodies,
                  anchor='e', width=val_width)
    label.grid(row=gridrow, column=gridcolumn+7, sticky=W)

    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Ring Longitude:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_ring_longitude = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_ring_longitude'] = label_ring_longitude

    label = Label(imgdisp.addon_control_frame, text='Ring Radius:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label_ring_radius = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_radius.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_ring_radius'] = label_ring_radius

    label = Label(imgdisp.addon_control_frame, text='Ring Radial Res:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    label_ring_resolution = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_resolution.grid(row=gridrow, column=gridcolumn+7, sticky=W)
    metadata['label_ring_resolution'] = label_ring_resolution
    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Ring Phase:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    label_ring_phase = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_ring_phase'] = label_ring_phase

    label = Label(imgdisp.addon_control_frame, text='Ring Emission:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    label_ring_emission = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_emission.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_ring_emission'] = label_ring_emission

    label = Label(imgdisp.addon_control_frame, text='Ring Incidence:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    label_ring_incidence = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_ring_incidence.grid(row=gridrow, column=gridcolumn+7, sticky=W)
    metadata['label_ring_incidence'] = label_ring_incidence
    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Moon Longitude:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    metadata['label_body_name_longitude'] = label
    label_longitude = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_body_longitude'] = label_longitude

    label = Label(imgdisp.addon_control_frame, text='Moon Latitude:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    metadata['label_body_name_latitude'] = label
    label_latitude = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_latitude.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_body_latitude'] = label_latitude

    label = Label(imgdisp.addon_control_frame, text='Moon Resolution:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    metadata['label_body_name_resolution'] = label
    label_resolution = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_resolution.grid(row=gridrow, column=gridcolumn+7, sticky=W)
    metadata['label_body_resolution'] = label_resolution
    gridrow += 1

    label = Label(imgdisp.addon_control_frame, text='Moon Phase:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    metadata['label_body_name_phase'] = label
    label_phase = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    metadata['label_body_phase'] = label_phase

    label = Label(imgdisp.addon_control_frame, text='Moon Emission:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+3, sticky=W)
    metadata['label_body_name_emission'] = label
    label_emission = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_emission.grid(row=gridrow, column=gridcolumn+4, sticky=W)
    metadata['label_body_emission'] = label_emission

    label = Label(imgdisp.addon_control_frame, text='Moon Incidence:', anchor='w', width=label_width)
    label.grid(row=gridrow, column=gridcolumn+6, sticky=W)
    metadata['label_body_name_incidence'] = label
    label_incidence = Label(imgdisp.addon_control_frame, text='', anchor='e', width=val_width)
    label_incidence.grid(row=gridrow, column=gridcolumn+7, sticky=W)
    metadata['label_body_incidence'] = label_incidence
    gridrow += 1


        
    callback_mousemove_func = (lambda x, y, metadata=metadata:
                               callback_mousemove(x, y, metadata))
    imgdisp.bind_mousemove(0, callback_mousemove_func)

    frame_toplevel.pack()

    tk.mainloop()


def process_random_file(root_path):
    filenames = sorted(os.listdir(root_path))
    dir_list = []
    filename_list = []
    for filename in filenames:
        full_path = os.path.join(root_path, filename)
        if os.path.isdir(full_path):
            dir_list.append(filename)
        else:
            if filename[-4:] == '.IMG':
                filename_list.append(filename)
    if len(dir_list) == 0:
        if len(filename_list) == 0:
            assert False
        file_no = random.randint(0, len(filename_list)-1)
        new_file = os.path.join(root_path, filename_list[file_no])
#        try:
        process_image(new_file, interactive=True)
#        except:
#            print new_file
#            print 'THREW EXCEPTION'
    else:
        dir_no = random.randint(0, len(dir_list)-1)
        new_dir = os.path.join(root_path, dir_list[dir_no])
        process_random_file(new_dir)

# OK
# F ring and main rings - lots of stars
# DATA SIZE (1024L, 1024L) TEXP 2.6 FILTERS CL1 CL2
#process_image(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1624836945_1625069379/N1624900314_1_CALIB.IMG')

# XXX            
# Star field through filters - Insufficient stars
# DATA SIZE (1024L, 1024L) TEXP 0.26 FILTERS BL1 GRN
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2068\data\1683279174_1683355540\N1683354649_1_CALIB.IMG')

# XXX
# Star field long exposure with Enceladus - star matching doesn't work
# DATA SIZE (1024L, 1024L) TEXP 680.0 FILTERS CL1 UV3
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2051\data\1608970573_1609104344\N1608970573_1_CALIB.IMG')

# OK
# Star field long exposure
# DATA SIZE (512L, 512L) TEXP 82.0 FILTERS CL1 GRN
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2053\data\1613598819_1613977956\N1613844349_1_CALIB.IMG')

##### SATURN

# XXX
# Saturn limb only
# DATA SIZE (1024L, 1024L) TEXP 18.0 FILTERS CL1 MT2
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2072\data\1704630602_1704833275\N1704832756_1_CALIB.IMG')

# Saturn plus rings and moons - should ignore Saturn for match - rings shadowed on Saturn
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2034\data\1564337660_1564348958\W1564345216_1_CALIB.IMG')

# Saturn closeup with rings edge on
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2021\data\1520611263_1520675998\N1520674835_1_CALIB.IMG')

# Cut off moon model - Tethys
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2069\data\1694646740_1694815141\N1694664125_1_CALIB.IMG')

# Moon terminator along edge of image
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2013\data\1498654338_1498712236\W1498658693_1_CALIB.IMG')

# All rings but bad correlation
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2080\data\1738387955_1738409545\N1738395265_1_CALIB.IMG')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2028\data\1546797712_1546867749\N1546863063_1_CALIB.IMG')

# Dione
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2027\data\1544835042_1544908839\N1544892287_1_CALIB.IMG')

# Hyperion rotated incorrectly - rotation is chaotic
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2023\data\1530158136_1530199809\N1530185228_1_CALIB.IMG')

# Saturn and full rings with shadow
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2001\data\1458023716_1458209270\N1458112889_1_CALIB.IMG', interactive=True)

# F ring with A ring
#process_image('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542047596_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054271_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054716_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_039RF_FMOVIE001_VIMS\N1551254314_1_CALIB.IMG', interactive=True)

# F ring without A ring
# FAILS
#process_image('T:/external/cassini/derived/COISS_2xxx/COISS_2031/data/1555449244_1555593613/N1555565413_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_059RF_FMOVIE001_VIMS\N1581945338_1_CALIB.IMG', interactive=True)

# All rings
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2055\data\1622272893_1622549559\N1622394132_1_CALIB.IMG')




# Phoebe
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2003/data/1465650156_1465674412/N1465650307_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2004/data/1465674475_1465709620/N1465679337_2_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2004/data/1465674475_1465709620/N1465677386_2_CALIB.IMG', interactive=True)

# Mimas
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2014/data/1501618408_1501647096/N1501630117_1_CALIB.IMG', interactive=True, use_lambert=False)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2027/data/1542749662_1542807100/N1542756630_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2027/data/1542749662_1542807100/N1542758143_1_CALIB.IMG', interactive=True, use_lambert=False)

# Enceladus
process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2009/data/1487182149_1487415680/N1487300482_1_CALIB.IMG', interactive=True)
process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2019/data/1516036945_1516171123/N1516168806_1_CALIB.IMG', interactive=True)
process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2047/data/1597149262_1597186268/N1597179218_2_CALIB.IMG', interactive=True)

# Tethys
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2007/data/1477601077_1477653092/N1477639356_1_CALIB.IMG', interactive=True)

# Bad DN calibration - DN is way too high!
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2012\data\1497164137_1497406737\N1497238879_1_CALIB.IMG')

# Mimas and Tethys + rings at a distance
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2032\data\1559710457_1559931672\W1559730511_1_CALIB.IMG')

# Rings and Pandora - offset from real position as seen by stars
#process_image('t:/external/cassini/derived/COISS_2xxx\COISS_2009\data\1484846724_1485147239\N1484916376_1_CALIB.IMG')

# A ring- too straight, shadow
#process_image(r'T:\external\cassini\derived\COISS_2xxx\COISS_2055/data/1622711732_1623166344/N1623166278_1_CALIB.IMG')

# Used to crash in finding ring shadow
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2008\data\1481719190_1481724981\W1481719983_2_CALIB.IMG', allow_stars=False, allow_moons=False, allow_saturn=False)

# Good starfield - IR1 filter
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2049\data\1602565432_1602671572\N1602583170_5_CALIB.IMG')

# Good star matching with F ring
#process_image(r't:/clumps/data/ISS_055RF_FMOVIE001_VIMS/N1577809417_1_CALIB.IMG')

# Good star matching 512x512 CLEAR
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2066\data\1671705825_1671884383\N1671716657_1_CALIB.IMG')

# Stars through the rings - doesn't work
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2054\data\1620034503_1620036662\N1620036101_1_CALIB.IMG')

# MT2+IRP0 used to crash
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2004\data\1466194667_1466198705\W1466197722_1_CALIB.IMG')

# Thinks Saturn present but it's not
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2063\data\1656229607_1656238823\N1656238607_1_CALIB.IMG')

# Saturn with ring shadows
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2043\data\1585391197_1585519454\W1585492454_1_CALIB.IMG')

# Saturn occluding rings but the rings are showing through
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2080\data\1738716783_1738795297\W1738780648_1_CALIB.IMG')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2012\data\1497079709_1497122802\W1497120023_1_CALIB.IMG')

# Star field doesn't meet photometry
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2071\data\1696329488_1696440486\N1696331406_1_CALIB.IMG')


# High-res rings - unlit
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2007/data/1477517306_1477600896/N1477600536_1_CALIB.IMG', rings_model_source='uvis')

# High-res rings - lit
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2004/data/1466584989_1467427246/N1467351187_2_CALIB.IMG', rings_model_source='uvis') 
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2044/data/1585780929_1585829570/N1585802590_1_CALIB.IMG', rings_model_source='uvis')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2054/data/1617917998_1618066143/N1617918718_1_CALIB.IMG', rings_model_source='uvis')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2056/data/1627301233_1627319153/N1627310905_1_CALIB.IMG')

# Pan shadow
#process_image(r'T:/external/cassini/derived/COISS_2xxx/COISS_2053/data/1613001873_1613171522/N1613101588_1_CALIB.IMG', rings_model_source='uvis', allow_stars=False)
  
#while True:
#    process_random_file('t:/external/cassini/derived/COISS_2xxx')




#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2042\data\1580930973_1581152891\W1581143861_1_CALIB.IMG')


# TODO:
#       Optimize speed for rings - are there any rings in the display in the first place?
#       Optimize stars - try with stars, then try with bright parts blocked out
#       How do we know if the model fits at all? Some of the A ring ones are way off the edge
#       Optimize star searching for WAC
#       Figure out max magnitude for a given TEXP
#       Figure out max distance for a small moon to be visible
#       If model of moon goes to the edge, we have a problem
#       Terminator instead of limb
#       Detected offset at edge of range is bad
#       Stars should pay attention to opacity of the rings
