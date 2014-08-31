'''
Created on Apr 11, 2014

@author: rfrench
'''

import numpy as np
import numpy.ma as ma
import oops.inst.cassini.iss as iss
import matplotlib.pyplot as plt
from imgdisp import *
import Tkinter as tk
from cb_offset import *
import os
import pickle

INTERACTIVE_OFFSET = True
INTERACTIVE_REPROJ = False
INTERACTIVE_MOSAIC = True

RADIUS_INNER = 139500 # km
RADIUS_OUTER = 141000 # km
RADIUS_RESOLUTION = 5 # km
LONGITUDE_RESOLUTION = 0.5 # degrees

ROOT_PATH = 't:/clumps/data'

RADIUS_PIXELS = int((RADIUS_OUTER-RADIUS_INNER) / RADIUS_RESOLUTION)

LONGITUDE_PIXELS = int(360 / LONGITUDE_RESOLUTION)

ROTATING_ET = cspice.utc2et("2007-1-1")
FRING_MEAN_MOTION = 581.964

def compute_longitude_shift(et): 
    return - (FRING_MEAN_MOTION * ((et - ROTATING_ET) / 86400.)) % 360.

def inertial_to_corotating(longitude, et):
    return (longitude + compute_longitude_shift(et)) % 360.

def corotating_to_inertial(co_long, et):
    return (co_long - compute_longitude_shift(et)) % 360.

def offset_filename(filename):
    return filename + '.OFF'

def write_offset(filename, offset_u, offset_v):
    off_filename = offset_filename(filename)
    offset_pickle_fp = open(off_filename, 'wb')
    pickle.dump((offset_u, offset_v), offset_pickle_fp)    
    offset_pickle_fp.close()

def read_offset(filename):
    off_filename = offset_filename(filename)
    if not os.path.exists(off_filename):
        return None, None 
    offset_pickle_fp = open(off_filename, 'rb')
    offset_u, offset_v = pickle.load(offset_pickle_fp)
    offset_pickle_fp.close()
    
    return offset_u, offset_v
    
def create_offset_one_file(filename):
    print 'Finding offset for', filename,

    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    obs.data = data
#     med = filt.median_filter(obs.data, 11)
#     perc = 50
#     mask = med > perc
#     obs.data[mask] = 0.

    print 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u, offset_v, ext_data, overlay = find_offset(obs, create_overlay=INTERACTIVE_OFFSET)

    print 'OFFSET', offset_u, offset_v
    
    write_offset(offset_filename(filename), offset_u, offset_v)
    
    if INTERACTIVE_OFFSET:
        imgdisp = ImageDisp([ext_data], [overlay], canvas_size=(1024,512), allow_enlarge=True)
        tk.mainloop()

def create_offset_obsid(obsid, suffix='_CALIB.IMG', max_files=10000, skip_files=0):
    abs_path = os.path.join(ROOT_PATH, obsid) 
    filenames = sorted(os.listdir(abs_path))
    for filename in filenames:
        full_path = os.path.join(abs_path, filename)
        if not os.path.isfile(full_path):
            continue
        if filename[-len(suffix):] != suffix:
            continue
        if skip_files:
            skip_files -= 1
            continue
        create_offset_one_file(full_path)

def reproject_one_file(filename):
    print 'Reprojecting', filename

    # XXX Convert to normal I/F
    
    offset_u, offset_v = read_offset(offset_filename(filename))
    if offset_u is None or offset_v is None:
        return None
    
    obs = iss.from_file(filename)
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(offset_u, offset_v))
    
    bp = oops.Backplane(obs)
    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
    bp_resolution = bp.ring_radial_resolution('saturn:ring').vals.astype('float')
    
    bp_longitude = inertial_to_corotating(bp_longitude, obs.midtime)

    min_longitude_pixel = (np.floor(np.min(bp_longitude) / LONGITUDE_RESOLUTION)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, LONGITUDE_PIXELS-1)
    max_longitude_pixel = (np.ceil(np.max(bp_longitude) / LONGITUDE_RESOLUTION)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, LONGITUDE_PIXELS-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    valid_radius = bp_radius.flatten()
    valid_longitude = bp_longitude.flatten()
    valid_data = obs.data.flatten()
    valid_resolution = np.mean(bp_resolution, axis=0) 
    valid_mean_longitude = np.mean(bp_longitude, axis=0)
    
    rad_bins = np.repeat(np.arange(RADIUS_PIXELS), num_longitude_pixel) # Bin numbers
    rad_bins_act = rad_bins * RADIUS_RESOLUTION + RADIUS_INNER      # Actual radius

    long_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1), RADIUS_PIXELS)  # Bin numbers
    long_bins_act = long_bins * LONGITUDE_RESOLUTION                # Actual longitude

    res_long_bins = np.arange(LONGITUDE_PIXELS)                     # Bin numbers
    res_long_bins_act = res_long_bins * LONGITUDE_RESOLUTION        # Actual longitude for resolution
    
    print 'RAD BINS RANGE', np.min(rad_bins_act), np.max(rad_bins_act)
    print 'LONG BINS RANGE', np.min(long_bins_act), np.max(long_bins_act)
    print 'RADIUS RANGE', np.min(valid_radius), np.max(valid_radius)
    print 'LONG RANGE', np.min(valid_longitude), np.max(valid_longitude)
    print 'RESOLUTION RANGE', np.min(valid_resolution), np.max(valid_resolution)
    print 'DATA RANGE', np.min(valid_data), np.max(valid_data)
    
    radlon_points = np.empty((rad_bins.shape[0], 2))
    radlon_points[:,0] = rad_bins_act
    radlon_points[:,1] = long_bins_act

    interp_data = interp.griddata((valid_radius, valid_longitude), valid_data,
                                  radlon_points, fill_value=1e300)
    interp_res = interp.griddata(valid_mean_longitude, valid_resolution,
                                 res_long_bins_act, fill_value=1e300)

    new_mosaic = np.zeros((RADIUS_PIXELS, LONGITUDE_PIXELS))+1e300
    
    new_mosaic[rad_bins,long_bins] = interp_data
    
    print 'NEW MOSAIC RANGE', np.min(interp_data), np.max(interp_data)

    temp_mosaic_data = new_mosaic.copy()
    temp_mosaic_data[temp_mosaic_data > 1000] = 0
    
    if INTERACTIVE_REPROJ:
        imgdisp = ImageDisp([temp_mosaic_data[::-1,:]], canvas_size=(1024,512), allow_enlarge=True, one_zoom=False)
        tk.mainloop()
    
    new_resolution = np.zeros(LONGITUDE_PIXELS)+1e300
    new_resolution[res_long_bins] = interp_res

    return new_mosaic, new_resolution

def create_mosaic_obsid(obsid, suffix='_CALIB.IMG', max_files=10000, skip_files=0):
    mosaic_data = np.zeros((RADIUS_PIXELS, LONGITUDE_PIXELS))+1e300
    mosaic_resolution = np.zeros(LONGITUDE_PIXELS)+1e300
    mosaic_valid_radius_count = np.zeros(LONGITUDE_PIXELS)
    
    abs_path = os.path.join(ROOT_PATH, obsid) 
    filenames = sorted(os.listdir(abs_path))
    for filename in filenames:
        full_path = os.path.join(abs_path, filename)
        if not os.path.isfile(full_path):
            continue
        if filename[-len(suffix):] != suffix:
            continue
        if skip_files:
            skip_files -= 1
            continue
        ret = reproject_one_file(full_path)
        if ret is None:
            continue
        new_mosaic, new_resolution = ret
        # Calculate number of good entries and where number is larger than before
        new_valid_radius_count = RADIUS_PIXELS-np.sum(new_mosaic == 1e300, axis=0)
        valid_radius_count_better_mask = new_valid_radius_count > mosaic_valid_radius_count
        valid_radius_count_equal_mask = new_valid_radius_count == mosaic_valid_radius_count
        # Calculate where the new resolution is better
        better_resolution_mask = new_resolution < mosaic_resolution
        # Final mask for which columns to replace mosaic values
        good_longitude_mask = np.logical_or(valid_radius_count_better_mask,
                    np.logical_and(valid_radius_count_equal_mask, better_resolution_mask))
    
        mosaic_data[:,good_longitude_mask] = new_mosaic[:,good_longitude_mask]
        mosaic_resolution[good_longitude_mask] = new_resolution[good_longitude_mask]
        mosaic_valid_radius_count[good_longitude_mask] = new_valid_radius_count[good_longitude_mask]

        temp_mosaic_data = mosaic_data.copy()
        temp_mosaic_data[temp_mosaic_data > 1000] = 0

        if INTERACTIVE_MOSAIC:
            imgdisp = ImageDisp([temp_mosaic_data[::-1,:]], canvas_size=(1024,512), allow_enlarge=True,
                                one_zoom=False)
            tk.mainloop()

# def process_plot_all_dir(abs_path):
#     filenames = sorted(os.listdir(abs_path))
#     for filename in filenames:
#         print filename
#         full_path = os.path.join(abs_path, filename)
#         if os.path.isdir(full_path):
#             if filename < 'ISS_007RI':
#                 continue
#             process_plot_dir(full_path, filename, max_files=50)

create_offset_obsid('ISS_036RF_FMOVIE002_VIMS', skip_files=0)
create_mosaic_obsid('ISS_036RF_FMOVIE002_VIMS')

# reproject_one_file(r't:/clumps/data\ISS_036RF_FMOVIE002_VIMS\N1546701061_1_CALIB.IMG')
# reproject_one_file(r't:/clumps/data\ISS_036RF_FMOVIE002_VIMS\N1546701434_1_CALIB.IMG')


# Limit longitude range to what's actually in the image
