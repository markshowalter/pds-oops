import copy
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.interpolate as interp

import imgdisp

import oops
import oops.inst.cassini.iss as iss

from cb_config import *
from cb_offset import *
from cb_util_file import *
from cb_util_image import *

TITAN_FILENAMES_CSV = os.path.join(SUPPORT_FILES_ROOT, 'titan', 
                                   'titan-file-list.csv')

FILTER_COLOR = {
    'CLEAR': (.7,.7,.7),
    
    'IR1': (0.5,0,0),
    'IR2': (0.6,0,0),
    'IR3': (0.7,0,0),
    'IR5': (0.8,0,0),
    'RED': (1,0,0),
    'GRN': (0,1,0),
    'BL1': (0,0,1),
    'VIO': (159/256.,0,1),
    
    'MT2': (1,204/256.,153/256.),
    'CB2': (1,.5,0),

    'MT3': (1,153/256.,204/256.),
    'CB3': (1,.3,1),
}    

FILTER_NUMBER = {
    'CLEAR': 0,
    
    'IR1': 1,
    'IR2': 2,
    'IR3': 3,
    'IR5': 4,
    'RED': 5,
    'GRN': 6,
    'BL1': 7,
    'VIO': 8,
    
    'MT2': 9,
    'CB2': 10,

    'MT3': 11,
    'CB3': 12,
}    

    
#===============================================================================
# 
# CREATE THE PROFILE FOR ONE IMAGE
#
#===============================================================================

def process_image(filename):
    print '>>> Computing profile for', filename

    _, filespec = os.path.split(filename)
    filespec = filespec.replace('_CALIB.IMG', '')
    
    pickle_path = os.path.join(SUPPORT_FILES_ROOT, 'titan', filespec+'.pickle')

    if os.path.exists(pickle_path):
        print 'Pickle file already exists'
        return
    
    full_filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
    obs = file_read_iss_file(full_filename)
    if obs.data.shape[0] < 512:
        print 'Skipping due to small image size'
        fp = open(pickle_path, 'wb')
        pickle.dump('Small image size', fp)
        fp.close()
        return

    if obs.filter1 == 'CL1' and obs.filter2 == 'CL2':
        filter = 'CLEAR'
    else:
        filter = obs.filter1
        if filter == 'CL1':
            filter = obs.filter2
        elif obs.filter2 != 'CL2':
            filter += '+' + obs.filter2

    # Titan parameters
    titan_inv_list = obs.inventory(['TITAN+ATMOSPHERE'], return_type='full')
    titan_inv = titan_inv_list['TITAN+ATMOSPHERE']
    titan_center = titan_inv['center_uv']
    titan_resolution = (titan_inv['resolution'][0]+
                        titan_inv['resolution'][1])/2
    titan_radius = titan_inv['outer_radius'] 
    titan_radius_pix = int(titan_radius / titan_resolution)

    # Backplanes
    set_obs_bp(obs)
    bp = obs.bp

    bp_incidence = bp.incidence_angle('TITAN+ATMOSPHERE') * oops.DPR
    full_mask = ma.getmaskarray(bp_incidence.mvals)
    
    # If the angles touch the edge, skip this image
    inv_mask = np.logical_not(full_mask)
    if (np.any(inv_mask[0,:]) or np.any(inv_mask[-1,:]) or
        np.any(inv_mask[:,0]) or np.any(inv_mask[:,-1])):
        print 'Skipping due to Titan off edge'
        fp = open(pickle_path, 'wb')
        pickle.dump('Titan off edge', fp)
        fp.close()
        return

    phase_angle = bp.center_phase_angle('TITAN+ATMOSPHERE').vals
    
    if phase_angle < oops.HALFPI:
        extreme_pos = np.argmin(bp_incidence.mvals)
    else:
        extreme_pos = np.argmax(bp_incidence.mvals)
    extreme_pos = np.unravel_index(extreme_pos, bp_incidence.mvals.shape)
    extreme_uv = (extreme_pos[1], extreme_pos[0])
    
    sun_angle = np.arctan2(extreme_uv[1]-titan_center[1], 
                           extreme_uv[0]-titan_center[0]) % oops.PI

    print 'PHASE', phase_angle * oops.DPR,
    print 'FILTER', filter, 'CENTER', titan_center, 'RADIUS', titan_radius,
    print 'RES', titan_resolution, 'RADIUS PIX', titan_radius_pix,
    print 'SUN ANGLE', sun_angle * oops.DPR

    bp_emission = bp.emission_angle('TITAN+ATMOSPHERE') * oops.DPR
    
    bp_incidence.vals[full_mask] = 0.
    bp_emission.vals[full_mask] = 0.

    # Create circles of incidence and emission
    total_intersect = None
    inc_list = []
    for inc_ang in np.arange(0, 180., 10):
        intersect = bp.border_atop(('incidence_angle', 'TITAN+ATMOSPHERE'), 
                                   inc_ang * oops.RPD).vals.astype('float')
        intersect[full_mask] = ma.masked
        inc_list.append(intersect)
        
    em_list = []
    for em_ang in np.arange(2.5, 90., 5):
        intersect = bp.border_atop(('emission_angle', 'TITAN+ATMOSPHERE'), 
                                   em_ang * oops.RPD).vals.astype('float')
        intersect[full_mask] = ma.masked
        em_list.append(intersect)

    ie_list = []

    gap_threshold = 10
    max_pixels = 10

    for inc_int in inc_list:
        for em_int in em_list:
            joint_intersect = np.logical_and(inc_int, em_int)
            pixels = np.where(joint_intersect) # in Y,X
            pixels = zip(*pixels) # List of (y,x) pairs
            # There have to be at least two points of intersection
            if len(pixels) < 2:
                continue
            pixels.sort() # Sort by increasing y then increasing x
            # There has to be a gap of "gap_threshold" in either x or y coords
            gap_y = None
            last_x = None
            last_y = None
            cluster_xy = None
            for y, x in pixels:
                if last_y is not None:
                    if y-last_y >= gap_threshold:
                        gap_y = True
                        cluster_xy = y
                        break
                last_y = y
            if gap_y is None:
                pixels.sort(key=lambda x: (x[1], x[0])) # Now sort by x then y
                for y, x in pixels:
                    if last_x is not None:
                        if abs(x-last_x) >= gap_threshold:
                            gap_y = False
                            cluster_xy = x
                            break
                    last_x = x
            if gap_y is None:
                # No gap!
                continue
            cluster1_list = []
            cluster2_list = []
            # pixels is sorted in the correct direction
            for pix in pixels:
                if ((gap_y and pix[0] >= cluster_xy) or 
                    (not gap_y and pix[1] >= cluster_xy)):
                    cluster2_list.append(pix)
                else:
                    cluster1_list.append(pix)
#            print cluster1_list
#            print cluster2_list
            if (len(cluster1_list) > max_pixels or 
                len(cluster2_list) > max_pixels):
                continue
            ie_list.append((cluster1_list, cluster2_list))

    data = obs.data
    best_rms = 1e38
    best_offset = None

    # Across Sun angle is perpendicular to main sun angle    
    a_sun_angle = (sun_angle + oops.HALFPI) % oops.PI
    
    for along_path_dist in xrange(-titan_radius_pix,titan_radius_pix+1):
        u_offset = int(np.round(along_path_dist * np.cos(a_sun_angle)))
        v_offset = int(np.round(along_path_dist * np.sin(a_sun_angle)))

        diff_list = []
        
        for cluster1_list, cluster2_list in ie_list:
            mean1_list = []
            mean2_list = []
            for pix in cluster1_list:
                if (not 0 <= pix[0]+v_offset < data.shape[0] or
                    not 0 <= pix[1]+u_offset < data.shape[1]):
                    continue
                mean1_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
            for pix in cluster2_list:
                if (not 0 <= pix[0]+v_offset < data.shape[0] or
                    not 0 <= pix[1]+u_offset < data.shape[1]):
                    continue
                mean2_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
            if len(mean1_list) == 0 or len(mean2_list) == 0:
                continue
            mean1 = np.mean(mean1_list)
            mean2 = np.mean(mean2_list)
            diff = np.abs(np.mean(mean2_list)-np.mean(mean1_list))
            diff_list.append(diff)
            
        diff_list = np.array(diff_list)

        rms = np.sqrt(np.sum(diff_list**2))
        
        if rms < best_rms:
            best_rms = rms
            best_offset = (u_offset, v_offset)

#        print along_path_dist, u_offset, v_offset, rms
        
    print 'FINAL RESULT', best_offset, best_rms

    # Now get the slice along the sun angle

    slice_x = []
    slice_y = []
    
    for along_path_dist in xrange(-titan_radius_pix,titan_radius_pix+1):
        # We want to go along the Sun angle
        u = (int(np.round(along_path_dist * np.cos(sun_angle))) + 
             best_offset[0] + titan_center[0])
        v = (int(np.round(along_path_dist * np.sin(sun_angle))) + 
             best_offset[1] + titan_center[1])
        if (not 0 <= u < data.shape[1] or
            not 0 <= v < data.shape[0]):
            continue
        
        slice_x.append(along_path_dist * titan_resolution)
        slice_y.append(data[v,u])
    
    slice_x = np.array(slice_x)
    slice_y = np.array(slice_y)
    
#    interp_func = interp.interp1d(slice_x, slice_y, bounds_error=False, fill_value=0., kind='cubic')
#    res_y = interp_func(TITAN_X)
    
#    plt.imshow(obs.data)
#    plt.figure()
#    plt.plot(slice_x, slice_y)
#    plt.show()

    fp = open(pickle_path, 'wb')
    pickle.dump(phase_angle, fp)
    pickle.dump(filter, fp)
    pickle.dump(titan_resolution, fp)
    pickle.dump(titan_radius, fp)
    pickle.dump(sun_angle, fp)
    pickle.dump(titan_center, fp)
    pickle.dump(best_offset, fp)
    pickle.dump(slice_x, fp)
    pickle.dump(slice_y, fp)
    fp.close()
    
    return


#==============================================================================
#
# MAIN ROUTINES
# 
#==============================================================================

# Make a new big Titan
   
titan = oops.Body.lookup('TITAN')
titan_atmos = copy.copy(titan)

titan_atmos.name = 'TITAN+ATMOSPHERE'
titan_atmos.radius += 700
titan_atmos.inner_radius += 700
surface = titan_atmos.surface
titan_atmos.surface = oops.surface.Spheroid(surface.origin, surface.frame, 
                                            (titan_atmos.radius, 
                                             titan_atmos.radius))
oops.Body.BODY_REGISTRY['TITAN+ATMOSPHERE'] = titan_atmos

TITAN_INCR = 1.
TITAN_SCAN_RADIUS = titan_atmos.radius
TITAN_X = np.arange(-TITAN_SCAN_RADIUS, TITAN_SCAN_RADIUS+TITAN_INCR, 
                    TITAN_INCR)        

# VIO
#process_image('COISS_2068/data/1680805782_1681997642/W1681926856_1_CALIB.IMG') # 16    
#process_image('COISS_2063/data/1655742537_1655905033/W1655808265_1_CALIB.IMG') # 30
#process_image('COISS_2068/data/1683372651_1683615321/W1683615178_1_CALIB.IMG') # 45
#process_image('COISS_2030/data/1552197101_1552225837/W1552216646_1_CALIB.IMG') # 60
#process_image('COISS_2082/data/1743902905_1744323160/W1743914297_1_CALIB.IMG') # 73
#process_image('COISS_2084/data/1753489519_1753577899/W1753508727_1_CALIB.IMG') # 90 
#process_image('COISS_2028/data/1548522106_1548756214/W1548712789_1_CALIB.IMG') # 104 
#process_image('COISS_2024/data/1532184650_1532257621/W1532185013_1_CALIB.IMG') # 120 
#process_image('COISS_2060/data/1643317802_1643406946/W1643375992_1_CALIB.IMG') # 135
#process_image('COISS_2076/data/1721802517_1721894741/W1721822901_1_CALIB.IMG') # 149 
#process_image('COISS_2033/data/1561668355_1561837358/W1561790952_1_CALIB.IMG') # 166

# RED
#process_image('COISS_2057/data/1629783492_1630072138/W1629929515_1_CALIB.IMG') # 16    
#process_image('COISS_2063/data/1655742537_1655905033/W1655808364_1_CALIB.IMG') # 30
#process_image('COISS_2068/data/1683372651_1683615321/W1683615321_1_CALIB.IMG') # 45
#process_image('COISS_2030/data/1552197101_1552225837/W1552216540_1_CALIB.IMG') # 60
#process_image('COISS_2082/data/1743902905_1744323160/W1743914117_1_CALIB.IMG') # 73
#process_image('COISS_2084/data/1753489519_1753577899/W1753508826_1_CALIB.IMG') # 90 
#process_image('COISS_2028/data/1548522106_1548756214/W1548712675_1_CALIB.IMG') # 104 
#process_image('COISS_2024/data/1532184650_1532257621/W1532184947_1_CALIB.IMG') # 120 
#process_image('COISS_2060/data/1643317802_1643406946/W1643376091_1_CALIB.IMG') # 135
#process_image('COISS_2076/data/1721802517_1721894741/W1721823033_1_CALIB.IMG') # 149 
#process_image('COISS_2033/data/1561668355_1561837358/W1561794185_1_CALIB.IMG') # 166

image_list = []

with open(TITAN_FILENAMES_CSV, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = csvreader.next()
    for colnum in xrange(len(header)):
        if (header[colnum] == 'primaryfilespec' or 
            header[colnum] == 'Primary File Spec'):
            break
    else:
        print 'Badly formatted CSV file', TITAN_FILENAMES_CSV
        sys.exit(-1)
    for row in csvreader:
        filespec = row[colnum]
        filespec = filespec.replace('.IMG', '_CALIB.IMG')
        image_list.append(filespec)

start_frac = 0
if len(sys.argv) == 2:
    start_frac = float(sys.argv[1])
    
for filename in image_list[int(start_frac*len(image_list)):]:
    try:
        process_image(filename)
    except RuntimeError:
        print 'Missing SPICE data'
    
