import colorsys
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
from cb_titan import *
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

FILTER_NUMBER = { # Table VIII
    'CLEAR': 0,         #  611 NAC  635 WAC
                # UV1-UV3  258-338 NAC
    'VIO': 1,           #           420 WAC
                    # BL2  440 NAC
    'BL1': 2,           #  451 NAC  460 WAC
    'GRN': 3,           #  568 NAC  567 WAC
                    # MT1  619 NAC
                    # CB1  619 NAC
                   # CB1a  635 NAC
                   # CB1b  603 NAC
    'RED': 4,           #  650 NAC  648 WAC
    'HAL': 5,           #  656 NAC  656 WAC
    'MT2': 6,           #  727 NAC  728 WAC
    'MT2+IRP0': 7,      #  727 NAC  728 WAC + 746 NAC 705 WAC
    'MT2+IRP90': 8,     #  727 NAC  728 WAC +         705 WAC
    'CB2': 9,           #  750 NAC  752 WAC
    'CB2+IRP0': 10,     #  750 NAC  752 WAC + 746 NAC 705 WAC
    'CB2+IRP90': 11,    #  750 NAC  752 WAC +         705 WAC
    'IR1': 12,          #  752 NAC  742 WAC
    'IR2': 13,          #  862 NAC  853 WAC
    'IR2+IR1': 14,      #  827 NAC  826 WAC (Table IX) 
    'MT3': 15,          #  889 NAC  890 WAC
    'MT3+IRP0': 16,     #  889 NAC  890 WAC + 746 NAC 705 WAC
    'MT3+IRP90': 17,    #  889 NAC  890 WAC +         705 WAC
    'CB3': 18,          #  938 NAC  939 WAC
    'CB3+IRP0': 19,     #  938 NAC  939 WAC + 746 NAC 705 WAC
    'CB3+IRP90': 20,    #  938 NAC  939 WAC +         705 WAC
    'IR3': 21,          #  930 NAC  918 WAC
    'IR4': 22,          # 1002 NAC 1001 WAC
    'IR5': 23,          #          1028 WAC    
}    

NUM_FILTERS = 24
    
#===============================================================================
# 
# CREATE THE PROFILE FOR ONE IMAGE
#
#===============================================================================

def process_image(filename, force=False, recompute=False):
    print '>>> Computing profile for', filename

    _, filespec = os.path.split(filename)
    filespec = filespec.replace('_CALIB.IMG', '')
    
    pickle_path = os.path.join(SUPPORT_FILES_ROOT, 'titan', filespec+'.pickle')

    if not force and not recompute and os.path.exists(pickle_path):
        print 'Pickle file already exists'
        return
    
    if recompute and os.path.exists(pickle_path):
        fp = open(pickle_path, 'rb')
        phase_angle = pickle.load(fp)
        fp.close()
        if type(phase_angle) == type(''):
            print 'Skipping due to previous error', phase_angle
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

    bp_incidence = bp.incidence_angle('TITAN+ATMOSPHERE')
    full_mask = ma.getmaskarray(bp_incidence.mvals)
    
    # If the angles touch or are near the edge, skip this image
    inv_mask = np.logical_not(full_mask)
    if (np.any(inv_mask[0:6,:]) or np.any(inv_mask[-6:-1,:]) or
        np.any(inv_mask[:,0:6]) or np.any(inv_mask[:,-6:-1])):
        print 'Skipping due to Titan off edge'
        fp = open(pickle_path, 'wb')
        pickle.dump('Titan off edge', fp)
        fp.close()
        return

    if np.any(obs.data[inv_mask] == 0.):
        print 'Skipping due to bad image data encroaching on Titan'
        fp = open(pickle_path, 'wb')
        pickle.dump('Bad image data', fp)
        fp.close()
        return

    # Find the projected angle of the solar illumination to the center of
    # Titan's disc.
    phase_angle = bp.center_phase_angle('TITAN+ATMOSPHERE').vals
    
    if phase_angle < oops.HALFPI:
        extreme_pos = np.argmin(bp_incidence.mvals)
    else:
        extreme_pos = np.argmax(bp_incidence.mvals)
    extreme_pos = np.unravel_index(extreme_pos, bp_incidence.mvals.shape)
    extreme_uv = (extreme_pos[1], extreme_pos[0])
    
    sun_angle = np.arctan2(extreme_uv[1]-titan_center[1], 
                           extreme_uv[0]-titan_center[0])

    print 'PHASE', phase_angle * oops.DPR,
    print 'FILTER', filter, 'CENTER', titan_center, 'RADIUS', titan_radius,
    print 'RES', titan_resolution, 'RADIUS PIX', titan_radius_pix,
    print 'SUN ANGLE', sun_angle * oops.DPR

#    plt.imshow(bp_incidence.mvals)
#    plt.show()
    
    bp_emission = bp.emission_angle('TITAN+ATMOSPHERE')
    
    cluster_gap_threshold = 10
    cluster_max_pixels = 10
    wac_offset_limit = 10
    offset_limit = int(np.ceil(wac_offset_limit*np.sqrt(2)))+1
    
    best_offset, best_rms = titan_find_symmetry_offset(
                obs, sun_angle,
                2.5*oops.RPD, oops.HALFPI, 5*oops.RPD,
                5.*oops.RPD, oops.PI, 5*oops.RPD,
                cluster_gap_threshold, cluster_max_pixels,
                offset_limit)
    
    if best_offset is None:
        print 'Symmetry offset finding failed'
        fp = open(pickle_path, 'wb')
        pickle.dump('Symmetry offset finding failed', fp)
        fp.close()
        return
        
    print 'FINAL RESULT', best_offset, best_rms

    # Now get the profile along the sun angle
    profile_x, profile_y = titan_along_track_profile(obs, best_offset,
                                                 sun_angle, titan_center,
                                                 titan_radius_pix,
                                                 titan_resolution)
    
#    interp_func = interp.interp1d(profile_x, profile_y, bounds_error=False, fill_value=0., kind='cubic')
#    res_y = interp_func(TITAN_X)
    
#     plt.imshow(obs.data)
#     plt.figure()
#    plt.plot(profile_x, profile_y)
#    plt.show()

    interp_func = interp.interp1d(profile_x, profile_y, bounds_error=False, 
                                  fill_value=0., kind='cubic')
    interp_profile = interp_func(TITAN_X)

    fp = open(pickle_path, 'wb')
    pickle.dump(phase_angle, fp)
    pickle.dump(filter, fp)
    pickle.dump(titan_resolution, fp)
    pickle.dump(titan_radius, fp)
    pickle.dump(sun_angle, fp)
    pickle.dump(titan_center, fp)
    pickle.dump(best_offset, fp)
    pickle.dump(profile_x, fp)
    pickle.dump(profile_y, fp)
    pickle.dump(interp_profile, fp)
    fp.close()
    
    return


def add_profile_to_list(filename, reinterp=False):
    _, filespec = os.path.split(filename)
    filespec = filespec.replace('_CALIB.IMG', '')
    
    pickle_path = os.path.join(SUPPORT_FILES_ROOT, 'titan', filespec+'.pickle')

    if not os.path.exists(pickle_path):
        return
    
#    print '>>> Reading profile for', filename

    fp = open(pickle_path, 'rb')
    phase_angle = pickle.load(fp)
    if type(phase_angle) == type(''):
#        print 'Error:', phase_angle
        fp.close()
        return
    filter = pickle.load(fp)
    titan_resolution = pickle.load(fp)
    titan_radius = pickle.load(fp)
    sun_angle = pickle.load(fp)
    titan_center = pickle.load(fp)
    best_offset = pickle.load(fp)
    profile_x = pickle.load(fp)
    profile_y = pickle.load(fp)
    try:
        profile = pickle.load(fp)
    except:
        reinterp = True
    fp.close()

    if reinterp:
        print 'Interpolating', filename
        interp_func = interp.interp1d(profile_x, profile_y, bounds_error=False, 
                                      fill_value=0., kind='cubic')
        profile = interp_func(TITAN_X)
        fp = open(pickle_path, 'wb')
        pickle.dump(phase_angle, fp)
        pickle.dump(filter, fp)
        pickle.dump(titan_resolution, fp)
        pickle.dump(titan_radius, fp)
        pickle.dump(sun_angle, fp)
        pickle.dump(titan_center, fp)
        pickle.dump(best_offset, fp)
        pickle.dump(profile_x, fp)
        pickle.dump(profile_y, fp)
        pickle.dump(profile, fp)
        fp.close()

    if profile[10] > 0.1:
#        print 'HIGH I/F VALUE', filter, phase_angle*oops.DPR, best_offset, sun_angle*oops.DPR, filename
        return
#        profile -= np.min(profile[100:-100])
#        profile = np.clip(profile, 0, 1e38)
#        plt.plot(profile)
#        plt.show()
    
#    if abs(best_offset[0]) > 10:
#        plt.plot(profile)
#        plt.show()
    
#    if abs(best_offset[0]) > 10:
#        print best_offset, filename
#        return

    OFFSET_LIST.append(best_offset)    
    PROFILE_LIST.append((filter, phase_angle, profile))

def bin_profiles(plot=False):
    for profile in PROFILE_LIST:
        filter, phase_angle, profile = profile
        if filter not in BY_FILTER_DB:
            entry = [[] for x in xrange(NUM_PHASE_BINS)]
            BY_FILTER_DB[filter] = entry
        entry = BY_FILTER_DB[filter]
        bin_num = int(phase_angle / PHASE_BIN_GRANULARITY)
        entry[bin_num].append(profile)
        
    for filter in sorted(BY_FILTER_DB):
        if plot:
            plt.figure()
        for bin_no, bin in enumerate(BY_FILTER_DB[filter]):
            if len(bin) == 0:
                BY_FILTER_DB[filter][bin_no] = None
                continue
            phase = bin_no*PHASE_BIN_GRANULARITY*oops.DPR
            if len(bin) == 1:
                med_res = bin[0]
            else:
                med_res = np.median(zip(*bin), axis=1)
            BY_FILTER_DB[filter][bin_no] = (TITAN_X, med_res, len(bin))
            print filter, bin_no*PHASE_BIN_GRANULARITY*oops.DPR, len(bin)
            if plot:
                color = colorsys.hsv_to_rgb(
                         phase/180, 1, 1)
                plt.plot(TITAN_X, med_res, label=('%.1f (%d)'%(phase, len(bin))), color=color)

        if plot:
            plt.legend()
            plt.title(filter)

    if plot:
        plt.show()

def plot_profiles_by_phase_filter():
    num_filters = len(BY_FILTER_DB)
    for bin_no in xrange(NUM_PHASE_BINS):
        phase = bin_no*PHASE_BIN_GRANULARITY*oops.DPR
        first_plot = True
        for filter_num, filter in sorted([(FILTER_NUMBER[x], x) for x in BY_FILTER_DB]):
            if filter[:2] != 'CB' and filter[:2] != 'MT':
                continue
            bin_contents = BY_FILTER_DB[filter][bin_no]
            if bin_contents is None:
                continue
            x, med_res, num_bins = bin_contents
            color = colorsys.hsv_to_rgb(
                     float(filter_num)/NUM_FILTERS, 1, 1)
            if first_plot:
                plt.figure()
                first_plot = False
            plt.plot(x, med_res, label=('%s (%d)'%(filter, num_bins)), color=color)

        plt.legend()
        plt.title('Phase %.2f' % phase)

    plt.show()

def plot_rescaled_filters():
    for filters in [('MT2', 'MT2+IRP0', 'MT2+IRP90'),
                    ('CB2', 'CB2+IRP0', 'CB2+IRP90'),
                    ('MT3', 'MT3+IRP0', 'MT3+IRP90'),
                    ('CB3', 'CB3+IRP0', 'CB3+IRP90')]:
        for bin_no in xrange(NUM_PHASE_BINS):
            phase = bin_no*PHASE_BIN_GRANULARITY*oops.DPR
            master = BY_FILTER_DB[filters[0]][bin_no]
            if master is None:
                continue
            master_x, master_y, master_num = master
            bad = False
            first_plot = True            
            for filter in filters[1:]:
                slave = BY_FILTER_DB[filter][bin_no]
                if slave is None:
                    bad = True
                    break
            if bad:
                continue
            plt.figure()
            color = colorsys.hsv_to_rgb(
                 float(FILTER_NUMBER[filters[0]])/NUM_FILTERS, 1, 1)
            plt.plot(master_x, master_y, 
                     label=('%s (%d)'%(filters[0], master_num)), 
                     color=color)
            for filter in filters[1:]:
                slave = BY_FILTER_DB[filter][bin_no]
                slave_x, slave_y, slave_num = slave
                scale = np.mean(master_y)/np.mean(slave_y)
                slave_y = slave_y * scale
                color = colorsys.hsv_to_rgb(
                     float(FILTER_NUMBER[filter])/NUM_FILTERS, 1, 1)
                plt.plot(slave_x, slave_y, 
                         label=('%s (%d)'%(filter, slave_num)), 
                         color=color)
                    
            plt.legend()
            plt.title('Phase %.2f' % phase)

    plt.show()
                    



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

#process_image('COISS_2023/data/1526706719_1526797307/W1526778201_1_CALIB.IMG', force=True)
#process_image('COISS_2049/data/1604402501_1604469049/W1604464865_1_CALIB.IMG', force=True)
#process_image('COISS_2049/data/1604402501_1604469049/W1604466257_1_CALIB.IMG', force=True)
#process_image('COISS_2049/data/1604402501_1604469049/W1604467653_1_CALIB.IMG', force=True)
#process_image('COISS_2049/data/1604402501_1604469049/W1604469049_1_CALIB.IMG', force=True)

#assert False

# VIO
# process_image('COISS_2068/data/1680805782_1681997642/W1681926856_1_CALIB.IMG') # 16    
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
    
if True:
    for filename in image_list[int(start_frac*len(image_list)):]:
        try:
            process_image(filename, recompute=True)
        except RuntimeError:
            print 'Missing SPICE data'

PROFILE_LIST = []
BY_FILTER_DB = {}    
PHASE_BIN_GRANULARITY = 10. * oops.RPD
NUM_PHASE_BINS = int(np.ceil(oops.PI / PHASE_BIN_GRANULARITY))
OFFSET_LIST = []

if True:
    for filename in image_list:#[:10]:
        add_profile_to_list(filename)
    bin_profiles(plot=False)

offset_list_x = [x[0] for x in OFFSET_LIST]
offset_list_y = [x[1] for x in OFFSET_LIST]

print 'TOTAL IMAGES', len(offset_list_x)
print 'MEAN OFFSET X', np.mean(offset_list_x)
print 'MEAN OFFSET Y', np.mean(offset_list_y)

#plot_profiles_by_phase_filter()
#plot_rescaled_filters()

pickle_file = os.path.join(SUPPORT_FILES_ROOT, 'titan-profiles.pickle')
fp = open(pickle_file, 'wb')
pickle.dump(PHASE_BIN_GRANULARITY, fp)
pickle.dump(BY_FILTER_DB, fp)
fp.close()
