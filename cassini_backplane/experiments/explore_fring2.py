'''
Created on Apr 11, 2014

@author: rfrench
'''

import numpy as np
import numpy.ma as ma
import oops.inst.cassini.iss as iss
import oops.inst.nh.lorri as lorri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cspice
from polymath import *
from imgdisp import *
import Tkinter as tk
import scipy.ndimage.filters as filt
from psfmodel.gaussian import GaussianPSF
from cb_correlate import *
from cb_stars import *
from cb_util_filters import *
import os

#plot_planck_vs_solar_flux()

#def find_best_offset(corr_data, phot_data, model, star_list):
#    image = corr_data.astype('float')
#    corr = correlate2d(image, model, retile=True)
#    
#    search_size_max_u = 30
#    search_size_max_v = 40
#    
#    slice = corr[corr.shape[0]//2-search_size_max_v:
#                 corr.shape[0]//2+search_size_max_v+1,
#                 corr.shape[1]//2-search_size_max_u:
#                 corr.shape[1]//2+search_size_max_u+1]
#
#    maxima = detect_local_maxima(slice)
#    
#    maxima_data = []
#    for offset_v, offset_u in zip(maxima[0], maxima[1]):
#        if (offset_v == 0 or offset_v == corr.shape[0]-1 or
#            offset_u == 0 or offset_u == corr.shape[1]-1):
#            continue
#
#        maxima_data.append((slice[offset_v, offset_u],
#                            offset_u-search_size_max_u,
#                            offset_v-search_size_max_v))
#
#    maxima_data.sort(reverse=True)
#    maxima_data = maxima_data[:20]
#
#    best_stars = -1
#    best_offset_u = None
#    best_offset_v = None
#    
#    for peak_val, offset_u, offset_v in maxima_data:
#        print 'CORRPEAK U V', peak_val, offset_u, offset_v
#        good_stars, confidence = star_perform_photometry(phot_data, star_list,
#                                             offset_u=offset_u, offset_v=offset_v,
#                                             fit_psf=False)
#    
#        if (good_stars == best_stars and
#            confidence == best_confidence and
#            (best_offset_u != offset_u or
#             best_offset_v != offset_v)):
#            # Same number of good stars and peak value, but different offset
#            # Don't trust this number of stars at all!
#            best_offset_u = None
#            best_offset_v = None
#            
#        if (good_stars > best_stars or
#            (good_stars == best_stars and
#             confidence > best_confidence)):
#            best_confidence = confidence
#            best_stars = good_stars
#            best_offset_u = offset_u
#            best_offset_v = offset_v
#            
#    return best_offset_u, best_offset_v, best_stars, best_confidence

#def find_best_offset(corr_data, phot_data, model, star_list):
#    image = corr_data.astype('float')
#    corr = correlate2d(image, model, retile=True)
#    
#    best_stars = -1
#    best_offset_u = None
#    best_offset_v = None
#    best_peak_val = None
#    
#    for search_size_min in xrange(0,28,3):
#        search_size_max = search_size_min+5
#        offset_u, offset_v, peak_val = find_correlated_offset(corr,
#                                          search_size_min=search_size_min,
#                                          search_size_max=search_size_max)
#
#        # We don't like offsets that are right at the edge because it's not a
#        # clean maximum
#        if (abs(offset_u) == search_size_max or
#            abs(offset_v) == search_size_max):
#            continue
#            
#        good_stars = star_perform_photometry(phot_data, star_list,
#                                             offset_u=offset_u, offset_v=offset_v,
#                                             fit_psf=False)
#        
#        if (good_stars == best_stars and
#            peak_val == best_peak_val and
#            (best_offset_u != offset_u or
#             best_offset_v != offset_v)):
#            # Same number of good stars and peak value, but different offset
#            # Don't trust this number of stars at all!
#            best_offset_u = None
#            best_offset_v = None
#            
#        if (good_stars > best_stars or
#            (good_stars == best_stars and peak_val > best_peak_val)):
#            best_stars = good_stars
#            best_offset_u = offset_u
#            best_offset_v = offset_v
#            best_peak_val = peak_val
#            
#    return best_offset_u, best_offset_v, best_stars
    
def process_image(filename, interactive=False):
    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    print filename, 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    star_list = star_list_for_obs(obs, num_stars=30)
    if len(star_list) == 0:
        return None, None, 0, 0, 0

    max_dn = star_list[0].dn

#    new_data = data.copy()
    
    filtered_data = filter_local_maximum(data, area_size=7)

    filtered_data[filtered_data < 0.] = 0.
    
    mask = filtered_data > max_dn
    
    mask = filt.maximum_filter(mask, 11)
    filtered_data[mask] = 0.
    model = star_create_model(obs, star_list)

    f_search_size_max_u = 40
    f_search_size_max_v = 40
    
    f_offset_u, f_offset_v, f_peak = find_correlation_and_offset(filtered_data, model, search_size_min=0,
                                                                 search_size_max=(f_search_size_max_u,
                                                                                  f_search_size_max_v))

    print 'FILTERED OFFSET', f_offset_u, f_offset_v

#    search_size_min_u = max(abs(f_offset_u)-2, 0)
#    search_size_max_u = min(abs(f_offset_u)+2, f_search_size_max_u)
#    search_size_min_v = max(abs(f_offset_v)-2, 0)
#    search_size_max_v = min(abs(f_offset_v)+2, f_search_size_max_v)
#
#    offset_u, offset_v, peak = find_correlation_and_offset(data, model, 
#                                                           search_size_min=(search_size_min_u,
#                                                                            search_size_min_v),
#                                                           search_size_max=(search_size_max_u,
#                                                                            search_size_max_v))
#
#    print 'NEW OFFSET', offset_u, offset_v
#
#    if abs(offset_u-f_offset_u) > 2 or abs(offset_v-f_offset_v) > 2:
#        print 'DO NOT TRUST THE NEW OFFSET!'
    
    offset_u = f_offset_u
    offset_v = f_offset_v
    
    good_stars, confidence = star_perform_photometry(data, star_list,
                                offset_u=offset_u, offset_v=offset_v,
                                fit_psf=False)
    
    if interactive:
        if offset_u is not None and offset_v is not None:        
            data = shift_image(data, offset_u, offset_v)
            filtered_data = shift_image(filtered_data, offset_u, offset_v)
        overlay = star_make_good_bad_overlay(data, star_list)
        imgdisp = ImageDisp([data,filtered_data], [overlay,overlay], canvas_size=(512,512), allow_enlarge=True)
        tk.mainloop()

    return offset_u, offset_v, len(star_list), good_stars, confidence

def process_dir(abs_path, suffix='_CALIB.IMG', max_files=10000, skip_files=0,
                interactive=False):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    confidence_list = []
    filename_list = []
    
    filenames = sorted(os.listdir(abs_path))
    for filename in filenames:
        full_path = os.path.join(abs_path, filename)
        if os.path.isdir(full_path):
            print 'SKIPPING DIRECTORY', full_path
            continue
            ret = process_dir(full_path)
            sub_offset_u_list, sub_offset_v_list, sub_num_stars_list, sub_num_good_stars_list, sub_confidence_list, sub_filename_list = ret
            offset_u_list += sub_offset_u_list
            offset_v_list += sub_offset_v_list
            num_stars_list += sub_num_stars_list
            num_good_stars_list += sub_num_good_stars_list
            confidence_list += sub_confidence_list
            filename_list += sub_filename_list
            continue
        if not os.path.isfile(full_path):
            continue
        if filename[-len(suffix):] != suffix:
            continue
        if skip_files:
            skip_files -= 1
            continue
        try:
            ret = process_image(full_path, interactive=interactive)
            offset_u, offset_v, num_stars, num_good_stars, confidence = ret
            offset_u_list.append(offset_u)
            offset_v_list.append(offset_v)
            num_stars_list.append(num_stars)
            num_good_stars_list.append(num_good_stars)
            confidence_list.append(confidence)
            filename_list.append(filename)
        except:
            print 'THROWN EXCEPTION'
        
        if len(offset_u_list) >= max_files:
            break
        
    print offset_u_list, offset_v_list
    
    return offset_u_list, offset_v_list, num_stars_list, num_good_stars_list, confidence_list, filename_list

def process_plot_dir(abs_path, basename, max_files=10000, skip_files=0,
                     interactive=False, plot=True):
    ret = process_dir(abs_path, max_files=max_files, skip_files=skip_files,
                      interactive=interactive)
    offset_u_list, offset_v_list, num_stars_list, num_good_stars_list, confidence_list, filename_list = ret
    
    if plot:
        x_min = -0.5
        x_max = len(offset_u_list)-0.5
        
        fig = plt.figure(figsize=(17,11))
        ax = fig.add_subplot(311)
        plt.plot(offset_u_list, '-', color='red', ms=5)
        plt.plot(offset_v_list, '-', color='green', ms=5)
        for i in xrange(len(offset_u_list)):
            if num_good_stars_list[i] >= 2:
                plt.plot(i, offset_u_list[i], 'o', mec='red', mfc='red', ms=5)
                plt.plot(i, offset_v_list[i], 'o', mec='green', mfc='green', ms=5)
            else:
                plt.plot(i, offset_u_list[i], 'x', mec='red', mfc='red', ms=5)
                plt.plot(i, offset_v_list[i], 'x', mec='green', mfc='green', ms=5)
        plt.xlim(x_min, x_max)
        plt.title('U/V Offset')
        
        ax = fig.add_subplot(312)
        plt.plot(num_stars_list, 'o', color='red', mec='red', mfc='red', ms=7)
        plt.plot(num_good_stars_list, 'o', color='black', mec='black', mfc='black', ms=4)
        plt.xlim(x_min, x_max)
        plt.ylim(-0.5, np.max(num_stars_list)+0.5)
        plt.title('Total stars vs. Good stars')
        
        ax = fig.add_subplot(313)
        plt.plot(confidence_list, 'o', color='black', mec='black', mfc='black', ms=4)
        plt.xlim(x_min, x_max)
        plt.ylim(-0.1, 1.1)
        plt.title('Confidence')
        
        plt.suptitle(basename)
        
        plt.subplots_adjust(left=0.025, right=0.975, top=0.925, bottom=0.025, hspace=0.15)
        plt.savefig('j:/Temp/plots/'+basename+'.png', bbox_inches='tight')

def process_plot_all_dir(abs_path):
    filenames = sorted(os.listdir(abs_path))
    for filename in filenames:
        print filename
        full_path = os.path.join(abs_path, filename)
        if os.path.isdir(full_path):
            if filename < 'ISS_007RI':
                continue
            process_plot_dir(full_path, filename, max_files=250)

#process_image('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542047596_1_CALIB.IMG')
#process_image('T:/external/cassini/derived/COISS_2xxx/COISS_2031/data/1555449244_1555593613/N1555565413_1_CALIB.IMG')
#process_image(r'T:\clumps\data\ISS_059RF_FMOVIE001_VIMS\N1581945338_1_CALIB.IMG')

#process_image(r'T:\clumps\data\ISS_039RF_FMOVIE001_VIMS\N1551254314_1_CALIB.IMG')
process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2042\data\1580930973_1581152891\W1581143861_1_CALIB.IMG')
#dir = 'ISS_039RF_FMOVIE001_VIMS'
#max_files = 10000

#dir = 'ISS_030RF_FMOVIE001_VIMS'
#max_files = 10000

#process_plot_all_dir('t:/clumps/data')

#process_plot_dir('t:/clumps/data/ISS_006RI_LPHRLFMOV001_PRIME', 'foo', max_files=100, skip_files=0, interactive=False, plot=True)
#process_plot_dir('t:/clumps/data/ISS_007RI_HPMRDFMOV001_PRIME', 'foo', max_files=100, skip_files=0, interactive=True, plot=False)
#process_plot_dir('t:/clumps/data/ISS_029RF_FMOVIE002_VIMS', 'foo', max_files=100, skip_files=7, interactive=True, plot=False)

#ret = process_dir('T:/clumps/data/'+dir, max_files=max_files)
#offset_u_list, offset_v_list, num_stars_list, num_good_stars_list, confidence_list, filename_list = ret
#
#print
#print
#
#for i in xrange(len(filename_list)):
#    print i, filename_list[i]
#    
#x_min = -0.5
#x_max = len(offset_u_list)-0.5
#
#fig = plt.figure()
#fig.subplots_adjust(left=0.08,bottom=0.06,right=0.96,top=0.96)
#ax = fig.add_subplot(311)
#plt.plot(offset_u_list, '-', color='red', ms=5)
#plt.plot(offset_v_list, '-', color='green', ms=5)
#for i in xrange(len(offset_u_list)):
#    if num_good_stars_list[i] >= 2:
#        plt.plot(i, offset_u_list[i], 'o', mec='red', mfc='red', ms=5)
#        plt.plot(i, offset_v_list[i], 'o', mec='green', mfc='green', ms=5)
#    else:
#        plt.plot(i, offset_u_list[i], 'x', mec='red', mfc='red', ms=5)
#        plt.plot(i, offset_v_list[i], 'x', mec='green', mfc='green', ms=5)
#plt.xlim(x_min, x_max)
#plt.title('U/V Offset')
#
#ax = fig.add_subplot(312)
#plt.plot(num_stars_list, 'o', color='red', mec='red', mfc='red', ms=7)
#plt.plot(num_good_stars_list, 'o', color='black', mec='black', mfc='black', ms=4)
#plt.xlim(x_min, x_max)
#plt.ylim(-0.5, np.max(num_stars_list)+0.5)
#plt.title('Total stars vs. Good stars')
#
#ax = fig.add_subplot(313)
#plt.plot(confidence_list, 'o', color='black', mec='black', mfc='black', ms=4)
#plt.xlim(x_min, x_max)
#plt.ylim(-0.1, np.max(confidence_list)+0.1)
#plt.title('Confidence')
#
#plt.suptitle(dir)
#plt.show()
