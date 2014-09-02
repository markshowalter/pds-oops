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


def process_image(filename, interactive=False):
    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    obs.data = data
    orig_data = obs.data.copy()
    
#    img_sorted = sorted(list(obs.data.flatten()))
#    perc = img_sorted[np.clip(int(len(img_sorted)*0.5), 0, len(img_sorted)-1)]
    perc = 50
    print perc
    med = filt.median_filter(obs.data, 11)
    mask = med > perc
    print mask
    obs.data[mask] = 0.

    print filename, 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u, offset_v, star_list, good_stars = star_find_offset(obs, extend_fov=(45,45))

    if offset_u is None:
        offset_u = 0
        offset_v = 0
            
    if interactive:
        overlay = star_make_good_bad_overlay(obs, star_list,
                                             offset_u, offset_v)
        imgdisp = ImageDisp([data, orig_data], [overlay,overlay], canvas_size=(768,768), allow_enlarge=True)
        tk.mainloop()

    return offset_u, offset_v, len(star_list), good_stars

def process_dir(abs_path, suffix='_CALIB.IMG', max_files=10000, skip_files=0,
                interactive=False):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    filename_list = []
    
    filenames = sorted(os.listdir(abs_path))
    for filename in filenames:
        if filename[0] == 'W': # XXX
            continue
        full_path = os.path.join(abs_path, filename)
        if os.path.isdir(full_path):
            print 'SKIPPING DIRECTORY', full_path
            continue
            ret = process_dir(full_path)
            sub_offset_u_list, sub_offset_v_list, sub_num_stars_list, sub_num_good_stars_list, sub_filename_list = ret
            offset_u_list += sub_offset_u_list
            offset_v_list += sub_offset_v_list
            num_stars_list += sub_num_stars_list
            num_good_stars_list += sub_num_good_stars_list
            filename_list += sub_filename_list
            continue
        if not os.path.isfile(full_path):
            continue
        if filename[-len(suffix):] != suffix:
            continue
        if skip_files:
            skip_files -= 1
            continue
#        try:
        ret = process_image(full_path, interactive=interactive)
        offset_u, offset_v, num_stars, num_good_stars = ret
        offset_u_list.append(offset_u)
        offset_v_list.append(offset_v)
        num_stars_list.append(num_stars)
        num_good_stars_list.append(num_good_stars)
        filename_list.append(filename)
#        except:
#            print 'THROWN EXCEPTION'
        
        if len(offset_u_list) >= max_files:
            break
        
    print offset_u_list, offset_v_list
    
    return offset_u_list, offset_v_list, num_stars_list, num_good_stars_list, filename_list

def process_plot_dir(abs_path, basename, max_files=10000, skip_files=0,
                     interactive=False, plot=True):
    ret = process_dir(abs_path, max_files=max_files, skip_files=skip_files,
                      interactive=interactive)
    offset_u_list, offset_v_list, num_stars_list, num_good_stars_list, filename_list = ret
    
    if len(offset_u_list) == 0:
        return
    
    if plot:
        x_min = -0.5
        x_max = len(offset_u_list)-0.5
        
        fig = plt.figure(figsize=(17,11))
        ax = fig.add_subplot(211)
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
        
        ax = fig.add_subplot(212)
        plt.plot(num_stars_list, 'o', color='red', mec='red', mfc='red', ms=7)
        plt.plot(num_good_stars_list, 'o', color='black', mec='black', mfc='black', ms=4)
        plt.xlim(x_min, x_max)
        plt.ylim(-0.5, np.max(num_stars_list)+0.5)
        plt.title('Total stars vs. Good stars')
        
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
            process_plot_dir(full_path, filename, max_files=50)

#process_image('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542047596_1_CALIB.IMG', interactive=True)
process_image('T:/external/cassini/derived/COISS_2xxx/COISS_2031/data/1555449244_1555593613/N1555565413_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_059RF_FMOVIE001_VIMS\N1581945338_1_CALIB.IMG', interactive=True)

#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054271_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054716_1_CALIB.IMG', interactive=True)

#process_image(r'T:\clumps\data\ISS_039RF_FMOVIE001_VIMS\N1551254314_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2042\data\1580930973_1581152891\W1581143861_1_CALIB.IMG')

# Mimas
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2008/data/1484506648_1484573247/N1484530421_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2014/data/1501618408_1501647096/N1501647096_1_CALIB.IMG', interactive=True)


#dir = 'ISS_039RF_FMOVIE001_VIMS'
#max_files = 10000

#dir = 'ISS_030RF_FMOVIE001_VIMS'
#max_files = 10000

#process_plot_all_dir('t:/clumps/data')

#process_plot_dir('t:/clumps/data/ISS_006RI_LPHRLFMOV001_PRIME', 'foo', max_files=100, skip_files=0, interactive=False, plot=True)
#process_plot_dir('t:/clumps/data/ISS_007RI_HPMRDFMOV001_PRIME', 'foo', max_files=100, skip_files=0, interactive=True, plot=False)
#process_plot_dir('t:/clumps/data/ISS_029RF_FMOVIE002_VIMS', 'foo', max_files=10, skip_files=0, interactive=False, plot=True)
process_plot_dir('t:/clumps/data/ISS_000RI_SATSRCHAP001_PRIME', 'foo', max_files=100, skip_files=9, interactive=True, plot=False)
process_plot_dir('t:/clumps/data/ISS_00ARI_SPKMOVPER001_PRIME', 'foo', max_files=10, skip_files=0, interactive=True, plot=False)

