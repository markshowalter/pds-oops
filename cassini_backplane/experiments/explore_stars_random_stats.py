'''
Created on Apr 11, 2014

@author: rfrench
'''

import random
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
    print filename
    
    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    print filename, 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u = None
    offset_v = None
    good_stars = 0
    confidence = 0
    
    star_list = star_list_for_obs(obs, num_stars=30)
    filtered_data = filter_local_maximum(data, area_size=7)
    filtered_data[filtered_data < 0.] = 0.
    
    if len(star_list) > 0:
        max_dn = star_list[0].dn
        mask = filtered_data > max_dn        
        mask = filt.maximum_filter(mask, 11)
        filtered_data[mask] = 0.
        model = star_create_model(obs, star_list)
    
        f_search_size_max_u = 40
        f_search_size_max_v = 40
        
        f_offset_u, f_offset_v, f_peak = find_correlation_and_offset(filtered_data, model, search_size_min=0,
                                                                     search_size_max=(f_search_size_max_u,
                                                                                      f_search_size_max_v))
    
        print 'OFFSET', f_offset_u, f_offset_v
        
        offset_u = f_offset_u
        offset_v = f_offset_v
        
        good_stars, confidence = star_perform_photometry(data, star_list,
                                    offset_u=offset_u, offset_v=offset_v,
                                    fit_psf=False)

        print 'FINAL OFFSET', offset_u, offset_v
        
    
    if interactive:
        if offset_u is not None and offset_v is not None:        
            data = shift_image(data, offset_u, offset_v)
            filtered_data = shift_image(filtered_data, offset_u, offset_v)
        overlay = star_make_good_bad_overlay(data, star_list)
        imgdisp = ImageDisp([data,filtered_data], [overlay,overlay], canvas_size=(512,512), allow_enlarge=True)
        tk.mainloop()

    return offset_u, offset_v, len(star_list), good_stars, confidence

def process_random_file(root_path):
    print root_path
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
        try:
            return process_image(new_file, interactive=False)
        except:
            print new_file
            print 'THREW EXCEPTION'
            return None, None, None, None, None
    else:
        dir_no = random.randint(0, len(dir_list)-1)
        new_dir = os.path.join(root_path, dir_list[dir_no])
        return process_random_file(new_dir)

def stats_from_list(thelist):
    print 'MIN', np.min(thelist), 'MAX', np.max(thelist),
    print 'MEAN', np.mean(thelist), 'STDEV', np.std(thelist),
    print 'MEDIAN', np.median(thelist)
    
star_list = []
good_star_list = []
confidence_list = []
num_images = 0
num_good_images = 0

while True:
    ret = process_random_file('t:/external/cassini/derived/COISS_2xxx')
    offset_u, offset_v, num_stars, good_stars, confidence = ret
    if offset_u is None:
        num_stars = 0
        good_stars = 0
        confidence = 0
    star_list.append(num_stars)
    good_star_list.append(good_stars)
    confidence_list.append(confidence)
    num_images += 1
    if good_stars >= 2:# and confidence > 0.75:
        num_good_images += 1
    
    print 'IMAGES', num_images, 'GOOD', num_good_images    
    print 'NUM STARS',
    stats_from_list(star_list)
    print 'GOOD STARS',
    stats_from_list(good_star_list)
    print 'CONFIDENCE', 
    stats_from_list(confidence_list)
    print
    