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

def process_image(filename, interactive=True):
    obs = iss.from_file(filename)
    print filename, 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u, offset_v, metadata = star_find_offset(obs)
    
    star_list = metadata['used_star_list']

    if offset_u is None:
        offset_u = 0
        offset_v = 0
    
    good_stars = metadata['num_good_stars']
        
    if interactive:
        overlay = star_make_good_bad_overlay(obs, star_list, offset_u, offset_v)
        imgdisp = ImageDisp([obs.data], [overlay], canvas_size=(512,512), allow_enlarge=True)
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

#process_image('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542047596_1_CALIB.IMG')

while True:
    process_random_file('t:/external/cassini/derived/COISS_2xxx')
