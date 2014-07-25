'''
Created on Apr 11, 2014

@author: rfrench
'''

import random
import numpy as np
import numpy.ma as ma
import oops.inst.cassini.iss as iss
from imgdisp import *
import Tkinter as tk
from cb_correlate import *
from cb_util_flux import *
from cb_util_filters import *
from cb_rings import *
import os

def process_image(filename, interactive=False):
    print filename
    
    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    print filename, 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u = None
    offset_v = None
    
    search_size_max_u = 40
    search_size_max_v = 40

    model = rings_create_model(obs)
        
    offset_u, offset_v, peak = find_correlation_and_offset(data, model, search_size_min=0,
                                                           search_size_max=(search_size_max_u,
                                                                            search_size_max_v))
    
    print 'OFFSET', offset_u, offset_v
    
    overlay = np.zeros(model.shape + (3,))
    overlay[:,:,0] = model / np.max(model)
    
    if interactive:
        if offset_u is not None and offset_v is not None:        
            data = shift_image(data, offset_u, offset_v)
        imgdisp = ImageDisp([data], [overlay], canvas_size=(512,512), allow_enlarge=True)
        tk.mainloop()

    return offset_u, offset_v, 0, 0, 0

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
        try:
            process_image(new_file, interactive=True)
        except:
            print new_file
            print 'THREW EXCEPTION'
    else:
        dir_no = random.randint(0, len(dir_list)-1)
        new_dir = os.path.join(root_path, dir_list[dir_no])
        process_random_file(new_dir)

process_image(r'T:\external\cassini\derived\COISS_2xxx\COISS_2012\data\1495425444_1495768586\N1495641779_1_CALIB.IMG', interactive=True)
#while True:
#    process_random_file('t:/external/cassini/derived/COISS_2xxx')
