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
from cb_gui_offset_data import *
from cb_logging import *
from cb_offset import *
from cb_util_file import *
from cb_util_oops import *

log_set_format(False)
log_set_default_level(logging.DEBUG)

def process_image(filename, interactive=True, **kwargs):
    filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
    print filename
    
    obs = read_iss_file(filename)
    print filename
    print 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', 
    print obs.filter1, obs.filter2

    metadata = master_find_offset(obs, create_overlay=True, **kwargs)

    display_offset_data(obs, metadata)

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


ringding_fp = open('t:/external/cassini/ringding_images.txt')
while True:
    fn = ringding_fp.readline().strip()
    fn = fn[:-4] + '_CALIB' + fn[-4:]
    fn = os.path.join('t:/external/cassini/derived/COISS_2xxx', fn)
    process_image(fn)
