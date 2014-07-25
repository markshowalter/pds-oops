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
from cb_offset import *
import os

def process_image(filename, interactive=False):
    print filename
    
    obs = iss.from_file(filename)
    data = calibrate_iof_image_as_dn(obs)
    print filename, 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u, offset_v, model, overlay = find_offset(obs)
    
    if interactive:
        if offset_u is not None and offset_v is not None:        
            data = shift_image(data, offset_u, offset_v)
        imgdisp = ImageDisp([data], [overlay], canvas_size=(1024,768), allow_enlarge=True)
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
#        try:
        process_image(new_file, interactive=True)
#        except:
#            print new_file
#            print 'THREW EXCEPTION'
    else:
        dir_no = random.randint(0, len(dir_list)-1)
        new_dir = os.path.join(root_path, dir_list[dir_no])
        process_random_file(new_dir)

# Saturn and full rings with shadow
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2001\data\1458023716_1458209270\N1458112889_1_CALIB.IMG', interactive=True)

# Phoebe
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2003/data/1465650156_1465674412/N1465650307_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2004/data/1465674475_1465709620/N1465679337_2_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2004/data/1465674475_1465709620/N1465677386_2_CALIB.IMG', interactive=True)

# Mimas
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2014/data/1501618408_1501647096/N1501630117_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2027/data/1542749662_1542807100/N1542756630_1_CALIB.IMG', interactive=True)

# Enceladus
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2009/data/1487182149_1487415680/N1487300482_1_CALIB.IMG', interactive=True)
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2019/data/1516036945_1516171123/N1516168806_1_CALIB.IMG', interactive=True)
process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2047/data/1597149262_1597186268/N1597179218_2_CALIB.IMG', interactive=True)

# Tethys
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2007/data/1477601077_1477653092/N1477639356_1_CALIB.IMG', interactive=True)

#while True:
#    process_random_file('t:/external/cassini/derived/COISS_2xxx')
