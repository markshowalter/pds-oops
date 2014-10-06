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
from cb_offset import *
import os
import cProfile, pstats, StringIO

def process_image(filename, interactive=True):
    print filename
    
    obs = iss.from_file(filename)
    print filename, 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

#    obs.data[:,:900] = 1 # XXX
#    obs.data[:250,:] = 1
#    obs.data[850:,:] = 1
#    obs.data[:,1020:] = 0
#    obs.data[obs.data > 0.02] = 1
#    obs.data[obs.data <= 0.02] = 0
    
    offset_u, offset_v, metadata = master_find_offset(obs, allow_saturn=False,
                                                      allow_moons=False, allow_stars=True,
                                                      create_overlay=True)
    
    print 'OFFSET', offset_u, offset_v
    
    ext_data = metadata['ext_data']
    overlay = metadata['ext_overlay']
    
    if interactive:
        imgdisp = ImageDisp([ext_data], [overlay], canvas_size=(1100,768), allow_enlarge=True)
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
        try:
            process_image(new_file, interactive=True)
        except:
            print new_file
            print 'THREW EXCEPTION'
    else:
        dir_no = random.randint(0, len(dir_list)-1)
        new_dir = os.path.join(root_path, dir_list[dir_no])
        process_random_file(new_dir)

pr = cProfile.Profile()
pr.enable()

process_image(r't:\clumps\data\ISS_041RF_FMOVIE002_VIMS\N1552839037_1_CALIB.IMG')

#process_image(r't:\clumps\data\ISS_006RI_LPHRLFMOV001_PRIME/N1492102189_1_CALIB.IMG')
#process_image(r't:\clumps\data\ISS_080RF_FMOVIE005_PRIME/N1597392780_1_CALIB.IMG')

#process_image(r't:\clumps\data\ISS_029RF_FMOVIE002_VIMS\N1538271171_1_CALIB.IMG')
#process_image(r't:\clumps\data\ISS_000RI_SATSRCHAP001_PRIME/N1466494781_1_CALIB.IMG', interactive=True)
#process_image(r't:\clumps\data\ISS_059RF_FMOVIE001_VIMS/N1581948682_1_CALIB.IMG', interactive=True)
#process_image(r't:\clumps\data\ISS_007RI_HPMRDFMOV001_PRIME/N1493860227_1_CALIB.IMG')
#process_image(r't:\clumps\data\ISS_007RI_HPMRDFMOV001_PRIME/N1493860577_1_CALIB.IMG')
#process_image(r't:\clumps\data\ISS_007RI_HPMRDFMOV001_PRIME/N1493860577_1_CALIB.IMG')
#process_image(r'T:\external\cassini\derived\COISS_2xxx\COISS_2012\data\1495425444_1495768586\N1495641779_1_CALIB.IMG', interactive=True)
#while True:
#    process_random_file('t:/external/cassini/derived/COISS_2xxx')

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
ps.print_callers()
print s.getvalue()
