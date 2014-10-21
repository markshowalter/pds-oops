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
from cb_util_image import *
from cb_offset import *
import os

def process_image(filename, interactive=True):
    print filename
    
    obs = iss.from_file(filename)
    obs.data = calibrate_iof_image_as_dn(obs)
    print filename, 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

    offset_u, offset_v, metadata = master_find_offset(obs, create_overlay=True)
#                                                      allow_stars=False, allow_rings=False)
    
    ext_data = metadata['ext_data']
    overlay = metadata['ext_overlay']
    
    if interactive and ext_data is not None:
        ext_u = (ext_data.shape[1]-obs.data.shape[1])/2
        ext_v = (ext_data.shape[0]-obs.data.shape[0])/2
        imgdisp = ImageDisp([ext_data], [overlay], canvas_size=(1024,768), allow_enlarge=True,
                            origin=(ext_u,ext_v))
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

# Star field through filters - stars visible that we think shouldn't be
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2068\data\1683279174_1683355540\N1683354649_1_CALIB.IMG')

# Star field long exposure - star matching doesn't work
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2051\data\1608970573_1609104344\N1608970573_1_CALIB.IMG')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2053\data\1613598819_1613977956\N1613844349_1_CALIB.IMG')

# Saturn limb only
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2072\data\1704630602_1704833275\N1704832756_1_CALIB.IMG')

# Saturn plus rings and moons - should ignore Saturn for match - rings shadowed on Saturn
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2034\data\1564337660_1564348958\W1564345216_1_CALIB.IMG')

# Saturn closeup with rings edge on
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2021\data\1520611263_1520675998\N1520674835_1_CALIB.IMG')

# Cut off moon model - Tethys
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2069\data\1694646740_1694815141\N1694664125_1_CALIB.IMG')

# Moon terminator along edge of image
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2013\data\1498654338_1498712236\W1498658693_1_CALIB.IMG')

# All rings but bad correlation
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2080\data\1738387955_1738409545\N1738395265_1_CALIB.IMG')
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2028\data\1546797712_1546867749\N1546863063_1_CALIB.IMG')

# Dione
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2027\data\1544835042_1544908839\N1544892287_1_CALIB.IMG')

# Hyperion rotated incorrectly - rotation is chaotic
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2023\data\1530158136_1530199809\N1530185228_1_CALIB.IMG')

# Saturn and full rings with shadow
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2001\data\1458023716_1458209270\N1458112889_1_CALIB.IMG', interactive=True)

# F ring with A ring
#process_image('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542047596_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054271_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_032RF_FMOVIE001_VIMS\N1542054716_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_039RF_FMOVIE001_VIMS\N1551254314_1_CALIB.IMG', interactive=True)

# F ring without A ring
# FAILS
#process_image('T:/external/cassini/derived/COISS_2xxx/COISS_2031/data/1555449244_1555593613/N1555565413_1_CALIB.IMG', interactive=True)
#process_image(r'T:\clumps\data\ISS_059RF_FMOVIE001_VIMS\N1581945338_1_CALIB.IMG', interactive=True)

# All rings
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2055\data\1622272893_1622549559\N1622394132_1_CALIB.IMG')




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
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2047/data/1597149262_1597186268/N1597179218_2_CALIB.IMG', interactive=True)

# Tethys
#process_image(r't:/external/cassini/derived/COISS_2xxx/COISS_2007/data/1477601077_1477653092/N1477639356_1_CALIB.IMG', interactive=True)

# Bad DN calibration - DN is way too high!
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2012\data\1497164137_1497406737\N1497238879_1_CALIB.IMG')

# Mimas and Tethys + rings at a distance
#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2032\data\1559710457_1559931672\W1559730511_1_CALIB.IMG')

# Rings and Pandora - offset from real position as seen by stars
#process_image('t:/external/cassini/derived/COISS_2xxx\COISS_2009\data\1484846724_1485147239\N1484916376_1_CALIB.IMG')

while True:
    process_random_file('t:/external/cassini/derived/COISS_2xxx')




#process_image(r't:/external/cassini/derived/COISS_2xxx\COISS_2042\data\1580930973_1581152891\W1581143861_1_CALIB.IMG')


# TODO:
#       Optimize speed for rings - are there any rings in the display in the first place?
#       Optimize stars - try with stars, then try with bright parts blocked out
#       How do we know if the model fits at all? Some of the A ring ones are way off the edge
#       Optimize star searching for WAC
#       Figure out max magnitude for a given TEXP
#       Figure out max distance for a small moon to be visible
#       If model of moon goes to the edge, we have a problem
#       Terminator instead of limb
#       Detected offset at edge of range is bad
#       Stars should pay attention to opacity of the rings
