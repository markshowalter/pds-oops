import cb_logging
import logging

import numpy as np
import numpy.ma as ma

from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

filespec = 'N1512515515_1.IMG'

obs = iss.from_file(filespec)
bp = oops.Backplane(obs)

data = obs.data.astype("float")
pylab.imshow(data)

pylab.imshow(data.clip(0,1900))
save_png(data, 'Figure_12a.png', 70, 220)

mask = bp.where_inside_shadow('saturn', 'saturn_main_rings')
mask2 = bp.where_in_front('saturn_main_rings', 'saturn')

save_png(mask.vals | mask2.vals, 'Figure_12b.png', 0, 1)
