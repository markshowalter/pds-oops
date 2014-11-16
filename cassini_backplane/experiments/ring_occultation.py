import numpy as np
import matplotlib.pyplot as plt

from pdstable import PdsTable
from tabulation import Tabulation

from cb_config import *

OCC_ROOT = os.path.join(COUVIS_8XXX_ROOT, 'COUVIS_8001', 'DATA',
                        'EASYDATA')


RING_MIN = 74658
RING_MAX = 136780
 
def read_occultation(filename):
    label = PdsTable(filename)
    print label.info.label['RING_OPEN_ANGLE']
    occ_tab = Tabulation(
           label.column_dict['RING_RADIUS'],
           label.column_dict['NORMAL OPTICAL DEPTH'])
    return occ_tab

def plot_all_occultations():
    radius_range = np.arange(74000., 139000.)
    filenames = sorted(os.listdir(OCC_ROOT))
    for filename in filenames:
        if not filename.endswith('01KM.LBL'):
            continue
        print filename
        full_path = os.path.join(OCC_ROOT, filename)
        if os.path.isfile(full_path):
            occ_tab = read_occultation(full_path)
            domain = occ_tab.domain()
            if domain[0] > RING_MIN or domain[1] < RING_MAX:
                print 'Incomplete range'
                continue
            results = occ_tab(radius_range)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(radius_range, results, '-', color='black')
            plt.show()
    
    
plot_all_occultations()
