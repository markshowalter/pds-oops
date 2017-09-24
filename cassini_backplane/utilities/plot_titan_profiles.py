import colorsys
import copy
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.interpolate as interp

import imgdisp

import oops
import oops.inst.cassini.iss as iss

from cb_config import *
from cb_util_file import *
from cb_util_flux import *

FILTER_COLOR = {
    'CLEAR': (.7,.7,.7),
    
    'IR1': (0.5,0,0),
    'IR2': (0.6,0,0),
    'IR3': (0.7,0,0),
    'IR4': (0.75,0,0),
    'IR5': (0.8,0,0),
    'RED': (1,0,0),
    'GRN': (0,1,0),
    'BL1': (0,0,1),
    'VIO': (159/256.,0,1),

    'IRP0': (0.3,0.3,0.3),
    'IRP90': (0.4,0.4,0.4),
        
    'HAL': (0.5,0.5,0.5),

    'MT2': (1,204/256.,153/256.),
    'CB2': (1,.5,0),

    'MT3': (1,153/256.,204/256.),
    'CB3': (1,.3,1),
}    

FILTER_NUMBER = { # Table VIII Porco et al. 2004
    'CLEAR': 0,         #  611 NAC  635 WAC
                # UV1-UV3  258-338 NAC
    'VIO': 1,           #           420 WAC
                    # BL2  440 NAC
    'BL1': 2,           #  451 NAC  460 WAC
    'GRN': 3,           #  568 NAC  567 WAC
                    # MT1  619 NAC
                    # CB1  619 NAC
                   # CB1a  635 NAC
                   # CB1b  603 NAC
    'RED': 4,           #  650 NAC  648 WAC
    'HAL': 5,           #  656 NAC  656 WAC
    'MT2': 6,           #  727 NAC  728 WAC
    'MT2+IRP0': 7,      #  727 NAC  728 WAC + 746 NAC 705 WAC
    'MT2+IRP90': 8,     #  727 NAC  728 WAC +         705 WAC
    'CB2': 9,           #  750 NAC  752 WAC
    'CB2+IRP0': 10,     #  750 NAC  752 WAC + 746 NAC 705 WAC
    'CB2+IRP90': 11,    #  750 NAC  752 WAC +         705 WAC
    'IR1': 12,          #  752 NAC  742 WAC
    'IR2': 13,          #  862 NAC  853 WAC
    'IR2+IR1': 14,      #  827 NAC  826 WAC (Table IX) 
    'MT3': 15,          #  889 NAC  890 WAC
    'MT3+IRP0': 16,     #  889 NAC  890 WAC + 746 NAC 705 WAC
    'MT3+IRP90': 17,    #  889 NAC  890 WAC +         705 WAC
    'CB3': 18,          #  938 NAC  939 WAC
    'CB3+IRP0': 19,     #  938 NAC  939 WAC + 746 NAC 705 WAC
    'CB3+IRP90': 20,    #  938 NAC  939 WAC +         705 WAC
    'IR3': 21,          #  930 NAC  918 WAC
    'IR4': 22,          # 1002 NAC 1001 WAC
    'IR5': 23,          #          1028 WAC    
}    

NUM_FILTERS = 24
    
def plot_all_filters_by_bin(restrict=None, dir='titan-by-phase'):
    for phase_bin in range(NUM_PHASE_BINS):
        phase = phase_bin * PHASE_BIN_GRANULARITY * oops.DPR
        x_list = []
        y_list = []
        color_list = []
        ls_list = []
        label_list = []
        for filter_number, filter in sorted([(FILTER_NUMBER[x], x) for x in BASELINE_DB]):
            if filter not in BASELINE_DB:
                continue
            if restrict is not None and filter not in restrict:
                continue
            entry = BASELINE_DB[filter][phase_bin]
            if entry is None:
                continue
            ls = '-'
            if filter.endswith('P0'):
                ls = '--'
            elif filter.endswith('P90') or filter.endswith('IRP90'):
                ls = ':'
            orig_filter = filter
            idx = filter.find('+')
            if idx != -1:
                filter = filter[:idx]
            profile_x, profile_y, resolution, num_images, bin_width = entry
#             profile_y = profile_y[profile_x >= 2000]
#             profile_x = profile_x[profile_x >= 2000]
            x_list.append(profile_x)
            y_list.append(profile_y)
            color_list.append(FILTER_COLOR[filter])
            ls_list.append(ls)
            label_list.append(('%s (%d)' % (orig_filter, num_images)))
        if len(x_list):
            plt.figure(figsize=(25,15))
            max_mean = 0.
            for i in range(len(x_list)):
                max_mean = np.max(np.mean(y_list[i]))
            for i in range(len(x_list)):
                plt.plot(x_list[i], y_list[i]*max_mean/np.mean(y_list[i]), 
                         ls_list[i], color=color_list[i],
                         label=label_list[i])
            plt.title('Phase %.0f' % phase)
            plt.legend(loc='upper left')
            plt.savefig(('/home/rfrench/Dropbox-SETI/%s/phase-%03d'%(dir, np.round(phase))), bbox_inches='tight')
    
#     plt.show()

def plot_compare_polarization_by_bin():
    for phase_bin in range(NUM_PHASE_BINS):
        phase = phase_bin * PHASE_BIN_GRANULARITY * oops.DPR
        for base_filter1, base_filter2, base_filter3 in [('CLEAR', 'IRP0', 'IRP90'),
                            ('MT2', 'MT2+IRP0', 'MT2+IRP90'),
                            ('MT3', 'MT3+IRP0', 'MT3+IRP90'),
                            ('CB2', 'CB2+IRP0', 'CB2+IRP90'),
                            ('CB3', 'CB3+IRP0', 'CB3+IRP90')]:
            if (base_filter1 not in BASELINE_DB or
                BASELINE_DB[base_filter1] is None or
                BASELINE_DB[base_filter1][phase_bin] is None or
                base_filter2 not in BASELINE_DB or
                BASELINE_DB[base_filter2] is None or
                BASELINE_DB[base_filter2][phase_bin] is None or
                base_filter3 not in BASELINE_DB or
                BASELINE_DB[base_filter3] is None or 
                BASELINE_DB[base_filter3][phase_bin] is None):
                continue
            if (BASELINE_DB[base_filter1][phase_bin][3] < 1 or
                BASELINE_DB[base_filter2][phase_bin][3] < 1 or
                BASELINE_DB[base_filter3][phase_bin][3] < 1):
                continue
            plt.figure(figsize=(25,15))
            for filter, color in [(base_filter1, 'red'),
                           (base_filter2, 'green'),
                           (base_filter3, 'blue')]:
                entry = BASELINE_DB[filter][phase_bin]
#                 if entry is None:
#                     continue
#                 ls = '-'
#                 if filter.endswith('P0'):
#                     ls = '--'
#                 elif filter.endswith('P90'):
#                     ls = ':'
                orig_filter = filter
                idx = filter.find('+')
                if idx != -1:
                    filter = filter[:idx]
                profile_x, profile_y, resolution, num_images, bin_width = entry
#                 profile_y = profile_y[profile_x >= 2000]
#                 profile_x = profile_x[profile_x >= 2000]
                max_idx = np.argmax(profile_y)
                max_x = profile_x[max_idx]
                if not orig_filter.endswith('0'):
                    max_y = np.max(profile_y)
                    mean_y = np.mean(profile_y)
                else:
                    profile_y = profile_y * mean_y / np.mean(profile_y)
                plt.plot(profile_x, profile_y, '-', color=color,
                         label=('%s (%d)' % (orig_filter, num_images)))
                plt.plot([max_x, max_x], [0,max_y], '-', lw=2, color=color)
            plt.title('Phase %.0f' % phase)
            plt.legend(loc='upper left')
            plt.savefig(('/home/rfrench/Dropbox-SETI/titan-pol/pol-%03d-%s'%(np.round(phase),filter)), bbox_inches='tight')
        
#     plt.show()

# plot_cassini_filter_transmission()

filename = os.path.join(CB_SUPPORT_FILES_ROOT,
                        'titan-profiles.pickle')
fp = open(filename, 'rb')
PHASE_BIN_GRANULARITY = pickle.load(fp)
PHASE_BIN_WIDTH = pickle.load(fp)
BASELINE_DB = pickle.load(fp)
fp.close()

print sorted(BASELINE_DB.keys())

NUM_PHASE_BINS = int(np.ceil(oops.PI / PHASE_BIN_GRANULARITY))+1

# plot_all_filters_by_bin()
# plot_all_filters_by_bin(restrict=['MT2', 'CB2', 'IR1'], dir='titan-by-phase-mt1')
plot_all_filters_by_bin(restrict=['BL1', 'VIO'], dir='titan-by-phase-bl1-vio')
# plot_compare_polarization_by_bin()
