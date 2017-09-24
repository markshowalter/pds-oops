import colorsys
import copy
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

import imgdisp

import oops
import oops.inst.cassini.iss as iss

from cb_util_file import *
from cb_util_misc import *

command_list = sys.argv[1:]

if len(command_list) == 0:
#     command_line_str = '--first-image-num 1454725799 --last-image-num 1455427331'# --use-reconstructed'
#     command_line_str = '--first-image-num 1455327968 --last-image-num 1455331808'# --use-reconstructed' # NAC vs. WAC
    command_line_str = '--volume COISS_2002'#--use-reconstructed'
#     command_line_str = '--last-image-num 1454843222'# --use-reconstructed'
#     command_line_str = '--first-image-num 1454843032 --last-image-num 1454843222'# --use-reconstructed'
#     command_line_str = '--first-image-num 1455335456 --last-image-num 1455367146'# --use-reconstructed' # 6 stares at Saturn from COISS_2001 with different filters
#     command_line_str = '--first-image-num 1455366956 --last-image-num 1455367146'# --use-reconstructed' # 6 stares at Saturn from COISS_2001 with different filters
#     command_line_str = '--first-image-num 1484506476 --last-image-num 1484581154'# --use-reconstructed' # Stares at Enceladus and Mimas with different filters
    
    # FMOVIES
#     command_line_str = '--first-image-num 1492052646 --last-image-num 1492102189 --use-reconstructed' # FMOVIE_006RI
#     command_line_str = '--first-image-num 1545556618 --last-image-num 1545613256'# --use-reconstructed' # FMOVIE_036RF_001
#     command_line_str = '--first-image-num 1598806665 --last-image-num 1598853071'# --use-reconstructed' # FMOVIE_083RI

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Plot RA/DEC offsets')

parser.add_argument(
    '--use-reconstructed', action='store_true',
    help='Use reconstructued kernels instead of predicted')
parser.add_argument(
    '--verbose', action='store_true',
    help='Be verbose')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

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
    'BL2': (0,0,0.8),
    'VIO': (159/256.,0,1),
    'UV3': (103/256.,0,1),

    'IRP0': (0.3,0.3,0.3),
    'IRP90': (0.4,0.4,0.4),
        
    'HAL': (0.5,0.5,0.5),

    'MT1': (1,103/256.,153/256.),
    'CB1': (1,.7,0),

    'MT2': (1,204/256.,153/256.),
    'CB2': (1,.5,0),

    'MT3': (1,153/256.,204/256.),
    'CB3': (1,.3,1),
}    


NAC_RAD_TO_PIX = 1/6e-6

scet_list = []
spice_ra_list = []
spice_dec_list = []
ra_diff = []
dec_diff = []
marker_list = []
color_list = []
scet_min = None

for image_path in file_yield_image_filenames_from_arguments(arguments):
    filename = file_clean_name(image_path)
    if filename[0] != 'N':
        continue
    
    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='no')
    if 'error' in metadata:
        metadata['status'] = 'error'
        metadata['status_detail1'] = metadata['error']
        metadata['status_detail2'] = metadata['error_traceback']
    elif 'status' not in metadata:
        metadata['status'] = 'ok'

    if 'status' not in metadata:
        continue

    if metadata['status'] != 'ok':
        continue
    if metadata['offset'] is None:
        continue

    filter = simple_filter_name_metadata(metadata, consolidate_pol=True)
    
    nav_ra, nav_dec = metadata['ra_dec_center_offset']
    nav_ra *= np.cos(nav_dec)
    if arguments.use_reconstructed:
        spice_ra, spice_dec = metadata['ra_dec_center_orig']
    else:
        pred_metadata = file_read_predicted_metadata(image_path)
        spice_ra = pred_metadata['ra_center_midtime']
        spice_dec = pred_metadata['dec_center_midtime']
    spice_ra *= np.cos(spice_dec)
    
    spice_ra_list.append(spice_ra * oops.DPR)
    spice_dec_list.append(spice_dec * oops.DPR)
    
    ra_pix = (nav_ra-spice_ra)*NAC_RAD_TO_PIX
    dec_pix = (nav_dec-spice_dec)*NAC_RAD_TO_PIX

    if 'xxxscet_midtime' not in metadata:
        scet = metadata['midtime']
    else:
        scet = metadata['scet_midtime']
    if scet_min is None:
        scet_min = scet
        print 'SCET MIN', scet_min
    print len(ra_diff), filename, 'SCET', scet-scet_min, 
    print 'RA+DEC DIFF %6.2f' % np.sqrt(ra_pix**2+dec_pix**2),
    print 'OFFSET %6.2f' % np.sqrt(metadata['offset'][0]**2+
                                   metadata['offset'][1]**2),
    print filter 
    print ra_pix, dec_pix, metadata['offset']
    scet_list.append(scet-scet_min)
    ra_diff.append(ra_pix)
    dec_diff.append(dec_pix)
    if filename[0] == 'N':
        marker = 'o'
    else:
        marker = 'x'
    if filter not in FILTER_COLOR:
        color = 'black'
    else:
        color = FILTER_COLOR[filter]
    marker_list.append(marker)
    color_list.append(color)

fig = plt.figure()

for ra_dec_num in xrange(2):
    if ra_dec_num == 0:
        ra_dec_list = ra_diff
        subplot_num = 211
        ylabel = 'Delta RA*cos(DEC) (NAC pixels)' 
    else:
        ra_dec_list = dec_diff
        subplot_num = 212
        ylabel = 'Delta DEC (NAC pixels)' 
        
    ax = fig.add_subplot(subplot_num)
    
    for i in xrange(len(ra_dec_list)):
        plt.plot(scet_list[i], ra_dec_list[i],
                 marker=marker_list[i], mec=color_list[i],
                 mfc='none', ms=5, mew=2, alpha=1)
    plt.ylabel(ylabel)
if arguments.use_reconstructed:
    title = 'Reconstructed vs. Navigated'
else:
    title = 'Predicted vs. Navigated'
plt.suptitle(title)

fig = plt.figure()

for ra_dec_num in xrange(2):
    if ra_dec_num == 0:
        ra_dec_list = spice_ra_list
        subplot_num = 211
        ylabel = 'SPICE RA*cos(DEC) (degrees)' 
    else:
        ra_dec_list = spice_dec_list
        subplot_num = 212
        ylabel = 'SPICE DEC (degrees)' 
        
    ax = fig.add_subplot(subplot_num)
    
    for i in xrange(len(ra_dec_list)):
        plt.plot(scet_list[i], ra_dec_list[i],
                 'o', color='black', ms=5)
    plt.ylabel(ylabel)

plt.xlabel('Delta SCET')
if arguments.use_reconstructed:
    title = 'Reconstructed SPICE Pointing'
else:
    title = 'Predicted SPICE Pointing'
plt.suptitle(title)
plt.show()
