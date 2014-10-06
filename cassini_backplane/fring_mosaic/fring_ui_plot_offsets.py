'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import fring_util
import os
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--verbose',
#                '-a',
'ISS_115RF_FMOVIEEQX001_PRIME',
#                'ISS_029RF_FMOVIE002_VIMS',
#                'ISS_041RF_FMOVIE002_VIMS',
#                'ISS_106RF_FMOVIE002_PRIME',
#                'ISS_132RI_FMOVIE001_VIMS',
#                'ISS_029RF_FMOVIE002_VIMS',
                ]

parser = OptionParser()

#
# The default behavior is to check the timestamps
# on the input file and the output file and recompute if the output file is out of date.
# Several options change this behavior:
#   --no-xxx: Don't recompute no matter what; this may leave you without an output file at all
#   --no-update: Don't recompute if the output file exists, but do compute if the output file doesn't exist at all
#   --recompute-xxx: Force recompute even if the output file exists and is current
#


##
## General options
##
parser.add_option('--allow-exception', dest='allow_exception',
                  action='store_true', default=False,
                  help="Allow exceptions to be thrown")

fring_util.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

#####################################################################################
#
# 
#
#####################################################################################

def plot_obsid(obsid, image_name_list, offset_path_list):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    used_objects_type_list = []
    
    for offset_path in offset_path_list:
        auto_offset, manual_offset, metadata = fring_util.read_offset(offset_path)
        if metadata is None:
            continue
        print auto_offset
        if auto_offset is None:
            offset_u_list.append(None)
            offset_v_list.append(None)
        else:
            offset_u_list.append(auto_offset[0])
            offset_v_list.append(auto_offset[1])
        try:
            num_stars_list.append(len(metadata['full_star_list']))
            num_good_stars_list.append(metadata['num_good_stars'])
            try:
                used_objects_type_list.append(metadata['used_objects_type'])
            except:
                if metadata['num_good_stars'] >= 3:
                    used_objects_type_list.append('stars')
                else:
                    used_objects_type_list.append('model')
        except:
            num_stars_list.append(-1)
            num_good_stars_list.append(-1)
            used_objects_type_list.append('stars')
    
    if len(offset_u_list) == 0:
        return
    
    x_min = -0.5
    x_max = len(offset_u_list)-0.5
    
    fig = plt.figure(figsize=(17,11))
    ax = fig.add_subplot(211)
    plt.plot(offset_u_list, '-', color='red', ms=5)
    plt.plot(offset_v_list, '-', color='green', ms=5)
    for i in xrange(len(offset_u_list)):
#        if num_good_stars_list[i] >= 3:
        if used_objects_type_list[i] == 'stars':
            plt.plot(i, offset_u_list[i], '*', mec='red', mfc='red', ms=10)
            plt.plot(i, offset_v_list[i], '*', mec='green', mfc='green', ms=10)
        else:
            plt.plot(i, offset_u_list[i], 'o', mec='red', mfc='none', ms=8, mew=1)
            plt.plot(i, offset_v_list[i], 'o', mec='green', mfc='none', ms=8, mew=1)
    plt.xlim(x_min, x_max)
    plt.title('U/V Offset')
    
    ax = fig.add_subplot(212)
    plt.plot(num_stars_list, '-o', color='red', mec='red', mfc='red', ms=7)
    plt.plot(num_good_stars_list, '-o', color='black', mec='black', mfc='black', ms=4)
    plt.xlim(x_min, x_max)
    plt.ylim(-0.5, np.max(num_stars_list)+0.5)
    plt.title('Total stars vs. Good stars')
    
    plt.suptitle(obsid)
    
    plt.subplots_adjust(left=0.025, right=0.975, top=0.925, bottom=0.025, hspace=0.15)
    plt.savefig('j:/Temp/plots/'+obsid+'.png', bbox_inches='tight')
    
#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

cur_obsid = None
image_name_list = []
image_path_list = []
offset_path_list = []
for obsid, image_name, image_path in fring_util.enumerate_files(options, args, '_CALIB.IMG'):
    print obsid, image_name
    offset_path = fring_util.offset_path(options, image_path, image_name)
    
    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(image_name_list) != 0:
            plot_obsid(cur_obsid, image_name_list, offset_path_list)
        obsid_list = []
        image_name_list = []
        image_path_list = []
        offset_path_list = []
        cur_obsid = obsid
    image_name_list.append(image_name)
    offset_path_list.append(offset_path)
    
# Final mosaic
if len(image_name_list) != 0:
    plot_obsid(cur_obsid, image_name_list, offset_path_list)
