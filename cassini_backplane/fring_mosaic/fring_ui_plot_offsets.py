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
                '-a',
#'ISS_115RF_FMOVIEEQX001_PRIME',
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

g_num_images = 0
g_num_no_attempt = 0
g_num_no_offset = 0
g_num_old_format = 0
g_num_used_stars = 0
g_num_used_model = 0
g_num_model_overrides = 0

verbose = True

def plot_obsid(obsid, image_name_list, offset_path_list):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    used_objects_type_list = []

    global g_num_images
    global g_num_no_attempt
    global g_num_no_offset
    global g_num_old_format
    global g_num_used_stars
    global g_num_used_model
    global g_num_model_overrides
    
    num_images = 0
    num_no_attempt = 0
    num_no_offset = 0
    num_old_format = 0
    num_used_stars = 0
    num_used_model = 0
    num_model_overrides = 0
        
    for offset_path in offset_path_list:
        auto_offset, manual_offset, metadata = fring_util.read_offset(offset_path)
        num_images += 1
        if metadata is None:
            num_no_attempt += 1
            continue
        old_format = 0
        object_type = None
        if auto_offset is None:
            num_no_offset += 1
            offset_u_list.append(None)
            offset_v_list.append(None)
            num_stars_list.append(-1)
            num_good_stars_list.append(-1)
            used_objects_type_list.append('stars')
        else:
            offset_u_list.append(auto_offset[0])
            offset_v_list.append(auto_offset[1])
            try:
                num_stars_list.append(len(metadata['full_star_list']))
                num_good_stars_list.append(metadata['num_good_stars'])
                try:
                    object_type = metadata['used_objects_type']
                except:
                    old_format = 1
                    if verbose:
                        print 'OLD FORMAT NO USED_OBJECTS_TYPE', offset_path
                    if metadata['num_good_stars'] >= 3:
                        object_type = 'stars'
                    else:
                        object_type = 'model'
            except:
                if verbose:
                    print 'OLD FORMAT NO NUM_GOOD_STARS', offset_path
                old_format = 1
                num_stars_list.append(-1)
                num_good_stars_list.append(-1)
                used_objects_type_list.append('oldformat')
            try:
                if metadata['model_overrides_stars']:
                    object_type = 'override'
                    num_model_overrides += 1
            except:
                if verbose:
                    print 'OLD FORMAT NO MODEL_OVERRIDES_STAR', offset_path
                old_format = 1
            
            num_old_format += old_format
            used_objects_type_list.append(object_type)
    
    if len(offset_u_list) == 0:
        return
    
    x_min = -0.5
    x_max = len(offset_u_list)-0.5
    
    fig = plt.figure(figsize=(17,11))
    ax = fig.add_subplot(211)
    plt.plot(offset_u_list, '-', color='red', ms=5)
    plt.plot(offset_v_list, '-', color='green', ms=5)
    for i in xrange(len(offset_u_list)):
        if offset_u_list[i] is not None and offset_v_list[i] is not None:
            if used_objects_type_list[i] == 'stars':
                num_used_stars += 1
                plt.plot(i, offset_u_list[i], '*', mec='red', mfc='red', ms=10)
                plt.plot(i, offset_v_list[i], '*', mec='green', mfc='green', ms=10)
            elif used_objects_type_list[i] == 'model':
                num_used_model += 1
                plt.plot(i, offset_u_list[i], 'o', mec='red', mfc='none', ms=8, mew=1)
                plt.plot(i, offset_v_list[i], 'o', mec='green', mfc='none', ms=8, mew=1)
            elif used_objects_type_list[i] == 'override':
                num_used_model += 1
                plt.plot(i, offset_u_list[i], 'o', mec='red', mfc='red', ms=8, mew=1)
                plt.plot(i, offset_v_list[i], 'o', mec='green', mfc='green', ms=8, mew=1)
            else:
                plt.plot(i, offset_u_list[i], '^', mec='red', mfc='none', ms=12, mew=2)
                plt.plot(i, offset_v_list[i], '^', mec='green', mfc='none', ms=12, mew=2)
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
    
    plt.close()
    
    print '%-40s %4d %4d %4d %4d %4d %4d %4d' % (obsid, num_images, num_no_attempt, num_no_offset,
                                             num_old_format, num_used_stars, num_used_model, num_model_overrides)
    
    g_num_images += num_images
    g_num_no_attempt += num_no_attempt
    g_num_no_offset += num_no_offset
    g_num_old_format += num_old_format
    g_num_used_stars += num_used_stars
    g_num_used_model += num_used_model
    g_num_model_overrides += num_model_overrides
    
    
#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

print '%40s #IMG EXST FOFF OLDF STAR MODL OVER' % ('OBSID')

cur_obsid = None
image_name_list = []
image_path_list = []
offset_path_list = []
for obsid, image_name, image_path in fring_util.enumerate_files(options, args, '_CALIB.IMG'):
#    print obsid, image_name
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

print
print '%-40s %4d %4d %4d %4d %4d %4d %4d' % ('TOTAL', g_num_images, g_num_no_attempt, g_num_no_offset,
                         g_num_old_format, g_num_used_stars, g_num_used_model, g_num_model_overrides)
