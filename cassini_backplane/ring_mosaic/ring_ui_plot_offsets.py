import argparse
import os
import os.path
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ring_util

from cb_util_file import *
from cb_config import *

POSTER = True

color_background = (1,1,1)
color_foreground = (0,0,0)
markersize = 8

if POSTER:
    markersize = 8
    matplotlib.rc('figure', facecolor=color_background)
    matplotlib.rc('axes', facecolor=color_background, edgecolor=color_foreground, labelcolor=color_foreground)
    matplotlib.rc('xtick', color=color_foreground, labelsize=18)
    matplotlib.rc('xtick.major', size=0)
    matplotlib.rc('xtick.minor', size=0)
    matplotlib.rc('ytick', color=color_foreground, labelsize=18)
    matplotlib.rc('ytick.major', size=0)
    matplotlib.rc('ytick.minor', size=0)
    matplotlib.rc('font', size=18)
    matplotlib.rc('legend', fontsize=24)

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--ring-type', 'BRING_MOUNTAINS', '--all-obsid']
#                 '-a',
#'ISS_115RF_FMOVIEEQX001_PRIME',
#                 'ISS_029RF_FMOVIE001_VIMS',
#                 'ISS_044RF_FMOVIE001_VIMS',
#                'ISS_106RF_FMOVIE002_PRIME',
#                'ISS_132RI_FMOVIE001_VIMS',
#                'ISS_029RF_FMOVIE002_VIMS',

parser = argparse.ArgumentParser()

ring_util.ring_add_parser_options(parser)

parser.add_argument('--include-stars', action='store_true', default=False,
                  help='Include # of stars subplot')

arguments = parser.parse_args(cmd_line)

ring_util.ring_init(arguments)

DIR_ROOT = os.path.join(CB_RESULTS_ROOT, 'ring_mosaic', 'offset_plots_'+arguments.ring_type)

try:
    os.mkdir(DIR_ROOT)
except:
    pass

#####################################################################################
#
# 
#
#####################################################################################

g_num_images = 0
g_num_no_attempt = 0
g_num_no_offset = 0
g_num_used_stars = 0
g_num_used_model = 0
g_num_model_overrides = 0

verbose = True

def plot_obsid(obsid, image_path_list):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    offset_winner_list = []

    global g_num_images
    global g_num_no_attempt
    global g_num_no_offset
    global g_num_used_stars
    global g_num_used_model
    global g_num_model_overrides
    
    num_images = 0
    num_no_attempt = 0
    num_no_offset = 0
    num_used_stars = 0
    num_used_model = 0
    num_model_overrides = 0
        
    for image_path in image_path_list:
        metadata = file_read_offset_metadata(image_path)
        num_images += 1
        status = metadata['status']
        if status != 'ok':
            num_no_attempt += 1
            continue
        offset = metadata['offset']
        object_type = None
        stars_metadata = metadata['stars_metadata']
        if offset is None:
            num_no_offset += 1
            offset_u_list.append(None)
            offset_v_list.append(None)
            num_stars_list.append(-1)
            num_good_stars_list.append(-1)
            offset_winner_list.append('STARS')
        else:
            offset_u_list.append(offset[0])
            offset_v_list.append(offset[1])
            if stars_metadata is not None:
                num_stars_list.append(stars_metadata['num_stars'])
                num_good_stars_list.append(stars_metadata['num_good_stars'])
            else:
                num_stars_list.append(-1)
                num_good_stars_list.append(-1)
            winner = metadata['offset_winner']
            offset_winner_list.append(winner)
    
    if len(offset_u_list) == 0:
        return
    
    x_min = -0.5
    x_max = len(offset_u_list)-0.5
    
    if POSTER:
        fig = plt.figure(figsize=(9,5))
    else:
        fig = plt.figure(figsize=(17,11))
    
    u_color = '#3399ff'
    v_color = '#0000cc'

    if arguments.include_stars:        
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_subplot(111)
    
    plt.plot(offset_u_list, '-', color=u_color, ms=5)
    plt.plot(offset_v_list, '-', color=v_color, ms=5)
    for i in xrange(len(offset_u_list)):
        if offset_u_list[i] is not None and offset_v_list[i] is not None:
            if offset_winner_list[i] == 'STARS':
                num_used_stars += 1
                plt.plot(i, offset_u_list[i], '*', mec=u_color, mfc=u_color, ms=markersize*1.25)
                plt.plot(i, offset_v_list[i], '*', mec=v_color, mfc=v_color, ms=markersize*1.25)
            elif offset_winner_list[i] == 'MODEL':
                num_used_model += 1
                plt.plot(i, offset_u_list[i], 'o', mec=u_color, mfc='none', ms=markersize, mew=1)
                plt.plot(i, offset_v_list[i], 'o', mec=v_color, mfc='none', ms=markersize, mew=1)
            else:
                plt.plot(i, offset_u_list[i], '^', mec=u_color, mfc='none', ms=markersize*1.5, mew=2)
                plt.plot(i, offset_v_list[i], '^', mec=v_color, mfc='none', ms=markersize*1.5, mew=2)
    plt.xlim(x_min, x_max)
    ax.set_xticklabels('')
#     if POSTER:
#         ax.get_yaxis().set_ticks([-30,-20,-10,0,10])
    plt.ylabel('Pixel Offset')
    if not arguments.include_stars:
        plt.xlabel('Image Number')

    if not POSTER:
        plt.title('X/Y Offset')
    
#     if not arguments.include_stars:
#         plt.title(obsid)
        
    ax.yaxis.set_label_coords(-0.055, 0.5)
    
    stars_color = '#ff8000'
    good_color = '#336600'
    
    if arguments.include_stars:
        ax = fig.add_subplot(212)
        plt.plot(num_stars_list, '-o', color=stars_color, mec=stars_color, mfc=stars_color, ms=markersize*.5)
        plt.plot(num_good_stars_list, '-o', color=good_color, mec=good_color, mfc=good_color, ms=markersize*.55)
        plt.xlim(x_min, x_max)
        plt.ylim(-0.5, max(np.max(num_good_stars_list),
                           np.max(num_stars_list))+0.5)
        plt.ylabel('# of Good Stars')
        plt.xlabel('Image Number')
        if POSTER:
            ax.get_xaxis().set_ticks([0,174])
            ax.get_yaxis().set_ticks([0,10,20,30])
            plt.xticks([0,174],['1','175'])
        if not POSTER:
            plt.title('Total Stars vs. Good Stars')
    
        ax.yaxis.set_label_coords(-0.055, 0.5)
        
    if not POSTER:
        plt.suptitle(obsid)
    
    plt.subplots_adjust(left=0.025, right=0.975, top=1., bottom=0.0, hspace=0.18)
    filename = os.path.join(DIR_ROOT, obsid+'.png')
    if POSTER:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.close()
    
    print '%-40s %4d %4d %4d %4d %4d %4d' % (obsid, num_images, num_no_attempt, num_no_offset,
                                             num_used_stars, num_used_model, num_model_overrides)
    
    g_num_images += num_images
    g_num_no_attempt += num_no_attempt
    g_num_no_offset += num_no_offset
    g_num_used_stars += num_used_stars
    g_num_used_model += num_used_model
    g_num_model_overrides += num_model_overrides
    
    
#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

print '%-40s #IMG  ERR NOFF STAR MODL OVER' % ('OBSID')

cur_obsid = None
image_path_list = []
for obsid, image_name, image_path in ring_util.ring_enumerate_files(arguments):
#    print obsid, image_name
    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(image_path_list) != 0:
            plot_obsid(cur_obsid, image_path_list)
        obsid_list = []
        image_path_list = []
        cur_obsid = obsid
    image_path_list.append(image_path)
    
# Final mosaic
if len(image_path_list) != 0:
    plot_obsid(cur_obsid, image_path_list)

print
print '%-40s %4d %4d %4d %4d %4d %4d' % ('TOTAL', g_num_images, g_num_no_attempt, g_num_no_offset,
                         g_num_used_stars, g_num_used_model, g_num_model_overrides)
