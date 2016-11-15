###############################################################################
# compare_botsim_offsets.py
#
# Compare offsets for WAC and NAC BOTSIM images.
###############################################################################

from cb_logging import *
import logging

import argparse
import math
import numpy as np
import os
import sys

from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Compare BOTSIM offsets',
    epilog='''Default behavior is to collect statistics on all images
              with associated offset files''')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

stars_diff_x_list = []
stars_diff_y_list = []
model_diff_x_list = []
model_diff_y_list = []

last_nac_filename = None
last_nac_image_path = None
last_nac_offset = None
last_nac_winner = None

for image_path in file_yield_image_filenames_from_arguments(arguments):
    status = ''
    _, base = os.path.split(image_path)

    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='prefer')
    if metadata is None:
        continue
        
    filename = file_clean_name(image_path)
    short_filename = file_img_to_short_img_path(image_path)

    if filename[0] == 'N':
        last_nac_filename = filename
        last_nac_image_path = image_path
        last_nac_offset = None
        
    # Fix up the metadata for old files - eventually this should
    # be removed! XXX
    if 'error' in metadata:
        metadata['status'] = 'error'
        metadata['status_detail1'] = metadata['error']
        metadata['status_detail2'] = metadata['error_traceback']
    elif 'status' not in metadata:
        metadata['status'] = 'ok'

    status = metadata['status']
    if status == 'error' or status == 'skipped':
        continue
    
    if metadata['image_shape'][0] != 1024:
        continue
    
    offset = metadata['offset']
    winner = metadata['offset_winner']
    if filename[0] == 'N':
        last_nac_offset = offset
        last_nac_winner = winner
    if offset is None:
        continue
    
    max_offset = MAX_POINTING_ERROR[(tuple(metadata['image_shape']),
                                     metadata['camera'])]
    if (abs(offset[0]) > max_offset[0] or
        abs(offset[1]) > max_offset[1]):
        print 'WARNING - ', filename, '-',
        print 'Offset', winner, offset, 'exceeds maximum', max_offset 

    if (last_nac_filename is None or 
        filename[0] != 'W' or
        filename[1:] != last_nac_filename[1:]):
        continue
        
    if (last_nac_offset is  None or offset is None or
        last_nac_winner == 'BOTSIM'):
        continue

    diff_x = last_nac_offset[0]-offset[0]*10
    diff_y = last_nac_offset[1]-offset[1]*10
    print short_filename, 'NAC', last_nac_winner, 'WAC', winner,
    print 'DIFF X', diff_x, 'DIFF Y', diff_y
    if last_nac_winner == 'STARS' and winner == 'STARS':
        stars_diff_x_list.append(diff_x)                
        stars_diff_y_list.append(diff_y)
    if last_nac_winner == 'MODEL' and winner == 'MODEL':
        model_diff_x_list.append(diff_x)                
        model_diff_y_list.append(diff_y)

if len(stars_diff_x_list) > 0:
    print 'STARS X DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f MEDIAN %.2f' % (
                                            np.min(stars_diff_x_list),
                                            np.max(stars_diff_x_list),
                                            np.mean(stars_diff_x_list),
                                            np.std(stars_diff_x_list),
                                            np.median(stars_diff_x_list))
    print 'STARS Y DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f MEDIAN %.2f' % (
                                            np.min(stars_diff_y_list),
                                            np.max(stars_diff_y_list),
                                            np.mean(stars_diff_y_list),
                                            np.std(stars_diff_y_list),
                                            np.median(stars_diff_y_list))
    print
    
if len(model_diff_x_list) > 0:
    print 'MODEL X DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f MEDIAN %.2f' % (
                                            np.min(model_diff_x_list),
                                            np.max(model_diff_x_list),
                                            np.mean(model_diff_x_list),
                                            np.std(model_diff_x_list),
                                            np.median(model_diff_x_list))
    print 'MODEL Y DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f MEDIAN %.2f' % (
                                            np.min(model_diff_y_list),
                                            np.max(model_diff_y_list),
                                            np.mean(model_diff_y_list),
                                            np.std(model_diff_y_list),
                                            np.median(model_diff_y_list))
