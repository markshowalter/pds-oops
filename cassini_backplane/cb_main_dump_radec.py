###############################################################################
# cb_main_dump_radec.py
#
# The main top-level driver for dumping results to a CSV file to create new
# CSPICE kernels.
###############################################################################

from cb_logging import *
import logging

import argparse
import os
import sys

import oops.inst.cassini.iss as iss
import oops

from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Dumping Results',
    epilog='''Default behavior is to dump results on all images
              with associated offset files''')

parser.add_argument(
    '--output-file', required=True,
    help='The output file')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

output_fp = open(arguments.output_file, 'w')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='prefer')  
    if not metadata:
        continue
      
    filename = file_clean_name(image_path)

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
    
    offset = metadata['offset']
    if offset is None:
        continue

    winner = metadata['offset_winner']
    if (metadata['bootstrapped']):
        main_nav = '*'
    else:
        main_nav = ''
    if winner == 'STARS':
        main_nav = 'STAR'
    elif winner == 'TITAN':
        main_nav = 'TITA'
    elif winner == 'BOTSIM':
        main_nav = 'BOTS'
    else:
        assert winner == 'MODEL'
        
        rings_metadata = metadata['rings_metadata']
        bodies_metadata = metadata['bodies_metadata']
        model_contents = metadata['model_contents']
        if 'RINGS' in model_contents:
            main_nav = 'RING'
        else:
            main_nav += model_contents[0][:4]
        if len(model_contents) > 1:
            main_nav += '+'
    
    ra, dec = metadata['ra_dec_center_offset']
    scet_midtime = metadata['midtime'] # XXX Needs to be changed to scet_midtime
    
    print >> output_fp, ('%s,%f,%s,%.16f,%.16f' % (filename,
                                                   scet_midtime,
                                                   main_nav,
                                                   ra, dec))
    
output_fp.close()
