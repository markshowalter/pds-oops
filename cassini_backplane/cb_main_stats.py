###############################################################################
# cb_main_offset.py
#
# The main top-level driver for all of CB.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
from datetime import datetime
import os
import subprocess
import sys
import time
import traceback

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file --verbose --volume COISS_2024'

    command_list = command_line_str.split()

## XXX Check restrict image list is included in first->last range 

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Statistics',
    epilog='''Default behavior is to collect statistics on all images
              with associated offset files''')

parser.add_argument(
    '--verbose', action='store_true',
    help='Be verbose')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

total_files = 0
total_offset = 0
total_spice_error = 0
total_other_error = 0
total_good_offset = 0
total_good_offset_list = []
total_good_star_offset = 0
total_good_model_offset = 0
total_body_only_db = {}
total_rings_only = 0
total_bootstrap_cand = 0
time_list = []

for image_path in file_yield_image_filenames_from_arguments(arguments):
    status = ''
    _, base = os.path.split(image_path)
    status += base + ': '
    total_files += 1

    metadata = file_read_offset_metadata(image_path)
    filename = file_clean_name(image_path)
    status = filename + ' - ' + offset_result_str(metadata)
    
    if metadata is not None:
        if 'error' in metadata:
            error = metadata['error']
            if error.startswith('SPICE(NOFRAMECONNECT)'):
                total_spice_error += 1
            else:
                total_other_error += 1
        else:
            time_list.append(metadata['end_time']-metadata['start_time'])
            
            offset = metadata['offset']
            if offset is not None:
                total_good_offset += 1
                total_good_offset_list.append(tuple(offset))
                
            if metadata['stars_metadata'] is not None:
                star_offset = metadata['stars_metadata']['offset']
                if star_offset is not None:
                    total_good_star_offset += 1
        
            model_offset = metadata['model_offset']
            if model_offset is not None:
                total_good_model_offset += 1
        
            body_only = metadata['body_only']
            if body_only:
                if body_only not in total_body_only_db:
                    total_body_only_db[body_only] = 0
                total_body_only_db[body_only] = total_body_only_db[body_only]+1
                
            if metadata['rings_only']:
                total_rings_only += 1
        
            if metadata['bootstrap_candidate']:
                total_bootstrap_cand += 1
                
    if arguments.verbose:
        print status

print 'Total image files:', total_files
print 'Total with offset file:', total_offset
print 'Run time: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (np.min(time_list),
                                                          np.max(time_list),
                                                          np.mean(time_list),
                                                          np.std(time_list))
if total_offset:
    print 'Spice error: %d (%.2f%%)' % (total_spice_error, float(total_spice_error)/total_offset*100)
    print 'Other error: %d (%.2f%%)' % (total_other_error, float(total_other_error)/total_offset*100)
    print
    print 'Good final offset: %d (%.2f%%)' % (total_good_offset, float(total_good_offset)/total_offset*100)
    print '  Good star offset: %d (%.2f%%)' % (total_good_star_offset, float(total_good_star_offset)/total_offset*100)
    print '  Good model offset: %d (%.2f%%)' % (total_good_model_offset, float(total_good_model_offset)/total_offset*100)
    print 'Bootstrap candidate: %d (%.2f%%)' % (total_bootstrap_cand, float(total_bootstrap_cand)/total_offset*100)
    failed = total_offset-total_good_offset-total_bootstrap_cand
    print 'Failed and not bootstrap candidate: %d (%.2f%%)' % (failed, float(failed)/total_offset*100)
    print
    print 'Total body-only:'
    for body_name in sorted(total_body_only_db):
        print '    %s: %d' % (body_name, total_body_only_db[body_name]) 
    print
    print 'Rings only: %d (%.2f%%)' % (total_rings_only, float(total_rings_only)/total_offset*100)
