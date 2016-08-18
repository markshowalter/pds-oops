###############################################################################
# cb_main_stats.py
#
# The main top-level driver for offset file statistics.
###############################################################################

from cb_logging import *
import logging

import argparse
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_titan import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--has-offset-file'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Statistics',
    epilog='''Default behavior is to collect statistics on all images
              with associated offset files''')

parser.add_argument(
    '--verbose', action='store_true',
    help='Be verbose')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

# From http://pyinsci.blogspot.com/2009/10/ascii-histograms.html
class Histogram(object):
    """
    Ascii histogram
    """
    def __init__(self, data, bins=10):
        """
        Class constructor
        
        :Parameters:
            - `data`: array like object
        """
        self.data = data
        self.bins = bins
        self.h = np.histogram(self.data, bins=self.bins)
    def horizontal(self, height=4, character ='|'):
        """Returns a multiline string containing a
        a horizontal histogram representation of self.data
        :Parameters:
            - `height`: Height of the histogram in characters
            - `character`: Character to use
        >>> d = normal(size=1000)
        >>> h = Histogram(d,bins=25)
        >>> print h.horizontal(5,'|')
        106            |||
                      |||||
                      |||||||
                    ||||||||||
                   |||||||||||||
        -3.42                         3.09
        """
        his = """"""
        bars = self.h[0]/float(max(self.h[0]))*height
        for l in reversed(range(1,height+1)):
            line = ""
            if l == height:
                line = '%s '%max(self.h[0]) #histogram top count
            else:
                line = ' '*(len(str(max(self.h[0])))+1) #add leading spaces
            for c in bars:
                if c >= math.ceil(l):
                    line += character
                else:
                    line += ' '
            line +='\n'
            his += line
        his += '%.2f'%self.h[1][0] + ' '*(self.bins) +'%.2f'%self.h[1][-1] + '\n'
        return his
    def vertical(self,height=20, character ='|'):
        """
        Returns a Multi-line string containing a
        a vertical histogram representation of self.data
        :Parameters:
            - `height`: Height of the histogram in characters
            - `character`: Character to use
        >>> d = normal(size=1000)
        >>> Histogram(d,bins=10)
        >>> print h.vertical(15,'*')
                              236
        -3.42:
        -2.78:
        -2.14: ***
        -1.51: *********
        -0.87: *************
        -0.23: ***************
        0.41 : ***********
        1.04 : ********
        1.68 : *
        2.32 :
        """
        his = """"""
        xl = ['%.2f'%n for n in self.h[1]]
        lxl = [len(l) for l in xl]
        bars = self.h[0]/float(max(self.h[0]))*height
        his += ' '*(max(bars)+2+max(lxl))+'%s\n'%max(self.h[0])
        for i,c in enumerate(bars):
            line = xl[i] +' '*(max(lxl)-lxl[i])+': '+ character*c+'\n'
            his += line
        return his

bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
max_num_longest_time = 10

total_files = 0
total_offset = 0
total_spice_error = 0
total_other_error = 0
other_error_db = {}
other_error_file_db = {}
total_good_offset = 0
total_good_offset_list = []
total_good_star_offset = 0
total_good_model_nobs_offset = 0
total_good_model_bs_offset = 0
total_good_titan_offset = 0
total_winner_star = 0
total_winner_model = 0
total_winner_titan = 0
total_winner_botsim = 0
body_only_db = {}
total_rings_only = 0
total_bootstrap_cand = 0
bootstrap_cand_db = {}
titan_status_db = {}
total_titan_attempt = 0
time_list = []
longest_time_filenames = []

for image_path in file_yield_image_filenames_from_arguments(arguments):
    status = ''
    _, base = os.path.split(image_path)
    status += base + ': '
    total_files += 1

    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='prefer')    
    filename = file_clean_name(image_path)
    status = filename + ' - ' + offset_result_str(metadata)
    
    if metadata is not None:
        total_offset += 1
        
        if 'error' in metadata:
            error = metadata['error']
            if error == '':
                error = metadata['error_traceback'].split('\n')[-2]
            if error.startswith('SPICE(NOFRAMECONNECT)'):
                total_spice_error += 1
            else:
                total_other_error += 1
                if error not in other_error_db:
                    other_error_db[error] = 0
                    other_error_file_db[error] = filename
                other_error_db[error] += 1
        else:
            total_time = metadata['end_time']-metadata['start_time']
            time_list.append(total_time)
            longest_time_filenames.append((total_time, filename))
            longest_time_filenames.sort(reverse=True)
            longest_time_filenames = longest_time_filenames[:max_num_longest_time]
                
            offset = metadata['offset']
            if offset is not None:
                total_good_offset += 1
                total_good_offset_list.append(tuple(offset))
                
            stars_offset = metadata['stars_offset']
            if stars_offset is not None:
                total_good_star_offset += 1
        
            model_offset = metadata['model_offset']
            if model_offset is not None:
                if metadata['bootstrapped']:
                    total_good_model_bs_offset += 1
                else:
                    total_good_model_nobs_offset += 1
        
            titan_offset = metadata['titan_offset']
            if titan_offset is not None:
                total_good_titan_offset += 1
        
            winner = metadata['offset_winner']
            if winner == 'STARS':
                total_winner_star += 1
            elif winner == 'MODEL':
                total_winner_model += 1
            elif winner == 'TITAN':
                total_winner_titan += 1
            elif winner == 'BOTSIM':
                total_winner_botsim += 1
            elif winner is not None:
                print image_path, 'Unknown offset winner', winner
            body_only = metadata['body_only']
            if body_only:
                if body_only not in body_only_db:
                    body_only_db[body_only] = 0
                body_only_db[body_only] += 1
                
            if metadata['rings_only']:
                total_rings_only += 1

            titan_metadata = metadata['titan_metadata']
            if titan_metadata is not None:
                total_titan_attempt += 1
                titan_status = titan_metadata_to_status(titan_metadata)
                if titan_status not in titan_status_db:
                    titan_status_db[titan_status] = 0
                titan_status_db[titan_status] += 1
                
            bootstrap_cand = False        
            if (metadata['bootstrap_candidate'] or
                metadata['bootstrapped']):
                bodies_metadata = metadata['bodies_metadata']
                if bodies_metadata is not None:
                    for body_name in metadata['large_bodies']:
                        if (body_name not in bootstrap_config['body_list'] or
                            body_name in FUZZY_BODY_LIST or
                            body_name == 'TITAN'):
                            continue
                        if body_name not in bodies_metadata:
                            continue
                        body_metadata = bodies_metadata[body_name]
                        if not body_metadata['size_ok']:
                            continue
                        if body_name not in bootstrap_cand_db:
                            cand_count = 0
                            bootstrap_status_db = {}
                        else:
                            (cand_count, 
                             bootstrap_status_db) = bootstrap_cand_db[body_name]
                        bootstrap_cand = True
                        bootstrap_status = 'Not bootstrapped'
                        if metadata['bootstrapped']:
                            bootstrap_status = metadata['bootstrap_status']
                        if bootstrap_status not in bootstrap_status_db:
                            bootstrap_status_db[bootstrap_status] = 0
                        bootstrap_status_db[bootstrap_status] += 1
                        
                        bootstrap_cand_db[body_name] = (cand_count+1, 
                                                        bootstrap_status_db)
                        status += '  %s: %s' % (body_name, bootstrap_status)
                    
                        break # Only allow one bootstrap body for now XXX

            if bootstrap_cand:
                total_bootstrap_cand += 1
                    
    if arguments.verbose:
        print status

sep = '-' * 55

print sep
print 'Total image files:                  %6d' % total_files
print 'Total with offset file:             %6d' % total_offset
print sep

if total_offset:
    print 'Spice error:                        %6d (%6.2f%%)' % (
                total_spice_error, 
                float(total_spice_error)/total_offset*100)
    print 'Other error:                        %6d (%6.2f%%)' % (
                total_other_error, 
                float(total_other_error)/total_offset*100)
    if total_other_error:
        for error in sorted(other_error_db.keys()):
            print '  %6d (%6.2f%%): %s (%s)' % (
                other_error_db[error],
                float(total_other_error)/total_other_error*100,
                error, 
                other_error_file_db[error])
    print sep

    print 'Good final offset: (%% of non-err)   %6d (%6.2f%%)' % (
                total_good_offset, 
                float(total_good_offset)/total_offset*100)
    print '  Good star  offset:   %6d (%6.2f%%, %6.2f%% of total)' % (
                total_good_star_offset, 
                float(total_good_star_offset)/total_good_offset*100,
                float(total_good_star_offset)/total_offset*100)
    total_good_model_offset = total_good_model_bs_offset+total_good_model_nobs_offset
    print '  Good model offset:   %6d (%6.2f%%, %6.2f%% of total)' % (
                total_good_model_offset, 
                float(total_good_model_offset)/total_good_offset*100,
                float(total_good_model_offset)/total_offset*100)
    print '    Bootstrapped:      %6d (%6.2f%%, %6.2f%% of total)' % (
                total_good_model_bs_offset, 
                float(total_good_model_bs_offset)/total_good_offset*100,
                float(total_good_model_bs_offset)/total_offset*100)
    print '    Not bootstrapped:  %6d (%6.2f%%, %6.2f%% of total)' % (
                total_good_model_nobs_offset, 
                float(total_good_model_nobs_offset)/total_good_offset*100,
                float(total_good_model_nobs_offset)/total_offset*100)
    print '  Good Titan offset:   %6d (%6.2f%%, %6.2f%% of total)' % (
                total_good_titan_offset, 
                float(total_good_titan_offset)/total_good_offset*100,
                float(total_good_titan_offset)/total_offset*100)
    print '  Good BOTSIM offset:  %6d (%6.2f%%, %6.2f%% of total)' % (
                total_winner_botsim, 
                float(total_winner_botsim)/total_good_offset*100,
                float(total_winner_botsim)/total_offset*100)
    print '  Winners:'
    print '    Star:   %6d (%6.2f%%, %6.2f%% of total)' % (
                total_winner_star, 
                float(total_winner_star)/total_good_offset*100,
                float(total_winner_star)/total_offset*100)
    print '    Model:  %6d (%6.2f%%, %6.2f%% of total)' % (
                total_winner_model, 
                float(total_winner_model)/total_good_offset*100,
                float(total_winner_model)/total_offset*100)
    print '    Titan:  %6d (%6.2f%%, %6.2f%% of total)' % (
                total_winner_titan, 
                float(total_winner_titan)/total_good_offset*100,
                float(total_winner_titan)/total_offset*100)
    print '    BOTSIM: %6d (%6.2f%%, %6.2f%% of total)' % (
                total_winner_botsim, 
                float(total_winner_botsim)/total_good_offset*100,
                float(total_winner_botsim)/total_offset*100)
    total_bad_offset = total_offset-total_good_offset
    print 'Bad final offset:                   %6d (%6.2f%%)' % (
                total_bad_offset, 
                float(total_bad_offset)/total_offset*100)
    failed = total_offset-total_good_offset-total_bootstrap_cand
    print '  Bootstrap candidate: %6d (%6.2f%%, %6.2f%% of total)' % (
                total_bootstrap_cand, 
                float(total_bootstrap_cand)/total_bad_offset*100,
                float(total_bootstrap_cand)/total_offset*100)
    print '  Not bootstrap cand:  %6d (%6.2f%%, %6.2f%% of total)' % (
                failed, 
                float(failed)/total_bad_offset*100,
                float(failed)/total_offset*100)
    
    print sep
    print 'Total Titan navigation attempts:    %6d (%6.2f%%)' % (
                total_titan_attempt, 
                float(total_titan_attempt)/total_offset*100)
    for titan_status in sorted(titan_status_db):
        print '  %-26s %6d (%6.2f%%)' % (
                           titan_status+':',
                           titan_status_db[titan_status],
                           float(titan_status_db[titan_status])/
                           total_titan_attempt*100)
    
    print sep
    print 'Valid bootstrap candidates:'
    for body_name in sorted(bootstrap_cand_db):
        cand_count, bootstrap_status_db = bootstrap_cand_db[body_name]
        print '  %-10s %6d' % (body_name+':', cand_count)
        for bootstrap_status in sorted(bootstrap_status_db):
            print '     %-28s %6d (%6.2f%%)' % (
                               bootstrap_status+':',
                               bootstrap_status_db[bootstrap_status],
                               float(bootstrap_status_db[bootstrap_status])/
                               cand_count*100) 
    print sep
    total_body_only = 0
    for body_name in sorted(body_only_db):
        total_body_only += body_only_db[body_name] 
    print 'Total filled by body:               %6d (%6.2f%%)' % (total_body_only, float(total_body_only)/total_offset*100)
    for body_name in sorted(body_only_db):
        print '    %-10s %6d' % (body_name+':', body_only_db[body_name]) 
    print 'Total filled by rings:              %6d (%6.2f%%)' % (total_rings_only, float(total_rings_only)/total_offset*100)
    print sep
    print 'Run time: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (np.min(time_list),
                                                              np.max(time_list),
                                                              np.mean(time_list),
                                                              np.std(time_list))
    print
    print 'Longest run times:'
    for total_time, filename in longest_time_filenames:
        print '%-13s: %9.2f' % (filename, total_time)
    print
    print 'Histogram of run times:'
    time_list = np.array(time_list)
    h = Histogram(time_list, bins=50)
    print h.vertical(50)
