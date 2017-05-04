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
import re
import os
import sys

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_rings import *
from cb_titan import *
from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = ''

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Statistics',
    epilog='''Default behavior is to collect statistics on all images
              with associated offset files''')

parser.add_argument(
    '--verbose', action='store_true',
    help='Be verbose')
parser.add_argument(
    '--top-bad', type=int, default=0,
    help='Show the top N files for each bad navigation')

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
        his += ' '*(int(max(bars)+2+max(lxl)))+'%s\n'%max(self.h[0])
        for i,c in enumerate(bars):
            line = xl[i] +' '*(int(max(lxl)-lxl[i]))+': '+ character*int(c)+'\n'
            his += line
        return his

def dump_body_info(body_db):    
    for body_name in sorted(body_db):
        (count, 
         no_metadata, no_metadata_filename_list,
         bad_size, bad_size_filename_list,
         bad_curvature, bad_curvature_filename_list,
         bad_limb, bad_limb_filename_list,
         ok, ok_filename_list,
         ok_but_bad_secondary, ok_but_bad_secondary_filename_list,
         ok_but_no_secondary, ok_but_no_secondary_filename_list,
         bad_body_blur, bad_body_blur_filename_list,
         bad_rings_blur, bad_rings_blur_filename_list
        ) = body_db[body_name]
        print '      %-15s  %6d (%6.2f%%, %6.2f%% of total)' % (
                    body_name, count,
                    float(count)/total_bad_offset*100,
                    float(count)/total_offset*100)
        if no_metadata:
            print '        No metadata:         %6d (%6.2f%%) [%s]' % (
                        no_metadata, float(no_metadata)/count*100,
                        no_metadata_filename_list[0])
            if arguments.top_bad:
                for filename in no_metadata_filename_list[:arguments.top_bad]:
                    print '          %s' % filename
        if bad_size:
            print '        Too small:           %6d (%6.2f%%) [%s]' % (
                        bad_size, float(bad_size)/count*100,
                        bad_size_filename_list[0])
            if arguments.top_bad:
                for filename in bad_size_filename_list[:arguments.top_bad]:
                    print '          %s' % filename
        if bad_limb:
            if bad_size:
                print '          (else...)'
            print '        Bad limb:            %6d (%6.2f%%) [%s]' % (
                        bad_limb, float(bad_limb)/count*100,
                        bad_limb_filename_list[0])
            if arguments.top_bad:
                for filename in bad_limb_filename_list[:arguments.top_bad]:
                    print '          %s' % filename
        if bad_curvature:
            if bad_size or bad_limb:
                print '          (else...)'
            print '        Bad curvature:       %6d (%6.2f%%) [%s]' % (
                        bad_curvature, float(bad_curvature)/count*100,
                        bad_curvature_filename_list[0])
            if arguments.top_bad:
                for filename in bad_curvature_filename_list[:arguments.top_bad]:
                    print '          %s' % filename
        if bad_body_blur:
            print '        Bad body blur:       %6d (%6.2f%%) [%6.3f %s]' % (
                        bad_body_blur, float(bad_body_blur)/count*100,
                        bad_body_blur_filename_list[0][0],
                        bad_body_blur_filename_list[0][1])
            if arguments.top_bad:
                for blur, filename in bad_body_blur_filename_list[:arguments.top_bad]:
                    print '          %6.3f %s' % (blur, filename) 
        if bad_rings_blur:
            print '        Bad rings blur:      %6d (%6.2f%%) [%6.3f %s]' % (
                        bad_rings_blur, float(bad_rings_blur)/count*100,
                        bad_rings_blur_filename_list[0][0],
                        bad_rings_blur_filename_list[0][1])
            if arguments.top_bad:
                for blur, filename in bad_rings_blur_filename_list[:arguments.top_bad]:
                    print '          %6.3f %s' % (blur, filename) 
            
        if ok_but_bad_secondary:
            if ok:
                print '        All OK Good Sec Corr %6d (%6.2f%%) [%s]' % (
                            ok, float(ok)/count*100,
                            ok_filename_list[0])
#             if arguments.top_bad:
#                 for filename in ok_filename_list[:arguments.top_bad]:
#                     print '          %s' % filename
            if ok_but_bad_secondary:
                print '        All OK Bad Sec Corr  %6d (%6.2f%%) [%s]' % (
                        ok_but_bad_secondary, float(ok_but_bad_secondary)/count*100,
                        ok_but_bad_secondary_filename_list[0])
                if arguments.top_bad:
                    for filename in ok_but_bad_secondary_filename_list[:arguments.top_bad]:
                        print '            %s' % filename
            if ok_but_no_secondary:
                print '        All OK No Sec Corr   %6d (%6.2f%%) [%s]' % (
                        ok_but_no_secondary, float(ok_but_no_secondary)/count*100,
                        ok_but_no_secondary_filename_list[0])
                if arguments.top_bad:
                    for filename in ok_but_no_secondary_filename_list[:arguments.top_bad]:
                        print '            %s' % filename


bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
max_num_longest_time = 10

total_files = 0
total_offset = 0
total_has_offset_result = 0
no_offset_file_list = []
total_spice_error = 0
total_other_error = 0
total_skipped = 0
other_error_db = {}
other_error_file_db = {}
skipped_db = {}
skipped_file_db = {}
total_bad_but_botsim_candidate = 0
total_botsim_candidate = 0
total_botsim_winner_excess_diff = 0
total_botsim_potential_excess_diff = 0
botsim_potential_excess_diff_x_list = []
botsim_potential_excess_diff_y_list = []
total_botsim_potential_stars_excess_diff = 0
botsim_potential_stars_excess_diff_x_list = []
botsim_potential_stars_excess_diff_y_list = []
total_good_offset = 0
total_good_offset_list = {('NAC',256): [],
                          ('NAC',512): [],
                          ('NAC',1024): [],
                          ('WAC',256): [],
                          ('WAC',512): [],
                          ('WAC',1024): []}
total_good_star_offset = 0
total_good_model_nobs_offset = 0
total_good_model_bs_offset = 0
total_good_titan_offset = 0
total_winner_star = 0
total_winner_model = 0
total_winner_titan = 0
total_winner_botsim = 0
good_offset_broken_assumptions = {}
good_body_blur_list = []
good_rings_blur_list = []
total_secondary_corr_failed = 0
total_images_marked_bad = 0
image_description_db = {}
secondary_corr_failed_filename_list = []
compare_good_nav_list = {}
body_only_db = {}
total_rings_entirely = 0
total_rings_entirely_no_offset = 0
total_rings_entirely_no_offset_bad_curvature = 0
total_rings_entirely_no_offset_bad_emission = 0
total_rings_entirely_no_offset_bad_features = 0
total_rings_entirely_no_offset_ok = 0
total_rings_only = 0
total_rings_only_no_offset = 0
total_rings_only_no_offset_bad_curvature = 0
total_rings_only_no_offset_bad_emission = 0
total_rings_only_no_offset_bad_features = 0
total_rings_only_no_offset_ok = 0
total_rings_only_no_offset_dring = 0
total_rings_only_no_offset_fring = 0
total_bad_offset_no_rings_or_bodies = 0
no_rings_single_body_db = {}
no_rings_multi_body_db = {}
with_rings_single_body_db = {}
with_rings_multi_body_db = {}
total_bootstrap_cand = 0
total_bootstrap_cand_no_offset = 0
bootstrap_cand_db = {}
titan_status_db = {}
titan_insuff_db = {}
total_titan_attempt = 0
time_list = []
longest_time_filenames = []
earliest_date = 1e38
latest_date = 0

last_nac_filename = None
last_nac_image_path = None
last_nac_offset = None
last_nac_metadata = None

for image_path in file_yield_image_filenames_from_arguments(arguments):
    status = ''
    _, base = os.path.split(image_path)

    status += base + ': '
    total_files += 1

    metadata = file_read_offset_metadata(image_path, overlay=False,
                                         bootstrap_pref='prefer')    
    filename = file_clean_name(image_path)
    status = filename + ' - ' + offset_result_str(metadata)

    if filename[0] == 'N':
        last_nac_filename = filename
        last_nac_image_path = image_path
        last_nac_offset = None
        last_nac_metadata = None
    
    if metadata is None:
        no_offset_file_list.append(filename)
        if arguments.verbose:
            print status
        continue
        
    total_offset += 1
    
    # Fix up the metadata for old files - eventually this should
    # be removed! XXX
    if 'error' in metadata:
        metadata['status'] = 'error'
        metadata['status_detail1'] = metadata['error']
        metadata['status_detail2'] = metadata['error_traceback']
    elif 'status' not in metadata:
        metadata['status'] = 'ok'

    status = metadata['status']
    if status == 'error':            
        error = metadata['status_detail1']
        if error == '':
            error = metadata['status_detail2'].split('\n')[-2]
        if error.startswith('SPICE(NOFRAMECONNECT)'):
            total_spice_error += 1
        else:
            total_other_error += 1
            if error not in other_error_db:
                other_error_db[error] = 0
                other_error_file_db[error] = filename
            other_error_db[error] += 1
    elif status == 'skipped':
        reason = metadata['status_detail1']
        total_skipped += 1
        if reason not in skipped_db:
            skipped_db[reason] = 0
            skipped_file_db[reason] = filename
        skipped_db[reason] += 1
    else:
        if status != 'ok':
            print 'BAD STATUS', status, filename
        earliest_date = min(metadata['start_time'], earliest_date)
        latest_date = max(metadata['end_time'], latest_date)
        total_time = metadata['end_time']-metadata['start_time']
        time_list.append(total_time)
        longest_time_filenames.append((total_time, filename))
        longest_time_filenames.sort(reverse=True)
        longest_time_filenames = longest_time_filenames[:max_num_longest_time]
            
        total_has_offset_result += 1
        offset = metadata['offset']
        if filename[0] == 'N':
            last_nac_offset = offset
            last_nac_metadata = metadata
        winner = metadata['offset_winner']
        stars_metadata = metadata['stars_metadata']
        rings_metadata = metadata['rings_metadata']
        bodies_metadata = metadata['bodies_metadata']
        titan_metadata = metadata['titan_metadata']
        
        # We don't use IF/ELSE constructs here to save on indentation
        description_ok = True
                        
        if offset is not None:
            total_good_offset += 1
            total_good_offset_list[metadata['camera'],
                                   metadata['image_shape'][0]].append(
                                                          tuple(offset))
            max_offset = MAX_POINTING_ERROR[(tuple(metadata['image_shape']),
                                             metadata['camera'])]
            if (abs(offset[0]) > max_offset[0] or
                abs(offset[1]) > max_offset[1]):
                print 'WARNING - ', filename, '-',
                print 'Offset', winner, offset, 'exceeds maximum', max_offset 

            broken_assumptions_list = []
            if metadata['model_override_bodies_curvature']:
                broken_assumptions_list.append('BodyCurv')
            if metadata['model_override_bodies_limb']:
                broken_assumptions_list.append('BodyLimb')
            if 'model_bodies_blur' in metadata and metadata['model_bodies_blur']:
                broken_assumptions_list.append('BodyBlur')
            if metadata['model_override_rings_curvature']:
                broken_assumptions_list.append('RingCurv')
            if metadata['model_override_fiducial_features']:
                broken_assumptions_list.append('RingFeat')
            if metadata['model_rings_blur']:
                broken_assumptions_list.append('RingBlur') 
                
            if len(broken_assumptions_list) > 0:
                broken_assumptions_str = '+'.join(broken_assumptions_list)
                if broken_assumptions_str not in good_offset_broken_assumptions:
                    good_offset_broken_assumptions[broken_assumptions_str] = []
                good_offset_broken_assumptions[broken_assumptions_str].append(filename)

            if (rings_metadata is not None and
                rings_metadata['fiducial_blur']):
                good_rings_blur_list.append((rings_metadata['fiducial_blur'], filename))

            if bodies_metadata is not None:
                for body_name in sorted(bodies_metadata):
                    body_metadata = bodies_metadata[body_name]
                    if body_metadata['body_blur'] is not None:
                        good_body_blur_list.append((body_metadata['body_blur'], body_name, filename))
            
        if offset is None:
            if ('description' in metadata and
                metadata['description'] != 'N/A'):
                description_ok = False
                description = metadata['description']
                description = re.sub('[NW][0-9]*_[0-9]*.IMG', '[IMAGE]', description)
                if description not in image_description_db:
                    image_description_db[description] = []
                image_description_db[description].append(filename)
                total_images_marked_bad += 1
                             
        if offset is None and description_ok:
            if metadata['secondary_corr_ok'] is False:
                # Beware - None means not performed
                total_secondary_corr_failed += 1
                secondary_corr_failed_filename_list.append(filename)

            bootstrap_cand = False  
            if (metadata['bootstrap_candidate'] or
                metadata['bootstrapped']):
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
                        bootstrap_status_db[bootstrap_status] = bootstrap_status_db.get(bootstrap_status,0)+1
                        
                        bootstrap_cand_db[body_name] = (cand_count+1, 
                                                        bootstrap_status_db)
                        status += '  %s: %s' % (body_name, bootstrap_status)
                    
                        break # Only allow one bootstrap body for now XXX
    
            if bootstrap_cand:
                total_bootstrap_cand += 1
                if offset is None:
                    total_bootstrap_cand_no_offset += 1
                
            has_rings = not (
                rings_metadata is None or
                (rings_metadata is not None and 
                 (('max_radius' in rings_metadata and
                   (rings_metadata['max_radius'] < RINGS_MIN_RADIUS_D or
                    rings_metadata['min_radius'] > RINGS_MAX_RADIUS_F)) or
                  ('max_radius' not in rings_metadata and
                   not rings_metadata['curvature_ok'] and
                   not rings_metadata['emission_ok'] and
                   not rings_metadata['fiducial_features_ok'])))) 

            closest_body_name = 'NONE'
            if len(metadata['large_bodies']):
                closest_body_name = metadata['large_bodies'][0]
            if (metadata['bootstrap_candidate'] and
                closest_body_name in BOOTSTRAP_DEFAULT_CONFIG['body_list']):
                closest_body_name += ' (BS)'

            body_db_to_update = None
            
            if metadata['rings_only']:
                assert has_rings
                # MAIN RINGS fill entire image (no bodies)
                total_rings_entirely_no_offset += 1
                if rings_metadata is not None:
                    if rings_metadata['curvature_ok'] == False:
                        total_rings_entirely_no_offset_bad_curvature += 1
                    if rings_metadata['emission_ok'] == False:
                        total_rings_entirely_no_offset_bad_emission += 1
                    if rings_metadata['fiducial_features_ok'] == False:
                        total_rings_entirely_no_offset_bad_features += 1
                    if (rings_metadata['curvature_ok'] and
                        rings_metadata['emission_ok'] and
                        rings_metadata['fiducial_features_ok']):
                        total_rings_entirely_no_offset_ok += 1
            elif (has_rings and len(metadata['large_bodies']) == 0):
                # ANY RINGS but NO BODIES and MAIN RINGS DON'T FILL ENTIRE IMAGE
                total_rings_only_no_offset += 1
                if rings_metadata['curvature_ok'] == False:
                    total_rings_only_no_offset_bad_curvature += 1
                if rings_metadata['emission_ok'] == False:
                    total_rings_only_no_offset_bad_emission += 1
                if rings_metadata['fiducial_features_ok'] == False:
                    total_rings_only_no_offset_bad_features += 1
                if (rings_metadata['curvature_ok'] and
                    rings_metadata['emission_ok'] and
                    rings_metadata['fiducial_features_ok']):
                    total_rings_only_no_offset_ok += 1
                if ('max_radius' in rings_metadata and
                    rings_metadata['max_radius'] > RINGS_F_RING_CORE and
                    (RINGS_MAX_RADIUS < rings_metadata['min_radius'] < 
                     RINGS_F_RING_CORE)):
                    total_rings_only_no_offset_fring += 1
                if ('max_radius' in rings_metadata and
                    (RINGS_MIN_RADIUS_D < rings_metadata['max_radius'] < 
                     RINGS_MIN_RADIUS)):
                    total_rings_only_no_offset_dring += 1
            elif has_rings and len(metadata['large_bodies']) == 1:
                # HAS RINGS and HAS SINGLE BODY
                body_db_to_update = with_rings_single_body_db
            elif has_rings and len(metadata['large_bodies']) > 1:
                # HAS RINGS and HAS MULTIPLE BODIES
                body_db_to_update = with_rings_multi_body_db
            elif not has_rings and len(metadata['large_bodies']) == 0:
                # NO RINGS or BODIES
                total_bad_offset_no_rings_or_bodies += 1
            elif not has_rings and len(metadata['large_bodies']) == 1:
                # NO RINGS but HAS SINGLE BODY
                body_db_to_update = no_rings_single_body_db
            elif not has_rings and len(metadata['large_bodies']) > 1:
                # NO RINGS but HAS MULTIPLE BODIES
                body_db_to_update = no_rings_multi_body_db
            else:
                assert False
        
            if body_db_to_update is not None:
                (count, 
                 no_metadata, no_metadata_filename_list,
                 bad_size, bad_size_filename_list,
                 bad_curvature, bad_curvature_filename_list,
                 bad_limb, bad_limb_filename_list,
                 ok, ok_filename_list,
                 ok_but_bad_secondary,
                 ok_but_bad_secondary_filename_list,
                 ok_but_no_secondary,
                 ok_but_no_secondary_filename_list,
                 bad_body_blur, bad_body_blur_filename_list,
                 bad_rings_blur, bad_rings_blur_filename_list
                ) = body_db_to_update.get(closest_body_name,
                                          (0,0,[],0,[],0,[],0,[],
                                           0,[],0,[],0,[],
                                           0,[],0,[]))
                count += 1
                if closest_body_name != 'NONE':
                    if closest_body_name.replace(' (BS)','') not in metadata['bodies_metadata']:
                        no_metadata += 1
                        no_metadata_filename_list.append(filename)
                    else:
                        body_metadata = metadata['bodies_metadata'][closest_body_name.replace(' (BS)','')]
                        if not body_metadata['size_ok']:
                            bad_size += 1
                            bad_size_filename_list.append(filename)
                        # Give preference to a bad limb, because usually if the limb is bad
                        # there wasn't any point in dealing with the curvature
                        elif not body_metadata['limb_ok']:
                            bad_limb += 1
                            bad_limb_filename_list.append(filename)
                        elif not body_metadata['curvature_ok']:
                            bad_curvature += 1
                            bad_curvature_filename_list.append(filename)
                        if (body_metadata['size_ok'] and
                            body_metadata['curvature_ok'] and
                            body_metadata['limb_ok']):
                            if metadata['secondary_corr_ok'] is False:
                                ok_but_bad_secondary += 1
                                ok_but_bad_secondary_filename_list.append(filename)
                            if metadata['secondary_corr_ok'] is None:
                                ok_but_no_secondary += 1
                                ok_but_no_secondary_filename_list.append(filename)
                            if metadata['secondary_corr_ok'] is True:
                                # This really should never happen
                                ok += 1
                                ok_filename_list.append(filename)
                        if (body_metadata['body_blur'] is not None and
                            body_metadata['body_blur'] > OFFSET_DEFAULT_CONFIG['maximum_blur']):
                            bad_body_blur += 1
                            bad_body_blur_filename_list.append((body_metadata['body_blur'],
                                                                filename))
                if (has_rings and
                    rings_metadata['fiducial_blur'] > OFFSET_DEFAULT_CONFIG['maximum_blur']):
                    bad_rings_blur += 1
                    bad_rings_blur_filename_list.append((rings_metadata['fiducial_blur'],
                                                         filename))

                body_db_to_update[closest_body_name] = (
                    count, 
                    no_metadata, no_metadata_filename_list,
                    bad_size, bad_size_filename_list,
                    bad_curvature, bad_curvature_filename_list,
                    bad_limb, bad_limb_filename_list,
                    ok, ok_filename_list,
                    ok_but_bad_secondary, ok_but_bad_secondary_filename_list,
                    ok_but_no_secondary, ok_but_no_secondary_filename_list,
                    bad_body_blur, bad_body_blur_filename_list,
                    bad_rings_blur, bad_rings_blur_filename_list)
        
        if (last_nac_filename is not None and
            filename[0] == 'W' and
            filename[1:] == last_nac_filename[1:]):
            total_botsim_candidate += 1
            
            if last_nac_offset is None and offset is not None:
                total_bad_but_botsim_candidate += 1
                
            if (last_nac_offset is not None and offset is not None):
                if (last_nac_metadata['offset_winner'] != 'BOTSIM' and
                    winner != 'BOTSIM'):
                    total_botsim_potential_excess_diff += 1
                    botsim_potential_excess_diff_x_list.append(
                                       last_nac_offset[0]-offset[0]*10)                
                    botsim_potential_excess_diff_y_list.append(
                                       last_nac_offset[1]-offset[1]*10)                
                if (last_nac_metadata['offset_winner'] == 'STARS' and
                    winner == 'STARS'):
                    total_botsim_potential_stars_excess_diff += 1
                    botsim_potential_stars_excess_diff_x_list.append(
                                       last_nac_offset[0]-offset[0]*10)                
                    botsim_potential_stars_excess_diff_y_list.append(
                                       last_nac_offset[1]-offset[1]*10)                
                if winner == 'BOTSIM':
                    if (abs(last_nac_offset[0]-offset[0]*10) > 10 or
                        abs(last_nac_offset[1]-offset[1]*10) > 10):
                        total_botsim_winner_excess_diff += 1
                        
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

        max_offset = MAX_POINTING_ERROR[(tuple(metadata['image_shape']),
                                         metadata['camera'])]
        if (stars_offset is not None and (abs(stars_offset[0]) > max_offset[0] or
                                          abs(stars_offset[1]) > max_offset[1])):
            print 'WARNING - ', filename, '-',
            print 'Stars Offset', stars_offset, 'exceeds maximum', max_offset 
        if (model_offset is not None and (abs(model_offset[0]) > max_offset[0] or
                                          abs(model_offset[1]) > max_offset[1])):
            print 'WARNING - ', filename, '-',
            print 'Model Offset', model_offset, 'exceeds maximum', max_offset 
        if (titan_offset is not None and (abs(titan_offset[0]) > max_offset[0] or
                                          abs(titan_offset[1]) > max_offset[1])):
            print 'WARNING - ', filename, '-',
            print 'Titan Offset', titan_offset, 'exceeds maximum', max_offset 
            
        if stars_offset is not None:
            if model_offset is not None:
                key = ('Stars vs. Model',
                       metadata['camera'], metadata['image_shape'][0])
                compare_good_nav_list[key] = compare_good_nav_list.get(
                       key, []) + [(stars_offset[0]-model_offset[0],
                                    stars_offset[1]-model_offset[1],
                                    filename)]
            if titan_offset is not None:
                key = ('Stars vs. Titan',
                       metadata['camera'], metadata['image_shape'][0])
                compare_good_nav_list[key] = compare_good_nav_list.get(
                       key, []) + [(stars_offset[0]-titan_offset[0],
                                    stars_offset[1]-titan_offset[1],
                                    filename)]
        if model_offset is not None and titan_offset is not None:
                key = ('Model vs. Titan',
                       metadata['camera'], metadata['image_shape'][0])
                compare_good_nav_list[key] = compare_good_nav_list.get(
                       key, []) + [(model_offset[0]-titan_offset[0],
                                    model_offset[1]-titan_offset[1],
                                    filename)]
    
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
            body_only_db[body_only] = body_only_db.get(body_only, 0)+1
            
        if metadata['rings_only']:
            total_rings_only += 1

        if titan_metadata is not None:
            total_titan_attempt += 1
            titan_status = titan_metadata_to_status(titan_metadata)
            titan_status_db[titan_status] = titan_status_db.get(titan_status, []) + [filename]
            if titan_status == 'Insufficient profile data':
                try:
                    filter = titan_metadata['mapped_filter']
                    phase = titan_metadata['mapped_phase']*oops.DPR
                except KeyError:
                    filter = 'Unknown'
                    phase = 0.
                titan_insuff_db[(filter, phase)] = titan_insuff_db.get((filter, phase), []) + [filename]
                
    if arguments.verbose:
        print status

sep = '-' * 55

print sep
print 'Total image files:                  %6d' % total_files
print 'Total with offset file:             %6d' % total_offset
if len(no_offset_file_list) != 0:
    print 'Missing offset files:'
    for filename in no_offset_file_list:
        print '    %s' % filename
print sep

if total_offset:
    print 'SPICE error:                        %6d (%6.2f%%)' % (
                total_spice_error, 
                float(total_spice_error)/total_offset*100)
    print 'Other error:                        %6d (%6.2f%%)' % (
                total_other_error, 
                float(total_other_error)/total_offset*100)
    if total_other_error:
        for error in sorted(other_error_db.keys()):
            print '  %6d (%6.2f%%): %s (%s)' % (
                other_error_db[error],
                float(other_error_db[error])/total_other_error*100,
                error, 
                other_error_file_db[error])
    print 'Skipped:                            %6d (%6.2f%%)' % (
                total_skipped, 
                float(total_skipped)/total_offset*100)
    if total_skipped:
        for reason in sorted(skipped_db.keys()):
            print '  %6d (%6.2f%%): %s (%s)' % (
                skipped_db[reason],
                float(skipped_db[reason])/total_skipped*100,
                reason, 
                skipped_file_db[reason])
    print 'Total remaining:                    %6d' % total_has_offset_result
    print sep

    if total_has_offset_result:
        print 'Good final offset: (%% of non-err)   %6d (%6.2f%%)' % (
                    total_good_offset, 
                    float(total_good_offset)/total_has_offset_result*100)
        if total_good_offset:
            print '  Good star   offset:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_good_star_offset, 
                        float(total_good_star_offset)/total_good_offset*100,
                        float(total_good_star_offset)/total_has_offset_result*100)
            total_good_model_offset = total_good_model_bs_offset+total_good_model_nobs_offset
            print '  Good model  offset:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_good_model_offset, 
                        float(total_good_model_offset)/total_good_offset*100,
                        float(total_good_model_offset)/total_has_offset_result*100)
            print '    Bootstrapped:      %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_good_model_bs_offset, 
                        float(total_good_model_bs_offset)/total_good_offset*100,
                        float(total_good_model_bs_offset)/total_has_offset_result*100)
            print '    Not bootstrapped:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_good_model_nobs_offset, 
                        float(total_good_model_nobs_offset)/total_good_offset*100,
                        float(total_good_model_nobs_offset)/total_has_offset_result*100)
            print '  Good Titan  offset:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_good_titan_offset, 
                        float(total_good_titan_offset)/total_good_offset*100,
                        float(total_good_titan_offset)/total_has_offset_result*100)
            print '  Good BOTSIM offset:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_winner_botsim, 
                        float(total_winner_botsim)/total_good_offset*100,
                        float(total_winner_botsim)/total_has_offset_result*100)
            print '  Winners:'
            print '    Star:   %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_winner_star, 
                        float(total_winner_star)/total_good_offset*100,
                        float(total_winner_star)/total_has_offset_result*100)
            print '    Model:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_winner_model, 
                        float(total_winner_model)/total_good_offset*100,
                        float(total_winner_model)/total_has_offset_result*100)
            print '    Titan:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_winner_titan, 
                        float(total_winner_titan)/total_good_offset*100,
                        float(total_winner_titan)/total_has_offset_result*100)
            print '    BOTSIM: %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_winner_botsim, 
                        float(total_winner_botsim)/total_good_offset*100,
                        float(total_winner_botsim)/total_has_offset_result*100)
        for offset_list_camera, offset_list_size in sorted(total_good_offset_list):
            offset_list = total_good_offset_list[(offset_list_camera,
                                                  offset_list_size)]
            if len(offset_list) == 0:
                continue
            print
            off_list = [x[0] for x in offset_list]
            print '  %s %4d X Offset: MIN %7.2f MAX %7.2f MEAN %7.2f STD %7.2f (%d)' % (
                                offset_list_camera, offset_list_size, 
                                np.min(off_list), np.max(off_list), 
                                np.mean(off_list), np.std(off_list), len(off_list))
            off_list = [x[1] for x in offset_list]
            print '  %s %4d Y Offset: MIN %7.2f MAX %7.2f MEAN %7.2f STD %7.2f' % (
                                offset_list_camera, offset_list_size,
                                np.min(off_list), np.max(off_list), 
                                np.mean(off_list), np.std(off_list))
    
        last_navsrc = None
        for (offset_list_navsrc, offset_list_camera, 
             offset_list_size) in sorted(compare_good_nav_list):
            offset_list = compare_good_nav_list[(offset_list_navsrc,
                                                 offset_list_camera,
                                                 offset_list_size)]
            if len(offset_list) == 0:
                continue
            if last_navsrc != offset_list_navsrc:
                print
                print '  %s' % offset_list_navsrc
                last_navsrc = offset_list_navsrc
            off_list = [x[0] for x in offset_list]
            print '    %s %4d X Offset: MIN %7.2f MAX %7.2f MEAN %7.2f STD %7.2f (%d)' % (
                                offset_list_camera, offset_list_size, 
                                np.min(off_list), np.max(off_list), 
                                np.mean(off_list), np.std(off_list), len(off_list))
            if arguments.top_bad:
                offset_list.sort(key=lambda x:-abs(x[0]))
                file_list = [('%s %7.2f'%(x[2],x[0])) for x in offset_list[:arguments.top_bad]]
                for file in file_list:
                    print '      %s' % file
            off_list = [x[1] for x in offset_list]
            print '    %s %4d Y Offset: MIN %7.2f MAX %7.2f MEAN %7.2f STD %7.2f' % (
                                offset_list_camera, offset_list_size,
                                np.min(off_list), np.max(off_list), 
                                np.mean(off_list), np.std(off_list))
            if arguments.top_bad:
                offset_list.sort(key=lambda x:-abs(x[1]))
                file_list = [('%s %7.2f'%(x[2],x[1])) for x in offset_list[:arguments.top_bad]]
                for file in file_list:
                    print '      %s' % file
        
        print sep
        total_bad_offset = total_has_offset_result-total_good_offset
        print 'No final offset:                    %6d (%6.2f%%)' % (
                    total_bad_offset, 
                    float(total_bad_offset)/total_has_offset_result*100)
        if total_bad_offset:
            if len(image_description_db) > 0:
                print '  Images marked bad:   %6d (%6.2f%%, %6.2f%% of total)' % (
                            total_images_marked_bad, 
                            float(total_images_marked_bad)/total_bad_offset*100,
                            float(total_images_marked_bad)/total_has_offset_result*100)
                for description in sorted(image_description_db):
                    print '    %s' % description
                    print '                            %6d (%6.2f%%) [%s]' % (
                                       len(image_description_db[description]),
                                       float(len(image_description_db[description]))/
                                         total_bad_offset*100.,
                                       image_description_db[description][0])
                    if arguments.top_bad:
                        for filename in image_description_db[description][:arguments.top_bad]:
                            print '      %s' % filename
                print
                print ' Otherwise...'

            print '  Secondary corr fail: %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_secondary_corr_failed, 
                        float(total_secondary_corr_failed)/total_bad_offset*100,
                        float(total_secondary_corr_failed)/total_has_offset_result*100)
            if arguments.top_bad:
                for filename in secondary_corr_failed_filename_list[:arguments.top_bad]:
                    print '      %s' % filename
            print '  BOTSIM candidate:    %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_bad_but_botsim_candidate, 
                        float(total_bad_but_botsim_candidate)/total_bad_offset*100,
                        float(total_bad_but_botsim_candidate)/total_has_offset_result*100)
            print '  Bootstrap candidate: %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_bootstrap_cand_no_offset, 
                        float(total_bootstrap_cand_no_offset)/total_bad_offset*100,
                        float(total_bootstrap_cand_no_offset)/total_has_offset_result*100)
    
            print
            print 'Reasons:'
            print '  No rings'
            print '    No bodies:         %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_bad_offset_no_rings_or_bodies, 
                        float(total_bad_offset_no_rings_or_bodies)/total_bad_offset*100,
                        float(total_bad_offset_no_rings_or_bodies)/total_has_offset_result*100)
            print '    Single body only:'
            dump_body_info(no_rings_single_body_db)
            print '    Multiple bodies, closest:'
            dump_body_info(no_rings_multi_body_db)
            print '  Has rings (D-F)'
            print '    Filled by main:    %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_rings_entirely_no_offset, 
                        float(total_rings_entirely_no_offset)/total_bad_offset*100,
                        float(total_rings_entirely_no_offset)/total_has_offset_result*100)
            if total_rings_entirely_no_offset:
                if total_rings_entirely_no_offset_bad_curvature:
                    print '      Bad curvature:         %6d (%6.2f%%)' % (
                                total_rings_entirely_no_offset_bad_curvature, 
                                float(total_rings_entirely_no_offset_bad_curvature)/total_rings_entirely_no_offset*100)
                if total_rings_entirely_no_offset_bad_emission:
                    print '      Bad emission:          %6d (%6.2f%%)' % (
                                total_rings_entirely_no_offset_bad_emission,
                                float(total_rings_entirely_no_offset_bad_emission)/total_rings_entirely_no_offset*100)
                if total_rings_entirely_no_offset_bad_features:
                    print '      Bad features:          %6d (%6.2f%%)' % (
                                total_rings_entirely_no_offset_bad_features, 
                                float(total_rings_entirely_no_offset_bad_features)/total_rings_entirely_no_offset*100)
                if total_rings_entirely_no_offset_ok:
                    print '      All OK:                %6d (%6.2f%%)' % (
                                total_rings_entirely_no_offset_ok, 
                                float(total_rings_entirely_no_offset_ok)/total_rings_entirely_no_offset*100)
            print '    No bodies:         %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_rings_only_no_offset, 
                        float(total_rings_only_no_offset)/total_bad_offset*100,
                        float(total_rings_only_no_offset)/total_has_offset_result*100)
            if total_rings_only_no_offset:
                if total_rings_only_no_offset_bad_curvature:
                    print '      Bad curvature:         %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_bad_curvature, 
                                float(total_rings_only_no_offset_bad_curvature)/total_rings_only_no_offset*100)
                if total_rings_only_no_offset_bad_emission:
                    print '      Bad emission:          %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_bad_emission,
                                float(total_rings_only_no_offset_bad_emission)/total_rings_only_no_offset*100)
                if total_rings_only_no_offset_bad_features:
                    print '      Bad features:          %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_bad_features, 
                                float(total_rings_only_no_offset_bad_features)/total_rings_only_no_offset*100)
                if total_rings_only_no_offset_ok:
                    print '      All OK:                %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_ok, 
                                float(total_rings_only_no_offset_ok)/total_rings_only_no_offset*100)
                if total_rings_only_no_offset_fring:
                    print '      F ring only:           %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_fring, 
                                float(total_rings_only_no_offset_fring)/total_rings_only_no_offset*100)
                if total_rings_only_no_offset_dring:
                    print '      D ring only:           %6d (%6.2f%%)' % (
                                total_rings_only_no_offset_dring, 
                                float(total_rings_only_no_offset_dring)/total_rings_only_no_offset*100)
            print '    Single body only:'
            dump_body_info(with_rings_single_body_db)
            print '    Multiple bodies, closest:'
            dump_body_info(with_rings_multi_body_db)
    
        if total_botsim_candidate:
            print sep
            print 'BOTSIM opportunity:                 %6d (%6.2f%%)' % (
                        total_botsim_candidate,
                        float(total_botsim_candidate)/total_has_offset_result*100)
            print '  Winner bad diff:     %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_botsim_winner_excess_diff, 
                        float(total_botsim_winner_excess_diff)/total_botsim_candidate*100,
                        float(total_botsim_winner_excess_diff)/total_has_offset_result*100)
            print '  Both nav OK diff:    %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_botsim_potential_excess_diff, 
                        float(total_botsim_potential_excess_diff)/total_botsim_candidate*100,
                        float(total_botsim_potential_excess_diff)/total_has_offset_result*100)
            if len(botsim_potential_excess_diff_x_list):
                print '    X DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (
                                                        np.min(botsim_potential_excess_diff_x_list),
                                                        np.max(botsim_potential_excess_diff_x_list),
                                                        np.mean(botsim_potential_excess_diff_x_list),
                                                        np.std(botsim_potential_excess_diff_x_list))
                print '    Y DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (
                                                        np.min(botsim_potential_excess_diff_y_list),
                                                        np.max(botsim_potential_excess_diff_y_list),
                                                        np.mean(botsim_potential_excess_diff_y_list),
                                                        np.std(botsim_potential_excess_diff_y_list))
    
            print '  Both OK stars diff:  %6d (%6.2f%%, %6.2f%% of total)' % (
                        total_botsim_potential_stars_excess_diff, 
                        float(total_botsim_potential_stars_excess_diff)/total_botsim_candidate*100,
                        float(total_botsim_potential_stars_excess_diff)/total_has_offset_result*100)
            if len(botsim_potential_stars_excess_diff_x_list):
                print '    X DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (
                                                        np.min(botsim_potential_stars_excess_diff_x_list),
                                                        np.max(botsim_potential_stars_excess_diff_x_list),
                                                        np.mean(botsim_potential_stars_excess_diff_x_list),
                                                        np.std(botsim_potential_stars_excess_diff_x_list))
                print '    Y DIFF: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (
                                                        np.min(botsim_potential_stars_excess_diff_y_list),
                                                        np.max(botsim_potential_stars_excess_diff_y_list),
                                                        np.mean(botsim_potential_stars_excess_diff_y_list),
                                                        np.std(botsim_potential_stars_excess_diff_y_list))
    
        print sep
        print 'Total bootstrap candidates:         %6d (%6.2f%%)' % (
                    total_bootstrap_cand, 
                    float(total_bootstrap_cand)/total_has_offset_result*100)
        for body_name in sorted(bootstrap_cand_db):
            cand_count, bootstrap_status_db = bootstrap_cand_db[body_name]
            print '  %-10s %6d' % (body_name+':', cand_count)
            for bootstrap_status in sorted(bootstrap_status_db):
                print '    %-28s %6d (%6.2f%%)' % (
                                   bootstrap_status+':',
                                   bootstrap_status_db[bootstrap_status],
                                   float(bootstrap_status_db[bootstrap_status])/
                                   cand_count*100)
        
        print sep
        print 'Total Titan navigation attempts:    %6d (%6.2f%%)' % (
                    total_titan_attempt, 
                    float(total_titan_attempt)/total_has_offset_result*100)
        for titan_status in sorted(titan_status_db):
            print '  %-26s %6d (%6.2f%%) [%s]' % (
                               titan_status+':',
                               len(titan_status_db[titan_status]),
                               float(len(titan_status_db[titan_status]))/
                                 total_titan_attempt*100,
                               titan_status_db[titan_status][0])
            if arguments.top_bad:
                for filename in titan_status_db[titan_status][:arguments.top_bad]:
                    print '      %s' % filename
            if titan_status == 'Insufficient profile data':
                for key in sorted(titan_insuff_db):
                    filter, phase = key
                    print '    %-9s %6.2f %6d (%6.2f%%) [%s]' % (
                                       filter, phase,
                                       len(titan_insuff_db[key]),
                                       float(len(titan_insuff_db[key]))/
                                         total_titan_attempt*100,
                                       titan_insuff_db[key][0])
                    if arguments.top_bad:
                        for filename in titan_insuff_db[key][:arguments.top_bad]:
                            print '        %s' % filename
            
    
    print sep
    total_body_only = 0
    for body_name in sorted(body_only_db):
        total_body_only += body_only_db[body_name] 
    print 'Total filled by body:               %6d (%6.2f%%)' % (total_body_only, float(total_body_only)/total_has_offset_result*100)
    for body_name in sorted(body_only_db):
        print '    %-10s %6d' % (body_name+':', body_only_db[body_name]) 
    print 'Total filled by rings:              %6d (%6.2f%%)' % (total_rings_only, float(total_rings_only)/total_has_offset_result*100)

    if total_good_model_offset:
        print sep
        print 'Good model, broken assumptions:'
        for broken_assumptions_str in sorted(good_offset_broken_assumptions):
            filelist = good_offset_broken_assumptions[broken_assumptions_str]
            print '  %s' % broken_assumptions_str
            print '                       %6d (%6.2f%%, %6.2f%% of total)' % (
                        len(filelist), 
                        float(len(filelist))/total_good_model_offset*100,
                        float(len(filelist)/total_has_offset_result*100))
            if arguments.top_bad:
                for filename in filelist[:arguments.top_bad]:
                    print '      %s' % filename

    if ((len(good_body_blur_list) > 0 or len(good_rings_blur_list) > 0) and
        arguments.top_bad):
        print sep
        if len(good_body_blur_list) > 0:
            print 'Body blur:'
            good_body_blur_list.sort(reverse=True)
            for body_blur, body_name, filename in good_body_blur_list[:arguments.top_bad]:
                print '  %6.3f %-10s %s' % (body_blur, body_name, filename)
        if len(good_rings_blur_list) > 0:
            print 'Rings blur:'
            good_rings_blur_list.sort(reverse=True)
            for rings_blur, filename in good_rings_blur_list[:arguments.top_bad]:
                print '  %6.3f %s' % (rings_blur, filename)
                      
    print sep    
    print 'Earliest offset:', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(earliest_date))
    print 'Latest offset:  ', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(latest_date))
    print
    print 'Run time: MIN %.2f MAX %.2f MEAN %.2f STD %.2f' % (np.min(time_list),
                                                              np.max(time_list),
                                                              np.mean(time_list),
                                                              np.std(time_list))
    print 'Total run time: %.2f' % np.sum(time_list)
    
    print
    print 'Longest run times:'
    for total_time, filename in longest_time_filenames:
        print '%-13s: %9.2f' % (filename, total_time)
    print
    print 'Histogram of run times:'
    time_list = np.array(time_list)
    h = Histogram(time_list, bins=50)
    print h.vertical(50)
