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

def verify(obsid, image_name_list, offset_path_list):
    for offset_path in offset_path_list:
        auto_offset, manual_offset, metadata = fring_util.read_offset(offset_path)
        if metadata is None:
            continue
        try:
            if metadata['used_objects_type'] != 'stars':
                continue
        except:
            continue
        star_list = metadata['full_star_list']
        if auto_offset is None:
            continue
        offset_u, offset_v = auto_offset
        bright_stars = 0
        good_bright_stars = 0
        for star in star_list:
            try:
                if not (0 <= star.u+offset_u <= 1023):
                    continue
                if not (0 <= star.v+offset_v <= 1023):
                    continue
                if not star.is_dim_enough:
                    continue
                if not star.is_bright_enough:
                    continue
                if star.conflicts:
                    continue
                if star.dn < 200:
                    continue
                bright_stars += 1
                if star.photometry_confidence == 1:
                    good_bright_stars += 1
            except:
                pass
        if good_bright_stars < bright_stars*2//3:
            print offset_path, bright_stars, good_bright_stars,
            print '***'
        
            
    
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
#    print obsid, image_name
    offset_path = fring_util.offset_path(options, image_path, image_name)
    
    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(image_name_list) != 0:
            verify(cur_obsid, image_name_list, offset_path_list)
        obsid_list = []
        image_name_list = []
        image_path_list = []
        offset_path_list = []
        cur_obsid = obsid
    image_name_list.append(image_name)
    offset_path_list.append(offset_path)
    
# Final mosaic
if len(image_name_list) != 0:
    verify(cur_obsid, image_name_list, offset_path_list)
