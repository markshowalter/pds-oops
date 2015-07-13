###############################################################################
# display_offset_statistics.py
###############################################################################

import argparse
import sys

import cspice

from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--first-image-num 1481738274 --last-image-num 1496491595'

    command_list = command_line_str.split()

## XXX Check restrict image list is included in first->last range 

parser = argparse.ArgumentParser(
    description='Cassini Backplane Offset Metadata Display')

def validate_image_name(name):
    valid = (len(name) == 13 and name[0] in 'NW' and name[11] == '_')
    if valid:
        try:
            _ = int(name[1:11])
            _ = int(name[12])
        except ValueError:
            valid = False
    if not valid:
        raise argparse.ArgumentTypeError(
             name+
             " is not a valid image name - format must be [NW]dddddddddd_d")
    return name

# Arguments about selecting the images to process
parser.add_argument(
    '--first-image-num', type=int, default='1', metavar='IMAGE_NUM',
    help="The starting image number")
parser.add_argument(
    '--last-image-num', type=int, default='9999999999', metavar='IMAGE_NUM',
    help="The ending image number")
nacwac_group = parser.add_mutually_exclusive_group()
nacwac_group.add_argument(
    '--nac-only', action='store_true', default=False,
    help="Only process NAC images")
nacwac_group.add_argument(
    '--wac-only', action='store_true', default=False,
    help="Only process WAC images")
parser.add_argument(
    'image_name', action='append', nargs='*', type=validate_image_name,
    help="Specific image names to process")
parser.add_argument(
    '--image-full-path', action='append',
    help="The full path for an image")

arguments = parser.parse_args(command_list)

restrict_camera = 'NW'
if arguments.nac_only:
    restrict_camera = 'N'
if arguments.wac_only:
    restrict_camera = 'W'

restrict_image_list = None
if arguments.image_name is not None and arguments.image_name != [[]]:
    restrict_image_list = arguments.image_name[0]

first_image_number = arguments.first_image_num
last_image_number = arguments.last_image_num

def str_offset_one_image(image_path):
    metadata = file_read_offset_metadata(image_path)
    filename = file_clean_name(image_path)
    
    ret = filename + ' '
    if metadata is None:
        ret += 'No offset file written'
        return ret

    if 'error' in metadata:
        ret += 'ERROR: '
        error = metadata['error']
        if error.startswith('SPICE(NOFRAMECONNECT)'):
            ret += 'SPICE KERNEL MISSING DATA AT ' + error[34:53]
        else:
            ret += error 
        return ret
    
    the_time = cspice.et2utc(metadata['midtime'], 'C', 0)
    ret += the_time + ' '

    filter1 = metadata['filter1']
    filter2 = metadata['filter2']
    ret += ('%4s'%filter1) + ' ' + ('%4s'%filter2) + ' '
    
    the_size = '%dx%d' % tuple(metadata['image_shape'])
    the_size = '%9s' % the_size
    ret += the_size + ' '

    
    offset = metadata['offset']
    if offset is None:
        offset_str = '  N/A  '
    else:
        offset_str = '%3d,%-3d' % tuple(offset)
    ret += offset_str + ' '

    star_offset_str = '  N/A  '
    if metadata['stars_metadata'] is not None:
        star_offset = metadata['stars_metadata']['offset']
        if star_offset is not None:
            if metadata['used_objects_type'] == 'stars':
                star_offset_str = '%3d*%-3d' % tuple(star_offset)
            else:
                star_offset_str = '%3d,%-3d' % tuple(star_offset)
    ret += star_offset_str + ' '

    model_offset = metadata['model_offset']
    if model_offset is None:
        model_offset_str = '  N/A  '
    else:
        if metadata['used_objects_type'] == 'model':
            model_offset_str = '%3d*%-3d' % tuple(model_offset)
        else:
            model_offset_str = '%3d,%-3d' % tuple(model_offset)
    ret += model_offset_str + ' '

    bootstrap_str = ' '*6
    if metadata['bootstrap_candidate']:
        bootstrap_str = '%-6s' % metadata['bootstrap_body'][:6]
    ret += bootstrap_str + ' '
        
    single_body_str = ' '*6
    if metadata['body_only']:
        single_body_str = '%-6s' % metadata['body_only'][:6]
    if metadata['rings_only']:
        single_body_str = 'RINGS '

    ret += single_body_str + ' '
        
    return ret
    
if arguments.image_full_path:
    for image_path in arguments.image_full_path:
        print str_offset_one_image(image_path)
    
if first_image_number <= last_image_number:
    for image_path in yield_image_filenames(
            first_image_number, last_image_number,
            camera=restrict_camera, restrict_list=restrict_image_list):
        print str_offset_one_image(image_path)
