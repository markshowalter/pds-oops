###############################################################################
# cb_util_file.py
#
# Routines related to reading and writing files.
#
# Exported routines:
#    file_clean_join
#    file_add_selection_arguments
#    file_log_arguments
#    file_yield_image_filenames_from_arguments
#    file_yield_image_filenames
#    file_clean_name
#    file_read_iss_file
#    file_img_to_log_path
#    file_img_to_offset_path
#    file_offset_to_img_path
#    file_read_offset_metadata
#    file_write_offset_metadata
#    file_img_to_png_file
#    file_write_png_from_image
#    file_mosaic_path
#    file_read_mosaic_metadata
#    file_write_mosaic_metadata
###############################################################################

import cb_logging
import logging

import argparse
import csv
import numpy as np
import msgpack
import msgpack_numpy
import os.path
from PIL import Image

import oops
import oops.inst.cassini.iss as iss

from cb_config import *

_LOGGING_NAME = 'cb.' + __name__


###############################################################################
#
#
# GENERAL FILE UTILITIES - SELECTING AND LISTING FILES
#
#
###############################################################################

def file_clean_join(*args):
    ret = os.path.join(*args)
    return ret.replace('\\', '/')

def _validate_image_name(name):
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
             ' is not a valid image name - format must be [NW]dddddddddd_d')
    return name

def file_add_selection_arguments(parser):
    parser.add_argument(
        '--first-image-num', type=int, default='1', metavar='IMAGE_NUM',
        help='The starting image number')
    parser.add_argument(
        '--last-image-num', type=int, default='9999999999', metavar='IMAGE_NUM',
        help='The ending image number')
    nacwac_group = parser.add_mutually_exclusive_group()
    nacwac_group.add_argument(
        '--nac-only', action='store_true', default=False,
        help='Only process NAC images')
    nacwac_group.add_argument(
        '--wac-only', action='store_true', default=False,
        help='Only process WAC images')
    parser.add_argument(
        'image_name', action='append', nargs='*', type=_validate_image_name,
        help='Specific image names to process')
    parser.add_argument(
        '--volume', action='append',
        help='An entire volume')
    parser.add_argument(
        '--image-full-path', action='append',
        help='The full path for an image')
    parser.add_argument(
        '--image-pds-csv', action='append',
        help='''A CSV file downloaded from PDS that contains filespecs of images
    to process''')
    parser.add_argument(
        '--has-offset-file', action='store_true', default=False,
        help='Only process images that already have an offset file')
    parser.add_argument(
        '--has-no-offset-file', action='store_true', default=False,
        help='Only process images that don\'t already have an offset file')
    parser.add_argument(
        '--has-png-file', action='store_true', default=False,
        help='Only process images that already have a PNG file')
    parser.add_argument(
        '--has-no-png-file', action='store_true', default=False,
        help='Only process images that don\'t already have a PNGfile')

def file_log_arguments(arguments, log):
    if arguments.image_full_path:
        log('*** Images explicitly from full paths:')
        for image_path in arguments.image_full_path:
            log('        %s', image_path)
    log('*** Image #s %010d - %010d',
        arguments.first_image_num,
        arguments.last_image_num)
    log('*** NAC only:                %s', arguments.nac_only)
    log('*** WAC only:                %s', arguments.wac_only)
    log('*** Already has offset file: %s', arguments.has_offset_file)
    log('*** Has no offset file:      %s', arguments.has_no_offset_file)
    log('*** Already has PNG file:    %s', arguments.has_png_file)
    log('*** Has no PNG file:         %s', arguments.has_no_png_file)
    if (arguments.image_name is not None and arguments.image_name != [] and
        arguments.image_name[0] != []):
        log('*** Images restricted to list:')
        for filename in arguments.image_name[0]:
            log('        %s', filename)
    if arguments.volume is not None and arguments.volume != []:
        log('*** Images restricted to volume:')
        for filename in arguments.volume:
            log('        %s', filename)
    if arguments.image_pds_csv:
        log('*** Images restricted to those from PDS CSV %s',
            arguments.image_pds_csv)

def file_yield_image_filenames_from_arguments(arguments):
    if arguments.image_full_path:
        for image_path in arguments.image_full_path:
            yield image_path
    
    restrict_image_list = []
    if arguments.image_name is not None and arguments.image_name != [[]]:
        restrict_image_list = arguments.image_name[0][:] # Copy the list
    
    if arguments.image_pds_csv:
        for filename in arguments.image_pds_csv:
            with open(filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                header = csvreader.next()
                for colnum in xrange(len(header)):
                    if header[colnum] == 'primaryfilespec':
                        break
                else:
                    main_logger.error('Badly formatted CSV file %s', filename)
                    sys.exit(-1)
                if arguments.image_name is None:
                    arguments.image_name = []
                    arguments.image_name.append([])
                for row in csvreader:
                    filespec = row[colnum]
                    filespec = filespec.replace('.IMG', '').replace('_CALIB', 
                                                                    '')
                    _, filespec = os.path.split(filespec)
                    restrict_image_list.append(filespec)
    
    restrict_camera = 'NW'
    if arguments.nac_only:
        restrict_camera = 'N'
    if arguments.wac_only:
        restrict_camera = 'W'

    first_image_number = arguments.first_image_num
    last_image_number = arguments.last_image_num
    volume = arguments.volume
    
    for image_path in file_yield_image_filenames(
                first_image_number, 
                last_image_number,
                volume,
                restrict_camera,
                restrict_image_list,
                force_has_offset_file=arguments.has_offset_file,
                force_has_no_offset_file=arguments.has_no_offset_file,
                force_has_png_file=arguments.has_png_file,
                force_has_no_png_file=arguments.has_no_png_file):
        yield image_path

def file_yield_image_filenames(img_start_num=0, img_end_num=9999999999,
                               volume=None, 
                               camera='NW', restrict_list=None,
                               force_has_offset_file=False,
                               force_has_no_offset_file=False,
                               force_has_png_file=False,
                               force_has_no_png_file=False,
                               image_root=COISS_2XXX_DERIVED_ROOT,
                               suffix='_CALIB.IMG'):
    search_root = COISS_2XXX_DERIVED_ROOT
    search_suffix = '_CALIB.IMG'
    searching_offset = False
    searching_png = False
    
    if force_has_offset_file:
        search_root = file_clean_join(RESULTS_ROOT, 'offsets')
        search_suffix = '-OFFSET.dat'
        searching_offset = True
    if force_has_png_file:
        search_root = file_clean_join(RESULTS_ROOT, 'png')
        search_suffix = '.png'
        searching_png = True      

    done = False
    if not os.path.isdir(search_root):
        return
    for search_dir in sorted(os.listdir(search_root)):
        search_fulldir = file_clean_join(search_root, search_dir)
        coiss_fulldir = file_clean_join(image_root, search_dir)
        if volume:
            found_vol_spec = False
            good_vol_spec = False
            for vol in volume:
                if vol.find('/') != -1:
                    continue
                found_vol_spec = True
                if search_dir == vol:
                    good_vol_spec = True
                    break
            if found_vol_spec and not good_vol_spec:
                continue
        if (not os.path.isdir(search_fulldir) or
            not os.path.isdir(coiss_fulldir)):
            continue
        if not searching_offset and not searching_png:
            search_fulldir = file_clean_join(search_fulldir, 'data')
        coiss_fulldir = file_clean_join(coiss_fulldir, 'data')
        for range_dir in sorted(os.listdir(search_fulldir)):
            if len(range_dir) != 21 or range_dir[10] != '_':
                continue
            if volume:
                found_full_spec = False
                good_full_spec = False
                for vol in volume:
                    if vol.find('/') == -1:
                        continue
                    found_full_spec = True
                    if search_dir+'/'+range_dir == vol:
                        good_full_spec = True
                        break
                if found_full_spec and not good_full_spec:
                    continue
            range1 = int(range_dir[:10])
            range2 = int(range_dir[11:])
            if range1 > img_end_num:
                done = True
                break
            if range2 < img_start_num:
                continue
            meta_dir = file_clean_join(search_fulldir, range_dir)
            img_dir = file_clean_join(coiss_fulldir, range_dir)
            # Sort by number then letter so N/W are together
            for img_name in sorted(os.listdir(meta_dir),
                                   key=lambda x: x[1:13]+x[0]):
                if not img_name.endswith(search_suffix):
                    continue
                img_name = img_name.replace(search_suffix, suffix)
                img_num = int(img_name[1:11])
                if img_num > img_end_num:
                    done = True
                    break
                if img_num < img_start_num:
                    continue
                if img_name[0] not in camera:
                    continue
                if restrict_list and img_name[:13] not in restrict_list:
                    continue
                img_path = file_clean_join(img_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                if force_has_offset_file and not searching_offset:
                    offset_path = file_img_to_offset_path(img_path)
                    if not os.path.isfile(offset_path):
                        continue
                if force_has_no_offset_file:
                    offset_path = file_img_to_offset_path(img_path)
                    if os.path.isfile(offset_path):
                        continue
                if force_has_png_file and not searching_png:
                    png_path = file_img_to_png_path(img_path)
                    if not os.path.isfile(png_path):
                        continue
                if force_has_no_png_file:
                    png_path = file_img_to_png_path(img_path)
                    if os.path.isfile(png_path):
                        continue
    
                yield img_path
            if done:
                break
        if done:
            break

def file_clean_name(path):
    _, filename = os.path.split(path)
    filename = filename.replace('.IMG', '')
    filename = filename.replace('_CALIB', '')
    return filename

def file_read_iss_file(path):
    obs = iss.from_file(path, fast_distortion=True)
    obs.full_path = path
    _, obs.filename = os.path.split(path) 
    return obs


###############################################################################
#
#
# READ AND WRITE DATA FILES IN THE RESULTS DIRECTORY
#
#
###############################################################################


def _results_path(img_path, file_type, root=RESULTS_ROOT, make_dirs=False):
    """Results path is of the form:
    
    <ROOT>/<file_type>/COISS_2<nnn>/nnnnnnnnnn_nnnnnnnnnn/filename
    """
    rdir, filename = os.path.split(img_path)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    assert dir2 == 'data'
    rdir, dir3 = os.path.split(rdir)
    filename = filename.upper()
    filename = filename.replace('.IMG', '')
    filename = filename.replace('_CALIB', '')

    assert os.path.exists(root)
    root = file_clean_join(root, file_type)
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    part_dir3 = file_clean_join(root, dir3)
    if make_dirs and not os.path.exists(part_dir3):
        os.mkdir(part_dir3)
    part_dir1 = file_clean_join(part_dir3, dir1)
    if make_dirs and not os.path.exists(part_dir1):
        os.mkdir(part_dir1)
    return file_clean_join(part_dir1, filename)

def file_img_to_log_path(img_path, bootstrap=False, make_dirs=True):
    fn = _results_path(img_path, 'logs', make_dirs=make_dirs)
    if bootstrap:
        fn += '-bootstrap'
    fn += '.log'
    return fn

def file_img_to_offset_path(img_path, make_dirs=False):
    fn = _results_path(img_path, 'offsets', make_dirs=make_dirs)
    fn += '-OFFSET.dat'
    return fn

def file_offset_to_img_path(offset_path):
    rdir, filename = os.path.split(offset_path)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    
    filename = filename.replace('-OFFSET.dat', '')
    filename += '_CALIB.IMG'

    img_path = file_clean_join(COISS_2XXX_DERIVED_ROOT, dir2, 'data',
                            dir1, filename)
        
    return img_path

def file_read_offset_metadata(img_path):
    offset_path = file_img_to_offset_path(img_path, make_dirs=False)
    
    if not os.path.exists(offset_path):
        return None

    offset_fp = open(offset_path, 'rb')
    metadata = msgpack.unpackb(offset_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    offset_fp.close()

    return metadata

def file_write_offset_metadata(img_path, metadata):
    """Write offset file for img_filename."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_offset_metadata')

    offset_path = file_img_to_offset_path(img_path, make_dirs=True)

    logger.debug('Writing offset file %s', offset_path)
    
    new_metadata = metadata.copy()
    if 'ext_data' in new_metadata:
        del new_metadata['ext_data']
    if 'ext_overlay' in new_metadata:
        del new_metadata['ext_overlay']
    # XXX ONLY UNTIL OFFSET WRITING IS FIXED TO HANDLE STARS
    if ('stars_metadata' in new_metadata and 
        new_metadata['stars_metadata'] is not None):
        if 'full_star_list' in new_metadata['stars_metadata']:
            new_metadata['stars_metadata'] = \
                    new_metadata['stars_metadata'].copy()
            del new_metadata['stars_metadata']['full_star_list']
            
    offset_fp = open(offset_path, 'wb')
    offset_fp.write(msgpack.packb(new_metadata, 
                                  default=msgpack_numpy.encode))    
    offset_fp.close()
    
    return offset_path

def file_img_to_png_path(img_path, make_dirs=False):
    fn = _results_path(img_path, 'png', make_dirs=make_dirs)
    fn += '.png'
    return fn

def file_write_png_from_image(img_path, image):
    fn = file_img_to_png_path(img_path, make_dirs=True)
    im = Image.fromarray(image)
    im.save(fn)
    
def file_mosaic_path(metadata, root=RESULTS_ROOT, make_dirs=True):
    """Mosaic filename is of the form:
    
    <ROOT>/mosaics/<bodyname>/<firstimg>_<lastimg>_<#img>.dat
    """
    assert os.path.exists(root)
    if len(metadata['path_list']) == 0:
        return None
    root = file_clean_join(root, 'mosaics')
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    root = file_clean_join(root, metadata['body_name'])
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    sorted_img_list = sorted(metadata['path_list'])
    _, first_filename = os.path.split(sorted_img_list[0])
    _, last_filename = os.path.split(sorted_img_list[-1])
    filename = '%s_%s_%04d.dat' % (first_filename[:13],
                                   last_filename[:13],
                                   len(sorted_img_list))
    
    return file_clean_join(root, filename)

def file_read_mosaic_metadata(path):
    if not os.path.exists(path):
        return None

    mosaic_fp = open(path, 'rb')
    metadata = msgpack.unpackb(mosaic_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    mosaic_fp.close()

    return metadata

def file_write_mosaic_metadata(metadata):
    """Write mosaic metadata."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_mosaic_metadata')

    mosaic_path = file_mosaic_path(metadata)

    logger.debug('Writing mosaic file %s', mosaic_path)
    
    mosaic_fp = open(mosaic_path, 'wb')
    mosaic_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    mosaic_fp.close()

    return mosaic_path
