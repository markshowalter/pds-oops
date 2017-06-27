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
#    file_img_to_short_img_path
#    file_results_path
#    file_img_to_log_path
#    file_img_to_offset_path
#    file_offset_to_img_path
#    file_img_to_overlay_path
#    file_read_offset_metadata
#    file_read_offset_metadata_path
#    file_write_offset_metadata
#    file_write_offset_metadata_path
#    file_img_to_png_path
#    file_write_png_from_image
#    file_write_png_path
#    file_img_to_predicted_path
#    file_read_predicted_metadata
#    file_write_predicted_metadata
#    file_bootstrap_good_image_path
#    file_bootstrap_candidate_image_path
#    file_bootstrap_shadow_to_str
#    file_bootstrap_status_image_path
#    file_img_to_reproj_body_path
#    file_read_reproj_body_path
#    file_read_reproj_body
#    file_write_reproj_body
#    file_mosaic_path
#    file_read_mosaic_metadata
#    file_write_mosaic_metadata
###############################################################################

import cb_logging
import logging

import argparse
import copy
import csv
import datetime
import numpy as np
import msgpack
import msgpack_numpy
import os.path
import sys
import zlib
from PIL import Image

import oops
import oops.inst.cassini.iss as iss

import starcat

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
    valid = ((len(name) == 13 or len(name) == 14) and
             name[0] in 'NW' and name[11] == '_')
    if valid:
        try:
            _ = int(name[1:11])
            _ = int(name[12:])
        except ValueError:
            valid = False
    if not valid:
        raise argparse.ArgumentTypeError(
             name+
             ' is not a valid image name - format must be [NW]dddddddddd_d{d}')
    return name

def file_add_selection_arguments(parser):
    parser.add_argument(
        '--first-image-num', type=int, default=1, metavar='IMAGE_NUM',
        help='The starting image number')
    parser.add_argument(
        '--last-image-num', type=int, default=9999999999, metavar='IMAGE_NUM',
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
        help='An entire volume or volume/range')
    parser.add_argument(
        '--first-volume-num', type=int, default=2001, metavar='VOL_NUM',
        help='The starting volume number')
    parser.add_argument(
        '--last-volume-num', type=int, default=2999, metavar='VOL_NUM',
        help='The ending volume number')
    parser.add_argument(
        '--image-full-path', action='append',
        help='The full path for an image')
    parser.add_argument(
        '--image-pds-csv', action='append',
        help='''A CSV file downloaded from PDS that contains filespecs of images
    to process''')
    parser.add_argument(
        '--image-filelist', action='append',
        help='''A file that contains image names of images to process''')
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
    parser.add_argument(
        '--selection-expr', type=str, metavar='EXPR',
        help='Expression to evaluate to decide whether to reprocess an offset')

def file_log_arguments(arguments, log):
    if arguments.image_full_path:
        log('*** Images explicitly from full paths:')
        for image_path in arguments.image_full_path:
            log('        %s', image_path)
    log('*** Image #s %010d - %010d',
        arguments.first_image_num,
        arguments.last_image_num)
    log('*** Volume #s %04d - %04d',
        arguments.first_volume_num,
        arguments.last_volume_num)
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
        log('*** Images restricted to volumes:')
        for filename in arguments.volume:
            log('        %s', filename)
    if arguments.image_pds_csv:
        log('*** Images restricted to those from PDS CSV:')
        for filename in arguments.image_pds_csv:
            log('        %s', filename)
    if arguments.image_filelist:
        log('*** Images restricted to those from file:')
        for filename in arguments.image_filelist:
            log('        %s', filename)

def file_yield_image_filenames_from_arguments(arguments, use_index_files=False):
    if arguments.image_full_path:
        for image_path in arguments.image_full_path:
            yield image_path
        return
    
    restrict_image_list = []
    if arguments.image_name is not None and arguments.image_name != [[]]:
        restrict_image_list = arguments.image_name[0][:] # Copy the list
    
    if arguments.image_pds_csv:
        for filename in arguments.image_pds_csv:
            with open(filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                header = csvreader.next()
                for colnum in xrange(len(header)):
                    if (header[colnum] == 'Primary File Spec' or
                        header[colnum] == 'primaryfilespec'):
                        break
                else:
                    print 'Badly formatted CSV file', filename
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

    if arguments.image_filelist:
        for filename in arguments.image_filelist:
            with open(filename, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if len(line) != 13 and len(line) != 14:
                        print 'BAD FILENAME', line
                        sys.exit(-1)
                    restrict_image_list.append(line)
    
    restrict_camera = 'NW'
    if arguments.nac_only:
        restrict_camera = 'N'
    if arguments.wac_only:
        restrict_camera = 'W'

    first_image_number = arguments.first_image_num
    last_image_number = arguments.last_image_num
    first_volume_number = arguments.first_volume_num
    last_volume_number = arguments.last_volume_num
    volumes = arguments.volume
    
    if len(restrict_image_list):
        first_image_number = max(first_image_number,
                                 min([int(x[1:11]) 
                                      for x in restrict_image_list]))
        last_image_number = min(last_image_number,
                                max([int(x[1:11]) 
                                     for x in restrict_image_list]))
        
    if not use_index_files:
        for image_path in file_yield_image_filenames(
                    first_image_number, 
                    last_image_number,
                    first_volume_number,
                    last_volume_number,
                    volumes,
                    restrict_camera,
                    restrict_image_list,
                    force_has_offset_file=arguments.has_offset_file,
                    force_has_no_offset_file=arguments.has_no_offset_file,
                    force_has_png_file=arguments.has_png_file,
                    force_has_no_png_file=arguments.has_no_png_file,
                    selection_expr=arguments.selection_expr):
            yield image_path
    else:
        for image_path in file_yield_image_filenames_index(
                    first_image_number, 
                    last_image_number,
                    first_volume_number,
                    last_volume_number,
                    volumes,
                    restrict_camera,
                    restrict_image_list,
                    force_has_offset_file=arguments.has_offset_file,
                    force_has_no_offset_file=arguments.has_no_offset_file,
                    force_has_png_file=arguments.has_png_file,
                    force_has_no_png_file=arguments.has_no_png_file,
                    selection_expr=arguments.selection_expr):
            yield image_path
        
def file_yield_image_filenames(img_start_num=0, img_end_num=9999999999,
                               vol_start_num=2001, vol_end_num=2999,
                               volumes=None, 
                               camera='NW', restrict_list=None,
                               force_has_offset_file=False,
                               force_has_no_offset_file=False,
                               force_has_png_file=False,
                               force_has_no_png_file=False,
                               selection_expr=None,
                               image_root=COISS_2XXX_DERIVED_ROOT,
                               suffix='_CALIB.IMG'):
    search_root = COISS_2XXX_DERIVED_ROOT
    search_suffix = '_CALIB.IMG'
    searching_offset = False
    searching_png = False
    
    if force_has_offset_file:
        search_root = file_clean_join(CB_RESULTS_ROOT, 'offsets')
        search_suffix = '-OFFSET.dat'
        searching_offset = True
    if force_has_png_file:
        search_root = file_clean_join(CB_RESULTS_ROOT, 'png')
        search_suffix = '.png'
        searching_png = True      

    done = False
    if not os.path.isdir(search_root):
        return
    for search_dir in sorted(os.listdir(search_root)):
        if not search_dir.startswith('COISS_') or len(search_dir) != 10:
            continue
        vol_num = int(search_dir[6:])
        if not vol_start_num <= vol_num <= vol_end_num:
            continue
        search_fulldir = file_clean_join(search_root, search_dir)
        coiss_fulldir = file_clean_join(image_root, search_dir)
        if volumes:
            found_vol_spec = False
            good_vol_spec = False
            for vol in volumes:
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
            if volumes:
                found_full_spec = False
                good_full_spec = False
                for vol in volumes:
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
                                   key=lambda x: x[1:11]+x[0]):
                if not img_name.endswith(search_suffix):
                    continue
                img_name = img_name.replace(search_suffix, suffix)
                img_num = int(img_name[1:11])
                if img_num > img_end_num:
                    return
                if img_num < img_start_num:
                    continue
                if img_name[0] not in camera:
                    continue
                if restrict_list and img_name[:img_name.find(suffix)] not in restrict_list:
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
                if selection_expr is not None:
                    metadata = file_read_offset_metadata(
                                 img_path, bootstrap_pref='no', overlay=False)
                    bs_metadata = file_read_offset_metadata(
                                 img_path, bootstrap_pref='force', overlay=False)
                    if metadata is None or not eval(selection_expr):
                        continue
                yield img_path
        if done:
            break
        
def file_yield_image_filenames_index(img_start_num=0, img_end_num=9999999999,
                                     vol_start_num=2001, vol_end_num=2999,
                                     volumes=None, 
                                     camera='NW', restrict_list=None,
                                     force_has_offset_file=False,
                                     force_has_no_offset_file=False,
                                     force_has_png_file=False,
                                     force_has_no_png_file=False,
                                     selection_expr=None,
                                     image_root='',
                                     suffix='_CALIB.IMG'):
    assert not force_has_offset_file
    assert not force_has_no_offset_file
    assert not force_has_png_file
    assert not force_has_no_png_file
    assert selection_expr is None
    
    search_root = COISS_2XXX_DERIVED_ROOT
    search_suffix = '.IMG'
    
    done = False
    if not os.path.isdir(search_root):
        return

    for vol_num in xrange(vol_start_num, vol_end_num+1):
        vol_name = 'COISS_%04d' % vol_num
        index_file = os.path.join(COISS_2XXX_DERIVED_ROOT, 
                                  vol_name+'-index.tab')
        if not os.path.exists(index_file):
            return
        coiss_fulldir = file_clean_join(image_root, vol_name)
        if volumes:
            found_vol_spec = False
            good_vol_spec = False
            for vol in volumes:
                if vol.find('/') != -1:
                    continue
                found_vol_spec = True
                if vol_name == vol:
                    good_vol_spec = True
                    break
            if found_vol_spec and not good_vol_spec:
                continue
        coiss_fulldir = file_clean_join(coiss_fulldir, 'data')
        with open(index_file, 'r') as index_fp:
            csvreader = csv.reader(index_fp)
            for row in csvreader:
                img_name = row[0].strip('" ')
                range_dir = row[1].strip('" ')
                range_dir, index_name = os.path.split(range_dir)
                data, range_dir = os.path.split(range_dir)
                assert data == 'data'
                assert img_name == index_name
                index_volume = row[2].strip('" ')
                assert index_volume == vol_name
                if len(range_dir) != 21 or range_dir[10] != '_':
                    continue
                if volumes:
                    found_full_spec = False
                    good_full_spec = False
                    for vol in volumes:
                        if vol.find('/') == -1:
                            continue
                        found_full_spec = True
                        if vol_name+'/'+range_dir == vol:
                            good_full_spec = True
                            break
                    if found_full_spec and not good_full_spec:
                        continue
                if not img_name.endswith(search_suffix):
                    continue
                img_name = img_name.replace(search_suffix, suffix)
                img_num = int(img_name[1:11])
                if img_num > img_end_num:
                    return
                if img_num < img_start_num:
                    continue
                if img_name[0] not in camera:
                    continue
                if restrict_list and img_name[:img_name.find(suffix)] not in restrict_list:
                    continue
                img_path = file_clean_join(image_root, vol_name, 'data',
                                           range_dir, img_name)
                yield img_path

def file_clean_name(path, keep_bootstrap=False):
    _, filename = os.path.split(path)
    filename = filename.replace('.IMG', '')
    filename = filename.replace('_CALIB', '')
    if not keep_bootstrap:
        filename = filename.replace('-BOOTSTRAP', '')
    return filename

def file_read_iss_file(path, orig_path=None):
    obs = iss.from_file(path, fast_distortion=True)
    if orig_path is None:
        orig_path = path
    obs.full_path = orig_path
    _, obs.filename = os.path.split(orig_path) 
    return obs


###############################################################################
#
#
# READ AND WRITE DATA FILES IN THE RESULTS DIRECTORY
#
#
###############################################################################


# IMAGE FILES

def file_img_to_short_img_path(img_path):
    rdir, filename = os.path.split(img_path)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    assert dir2 == 'data'
    rdir, dir3 = os.path.split(rdir)
    return dir3+'/'+dir2+'/'+dir1+'/'+filename

# RESULTS FILES
    
def file_results_path(img_path, file_type, root=CB_RESULTS_ROOT, 
                      make_dirs=False, include_image_filename=True):
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

    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass
        
    root = file_clean_join(root, file_type)
    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass
    part_dir3 = file_clean_join(root, dir3)
    if make_dirs and not os.path.exists(part_dir3):
        try:
            os.mkdir(part_dir3)
        except OSError:
            pass
    part_dir1 = file_clean_join(part_dir3, dir1)
    if make_dirs and not os.path.exists(part_dir1):
        try:
            os.mkdir(part_dir1)
        except OSError:
            pass
    if include_image_filename:
        ret = file_clean_join(part_dir1, filename)
    else:
        ret = part_dir1
    return ret


### LOG FILES

def file_img_to_log_path(img_path, log_type, root=CB_RESULTS_ROOT, 
                         bootstrap=False, make_dirs=True):
    fn = file_results_path(img_path, 'logs', root=root, make_dirs=make_dirs)
    fn += '-'+log_type
    if bootstrap:
        fn += '-BOOTSTRAP'
    log_datetime = datetime.datetime.now().isoformat()[:-7]
    log_datetime = log_datetime.replace(':','-')
    fn += '-' + log_datetime + '.log'
    return fn


### OFFSETS AND OVERLAYS

def file_img_to_offset_path(img_path, root=CB_RESULTS_ROOT, 
                            bootstrap=False, make_dirs=False):
    fn = file_results_path(img_path, 'offsets', root=root, make_dirs=make_dirs)
    if bootstrap:
        fn += '-BOOTSTRAP'
    fn += '-OFFSET.dat'
    return fn

def file_offset_to_img_path(offset_path):
    rdir, filename = os.path.split(offset_path)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    
    filename = filename.replace('-BOOTSTRAP', '')
    filename = filename.replace('-OFFSET.dat', '')
    filename += '_CALIB.IMG'

    img_path = file_clean_join(COISS_2XXX_DERIVED_ROOT, dir2, 'data',
                            dir1, filename)
        
    return img_path

def file_img_to_overlay_path(img_path, root=CB_RESULTS_ROOT, 
                             bootstrap=False, make_dirs=False):
    fn = file_results_path(img_path, 'overlays', root=root, make_dirs=make_dirs)
    if bootstrap:
        fn += '-BOOTSTRAP'
    fn += '-OVERLAY.dat'
    return fn

def _compress_bool(a):
    if a is None:
        return None
    flat = a.astype('bool').flatten()
    res = zlib.compress(flat.data)
    return (a.shape, res)

def _uncompress_bool(comp):
    if comp is None:
        return None
    shape, flat = comp
    res = np.frombuffer(zlib.decompress(flat), dtype='bool')
    res = res.reshape(shape)
    return res

def file_read_offset_metadata(img_path, bootstrap_pref='prefer', overlay=True):
    # bootstrap is one of:
    #    no        Don't use the bootstrap file
    #    force     Force use of the bootstrap file
    #    prefer    Prefer use of the bootstrap file, if it exists
    assert bootstrap_pref in ('no', 'force', 'prefer')
    
    if bootstrap_pref == 'no':
        offset_path = file_img_to_offset_path(img_path, bootstrap=False,
                                               make_dirs=False)
    elif bootstrap_pref == 'force':
        offset_path = file_img_to_offset_path(img_path, bootstrap=True,
                                               make_dirs=False)
    else:
        offset_path = file_img_to_offset_path(img_path, bootstrap=True,
                                               make_dirs=False)
        if not os.path.exists(offset_path):
            offset_path = file_img_to_offset_path(img_path, bootstrap=False,
                                                   make_dirs=False)
            
    return file_read_offset_metadata_path(offset_path, overlay=overlay)

def file_read_offset_metadata_path(offset_path, overlay=True):    
    if not os.path.exists(offset_path):
        return None

    offset_fp = open(offset_path, 'rb')
    metadata = msgpack.unpackb(offset_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    offset_fp.close()

    metadata['offset_path'] = offset_path
    
    # UCAC4Star class can't be directly serialized by msgpack
    if ('stars_metadata' in metadata and 
        metadata['stars_metadata'] is not None):
        if 'full_star_list' in metadata['stars_metadata']:
            new_list = []
            for star in metadata['stars_metadata']['full_star_list']:
                new_star = starcat.UCAC4Star()
                new_star.from_dict(star)
                new_list.append(new_star)
            metadata['stars_metadata']['full_star_list'] = new_list

    # Uncompress all body reprojection data
    if 'bodies_metadata' in metadata:
        bodies_metadata = metadata['bodies_metadata']
        for body_name in bodies_metadata:
            if 'reproj' not in bodies_metadata[body_name]:
                continue
            reproj = bodies_metadata[body_name]['reproj']
            if reproj is None:
                continue
            mask = _uncompress_bool(reproj['full_mask'])
            reproj['full_mask'] = mask
            for data_type in ['eff_resolution', 'phase', 
                              'emission', 'incidence']:
                data = reproj[data_type]
                if data_type in ['phase', 'emission', 'incidence']:
                    data = data.astype('float')*oops.RPD
                new_data = np.zeros(mask.shape, dtype='float32')
                new_data[mask] = data
                reproj[data_type] = new_data
                
    if overlay:
        bootstrap = offset_path.find('-BOOTSTRAP') != -1
        overlay_path = file_img_to_overlay_path(
                                 file_offset_to_img_path(offset_path),
                                 make_dirs=False, bootstrap=bootstrap)
        if os.path.exists(overlay_path):
            overlay_fp = open(overlay_path, 'rb')
            metadata_overlay = msgpack.unpackb(overlay_fp.read(), 
                                               object_hook=msgpack_numpy.decode)
            overlay_fp.close()
            # Uncompress all boolean text overlay arrays
            for field in ['stars_overlay_text', 'bodies_overlay_text',
                          'rings_overlay_text']:
                if field in metadata_overlay:
                    metadata_overlay[field] = _uncompress_bool(
                                                       metadata_overlay[field])
            metadata.update(metadata_overlay)

    return metadata

def file_write_offset_metadata(img_path, metadata, root=CB_RESULTS_ROOT,
                               overlay=True):
    """Write offset file for img_path."""

    bootstrap = False
    if 'bootstrapped' in metadata:
        bootstrap = metadata['bootstrapped']
    offset_path = file_img_to_offset_path(img_path, root=root,
                                          bootstrap=bootstrap, make_dirs=True)
    overlay_path = None
    if overlay:
        overlay_path = file_img_to_overlay_path(img_path, root=root,
                                                bootstrap=bootstrap,
                                                make_dirs=True)
    return file_write_offset_metadata_path(offset_path, metadata, 
                                           overlay_path=overlay_path,
                                           overlay=overlay)

def file_write_offset_metadata_path(offset_path, metadata, overlay_path=None,
                                    overlay=True):
    """Write offset file to offset_path."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_offset_metadata_path')
    logger.debug('Writing offset file %s', offset_path)
    
    metadata = copy.deepcopy(metadata)
    metadata['offset_path'] = offset_path
    if 'ext_data' in metadata:
        del metadata['ext_data']
    if 'ext_overlay' in metadata:
        del metadata['ext_overlay']
        
    metadata_overlay = {}
    for field in ['overlay', 'stars_overlay', 'bodies_overlay',
                  'rings_overlay']:
        if field in metadata:
            metadata_overlay[field] = metadata[field]
            del metadata[field]
    # Compress all boolean text overlay arrays
    for field in ['stars_overlay_text', 'bodies_overlay_text',
                  'rings_overlay_text']:
        if field in metadata:
            metadata_overlay[field] = _compress_bool(metadata[field])
            del metadata[field]
    
    # UCAC4Star class can't be directly serialized by msgpack
    if ('stars_metadata' in metadata and 
        metadata['stars_metadata'] is not None):
        if 'full_star_list' in metadata['stars_metadata']:
            new_list = []
            for star in metadata['stars_metadata']['full_star_list']:
                new_star = star.to_dict()
                new_list.append(new_star)
            metadata['stars_metadata']['full_star_list'] = new_list

    # Compress all body reprojection data
    if 'bodies_metadata' in metadata:
        bodies_metadata = metadata['bodies_metadata']
        for body_name in bodies_metadata:
            if 'reproj' not in bodies_metadata[body_name]:
                continue
            reproj = bodies_metadata[body_name]['reproj']
            if reproj is None:
                continue
            mask = reproj['full_mask']
            reproj['full_mask'] = _compress_bool(mask)
            for data_type in ['eff_resolution', 'phase', 
                              'emission', 'incidence']:
                data = reproj[data_type]
                if data_type in ['phase', 'emission', 'incidence']:
                    data = (data*oops.DPR).astype('int8')
                reproj[data_type] = data[mask]
            
    offset_fp = open(offset_path, 'wb')
    offset_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    offset_fp.close()

    if overlay:
        overlay_fp = open(overlay_path, 'wb')
        overlay_fp.write(msgpack.packb(metadata_overlay, 
                                       default=msgpack_numpy.encode))    
        overlay_fp.close()

    return offset_path


### PNG

def file_img_to_png_path(img_path, root=CB_RESULTS_ROOT, 
                         bootstrap=False, make_dirs=False):
    fn = file_results_path(img_path, 'png', root=root, make_dirs=make_dirs)
    if bootstrap:
        fn += '-BOOTSTRAP'
    fn += '.png'
    return fn

def file_write_png_from_image(img_path, image, root=CB_RESULTS_ROOT, 
                              bootstrap=False):
    png_path = file_img_to_png_path(img_path, root=root, 
                                    bootstrap=bootstrap,
                                    make_dirs=True)
    file_write_png_path(png_path, image)

def file_write_png_path(png_path, image):
    im = Image.fromarray(image)
    im.save(png_path)


### PREDICTED KERNEL METADATA

def file_img_to_predicted_path(img_path, make_dirs=False):
    fn = file_results_path(img_path, 'pred', make_dirs=make_dirs, 
                       include_image_filename=False,
                       root=CB_SUPPORT_FILES_ROOT)
    fn = file_clean_join(fn, 'PREDICTED-METADATA.dat')
    return fn

def file_read_predicted_metadata(img_path):
    pred_path = file_img_to_predicted_path(img_path)
    if not os.path.exists(pred_path):
        return None

    pred_fp = open(pred_path, 'rb')
    metadata = msgpack.unpackb(pred_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    pred_fp.close()

    filename = file_clean_name(img_path)
    
    if filename not in metadata:
        return None
    
    return metadata[filename]

def file_write_predicted_metadata(img_path, metadata):
    pred_path = file_img_to_predicted_path(img_path, make_dirs=True)
    pred_metadata = {}
    if os.path.exists(pred_path):
        pred_fp = open(pred_path, 'rb')
        pred_metadata = msgpack.unpackb(pred_fp.read(), 
                                        object_hook=msgpack_numpy.decode)
        pred_fp.close()
    
    filename = file_clean_name(img_path)
    
    pred_metadata[filename] = metadata
    
    pred_fp = open(pred_path, 'wb')
    pred_fp.write(msgpack.packb(pred_metadata, 
                                default=msgpack_numpy.encode))    
    pred_fp.close()
    

### BOOTSTRAP DATA FILES

def file_bootstrap_good_image_path(body_name, make_dirs=False):
    assert os.path.exists(CB_RESULTS_ROOT)
    root = file_clean_join(CB_RESULTS_ROOT, 'bootstrap')
    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass

    return file_clean_join(CB_RESULTS_ROOT, 'bootstrap', body_name+'-good.dat')

def file_bootstrap_candidate_image_path(body_name, make_dirs=False):
    assert os.path.exists(CB_RESULTS_ROOT)
    root = file_clean_join(CB_RESULTS_ROOT, 'bootstrap')
    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass

    return file_clean_join(CB_RESULTS_ROOT, 'bootstrap', 
                           body_name+'-candidate.dat')

def file_bootstrap_shadow_to_str(lat_shadow_dir, lon_shadow_dir):    
    ew = 'EAST'
    if lon_shadow_dir:
        ew = 'WEST'
    ns = 'SOUTH'
    if lat_shadow_dir:
        ns = 'NORTH'
        
    return ns, ew

def file_bootstrap_status_image_path(body_name, lat_shadow_dir, 
                                     lon_shadow_dir, make_dirs=False):
    assert os.path.exists(CB_RESULTS_ROOT)
    root = file_clean_join(CB_RESULTS_ROOT, 'bootstrap')
    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass

    ns, ew = file_bootstrap_shadow_to_str(lat_shadow_dir, lon_shadow_dir)
    
    return file_clean_join(CB_RESULTS_ROOT, 'bootstrap', 
                           body_name+'-'+ns+'-'+ew+'-status.dat')
    
    
### REPROJECTED BODIES

def file_img_to_reproj_body_path(img_path, body_name, lat_res, lon_res, 
                                 latlon_type, lon_dir,
                                 make_dirs=False):
    fn = file_results_path(img_path, 'reproj', make_dirs=make_dirs)
    fn += '-%.3f_%.3f-%s-%s-REPROJBODY-%s.dat' % (
              lat_res*oops.DPR, lon_res*oops.DPR, latlon_type, lon_dir,
              body_name.upper())
    return fn

def file_read_reproj_body_path(reproj_path):
    if not os.path.exists(reproj_path):
        return None

    reproj_fp = open(reproj_path, 'rb')
    metadata = msgpack.unpackb(reproj_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    reproj_fp.close()
    
    return metadata

def file_read_reproj_body(img_path, body_name, lat_res, lon_res, 
                          latlon_type, lon_dir):
    """Read reprojection metadata file for img_path."""
    reproj_path = file_img_to_reproj_body_path(img_path, body_name,
                                               lat_res, lon_res, 
                                               latlon_type, lon_dir, 
                                               make_dirs=False)

    return file_read_reproj_body_path(reproj_path)

def file_write_reproj_body(img_path, metadata):
    """Write reprojection metadata file for img_path."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_reproj_body')

    reproj_path = file_img_to_reproj_body_path(img_path, metadata['body_name'],
                                               metadata['lat_resolution'],
                                               metadata['lon_resolution'],
                                               metadata['latlon_type'],
                                               metadata['lon_direction'],
                                               make_dirs=True)
    logger.debug('Writing body reprojection file %s', reproj_path)
    
    reproj_fp = open(reproj_path, 'wb')
    reproj_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    reproj_fp.close()

### MOSAICS
    
def file_mosaic_path(body_name, mosaic_root, root=CB_RESULTS_ROOT, 
                     make_dirs=True, next_num=False, reset_num=False):
    """Mosaic filename is of the form:
    
        <ROOT>/mosaics/<bodyname>/<mosaic_root>_<#img>.dat
    
    The current maximum image number is stored in:

        <ROOT>/mosaics/<bodyname>/last_mosaic_num.txt
    """
    assert os.path.exists(root)
    root = file_clean_join(root, 'mosaics')
    if make_dirs and not os.path.exists(root):
        try:
            os.mkdir(root)
        except OSError:
            pass
    root = file_clean_join(root, body_name)
    if make_dirs and not os.path.exists(root):
        try:
            os.mkdir(root)
        except OSError:
            pass

    num_path = file_clean_join(root, mosaic_root+'-idx.txt')
    if not reset_num and os.path.exists(num_path):
        num_fp = open(num_path, 'r')
        max_num = int(num_fp.readline())
        num_fp.close()
    else:
        max_num = 0
    
    if next_num:
        max_num += 1
    
        num_fp = open(num_path, 'w')
        num_fp.write(str(max_num))
        num_fp.close()
        
    mosaic_path = file_clean_join(root, '%s_%04d-MOSAIC.dat' % (mosaic_root,max_num))
    
    return mosaic_path

def file_read_mosaic_metadata(body_name, mosaic_root):
    """Read mosaic metadata."""
    mosaic_path = file_mosaic_path(body_name, mosaic_root, next_num=False)
    
    return file_read_mosaic_metadata_path(mosaic_path)

def file_read_mosaic_metadata_path(mosaic_path):
    logger = logging.getLogger(_LOGGING_NAME+'.file_read_mosaic_metadata')
    logger.debug('Reading mosaic file %s', mosaic_path)

    if not os.path.exists(mosaic_path):
        return None

    mosaic_fp = open(mosaic_path, 'rb')
    metadata = msgpack.unpackb(mosaic_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    mosaic_fp.close()

    metadata['full_path'] = mosaic_path    

    return metadata

def file_write_mosaic_metadata(body_name, mosaic_root, metadata, 
                               reset_num=False):
    """Write mosaic metadata."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_mosaic_metadata')

    mosaic_path = file_mosaic_path(body_name, mosaic_root, next_num=True,
                                   reset_num=reset_num)

    logger.debug('Writing mosaic file %s', mosaic_path)

    mosaic_fp = open(mosaic_path, 'wb')
    mosaic_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    mosaic_fp.close()

    return mosaic_path
