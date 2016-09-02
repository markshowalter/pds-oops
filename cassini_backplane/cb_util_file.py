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
#    file_offset_to_img_path
#    file_read_offset_metadata
#    file_write_offset_metadata
#    file_img_to_png_file
#    file_write_png_from_image
#    file_read_predicted_metadata
#    file_write_predicted_metadata
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
                    offset_path = _file_img_to_offset_path(img_path)
                    if not os.path.isfile(offset_path):
                        continue
                if force_has_no_offset_file:
                    offset_path = _file_img_to_offset_path(img_path)
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

def file_clean_name(path, keep_bootstrap=False):
    _, filename = os.path.split(path)
    filename = filename.replace('.IMG', '')
    filename = filename.replace('_CALIB', '')
    if not keep_bootstrap:
        filename = filename.replace('-BOOTSTRAP', '')
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


def _results_path(img_path, file_type, root=CB_RESULTS_ROOT, make_dirs=False,
                  include_image_filename=True):
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

def file_img_to_log_path(img_path, log_type, bootstrap=False, make_dirs=True):
    fn = _results_path(img_path, 'logs', make_dirs=make_dirs)
    fn += '-'+log_type
    if bootstrap:
        fn += '-BOOTSTRAP'
    log_datetime = datetime.datetime.now().isoformat()[:-7]
    log_datetime = log_datetime.replace(':','-')
    fn += '-' + log_datetime + '.log'
    return fn


### OFFSETS AND OVERLAYS

def _file_img_to_offset_path(img_path, bootstrap=False, make_dirs=False):
    fn = _results_path(img_path, 'offsets', make_dirs=make_dirs)
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

def _file_img_to_overlay_path(img_path, bootstrap=False, make_dirs=False):
    fn = _results_path(img_path, 'overlays', make_dirs=make_dirs)
    if bootstrap:
        fn += '-BOOTSTRAP'
    fn += '-OVERLAY.dat'
    return fn

def _compress_bool(a):
    if a is None:
        return None
    flat = a.astype('bool').flatten()
    assert (flat.shape[0] % 8) == 0

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
        offset_path = _file_img_to_offset_path(img_path, bootstrap=False,
                                               make_dirs=False)
    elif bootstrap_pref == 'force':
        offset_path = _file_img_to_offset_path(img_path, bootstrap=True,
                                               make_dirs=False)
    else:
        offset_path = _file_img_to_offset_path(img_path, bootstrap=True,
                                               make_dirs=False)
        if not os.path.exists(offset_path):
            offset_path = _file_img_to_offset_path(img_path, bootstrap=False,
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

    # Uncompress all body latlon_masks
    if 'bodies_metadata' in metadata:
        bodies_metadata = metadata['bodies_metadata']
        for key in bodies_metadata:
            mask = bodies_metadata[key]['latlon_mask']
            mask = _uncompress_bool(mask)
            bodies_metadata[key]['latlon_mask'] = mask

    if overlay:
        bootstrap = offset_path.find('-BOOTSTRAP') != -1
        overlay_path = _file_img_to_overlay_path(
                                 file_offset_to_img_path(offset_path),
                                 make_dirs=False, bootstrap=bootstrap)
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

def file_write_offset_metadata(img_path, metadata, overlay=True):
    """Write offset file for img_path."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_offset_metadata')

    bootstrap = False
    if 'bootstrapped' in metadata:
        bootstrap = metadata['bootstrapped']
    offset_path = _file_img_to_offset_path(img_path, bootstrap=bootstrap, make_dirs=True)
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

    # Compress all body latlon_masks
    if 'bodies_metadata' in metadata:
        bodies_metadata = metadata['bodies_metadata']
        for key in bodies_metadata:
            mask = bodies_metadata[key]['latlon_mask']
            mask = _compress_bool(mask)
            bodies_metadata[key]['latlon_mask'] = mask
            
    offset_fp = open(offset_path, 'wb')
    offset_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    offset_fp.close()

    if overlay:
        overlay_path = _file_img_to_overlay_path(img_path, bootstrap=bootstrap,
                                                 make_dirs=True)
        overlay_fp = open(overlay_path, 'wb')
        overlay_fp.write(msgpack.packb(metadata_overlay, 
                                       default=msgpack_numpy.encode))    
        overlay_fp.close()

    return offset_path


### PNG

def file_img_to_png_path(img_path, bootstrap=False, make_dirs=False):
    fn = _results_path(img_path, 'png', make_dirs=make_dirs)
    if bootstrap:
        fn += '-BOOTSTRAP'
    fn += '.png'
    return fn

def file_write_png_from_image(img_path, image, bootstrap=False):
    fn = file_img_to_png_path(img_path, bootstrap=bootstrap,
                              make_dirs=True)
    im = Image.fromarray(image)
    im.save(fn)


### PREDICTED KERNEL METADATA

def file_img_to_predicted_path(img_path, make_dirs=False):
    fn = _results_path(img_path, 'pred', make_dirs=make_dirs, 
                       include_image_filename=False)
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

    return file_clean_join(CB_RESULTS_ROOT, 'bootstrap', body_name+'-good.data')

def file_bootstrap_candidate_image_path(body_name, make_dirs=False):
    assert os.path.exists(CB_RESULTS_ROOT)
    root = file_clean_join(CB_RESULTS_ROOT, 'bootstrap')
    if make_dirs and not os.path.exists(root):
        try: # Necessary for multi-process race conditions
            os.mkdir(root)
        except OSError:
            pass

    return file_clean_join(CB_RESULTS_ROOT, 'bootstrap', body_name+'-candidate.data')
    
    
### REPROJECTED BODIES

def file_img_to_reproj_body_path(img_path, body_name, lat_res, lon_res, 
                                 latlon_type, lon_dir,
                                 make_dirs=False):
    fn = _results_path(img_path, 'reproj', make_dirs=make_dirs)
    fn += '-%.3f_%.3f-%s-%s-REPROJBODY-%s.dat' % (
              lat_res*oops.DPR, lon_res*oops.DPR, latlon_type, lon_dir,
              body_name.upper())
    return fn

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

def file_read_reproj_body(img_path, body_name, lat_res, lon_res, 
                          latlon_type, lon_dir):
    """Read reprojection metadata file for img_path."""
    reproj_path = file_img_to_reproj_body_path(img_path, body_name,
                                               lat_res, lon_res, 
                                               latlon_type, lon_dir, 
                                               make_dirs=False)
    
    if not os.path.exists(reproj_path):
        return None

    reproj_fp = open(reproj_path, 'rb')
    metadata = msgpack.unpackb(reproj_fp.read(), 
                               object_hook=msgpack_numpy.decode)
    reproj_fp.close()
    
    return metadata


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
