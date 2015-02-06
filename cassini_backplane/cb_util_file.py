###############################################################################
# cb_util_file.py
#
# Routines related to reading and writing files.
#
# Exported routines:
#    file_offset_path
#    file_overlay_path
#    file_read_offset_metadata
#    file_write_offset_metadata
#    yield_image_filenames
###############################################################################

import cb_logging
import logging

import numpy as np
import msgpack
import msgpack_numpy
import os.path

import oops
import oops.inst.cassini.iss as iss

from cb_config import *

_LOGGING_NAME = 'cb.' + __name__


###############################################################################
#
#
# GENERAL FILE UTILITIES
#
#
###############################################################################

def yield_image_filenames(img_start_num, img_end_num, camera='NW',
                          restrict_list=None):
    done = False
    for coiss_dir in sorted(os.listdir(COISS_2XXX_DERIVED_ROOT)):
        coiss_fulldir = os.path.join(COISS_2XXX_DERIVED_ROOT, coiss_dir)
        if not os.path.isdir(coiss_fulldir):
            continue
        coiss_dir = os.path.join(coiss_fulldir, 'data')
        for range_dir in sorted(os.listdir(coiss_dir)):
            if len(range_dir) != 21 or range_dir[10] != '_':
                continue
            range1 = int(range_dir[:10])
            range2 = int(range_dir[11:])
            if range1 > img_end_num:
                done = True
                break
            if range2 < img_start_num:
                continue
            img_dir = os.path.join(coiss_dir, range_dir)
            # Sort by number then letter so N/W are together
            for img_name in sorted(os.listdir(img_dir),
                                   key=lambda x: x[1:13]+x[0]):
                if not img_name.endswith('_CALIB.IMG'):
                    continue
                img_num = int(img_name[1:11])
                if img_num > img_end_num:
                    done = True
                    break
                if img_num < img_start_num:
                    continue
                if img_name[0] not in camera:
                    continue
                yield os.path.join(img_dir, img_name)
            if done:
                break
        if done:
            break

def read_iss_file(path):
    obs = iss.from_file(path, fast_distortion=True)
    obs.full_path = path
    _, obs.filename = os.path.split(path) 
    return obs


###############################################################################
#
#
# READ AND WRITE BINARY DATA FILES IN THE RESULTS DIRECTORY
#
#
###############################################################################


def _results_root_for_file(img_path, root=RESULTS_ROOT,
                           make_dirs=True):
    """Results root is of the form:
    
    <ROOT>/COISS_2xxx/COISS_2<nnn>/nnnnnnnnnn_nnnnnnnnnn/filename
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
    root = os.path.join(root, 'COISS_2xxx')
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    part_dir3 = os.path.join(root, dir3)
    if make_dirs and not os.path.exists(part_dir3):
        os.mkdir(part_dir3)
    part_dir1 = os.path.join(part_dir3, dir1)
    if make_dirs and not os.path.exists(part_dir1):
        os.mkdir(part_dir1)
    return os.path.join(part_dir1, filename)

def file_img_to_offset_path(img_path, make_dirs=True):
    fn = _results_root_for_file(img_path, RESULTS_ROOT, make_dirs)
    fn += '-OFFSET.dat'
    return fn

def file_offset_to_img_path(offset_path):
    rdir, filename = os.path.split(offset_path)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    
    filename = filename.replace('-OFFSET.dat', '')
    filename += '_CALIB.IMG'

    img_path = os.path.join(COISS_2XXX_DERIVED_ROOT, dir2, 'data',
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

    offset_path = file_img_to_offset_path(img_path)

    logger.debug('Writing offset file %s', offset_path)
    
    new_metadata = metadata.copy()
    if 'ext_data' in new_metadata:
        del new_metadata['ext_data']
    if 'ext_overlay' in new_metadata:
        del new_metadata['ext_overlay']

    offset_fp = open(offset_path, 'wb')
    offset_fp.write(msgpack.packb(new_metadata, 
                                  default=msgpack_numpy.encode))    
    offset_fp.close()

def file_mosaic_path(metadata, root=RESULTS_ROOT, make_dirs=True):
    """Mosaic filename is of the form:
    
    <ROOT>/mosaics/<bodyname>/<firstimg>_<lastimg>_<#img>.dat
    """
    assert os.path.exists(root)
    root = os.path.join(root, 'mosaics')
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    root = os.path.join(root, metadata['body_name'])
    if make_dirs and not os.path.exists(root):
        os.mkdir(root)
    sorted_img_list = sorted(metadata['filename_list'])
    filename = '%s_%s_%04d.dat' % (sorted_img_list[0][:13],
                                   sorted_img_list[-1][:13],
                                   len(sorted_img_list))
    
    return os.path.join(root, filename)

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
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_moasic_metadata')

    mosaic_path = file_mosaic_path(metadata)

    logger.debug('Writing mosaic file %s', mosaic_path)
    
    mosaic_fp = open(mosaic_path, 'wb')
    mosaic_fp.write(msgpack.packb(metadata, 
                                  default=msgpack_numpy.encode))    
    mosaic_fp.close()
