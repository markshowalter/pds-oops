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
###############################################################################

import cb_logging
import logging

import pickle
import numpy as np
import os.path

import oops
import oops.inst.cassini.iss as iss

from cb_config import *

_LOGGING_NAME = 'cb.' + __name__

_OFFSET_FILE_VERSION = 0


def _results_root_for_file(img_filename, root=RESULTS_ROOT):
    rdir, filename = os.path.split(img_filename)
    rdir, dir1 = os.path.split(rdir)
    rdir, dir2 = os.path.split(rdir)
    assert dir2 == 'data'
    rdir, dir3 = os.path.split(rdir)
    filename = filename.upper()
    filename = filename.replace('.IMG', '')
    filename = filename.replace('_CALIB', '')

    assert os.path.exists(root)
    root = os.path.join(root, 'COISS_2xxx')
    assert os.path.exists(root)
    
    part_dir3 = os.path.join(root, dir3)
    if not os.path.exists(part_dir3):
        os.mkdir(part_dir3)
    part_dir1 = os.path.join(part_dir3, dir1)
    if not os.path.exists(part_dir1):
        os.mkdir(part_dir1)
    return os.path.join(root, dir3, dir1, filename)

def file_offset_path(img_filename):
    fn = _results_root_for_file(img_filename, RESULTS_ROOT) + '-OFFSET.pickle'
    return fn

def file_overlay_path(img_filename):
    fn = _results_root_for_file(img_filename, RESULTS_ROOT) + '-OVERLAY'
    fn = os.path.join(RESULTS_ROOT, fn)
    return fn, fn+'.npy'

_OFFSET_FILE_VERSION = 0

def file_read_offset_metadata(img_filename, read_overlay=False):
    offset_path = file_offset_path(img_filename)
    overlay_path_save, overlay_path_load = file_overlay_path(img_filename)
    
    if not os.path.exists(offset_path):
        return None
    offset_pickle_fp = open(offset_path, 'rb')
    offset_file_version = pickle.load(offset_pickle_fp)
    assert offset_file_version == _OFFSET_FILE_VERSION
    metadata = pickle.load(offset_pickle_fp)
    offset_pickle_fp.close()

    metadata['overlay'] = None
    if read_overlay and os.path.exists(overlay_path_load):
        overlay = np.load(overlay_path_load)
        if overlay.shape[0] != 0:
            metadata['overlay'] = overlay 
        
    return metadata

def file_write_offset_metadata(img_filename, metadata):
    """Write offset/overlay files for img_filename."""
    logger = logging.getLogger(_LOGGING_NAME+'.file_write_offset_metadata')

    offset_path = file_offset_path(img_filename)
    overlay_path_save, overlay_path_load = file_overlay_path(img_filename)

    logger.debug('Writing offset file %s', offset_path)
    
    new_metadata = metadata.copy()
    if 'ext_data' in new_metadata:
        del new_metadata['ext_data']
    if 'ext_overlay' in new_metadata:
        del new_metadata['ext_overlay']
    overlay = np.array([])
    if 'overlay' in new_metadata and new_metadata['overlay'] is not None:
        overlay = new_metadata['overlay']
        del new_metadata['overlay']

    offset_pickle_fp = open(offset_path, 'wb')
    pickle.dump(_OFFSET_FILE_VERSION, offset_pickle_fp)
    pickle.dump(new_metadata, offset_pickle_fp)    
    offset_pickle_fp.close()

    logger.debug('Writing overlay file %s', overlay_path_save)
    
    np.save(overlay_path_save, overlay)

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

def read_iss_file(filename):
    obs = iss.from_file(filename, fast_distortion=True)
    obs.full_filename = filename
    _, obs.filename = os.path.split(filename) 
    return obs
