###############################################################################
# cb_bootstrap.py
#
# Routines related to bootstrapping.
#
# Exported routines:
#    bootstrap_viable
###############################################################################

import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp

from imgdisp import *
import Tkinter as tk

from cb_bodies import *
from cb_gui_body_mosaic import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__


_BOOTSTRAP_ANGLE_TOLERANCE = 0.5 * oops.RPD

_BOOTSTRAP_INIT_KNOWNS = {}
_BOOTSTRAP_CANDIDATES = {}
_BOOTSTRAP_MOSAICS = {}
    

def bootstrap_viable(ref_path, ref_metadata, cand_path, cand_metadata):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_viable')

    if (ref_metadata['filter1'] != cand_metadata['filter1'] or
        ref_metadata['filter2'] != cand_metadata['filter2']):
        logger.debug('Incompatible - different filters')
        return False
    
    if ref_metadata['bootstrap_body'] != cand_metadata['bootstrap_body']:
        logger.debug('Incompatible - different bodies')
        return False
    
    return True

def _bootstrap_mask_overlap(mask1, mask2):
    # Scale the masks along each dimension to be the size of the maximum
    scale1 = float(mask1.shape[0]) / mask2.shape[0]
    scale2 = float(mask1.shape[1]) / mask2.shape[1]
    
    if scale1 < 1. and scale2 < 1.:
        mask1 = ndinterp.zoom(mask1, (1./scale1,1./scale2), order=0)
    elif scale1 > 1. and scale2 > 1.:
        mask2 = ndinterp.zoom(mask2, (scale1,scale2), order=0)
    else:
        if scale1 < 1.:
            mask1 = ndinterp.zoom(mask1, (1./scale1,1), order=0)
        elif scale1 > 1.:
            mask2 = ndinterp.zoom(mask2, (scale1,1), order=0)
        
        if scale2 < 1.:
            mask2 = ndinterp.zoom(mask2, (1,1./scale2), order=0)
        elif scale2 > 1.:
            mask1 = ndinterp.zoom(mask1, (1,scale2), order=0)

    # Deal with roundoff error
    if mask1.shape != mask2.shape:
        if mask1.shape[0] < mask2.shape[0]:
            mask2 = mask2[:mask1.shape[0],:]
        elif mask1.shape[0] > mask2.shape[0]:
            mask1 = mask1[:mask2.shape[0],:]
        if mask1.shape[1] < mask2.shape[1]:
            mask2 = mask2[:,mask1.shape[1]]
        elif mask1.shape[1] > mask2.shape[1]:
            mask1 = mask1[:,mask2.shape[1]]
    
    return np.logical_and(mask1, mask2)
    
def _bootstrap_bodies_reproject(obs, body_name, offset, bootstrap_config):
    data = bodies_interpolate_missing_stripes(obs.data)
    repro_metadata = bodies_reproject(
          obs, body_name, data=data, offset=offset,
          latitude_resolution=bootstrap_config['lat_resolution'], 
          longitude_resolution=bootstrap_config['lon_resolution'],
          latlon_type=bootstrap_config['latlon_type'],
          lon_direction=bootstrap_config['lon_direction'],
          mask_bad_areas=True)
    return repro_metadata
    
def _bootstrap_find_offset_and_update(cand_path, cand_metadata, 
                                      bootstrap_config, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+'._bootstrap_find_offset_and_update')

    _, cand_filename = os.path.split(cand_path)
    logger.info('Bootstrapping candidate %s', cand_filename)

    body_name = cand_metadata['bootstrap_body']

    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
    
    overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                      cand_body_metadata['latlon_mask'])

    if not np.any(overlap):
        logger.debug('No overlap with current mosaic - aborting')
        cand_metadata['bootstrap_mosaic_path'] = file_mosaic_path(
                                                          mosaic_metadata)
        cand_metadata['bootstrap_mosaic_filenames'] = mosaic_metadata[
                                                          'filename_list']
        cand_metadata['bootstrap_status'] = 'No overlap'
        file_write_offset_metadata(cand_path, cand_metadata)
        return None
    
    cand_obs = read_iss_file(cand_path)

    cart_dict = {body_name: mosaic_metadata}
    
    new_metadata = master_find_offset(cand_obs, create_overlay=True,
                                      bodies_cartographic_data=cart_dict,
                                      **kwargs) # XXX

    if new_metadata['offset'] is None:
        logger.debug('Bootstrapping failed')
        cand_metadata['bootstrap_mosaic_path'] = file_mosaic_path(
                                                          mosaic_metadata)
        cand_metadata['bootstrap_mosaic_filenames'] = mosaic_metadata[
                                                          'filename_list']
        cand_metadata['bootstrap_status'] = 'Offset failed'
        file_write_offset_metadata(cand_path, cand_metadata)
        return None
    
    # Store the mosaic path so we can reproduce this navigation in the future
    new_metadata['bootstrap_mosaic_path'] = file_mosaic_path(mosaic_metadata)
    new_metadata['bootstrap_mosaic_filenames'] = mosaic_metadata[
                                                         'filename_list']
    new_metadata['bootstrap_status'] = 'Success'
    
    logger.debug('Bootstrapping successful - updating mosaic')
    new_metadata['bootstrapped'] = True
    file_write_offset_metadata(cand_path, new_metadata)
    
    if (new_metadata['filter1'] != 'CL1' or 
        new_metadata['filter2'] != 'CL2'):
        logger.debug('Filter not CLEAR - not updating mosaic')
        return new_metadata
    
    repro_metadata = _bootstrap_bodies_reproject(
          cand_obs, body_name, new_metadata['offset'], bootstrap_config)

    bodies_mosaic_add(mosaic_metadata, repro_metadata, 
                      resolution_threshold=1.05)

    file_write_mosaic_metadata(mosaic_metadata)
        
#         plt.figure()
#         plt.imshow(mosaic_metadata['img'])
#         plt.show()
#         display_body_mosaic(mosaic_metadata)
        
#    display_offset_data(ref_obs, ref_metadata, show_rings=False, show_bodies=False)
#     display_offset_data(cand_obs, new_metadata, show_rings=False, show_bodies=False)

    return new_metadata

def _bootstrap_make_initial_mosaic(body_name, bootstrap_config):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_make_initial_mosaic')

    mosaic_metadata = bodies_mosaic_init(
        body_name,
        latitude_resolution=bootstrap_config['lat_resolution'], 
        longitude_resolution=bootstrap_config['lon_resolution'],
        latlon_type=bootstrap_config['latlon_type'],
        lon_direction=bootstrap_config['lon_direction'])

    for ref_path, ref_metadata in _BOOTSTRAP_INIT_KNOWNS[body_name]:
        if ref_metadata['bootstrap_body'] != body_name:
            continue
        ref_obs = read_iss_file(ref_path)
        _, ref_filename = os.path.split(ref_path)
        logger.debug('Adding reference to mosaic %s', ref_filename)

#        display_offset_data(ref_obs, ref_metadata, show_rings=False, show_bodies=False)

        repro_metadata = _bootstrap_bodies_reproject(
              ref_obs, body_name, ref_metadata['offset'], bootstrap_config)
        
        bodies_mosaic_add(mosaic_metadata, repro_metadata, resolution_threshold=1.05,
                          copy_slop=2)

        file_write_mosaic_metadata(mosaic_metadata)
        
    _BOOTSTRAP_MOSAICS[body_name] = mosaic_metadata

    file_write_mosaic_metadata(mosaic_metadata)
    
#     plt.imshow(mosaic_metadata['img'])
#     plt.show()
    
#     print 'Initial mosaic'    
#     display_body_mosaic(mosaic_metadata)
    
def _bootstrap_update_lists(body_name, cand_idx, new_metadata):
    
    _BOOTSTRAP_INIT_KNOWNS[body_name].append((cand_path, new_metadata))
        
    _BOOTSTRAP_INIT_KNOWNS[body_name].sort(key=lambda x: 
                                            abs(x[1]['midtime']))

def _bootstrap_time_expired(body_name):
    return False # XXX
    
def _bootstrap_sort_candidates(body_name):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_sort_candidates')

    candidates = _BOOTSTRAP_CANDIDATES[body_name]
    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    for cand_path, cand_metadata in candidates:
        cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
        overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                          cand_body_metadata['latlon_mask'])
        count = np.sum(overlap)
        cand_metadata['.sort_metric'] = count 

    candidates.sort(key=lambda x: x[1]['.sort_metric'], reverse=True)
    
    logger.debug('Sorted candidate list:')
    for i in xrange(len(candidates)):
        _, filename = os.path.split(candidates[i][0])
        logger.debug('  %s - %s', filename, candidates[i][1]['.sort_metric'])
        
        
def _bootstrap_process_all(force, bootstrap_config, **kwargs):
    for body_name in sorted(_BOOTSTRAP_CANDIDATES):
        if body_name not in BOOTSTRAP_BODY_LIST:
            continue
        if force or _bootstrap_time_expired(body_name):
            _bootstrap_process_one(body_name, bootstrap_config, **kwargs)
            
def _bootstrap_process_one(body_name, bootstrap_config, **kwargs):
    if body_name not in _BOOTSTRAP_INIT_KNOWNS:
        return
    
    _bootstrap_make_initial_mosaic(body_name, bootstrap_config)
    candidates = _BOOTSTRAP_CANDIDATES[body_name]
    go_again = True
    while go_again:
        go_again = False
        _bootstrap_sort_candidates(body_name)
        for cand_idx in xrange(len(candidates)):
            cand_path, cand_metadata = candidates[cand_idx]
            if cand_metadata['bootstrap_body'] != body_name:
                continue
            offset_metadata = _bootstrap_find_offset_and_update(
                       cand_path, cand_metadata, bootstrap_config,
                       allow_stars=False, **kwargs)
            if (offset_metadata is not None and
                offset_metadata['offset'] is not None):
                # Success! No longer a candidate
                del candidates[cand_idx]
                go_again = True
                break    
    
def bootstrap_add_file(image_path, metadata, bootstrap_config=None,
                       redo_bootstrapped=False, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_add_file')

    if bootstrap_config is None:
        bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
        
    if metadata is None:
        _bootstrap_process_all(True, bootstrap_config, **kwargs)
        return
        
    body_name = metadata['bootstrap_body']
    if body_name is None:
        return

    _, image_filename = os.path.split(image_path)
    already_bootstrapped = ('bootstrapped' in metadata and 
                            metadata['bootstrapped'])

    if metadata['offset'] is not None and not already_bootstrapped:
#        if metadata['camera'] == 'WAC':
#            logger.debug('Known offset but ignoring WAC - %s', image_filename)
#            return
        if metadata['filter1'] != 'CL1' or metadata['filter2'] != 'CL2':
            logger.debug('Known offset but not clear filter - %s', 
                         image_filename)
            return
        if body_name not in _BOOTSTRAP_INIT_KNOWNS:
            _BOOTSTRAP_INIT_KNOWNS[body_name] = []
        _BOOTSTRAP_INIT_KNOWNS[body_name].append((image_path,metadata))
        _BOOTSTRAP_INIT_KNOWNS[body_name].sort(key=lambda x: 
                                               abs(x[1]['midtime']))
        logger.debug('Known offset %s', image_filename)
    elif (metadata['bootstrap_candidate'] or 
          (already_bootstrapped and redo_bootstrapped)):
        if body_name not in _BOOTSTRAP_CANDIDATES:
            _BOOTSTRAP_CANDIDATES[body_name] = []
        _BOOTSTRAP_CANDIDATES[body_name].append((image_path,metadata))
        _BOOTSTRAP_CANDIDATES[body_name].sort(key=lambda x: 
                                                    abs(x[1]['midtime']))
        logger.debug('Candidate %s - %s', image_filename, body_name)
    else:
        logger.debug('No offset and not a candidate %s', image_filename)

    _bootstrap_process_all(metadata is None, bootstrap_config, **kwargs)
