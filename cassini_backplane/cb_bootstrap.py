#### XXX MAKE SLIDING WINDOW FOR TIME SLOT - DON'T JUST DELETE THE WHOLE LIST

###############################################################################
# cb_bootstrap.py
#
# Routines related to bootstrapping.
#
# Exported routines:
#    bootstrap_add_file
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

_LOGGING_MODULE_NAME = __name__


_BOOTSTRAP_INIT_KNOWNS = {}
_BOOTSTRAP_CANDIDATES = {}
_BOOTSTRAP_MOSAICS = {}
    

def _bootstrap_mask_overlap(mask1, mask2):
    if mask2 is None:
        return np.zeros(mask1.shape)
    
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
    if obs.filename[:13] == 'N1483279205_1':
        pass
    
    repro_metadata = bodies_reproject(
          obs, body_name, data=data, offset=offset,
          latitude_resolution=bootstrap_config['lat_resolution'], 
          longitude_resolution=bootstrap_config['lon_resolution'],
          latlon_type=bootstrap_config['latlon_type'],
          lon_direction=bootstrap_config['lon_direction'],
          mask_bad_areas=True,
          max_resolution = bootstrap_config['body_list'][body_name][1])
    if not np.any(repro_metadata['full_mask']):
        return None
    return repro_metadata
    
def _bootstrap_find_offset_and_update(cand_path, cand_metadata, 
                                      image_logfile_level,
                                      bootstrap_config, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+
                               '._bootstrap_find_offset_and_update')

    cand_filename = file_clean_name(cand_path)
    logger.info('Bootstrapping candidate %s', cand_filename)

    body_name = cand_metadata['bootstrap_body']

    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
    
    overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                      cand_body_metadata['latlon_mask'])

    if not np.any(overlap):
        logger.info('No overlap with current mosaic - aborting')
        cand_metadata['bootstrap_mosaic_path'] = file_mosaic_path(
                                                          mosaic_metadata)
        cand_metadata['bootstrap_mosaic_path_list'] = mosaic_metadata[
                                                          'path_list']            
        cand_metadata['bootstrap_status'] = 'No overlap'
        file_write_offset_metadata(cand_path, cand_metadata)
        return None
    
    # Set up per-image logging
    image_log_filehandler = None
    if image_logfile_level is not None:
        image_log_path = file_img_to_log_path(cand_path, bootstrap=True)
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this
       
        # This is added to the "cb." hierarchy, which is used by all other
        # cb_ modules     
        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    
    try:   
        cand_obs = read_iss_file(cand_path)
    except:
        logger.exception('File reading failed - %s', cand_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return None

    try:
        cart_dict = {body_name: mosaic_metadata}
        
        new_metadata = master_find_offset(cand_obs, create_overlay=True,
                                          bodies_cartographic_data=cart_dict,
                                          **kwargs) # XXX
    except:
        logger.exception('Offset finding failed - %s', cand_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return None

    cb_logging.log_remove_file_handler(image_log_filehandler)        
    
    if new_metadata['offset'] is None:
        logger.info('Bootstrapping failed')
        cand_metadata['bootstrap_mosaic_path'] = file_mosaic_path(
                                                          mosaic_metadata)
        cand_metadata['bootstrap_mosaic_path_list'] = mosaic_metadata[
                                                          'path_list']
        cand_metadata['bootstrap_status'] = 'Offset failed'
        
        try:
            file_write_offset_metadata(cand_path, cand_metadata)
        except:
            logger.exception('Offset file writing failed - %s', 
                                   cand_path)
            cb_logging.log_remove_file_handler(image_log_filehandler)
            return None
        
        return None

    # Store the mosaic path so we can reproduce this navigation in the future
    new_metadata['bootstrap_mosaic_path'] = file_mosaic_path(mosaic_metadata)
    new_metadata['bootstrap_mosaic_path_list'] = mosaic_metadata[
                                                         'path_list']
    new_metadata['bootstrap_status'] = 'Success'
    
    logger.info('Bootstrapping successful')
    new_metadata['bootstrapped'] = True
    new_metadata['original_metadata'] = cand_metadata

    try:
        file_write_offset_metadata(cand_path, new_metadata)
    except:
        logger.exception('Offset file writing failed - %s', cand_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return None
    
    if (new_metadata['filter1'] != 'CL1' or 
        new_metadata['filter2'] != 'CL2'):
        logger.info('Filter not CLEAR - not updating mosaic')
        return new_metadata
    
    repro_metadata = _bootstrap_bodies_reproject(
          cand_obs, body_name, new_metadata['offset'], bootstrap_config)

    if repro_metadata is None:
        logger.info('Reprojection is empty - ignoring for mosaic')
    else:
        bodies_mosaic_add(mosaic_metadata, repro_metadata, 
                          resolution_threshold=1.05, copy_slop=2)
        path = file_write_mosaic_metadata(mosaic_metadata)
        logger.info('Adding to mosaic and writing %s', path)

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
        ref_filename = file_clean_name(ref_path)
        inv = ref_obs.inventory([body_name], return_type='full')
        
        if body_name not in inv:
            logger.info('%s - %s not in observation body inventory - ignoring',
                         ref_filename, body_name)
            continue
        
        # First pass cutoff for resolution. Even if we pass this there may be
        # individual pixels that don't pass.
        max_res = bootstrap_config['body_list'][body_name][1]
        body_res = inv[body_name]['resolution'].to_scalar(0).vals
        if body_res > max_res:
            logger.info('%s - Resolution %.2f greater than maximum allowable %.2f',
                         ref_filename, body_res, max_res)
            continue 

        logger.info('%s - Adding reference to initial mosaic', ref_filename)

        repro_metadata = _bootstrap_bodies_reproject(
              ref_obs, body_name, ref_metadata['offset'], bootstrap_config)

        if repro_metadata is None:
            logger.info('Reprojection is empty - ignoring for mosaic')
        else:
            bodies_mosaic_add(mosaic_metadata, repro_metadata, resolution_threshold=1.05,
                              copy_slop=2)    
            path = file_write_mosaic_metadata(mosaic_metadata)
            logger.info('Adding to mosaic and writing %s', path)
        
    _BOOTSTRAP_MOSAICS[body_name] = mosaic_metadata

    if not np.any(mosaic_metadata['full_mask']):
        # Even when the mosaic is blank we let the whole process run because
        # this way the candidate offset metadata will get updated to
        # indicate that the bootstrapping was at least attempted.
        logger.info('No valid known-offset images - mosaic is blank')
    
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
        filename = file_clean_name(candidates[i][0])
        logger.debug('  %s - %s', filename, candidates[i][1]['.sort_metric'])
        
        
def _bootstrap_process_one_body(body_name,
                                image_logfile_level,
                                bootstrap_config, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_process_one_body')

    if body_name not in _BOOTSTRAP_INIT_KNOWNS:
        _BOOTSTRAP_INIT_KNOWNS[body_name] = []
    if body_name not in _BOOTSTRAP_CANDIDATES:
        _BOOTSTRAP_CANDIDATES[body_name] = []

    if len(_BOOTSTRAP_CANDIDATES[body_name]) == 0:
        _BOOTSTRAP_INIT_KNOWNS[body_name] = []
        return
        
    logger.info('Processing %s', body_name)
    logger.info('Known list:')
    for known_path, known_metadata in _BOOTSTRAP_INIT_KNOWNS[body_name]:
        logger.info('  %s', file_clean_name(known_path))
    logger.info('Candidate list:')
    for candidate_path, candidate_metadata in _BOOTSTRAP_CANDIDATES[body_name]:
        logger.info('  %s', file_clean_name(candidate_path))

    if len(_BOOTSTRAP_INIT_KNOWNS[body_name]) == 0:
        logger.info('No known images to make mosaic - aborting')
        _BOOTSTRAP_CANDIDATES[body_name] = []
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
                       cand_path, cand_metadata, 
                       image_logfile_level,
                       bootstrap_config,
                       allow_stars=False, **kwargs)
            if (offset_metadata is not None and
                offset_metadata['offset'] is not None):
                # Success! No longer a candidate
                del candidates[cand_idx]
                go_again = True
                break    

    _BOOTSTRAP_INIT_KNOWNS[body_name] = []
    _BOOTSTRAP_CANDIDATES[body_name] = []
    del _BOOTSTRAP_MOSAICS[body_name]
    
