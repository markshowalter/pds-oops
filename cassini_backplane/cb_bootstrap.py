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

_BOOTSTRAP_KNOWNS = {}
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
        
def _bootstrap_find_offset(cand_path, cand_metadata, bootstrap_config, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+'._bootstrap_find_offset')

    _, cand_filename = os.path.split(cand_path)
    logger.info('Bootstrapping candidate %s', cand_filename)

    body_name = cand_metadata['bootstrap_body']

    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
    
    overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                      cand_body_metadata['latlon_mask'])

    if not np.any(overlap):
        logger.debug('No overlap with current mosaic - aborting')
        return None
    
    cand_obs = read_iss_file(cand_path)

    cart_dict = {body_name: mosaic_metadata}
    
    new_metadata = master_find_offset(cand_obs, create_overlay=True,
                                      bodies_cartographic_data=cart_dict,
                                      **kwargs) # XXX

    if new_metadata['offset'] is not None:
        logger.debug('Bootstrapping successful - updating mosaic')
        new_metadata['bootstrapped'] = True
        file_write_offset_metadata(cand_path, new_metadata)
        
        repro_metadata = bodies_reproject(
              cand_obs, body_name,
              offset=new_metadata['offset'],
              latitude_resolution=bootstrap_config['lat_resolution'], 
              longitude_resolution=bootstrap_config['lon_resolution'],
              latlon_type=bootstrap_config['latlon_type'],
              lon_direction=bootstrap_config['lon_direction'])
        
        bodies_mosaic_add(mosaic_metadata, repro_metadata)

        print 'Updated mosaic'
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

    for ref_path, ref_metadata in _BOOTSTRAP_KNOWNS[body_name]:
        if ref_metadata['bootstrap_body'] != body_name:
            continue
        ref_obs = read_iss_file(ref_path)
        _, ref_filename = os.path.split(ref_path)
        logger.debug('Adding reference to mosaic %s', ref_filename)

#        display_offset_data(ref_obs, ref_metadata, show_rings=False, show_bodies=False)

        repro_metadata = bodies_reproject(
              ref_obs, body_name,
              offset=ref_metadata['offset'],
              latitude_resolution=bootstrap_config['lat_resolution'], 
              longitude_resolution=bootstrap_config['lon_resolution'],
              latlon_type=bootstrap_config['latlon_type'],
              lon_direction=bootstrap_config['lon_direction'])
        
        bodies_mosaic_add(mosaic_metadata, repro_metadata)
        
    _BOOTSTRAP_MOSAICS[body_name] = mosaic_metadata

    file_write_mosaic_metadata(mosaic_metadata)
    
#     plt.imshow(mosaic_metadata['img'])
#     plt.show()
    
#     print 'Initial mosaic'    
#     display_body_mosaic(mosaic_metadata)
    
def _bootstrap_update_lists(body_name, cand_idx, new_metadata):
    candidates = _BOOTSTRAP_CANDIDATES[body_name]
    cand_path, cand_metadata = candidates[cand_idx]

    del candidates[cand_idx]
    
    _BOOTSTRAP_KNOWNS[body_name].append((cand_path, new_metadata))
        
    _BOOTSTRAP_KNOWNS[body_name].sort(key=lambda x: 
                                            abs(x[1]['midtime']))

def _bootstrap_time_expired(body_name):
    return False # XXX
    
def _bootstrap_sort_candidates(body_name):
    candidates = _BOOTSTRAP_CANDIDATES[body_name]
    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    for cand_path, cand_metadata in candidates:
        cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
        overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                          cand_body_metadata['latlon_mask'])
        count = np.sum(overlap)
        cand_metadata['.sort_metric'] = count 

    print 'Unsorted candidate list:'
    for i in xrange(len(candidates)):
        print '  ', candidates[i][0]
    print
    
    candidates.sort(key=lambda x: x[1]['.sort_metric'], reverse=True)
    
    print 'Sorted candidate list:'
    for i in xrange(len(candidates)):
        print '  ', candidates[i][0], candidates[i][1]['.sort_metric']
    print
        
        
def _bootstrap_process(force, bootstrap_config, **kwargs):
    for body_name in sorted(_BOOTSTRAP_CANDIDATES):
        if body_name not in BOOTSTRAP_BODY_LIST:
            continue
        candidates = _BOOTSTRAP_CANDIDATES[body_name]
        if force or _bootstrap_time_expired(body_name):
            # Process one body
            _bootstrap_make_initial_mosaic(body_name, bootstrap_config)
            go_again = True
            while go_again:
                go_again = False
                _bootstrap_sort_candidates(body_name)
                for cand_idx in xrange(len(candidates)):
                    cand_path, cand_metadata = candidates[cand_idx]
                    if cand_metadata['bootstrap_body'] != body_name:
                        continue
                    offset_metadata = _bootstrap_find_offset(
                               cand_path, cand_metadata, bootstrap_config,
                               allow_stars=False, **kwargs)
                    if (offset_metadata is not None and
                        offset_metadata['offset'] is not None):
                        _bootstrap_update_lists(
                            body_name, cand_idx, offset_metadata)
                        go_again = True
                        break    
    
def bootstrap_add_file(image_path, metadata, bootstrap_config=None, **kwargs):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_add_file')

    if bootstrap_config is None:
        bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
    if metadata is not None:
        _, image_filename = os.path.split(image_path)
        body_name = metadata['bootstrap_body']
        if (metadata['offset'] is not None and 
            ('bootstrapped' not in metadata or 
             not metadata['bootstrapped'])): # XXX
            if body_name not in _BOOTSTRAP_KNOWNS:
                _BOOTSTRAP_KNOWNS[body_name] = []
            _BOOTSTRAP_KNOWNS[body_name].append((image_path,metadata))
            _BOOTSTRAP_KNOWNS[body_name].sort(key=lambda x: 
                                                    abs(x[1]['midtime']))
            logger.debug('Known offset %s', image_filename)
        elif metadata['bootstrap_candidate']:
            if body_name not in _BOOTSTRAP_CANDIDATES:
                _BOOTSTRAP_CANDIDATES[body_name] = []
            _BOOTSTRAP_CANDIDATES[body_name].append((image_path,metadata))
            _BOOTSTRAP_CANDIDATES[body_name].sort(key=lambda x: 
                                                        abs(x[1]['midtime']))
            logger.debug('Candidate %s - %s', image_filename, body_name)
        else:
            logger.debug('No offset and not a candidate %s', image_filename)

    _bootstrap_process(metadata is None, bootstrap_config, **kwargs)
