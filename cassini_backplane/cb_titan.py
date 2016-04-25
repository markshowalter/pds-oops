###############################################################################
# cb_titan.py
#
# Routines related to Titan navigation.
#
# Exported routines:
#    titan_navigate
###############################################################################

import cb_logging
import logging

import copy
import os
import time

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import oops

from cb_config import *
from cb_correlate import *
from cb_util_image import *
from cb_util_oops import *


_LOGGING_NAME = 'cb.' + __name__




def _titan_test_offset(data, ie_list, u_offset, v_offset):
    diff_list = []
    
    for cluster1_list, cluster2_list in ie_list:
        mean1_list = []
        mean2_list = []
        for pix in cluster1_list:
            if (not 0 <= pix[0]+v_offset < data.shape[0] or
                not 0 <= pix[1]+u_offset < data.shape[1]):
                return None
            mean1_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
        for pix in cluster2_list:
            if (not 0 <= pix[0]+v_offset < data.shape[0] or
                not 0 <= pix[1]+u_offset < data.shape[1]):
                return None
            mean2_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
        mean1 = np.mean(mean1_list)
        mean2 = np.mean(mean2_list)
#                if mean1 < min_threshold or mean2 < min_threshold:
#                    continue
        diff = np.abs(np.mean(mean2_list)-np.mean(mean1_list)) / np.mean(mean1_list)
        diff_list.append(diff)
        
    diff_list = np.array(diff_list)

    rms = np.sqrt(np.sum(diff_list**2))

    return rms

def titan_navigate(obs, other_model, titan_config=None):
    """Navigate Titan photometrically.
    
    Inputs:
        obs                The Observation.
        other_model        Optional array giving the locations (where non-zero)
                           of other bodies, rings, etc. in the image. These 
                           pixels will be masked out of the Titan model in
                           case they are in front of Titan and eclipsing it.
        titan_config       Configuration parameters.

    Returns:
        offset
        
        metadata is a dictionary containing

        'start_time'       The time (s) when titan_navigate was called.
        'end_time'         The time (s) when titan_navigate returned.
    """
    start_time = time.time()
    
    logger = logging.getLogger(_LOGGING_NAME+'.titan_navigate')

    if titan_config is None:
        titan_config = TITAN_DEFAULT_CONFIG
        
    metadata = {}
    metadata['start_time'] = start_time 

    set_obs_bp(obs)
    data = obs.data
        
    atmos_height = titan_config['atmosphere_height']
    
    logger.info('Performing Titan photometric navigation, '+
                'atmosphere height %.0f', atmos_height)

    body_titan = oops.Body.lookup('TITAN')
    try:
        body_atmos = oops.Body.lookup('TITAN+ATMOSPHERE')
    except KeyError:
        body_atmos = copy.copy(body_titan)
        body_atmos.name = 'TITAN+ATMOSPHERE'
        oops.Body.BODY_REGISTRY['TITAN+ATMOSPHERE'] = body_atmos
        
    radius = body_titan.radius + atmos_height
    surface = body_atmos.surface
    body_atmos.radius = radius
    body_atmos.inner_radius = radius
    body_atmos.surface = oops.surface.Spheroid(surface.origin, surface.frame, 
                                               (radius, radius))

    ext_bp_lambert = obs.ext_bp.lambert_law('TITAN+ATMOSPHERE')
    lambert = ext_bp_lambert.vals.astype('float')
    lambert[ma.getmaskarray(ext_bp_lambert.mvals)] = 0

    bp_incidence = obs.bp.incidence_angle('TITAN+ATMOSPHERE')
    bp_emission = obs.bp.emission_angle('TITAN+ATMOSPHERE')
    if other_model is not None:
        other_model_mask = other_model != 0.    
        bp_incidence = bp_incidence.mask_where(other_model_mask)
        bp_emission = bp_emission.mask_where(other_model_mask)
    full_mask = ma.getmaskarray(bp_incidence.mvals)
#    print 'Full mask', np.min(full_mask), np.max(full_mask)
#    plt.imshow(full_mask)
#    plt.show()
#    print 'Incudence'
#    plt.imshow(bp_incidence.mvals)
#    plt.show()
#    print 'Emission'
#    plt.imshow(bp_emission.mvals)
#    plt.show()
    
#    full_mask[:offset_limit,:] = True
#    full_mask[-offset_limit:,:] = True
#    full_mask[:,:offset_limit] = True
#    full_mask[:,-offset_limit:] = True
#
#    bp_incidence.vals[full_mask] = 0.
#    bp_emission.vals[full_mask] = 0.

    inc_incr = titan_config['incidence_increment']
    em_incr = titan_config['emission_increment']
    
    min_inc = bp_incidence.min()
    max_inc = bp_incidence.max()
    
    print 'Inc', min_inc, max_inc
    
    show_intersect = True
    total_intersect = None
    inc_list = []
    for inc_ang in np.arange(0, oops.PI, inc_incr):
        if inc_ang < min_inc:
            continue
        if inc_ang > max_inc:
            break
        intersect = obs.bp.border_atop(('incidence_angle', 'TITAN+ATMOSPHERE'), 
                                       inc_ang).vals
        intersect[full_mask] = 0
        print 'I', inc_ang
#        plt.imshow(intersect)
#        plt.show()
        inc_list.append(intersect)
        if show_intersect:
            if total_intersect is None:
                total_intersect = intersect
            else:
                total_intersect = np.logical_or(total_intersect, intersect)
        
    em_list = []
    max_emission = titan_config['max_emission_angle']
    for em_ang in np.arange(em_incr/2, max_emission, em_incr):
        intersect = obs.bp.border_atop(('emission_angle', 'TITAN+ATMOSPHERE'), 
                                       em_ang).vals
        intersect[full_mask] = 0
        em_list.append(intersect)
        if show_intersect:
            total_intersect = np.logical_or(total_intersect, intersect)

    if show_intersect:
        plt.imshow(total_intersect)
        plt.show()
    
    ie_list = []
    
    gap_threshold = titan_config['min_gap_pixels']
    max_cluster_size = titan_config['max_cluster_size']
    
    for inc_int in inc_list:
        for em_int in em_list:
            joint_intersect = np.logical_and(inc_int, em_int)
            pixels = np.where(joint_intersect) # in Y,X
            pixels = zip(*pixels)
            if len(pixels) < 2:
                continue
            pixels.sort()
            # There has to be a gap of "gap_threshold" in either x or y coords
            gap_y = None
            gap_x_pos = False
            last_x = None
            last_y = None
            cluster_xy = None
            for y, x in pixels:
                if last_y is not None:
                    if y-last_y >= gap_threshold:
                        gap_y = True
                        cluster_xy = y
                        break
                if last_x is not None:
                    if abs(x-last_x) >= gap_threshold:
                        gap_y = False
                        cluster_xy = x
                        gap_x_pos = x-last_x > 0
                        break
                last_x = x
                last_y = y
            if gap_y is None:
                # No gap!
                continue
            cluster1_list = []
            cluster2_list = []
            for pix in pixels:
                if ((gap_y and pix[0] >= cluster_xy) or 
                    (not gap_y and gap_x_pos and pix[1] >= cluster_xy) or
                    (not gap_y and not gap_x_pos and pix[1] <= cluster_xy)):
                    cluster2_list.append(pix)
                else:
                    cluster1_list.append(pix)
            if (len(cluster1_list) > max_cluster_size or 
                len(cluster2_list) > max_cluster_size):
                continue
            ie_list.append((cluster1_list, cluster2_list))

    if obs.detector == 'NAC':
        search_size_max = titan_config['search_size_max'][obs.detector]
        offset_list = find_correlation_and_offset(
                                  obs.ext_data, lambert,
                                  search_size_max=search_size_max)
        
        if len(offset_list) == 0:
            logger.info('Failed to find Titan+atmosphere model offset - aborting')
            metadata['end_time'] = time.time()
            return None

        starting_offset = offset_list[0][0]
        
        logger.info('Titan+atmosphere seed model offset U,V %d,%d', 
                    starting_offset[0], starting_offset[1])

        offset_limit_frac = titan_config['offset_limit'][obs.detector]
        titan_area = bp_incidence.mvals.count()
        titan_diameter = np.sqrt(titan_area / oops.PI)*2
        offset_limit1 = max(int(np.ceil(titan_diameter * offset_limit_frac)), 5)
        offset_limit = (offset_limit1, offset_limit1)
    else:
        starting_offset = (0,0)
        offset_limit = MAX_POINTING_ERROR_WAC

    starting_offset=(0,0)#XXX
    offset_limit=(15,15)#XXX
    logger.debug('Photometric search limit size U,V %d,%d', offset_limit[0], offset_limit[1])
    
    search_granularity = titan_config['search_granularity']

    offset_boundary_u = (starting_offset[0]-offset_limit[0],
                         starting_offset[0]+offset_limit[0]+1)
    offset_boundary_v = (starting_offset[1]-offset_limit[1],
                         starting_offset[1]+offset_limit[1]+1)
    
    for cur_granularity in [1]:#[search_granularity, 1]: 
        best_rms = 1e38
        best_offset = None
        u_min = max(starting_offset[0]-offset_limit[0], offset_boundary_u[0])
        u_max = min(starting_offset[0]+offset_limit[0]+1, offset_boundary_u[1])
        v_min = max(starting_offset[1]-offset_limit[1], offset_boundary_v[0])
        v_max = min(starting_offset[1]+offset_limit[1]+1, offset_boundary_v[1])
        res = np.zeros((v_max-v_min+1, u_max-u_min+1))
        for u_offset in xrange(u_min, u_max, cur_granularity):
            for v_offset in xrange(v_min, v_max, cur_granularity):
                rms = _titan_test_offset(data, ie_list, u_offset, v_offset)
                res[v_offset-v_min, u_offset-u_min] = rms
                if rms is not None and rms < best_rms:
                    best_rms = rms
                    best_offset = (u_offset, v_offset)

        plt.imshow(res)
        plt.show()
        
        if best_rms is None:
            break

        starting_offset = best_offset
        offset_limit = (cur_granularity, cur_granularity)

    if best_rms is None:
        logger.info('No final Titan offset found')
        best_offset = None
    else:
        logger.info('Final Titan offset U,V %d,%d RMS %f', 
                    best_offset[0], best_offset[1], best_rms)
    
    metadata['end_time'] = time.time()

    return best_offset
