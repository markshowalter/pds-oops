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
import pickle
import time

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import Tkinter as tk

from imgdisp import *
import oops

from cb_config import *
from cb_correlate import *
from cb_util_image import *
from cb_util_misc import *
from cb_util_oops import *


_LOGGING_NAME = 'cb.' + __name__

_BASELINE_DB = None
_PHASE_BIN_GRANULARITY = None

def titan_find_symmetry_offset(
           obs, sun_angle,
           min_emission_angle, max_emission_angle, incr_emission_angle,
           min_incidence_angle, max_incidence_angle, incr_incidence_angle,
           cluster_gap_threshold, cluster_max_pixels, 
           offset_limit, titan_size_u=None, titan_size_v=None, mask=None,
           display_total_intersect=False):
    """Find the axis of symmetry along the solar angle.
    
    Inputs:
        obs                    The Observation.
        
        sun_angle              The projected angle of the solar illumination
                               towards the center of Titan. 0 means direct
                               right-ward illumination and the angle moves
                               clockwise.
        
        min_emission_angle     The series of emission angles used to find
        max_emission_angle     intersection points.
        incr_emission_angle
        
        min_incidence_angle    The series of incidence angles used to find
        max_incidence_angle    intersection points.
        incr_incidence_angle
        
        cluster_gap_threshold  The minimum number of pixels required to 
                               separate two clusters of emission/incidence
                               intersection points.
                               
        cluster_max_pixels     The maximum number of pixels allowed in a 
                               cluster.
                               
        offset_limit           The number of pixels to search on each side
                               of the centerline.
                               
        mask                   A mask of pixels to avoid when looking for
                               intersect points.
                               
        titan_size_u           The size of Titan's bounding box in pixels.
        titan_size_v           This is used to make sure the symmetry finding
                               doesn't search an area that is even bigger
                               than Titan, making it more likely that the black
                               void will show up as the symmetric axis. This
                               is not a perfect solution to the problem.
                               
    Returns:
        metadata, a dictionary containing
        
        'offset'               The final offset, None if no offset was found.
        'confidence'           The confidence level (0-1) of the navigation.
    """ 
    set_obs_bp(obs)
    data = obs.data

    bp_incidence = obs.bp.incidence_angle('TITAN+ATMOSPHERE')
    min_inc = bp_incidence.min()
    max_inc = bp_incidence.max()

    full_mask = ma.getmaskarray(bp_incidence.mvals)
    if mask is not None:
        full_mask = np.logical_or(full_mask, mask != 0.)    
    
    # Create circles of incidence and emission
    total_intersect = None
    inc_list = []
    for inc_ang in np.arange(min_incidence_angle,
                             min(max_inc, max_incidence_angle),
                             incr_incidence_angle):
        if inc_ang < min_inc:
            continue
        intersect = obs.bp.border_atop(('incidence_angle', 'TITAN+ATMOSPHERE'), 
                                       inc_ang).vals
        intersect[full_mask] = 0
        inc_list.append(intersect)
        if total_intersect is None:
            total_intersect = intersect
        else:
            total_intersect = np.logical_or(total_intersect, intersect)
        
    em_list = []
    for em_ang in np.arange(min_emission_angle,
                            max_emission_angle,
                            incr_emission_angle):
        intersect = obs.bp.border_atop(('emission_angle', 'TITAN+ATMOSPHERE'), 
                                       em_ang).vals
        intersect[full_mask] = 0
        em_list.append(intersect)
        if total_intersect is None:
            total_intersect = intersect
        else:
            total_intersect = np.logical_or(total_intersect, intersect)

    if display_total_intersect:
        plt.figure()
        plt.imshow(total_intersect)
        plt.show()
                
    # Find the intersections of the circles of I and E. We need two distinct
    # clusters of points so we can make comparisons between them. The clusters
    # have to be separated by at least cluster_gap_threshold pixels and there
    # can be at most cluster_max_pixels in each cluster. These limits to take
    # of the cases where the circles are almost directly on top of each other.
    ie_list = []

    for inc_int in inc_list:
        for em_int in em_list:
            joint_intersect = np.logical_and(inc_int, em_int)
            pixels = np.where(joint_intersect) # in Y,X
            pixels = zip(*pixels) # List of (y,x) pairs
            # There have to be at least two points of intewunrsection
            if len(pixels) < 2:
                continue
            pixels.sort() # Sort by increasing y then increasing x
            # There has to be a gap of "gap_threshold" in either x or y coords
            gap_y = None
            last_x = None
            last_y = None
            cluster_xy = None
            for y, x in pixels:
                if last_y is not None:
                    if y-last_y >= cluster_gap_threshold:
                        gap_y = True
                        cluster_xy = y
                        break
                last_y = y
            if gap_y is None:
                pixels.sort(key=lambda x: (x[1], x[0])) # Now sort by x then y
                for y, x in pixels:
                    if last_x is not None:
                        if abs(x-last_x) >= cluster_gap_threshold:
                            gap_y = False
                            cluster_xy = x
                            break
                    last_x = x
            if gap_y is None:
                # No gap!
                continue
            cluster1_list = []
            cluster2_list = []
            # pixels is sorted in the correct direction
            for pix in pixels:
                if ((gap_y and pix[0] >= cluster_xy) or 
                    (not gap_y and pix[1] >= cluster_xy)):
                    cluster2_list.append(pix)
                else:
                    cluster1_list.append(pix)
            if (len(cluster1_list) > cluster_max_pixels or 
                len(cluster2_list) > cluster_max_pixels):
                continue
            ie_list.append((cluster1_list, cluster2_list))
#            print cluster1_list
#            print cluster2_list
#            print
            
    best_rms = 1e38
    best_offset = None
    best_along_path_dist = None
    
    # Across Sun angle is perpendicular to main sun angle    
    a_sun_angle = sun_angle + oops.HALFPI

    for along_path_dist in xrange(-offset_limit,offset_limit+1):
        u_offset = int(np.round(along_path_dist * np.cos(a_sun_angle)))
        v_offset = int(np.round(along_path_dist * np.sin(a_sun_angle)))
        if ((titan_size_u is not None and abs(u_offset) > titan_size_u) or
            (titan_size_v is not None and abs(v_offset) > titan_size_v)):
            continue
            
#         print along_path_dist, u_offset, v_offset

        diff_list = []
        
        for cluster1_list, cluster2_list in ie_list:
            mean1_list = []
            mean2_list = []
            bad_cluster = False
            for pix in cluster1_list:
                if (not 0 <= pix[0]+v_offset < data.shape[0] or
                    not 0 <= pix[1]+u_offset < data.shape[1]):
                    # We need to be able to analyze all the data
                    bad_cluster = True
                    break
                mean1_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
            if bad_cluster:
                continue
            for pix in cluster2_list:
                if (not 0 <= pix[0]+v_offset < data.shape[0] or
                    not 0 <= pix[1]+u_offset < data.shape[1]):
                    bad_cluster = True
                    break
                mean2_list.append(data[pix[0]+v_offset, pix[1]+u_offset])
            if bad_cluster:
                continue
            if len(mean1_list) == 0 or len(mean2_list) == 0:
                continue
            mean1 = np.mean(mean1_list)
            mean2 = np.mean(mean2_list)
            diff = np.abs(np.mean(mean2_list)-np.mean(mean1_list)) / np.mean(mean1_list)
            diff_list.append(diff)

        diff_list = np.array(diff_list)
        rms = np.sqrt(np.sum(diff_list**2))
        if rms < best_rms:
            best_rms = rms
            best_offset = (u_offset, v_offset)
            best_along_path_dist = along_path_dist

    if (best_along_path_dist is None or 
        abs(best_along_path_dist) == offset_limit):
        return None, None

    return best_offset, best_rms
    
def titan_along_track_profile(obs, offset, sun_angle, titan_center,
                              titan_radius_pix, titan_resolution):
    """Create a profile along the axis of symmetry.
    
    Inputs:
        obs                The Observation.

        offset             The image offset used to find the axis of symmetry.

        sun_angle          The projected angle of the solar illumination
                           towards the center of Titan. 0 means direct
                           right-ward illumination and the angle moves
                           clockwise.

        titan_center       The center of Titan (U,V) before the offset is
                           applied.
                           
        titan_radius_pix   The radius of Titan in pixels.
        
        titan_resolution   The image resolution of titan in km/pix.
        
    Returns:
        profile_x, profile_y
        
        profile_x          An array of distances in km from the center of 
                           Titan.
                           
        profile_y          The data values along the track.
    """
    data = obs.data
    
    profile_x = []
    profile_y = []
    
    for along_path_dist in xrange(-titan_radius_pix,titan_radius_pix+1):
        # We want to go along the Sun angle
        u = (int(np.round(along_path_dist * np.cos(sun_angle))) + 
             offset[0] + titan_center[0])
        v = (int(np.round(along_path_dist * np.sin(sun_angle))) + 
             offset[1] + titan_center[1])
        if (not 0 <= u < data.shape[1] or
            not 0 <= v < data.shape[0]):
            continue
        
        profile_x.append(along_path_dist * titan_resolution)
        profile_y.append(data[v,u])
    
    profile_x = np.array(profile_x)
    profile_y = np.array(profile_y)
    
    return profile_x, profile_y

def _find_baseline(filter, phase_angle):
    global _BASELINE_DB, _PHASE_BIN_GRANULARITY
    if _BASELINE_DB is None:
        filename = os.path.join(SUPPORT_FILES_ROOT,
                                'titan-profiles.pickle')
        fp = open(filename, 'rb')
        _PHASE_BIN_GRANULARITY = pickle.load(fp)
        _BASELINE_DB = pickle.load(fp)
        fp.close()
    
    if filter not in _BASELINE_DB:
        return None
    
    phase_bin = int(phase_angle / _PHASE_BIN_GRANULARITY)
    entry = _BASELINE_DB[filter][phase_bin]
    if entry is None:
        return None
    
    profile_x, profile_y, num_images = entry
     
    return profile_x, profile_y, num_images

def titan_navigate(obs, other_model, extend_fov=(0,0), titan_config=None):
    """Navigate Titan photometrically.
    
    We don't deal with models and text labels here; those are assumed to have 
    been taken care of during the earlier general bodies pass.
    
    Inputs:
        obs                The Observation.
        other_model        Optional array giving the locations (where non-zero)
                           of other bodies, rings, etc. in the image. These 
                           pixels will be masked out of the Titan model in
                           case they are in front of Titan and eclipsing it.
        extend_fov         The amount beyond the image in the (U,V) dimension
                           to consider when seeing if Titan is in the FOV.
        titan_config       Configuration parameters.

    Returns:
        metadata
        
        metadata is a dictionary containing

        'offset'           The final navigated offset.
        'confidence'       The confidence level (0-1) of the navigation.
        'symmetry_offset'  The offset used to create the axis of symmetry.
        'num_images'       The number of WAC images that went into making the
                           baseline profile used for navigation.
        'start_time'       The time (s) when titan_navigate was called.
        'end_time'         The time (s) when titan_navigate returned.
    """
    start_time = time.time()
    
    logger = logging.getLogger(_LOGGING_NAME+'.titan_navigate')

    if titan_config is None:
        titan_config = TITAN_DEFAULT_CONFIG
        
    metadata = {}
    metadata['start_time'] = start_time 
    metadata['symmetry_offset'] = None
    metadata['offset'] = None
    metadata['num_images'] = None
    metadata['confidence'] = 0.
    set_obs_bp(obs)
    data = obs.data

    if obs.filter1 == 'CL1' and obs.filter2 == 'CL2':
        filter = 'CLEAR'
    else:
        filter = obs.filter1
        if filter == 'CL1':
            filter = obs.filter2
        elif obs.filter2 != 'CL2':
            filter += '+' + obs.filter2

    titan_orig_inv_list = obs.inventory(['TITAN'], return_type='full')

    # Create the enlarged Titan
    atmos_height = titan_config['atmosphere_height']
    logger.info('Atmosphere height %.0f', atmos_height)

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

    # Titan parameters
    titan_inv_list = obs.inventory(['TITAN+ATMOSPHERE'], return_type='full')
    titan_inv = titan_inv_list['TITAN+ATMOSPHERE']
    titan_center = titan_inv['center_uv']
    titan_size_u = titan_inv['u_pixel_size']
    titan_size_v = titan_inv['v_pixel_size']
    titan_resolution = (titan_inv['resolution'][0]+
                        titan_inv['resolution'][1])/2
    titan_radius = titan_inv['outer_radius'] 
    titan_radius_pix = int(titan_radius / titan_resolution)

    logger.debug('Titan+atmosphere center U,V %d,%d / Resolution %.2f / Radius (km) %.2f / '+
                 'Radius (pix) %d', titan_center[0], titan_center[1],
                 titan_resolution, titan_radius, titan_radius_pix)

    # Hierarchy of confidence that Titan is really fully visible in the image
    u_min = titan_inv['u_min_unclipped']
    u_max = titan_inv['u_max_unclipped']
    v_min = titan_inv['v_min_unclipped']
    v_max = titan_inv['v_max_unclipped']

    titan_orig_inv = titan_orig_inv_list['TITAN']
    u_min2 = titan_orig_inv['u_min_unclipped']
    u_max2 = titan_orig_inv['u_max_unclipped']
    v_min2 = titan_orig_inv['v_min_unclipped']
    v_max2 = titan_orig_inv['v_max_unclipped']

    if (u_min >= extend_fov[0] and 
        u_max < obs.data.shape[1]-extend_fov[0] and
        v_min >= extend_fov[1] and
        v_max < obs.data.shape[0]-extend_fov[1]):
        logger.info('Titan+atmosphere entirely visible with extended FOV')
        confidence_size = 1.
    elif (u_min >= 0 and 
          u_max < obs.data.shape[1] and
          v_min >= 0 and
          v_max < obs.data.shape[0]):
        logger.info('Titan+atmosphere entirely visible only without extended FOV')
        confidence_size = 0.8
    elif (u_min2 >= 0 and 
          u_max2 < obs.data.shape[1] and
          v_min2 >= 0 and
          v_max2 < obs.data.shape[0]):
        logger.info('Titan body entirely visible only without extended FOV')
        confidence_size = 0.2
    else:
        logger.info('Titan body not entirely visible - aborting')
        metadata['end_time'] = time.time()
        return metadata
        
    phase_angle = obs.bp.center_phase_angle('TITAN+ATMOSPHERE').vals
    ret = _find_baseline(filter, phase_angle)

    if ret is None:
        logger.info('No baseline profile for filter %s and '+
                    'phase angle %.2f', filter, phase_angle*oops.DPR)
        metadata['end_time'] = time.time()
        return metadata

    baseline_x, baseline_profile, num_images = ret
    metadata['num_images'] = num_images
    
    logger.info('Baseline profile for filter %s phase angle %.2f was '+
                'constructed from %d images', filter, phase_angle*oops.DPR,
                num_images)

    confidence_images = 1. - 0.8 / num_images
        
    # Find the projected angle of the solar illumination to the center of
    # Titan's disc.
    bp_incidence = obs.bp.incidence_angle('TITAN+ATMOSPHERE')
    
    if phase_angle < oops.HALFPI:
        extreme_pos = np.argmin(bp_incidence.mvals)
    else:
        extreme_pos = np.argmax(bp_incidence.mvals)
    extreme_pos = np.unravel_index(extreme_pos, bp_incidence.mvals.shape)
    extreme_uv = (extreme_pos[1], extreme_pos[0])
    
    sun_angle = np.arctan2(extreme_uv[1]-titan_center[1], 
                           extreme_uv[0]-titan_center[0])

    logger.debug('Sun illumination angle %.2f', sun_angle*oops.DPR)

    em_min = titan_config['emission_min']
    em_max = titan_config['emission_max']
    em_incr = titan_config['emission_increment']
    inc_min = titan_config['incidence_min']
    inc_max = titan_config['incidence_max']
    inc_incr = titan_config['incidence_increment']
    
    cluster_gap_threshold = titan_config['cluster_gap_threshold']
    cluster_max_pixels = titan_config['cluster_max_pixels']
    
    max_error_u, max_error_v = MAX_POINTING_ERROR[(obs.data.shape, 
                                                   obs.detector)]
    
    offset_limit = int(np.ceil(np.sqrt(max_error_u**2+max_error_v**2)))+1
    
    model_mask = None
    if other_model is not None:
        model_mask = other_model != 0
        
    offset, rms = titan_find_symmetry_offset(
                     obs, sun_angle,
                     em_min, em_max, em_incr, inc_min, inc_max, inc_incr,
                     cluster_gap_threshold, cluster_max_pixels,
                     offset_limit, mask=model_mask,
                     titan_size_u=titan_size_u, titan_size_v=titan_size_v)
    
    if offset is None:
        logger.debug('No axis of symmetry found - aborting')
        metadata['end_time'] = time.time()
        return metadata
    
    metadata['symmetry_offset'] = offset
    logger.debug('Max symmetry offset U,V %d,%d', offset[0], offset[1])

    profile_x, profile_y = titan_along_track_profile(
                     obs, offset, sun_angle, titan_center,
                     titan_radius_pix, titan_resolution)

    interp_func = interp.interp1d(profile_x, profile_y, bounds_error=False, 
                                  fill_value=0., kind='cubic')
    profile = interp_func(baseline_x)
    
    along_track_distance = find_shift_1d(baseline_profile, 
                                         profile, 
                                         offset_limit)
    along_track_pixel = int(np.round(along_track_distance / titan_resolution))

    logger.debug('Along symmetry distance %.f km, %d pixels',
                 along_track_distance, along_track_pixel)
    
    new_offset = (int(offset[0] - np.cos(sun_angle)*along_track_pixel),
                  int(offset[1] - np.sin(sun_angle)*along_track_pixel))

    confidence = confidence_size * confidence_images
    
    logger.info('Final Titan offset U,V %d,%d (%.2f)', 
                new_offset[0], new_offset[1], confidence)
    metadata['offset'] = new_offset
    metadata['confidence'] = confidence

    new_titan_x = np.roll(baseline_x, -along_track_distance)
    new_titan_x = new_titan_x[:-abs(along_track_distance)]
    new_profile = profile[:-abs(along_track_distance)]
    
    if False:
        plt.figure()
        plt.plot(baseline_x, baseline_profile, '-', color='black')
        plt.plot(baseline_x, profile, '-', color='red')
        plt.plot(new_titan_x, new_profile, '-', color='green')
        plt.show()

    # We can't leave this hanging around but it confuses things...especially
    # inventory()
    del oops.Body.BODY_REGISTRY['TITAN+ATMOSPHERE']

    metadata['end_time'] = time.time()

    return metadata
