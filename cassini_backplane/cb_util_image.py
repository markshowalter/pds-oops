import cb_logging
import logging

import numpy as np
import scipy.ndimage.filters as filt
import scipy.ndimage.morphology as morphology

LOGGING_NAME = 'cb.' + __name__


#===============================================================================
# 
# IMAGE MANIPULATION
#
#===============================================================================

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset."""
    if offset_u == 0 and offset_v == 0:
        return image
    
    image = np.roll(image, -offset_u, 1)
    image = np.roll(image, -offset_v, 0)

    if offset_u != 0:    
        if offset_u < 0:
            image[:,:-offset_u] = 0
        else:
            image[:,-offset_u:] = 0
    if offset_v != 0:
        if offset_v < 0:
            image[:-offset_v,:] = 0
        else:
            image[-offset_v:,:] = 0
    
    return image

def pad_image(image, margin):
    if margin[0] == 0 and margin[1] == 0:
        return image
    new_image = np.zeros((image.shape[0]+margin[1]*2,image.shape[1]+margin[0]*2),
                         dtype=image.dtype)
    new_image[margin[1]:margin[1]+image.shape[0],
              margin[0]:margin[0]+image.shape[1], ...] = image
    return new_image

def unpad_image(image, margin):
    if margin[0] == 0 and margin[1] == 0:
        return image
    return image[margin[1]:image.shape[0]-margin[1],
                 margin[0]:image.shape[1]-margin[0], ...]

def compress_saturated_overlay(overlay):
    # Compress an RGB overlay assuming everything is either 0 or 255
    ret = np.empty((overlay.shape[0]/2, overlay.shape[1]), dtype=np.uint8)
    ret[:,:] = ( (overlay[ ::2,:,0] > 127) |
                ((overlay[ ::2,:,1] > 127) << 1) |
                ((overlay[ ::2,:,2] > 127) << 2) |
                ((overlay[1::2,:,0] > 127) << 3) |
                ((overlay[1::2,:,1] > 127) << 4) |
                ((overlay[1::2,:,2] > 127) << 5))
    return ret

def uncompress_saturated_overlay(overlay):
    ret = np.empty((overlay.shape[0]*2, overlay.shape[1], 3), dtype=np.uint8)
    ret[ ::2,:,0] =  overlay & 1
    ret[ ::2,:,1] = (overlay & 2) >> 1
    ret[ ::2,:,2] = (overlay & 4) >> 2
    ret[1::2,:,0] = (overlay & 8) >> 3
    ret[1::2,:,1] = (overlay & 16) >> 4
    ret[1::2,:,2] = (overlay & 32) >> 5
    ret *= 255
    return ret

#===============================================================================
# 
# FILTERS
#
#===============================================================================

def filter_local_maximum(data, maximum_boxsize=3, median_boxsize=11,
                         maximum_blur=0, maximum_tolerance=1.,
                         minimum_boxsize=0, gaussian_blur=0.):
    if median_boxsize:
        flat = data - filt.median_filter(data, median_boxsize)
    else:
        flat = data
    
    assert maximum_boxsize > 0
    max_filter = filt.maximum_filter(data, maximum_boxsize)
    mask = data == max_filter
    
    if minimum_boxsize:
        min_filter = filt.minimum_filter(data, minimum_boxsize)
        tol_mask = data >= min_filter * maximum_tolerance
        mask = np.logical_and(mask, tol_mask)
        
    if maximum_blur:
        mask = filt.maximum_filter(mask, maximum_blur)
        
    filtered = np.zeros(data.shape, dtype=data.dtype)
    filtered[mask] = flat[mask]
    
    if gaussian_blur:
        filtered = filt.gaussian_filter(filtered, gaussian_blur)

    return filtered

def filter_sub_median(data, median_boxsize=11, gaussian_blur=0.):
    if not median_boxsize and not gaussian_blur:
        return data
    
    sub = data
    
    if median_boxsize:
        sub = filt.median_filter(sub, median_boxsize)
    
    if gaussian_blur:
        sub = filt.gaussian_filter(sub, gaussian_blur)

    return data - sub


def detect_local_maxima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # Modified to detect maxima instead of minima
    """
    Takes an array and detects the peaks using the local minimum filter.
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = (filt.maximum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_maxima = local_max - eroded_background
    return np.where(detected_maxima)       
