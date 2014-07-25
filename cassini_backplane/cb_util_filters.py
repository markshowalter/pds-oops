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


#===============================================================================
# 
# FILTERS
#
#===============================================================================

def filter_local_maximum(data, boxsize=11, area_size=3, gaussian_blur=0.):
    mask = (data == filt.maximum_filter(data, boxsize))
    
    flat = data - filt.median_filter(data, boxsize)
    
    mask = filt.maximum_filter(mask, area_size)
    
    filtered = np.zeros(data.shape)
    
    filtered[mask] = flat[mask]
    
#    for x in xrange(area_halfsize+1):
#        for y in xrange(area_halfsize+1):
#            nx = data.shape[1]-x
#            if nx == 0:
#                nx = data.shape[1]
#            ny = data.shape[0]-y
#            if ny == 0:
#                ny = data.shape[0]
#            
#            filtered[y:,x:][mask[:ny,:nx]] = flat[y:,x:][mask[:ny,:nx]]
#            filtered[y:,:nx][mask[:ny,x:]] = flat[y:,:nx][mask[:ny,x:]]
#            filtered[:ny,x:][mask[y:,:nx]] = flat[:ny,x:][mask[y:,:nx]]
#            filtered[:ny,:nx][mask[y:,x:]] = flat[:ny,:nx][mask[y:,x:]]
            
    if gaussian_blur:
        filtered = filt.gaussian_filter(filtered, gaussian_blur)

    filtered = flat
    
    return filtered


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
