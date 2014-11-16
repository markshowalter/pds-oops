###############################################################################
# cb_util_image.py
#
# Routines related to image manipulation.
#
# Exported routines:
#    shift_image
#    pad_image
#    unpad_image
#    compress_saturated_overlay
#    uncompress_saturated_overlay
#    filter_local_maximum
#    filter_sub_median
###############################################################################

import cb_logging
import logging

import numpy as np
import scipy.ndimage.filters as filt

_LOGGING_NAME = 'cb.' + __name__


#==============================================================================
# 
# IMAGE MANIPULATION
#
#==============================================================================

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset.
    
    Pad the new area with zero and throw away the data moved off the edge."""
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
    """Pad an image with a zero-filled (U,V) margin on each edge."""
    if margin[0] == 0 and margin[1] == 0:
        return image
    new_image = np.zeros((image.shape[0]+margin[1]*2,image.shape[1]+margin[0]*2),
                         dtype=image.dtype)
    new_image[margin[1]:margin[1]+image.shape[0],
              margin[0]:margin[0]+image.shape[1], ...] = image
    return new_image

def unpad_image(image, margin):
    """Remove a padded margin (U,V) from each edge."""
    if margin[0] == 0 and margin[1] == 0:
        return image
    return image[margin[1]:image.shape[0]-margin[1],
                 margin[0]:image.shape[1]-margin[0], ...]

def compress_saturated_overlay(overlay):
    """Compress a 2-D RGB array making each color a single bit."""
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
    """Uncompress a 2-D RGB array."""
    ret = np.empty((overlay.shape[0]*2, overlay.shape[1], 3), dtype=np.uint8)
    ret[ ::2,:,0] =  overlay & 1
    ret[ ::2,:,1] = (overlay & 2) >> 1
    ret[ ::2,:,2] = (overlay & 4) >> 2
    ret[1::2,:,0] = (overlay & 8) >> 3
    ret[1::2,:,1] = (overlay & 16) >> 4
    ret[1::2,:,2] = (overlay & 32) >> 5
    ret *= 255
    return ret

#==============================================================================
# 
# FILTERS
#
#==============================================================================

def filter_local_maximum(data, maximum_boxsize=3, median_boxsize=11,
                         maximum_blur=0, maximum_tolerance=1.,
                         minimum_boxsize=0, gaussian_blur=0.):
    """Filter an image to find local maxima.
    
    Process:
        1) Create a mask consisting of the pixels that are local maxima
           within maximum_boxsize
        2) Find the minimum value for the area around each image pixel
           using minimum_boxsize
        3) Remove maxima that are not at least maximum_tolerange times
           the local minimum
        4) Blur the maximum pixels mask to make each a square area of
           maximum_blur X maximum_blur
        5) Compute the median-subtracted value for each image pixel
           using median_boxsize
        6) Copy the median-subtracted image pixels to a new zero-filled
           image where the maximum mask is true
        7) Gaussian blur this final result
        
    Inputs:
        data                The image
        maximum_boxsize     The box size to use when finding the maximum
                            value for the area around each pixel.
        median_boxsize      The box size to use when finding the median
                            value for the area around each pixel.
        maximum_blur        The amount to blur the maximum filter. If a pixel
                            is marked as a maximum, then the pixels in a
                            blur X blur square will also be marked as a
                            maximum.
        maximum_tolerance   The factor above the local minimum that a
                            maximum pixel has to be in order to be included
                            in the final result.
        minimum_boxsize     The box size to use when finding the minimum
                            value for the area around each pixel.
        gaussian_blur       The amount to blur the final result.
    """
                            
    assert maximum_boxsize > 0
    max_filter = filt.maximum_filter(data, maximum_boxsize)
    mask = data == max_filter
    
    if minimum_boxsize:
        min_filter = filt.minimum_filter(data, minimum_boxsize)
        tol_mask = data >= min_filter * maximum_tolerance
        mask = np.logical_and(mask, tol_mask)
        
    if maximum_blur:
        mask = filt.maximum_filter(mask, maximum_blur)
        
    if median_boxsize:
        flat = data - filt.median_filter(data, median_boxsize)
    else:
        flat = data
    
    filtered = np.zeros(data.shape, dtype=data.dtype)
    filtered[mask] = flat[mask]
    
    if gaussian_blur:
        filtered = filt.gaussian_filter(filtered, gaussian_blur)

    return filtered

def filter_sub_median(data, median_boxsize=11, gaussian_blur=0.):
    """Compute the median-subtracted value for each pixel.
    
    Inputs:
        data                The image
        median_boxsize      The box size to use when finding the median
                            value for the area around each pixel.
        gaussian_blur       The amount to blur the median value before
                            subtracting it from the image.
    """ 
    if not median_boxsize and not gaussian_blur:
        return data
    
    sub = data
    
    if median_boxsize:
        sub = filt.median_filter(sub, median_boxsize)
    
    if gaussian_blur:
        sub = filt.gaussian_filter(sub, gaussian_blur)

    return data - sub
