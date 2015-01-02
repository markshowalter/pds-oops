###############################################################################
# cb_correlate.py
#
# Routines related to correlating an image with a model and finding the
# pointing offset.
#
# Exported routines:
#    find_correlation_and_offset
###############################################################################

import cb_logging
import logging

import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from imgdisp import ImageDisp
import Tkinter

from cb_util_image import *

_LOGGING_NAME = 'cb.' + __name__

DEBUG_CORRELATE_PLOT = False
DEBUG_CORRELATE_IMGDISP = False


#==============================================================================
#
# INTERNAL HELPER ROUTINES
# 
#==============================================================================

def _next_power_of_2(n):
    """Compute the power of 2 >= n."""
    s = bin(n)[2:]
    if s.count('1') == 1: # Already power of 2
        return n
    s = '1' + '0' * len(s) 
    return int(s, 2)
    
def _pad_to_power_of_2(data):
    """Pad a 2-D array to be a power of 2 in each dimension."""
    s0 = _next_power_of_2(data.shape[0])
    s1 = _next_power_of_2(data.shape[1])
    
    if s0 == data.shape[0] and s1 == data.shape[1]:
        return data, (0,0)

    offset0 = (s0-data.shape[0])//2
    offset1 = (s1-data.shape[1])//2
    padding = (offset0,offset1)
    return pad_image(data, padding), padding    

#==============================================================================
#
# CORRELATION ROUTINES
# 
#==============================================================================

def _correlate2d(image, model, normalize=False, retile=False):
    """Correlate the image with the model; normalization to [-1,1] is optional.

    Correlation is performed using the 'correlation theorem' that equates
    correlation with a Fourier Transform.
    
    Inputs:
        image              The image.
        model              The model to correlate against image.
        normalize          If True, normalize the correlation result to [-1,1].
        retile             If True, the resulting correlation matrix is
                           shifted by 1/2 along each dimension so that
                           (0,0) is now in the center pixel:
                           (shape[0]//2,shape[1]//2).
                       
    Returns:
        The 2-D correlation matrix.
    """
    
    assert image.shape == model.shape

    if DEBUG_CORRELATE_IMGDISP:
        toplevel = Tkinter.Tk()
        frame_toplevel = Tkinter.Frame(toplevel)
        imdisp = ImageDisp([image,model], parent=frame_toplevel,
                           canvas_size=(512,512),
                           allow_enlarge=True, enlarge_limit=10,
                           auto_update=True)
        frame_toplevel.pack()
        Tkinter.mainloop()
    
    # Padding to a power of 2 makes FFT _much_ faster
    newimage, _ = _pad_to_power_of_2(image)
    newmodel, padding = _pad_to_power_of_2(model)
    
    image_fft = fftpack.fft2(newimage)
    model_fft = fftpack.fft2(newmodel)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))

    if normalize:
        norm_amt = np.sqrt(np.sum(image**2) * np.sum(model**2))
        if norm_amt != 0:
            corr /= norm_amt
    
    if retile:
        # This maps (0,0) into (-y,-x) == (shape[0]//2, shape[1]//2)
        y = corr.shape[0] // 2
        x = corr.shape[1] // 2
        offset_image = np.empty(corr.shape, image.dtype)
        offset_image[ 0:y, 0:x] = corr[-y: ,-x: ]
        offset_image[ 0:y,-x: ] = corr[-y: , 0:x]
        offset_image[-y: , 0:x] = corr[ 0:y,-x: ]
        offset_image[-y: ,-x: ] = corr[ 0:y, 0:x]
        corr = offset_image

    corr = unpad_image(corr, padding)
        
    return corr
    
def _find_correlated_offset(corr, search_size_min, search_size_max,
                            max_offsets, peak_margin):
    """Find the offset that best aligns an image and a model given the
    correlation.
    
    The offset is found by looking for the maximum correlation value within
    the given search range. Multiple offsets may be returned, in which case
    each peak and the area around it is eliminated from future consideration
    before the next peak is found.

    Inputs:
        corr               A 2-D correlation matrix with (0,0) located in the
                           center pixel (shape[0]//2,shape[1]//2).
        search_size_min    The number of pixels from an offset of zero to
        search_size_max    search. If either is a single number, the same
                           search size is used in each dimension. Otherwise it
                           is (search_size_u, search_size_v). The returned
                           offsets will always be within the range [min,max].
        max_offsets        The maximum number of offsets to return.
        peak_margin        The number of correlation pixels around a peak to
                           remove from future consideration before finding
                           the next peak.
                           
    Returns:
        List of
            offset_u, offset_v, peak_value
        
        offset_u           The offset in the U direction.
        offset_v           The offset in the V direction.
        peak_value         The correlation value at the peak in the range
                           [-1,1].
    """
    logger = logging.getLogger(_LOGGING_NAME+'._find_correlated_offset')

    if np.shape(search_size_min) == ():
        search_size_min_u = search_size_min
        search_size_min_v = search_size_min
    else:
        search_size_min_u, search_size_min_v = search_size_min

    if np.shape(search_size_max) == ():
        search_size_max_u = search_size_max
        search_size_max_v = search_size_max
    else:
        search_size_max_u, search_size_max_v = search_size_max

    logger.debug('Search U %d to %d, V %d to %d #PEAKS %d PEAKMARGIN %d',
                 search_size_min_u, search_size_max_u,
                 search_size_min_v, search_size_max_v,
                 max_offsets, peak_margin)
        
    assert 0 <= search_size_min_u <= search_size_max_u
    assert 0 <= search_size_min_v <= search_size_max_v
    assert 0 <= search_size_max_u <= corr.shape[1]//2 
    assert 0 <= search_size_max_v <= corr.shape[0]//2 

    # Extract a slice from the correlation matrix that is the maximum
    # search size and then make a "hole" in the center to represent
    # the minimum search size.
    slice = corr[corr.shape[0]//2-search_size_max_v:
                 corr.shape[0]//2+search_size_max_v+1,
                 corr.shape[1]//2-search_size_max_u:
                 corr.shape[1]//2+search_size_max_u+1].copy()

    global_min = np.min(slice)
    
    if search_size_min_u != 0 and search_size_min_v != 0:
        slice[slice.shape[0]//2-search_size_min_v+1:
              slice.shape[0]//2+search_size_min_v,
              slice.shape[1]//2-search_size_min_u+1:
              slice.shape[1]//2+search_size_min_u] = global_min

    # Iteratively search for the next peak.
    ret_list = []
    all_offset_u = []
    all_offset_v = []
    
    while len(ret_list) != max_offsets:
        peak = np.where(slice == slice.max())

        if DEBUG_CORRELATE_PLOT:
            plt.jet()
            plt.imshow(slice, interpolation='none')
            plt.contour(slice)
            plt.plot((search_size_max_u,search_size_max_u),
                     (0,2*search_size_max_v),'k')
            plt.plot((0,2*search_size_max_u),
                     (search_size_max_v,search_size_max_v),'k')
            if len(peak[0]) == 1:
                plt.plot(peak[1], peak[0], 'ko')
            plt.xticks(range(0,2*search_size_max_u+1,2), 
                       [str(x) for x in range(-search_size_max_u,
                                              search_size_max_u+1,2)])
            plt.yticks(range(0,2*search_size_max_v+1,2), 
                       [str(y) for y in range(-search_size_max_v,
                                              search_size_max_v+1,2)])
            plt.xlabel('U')
            plt.ylabel('V')
            plt.show()
        
        if DEBUG_CORRELATE_IMGDISP > 1:
            toplevel = Tkinter.Tk()
            frame_toplevel = Tkinter.Frame(toplevel)
            imdisp = ImageDisp([slice], parent=frame_toplevel,
                               canvas_size=(512,512),
                               allow_enlarge=True, enlarge_limit=10,
                               auto_update=True)
            frame_toplevel.pack()
            Tkinter.mainloop()
            
        if len(peak[0]) != 1:
            logger.debug('Peak # %d - No unique peak found - aborting',
                         len(ret_list)+1) 
            break

        peak_v = peak[0][0]
        peak_u = peak[1][0]
        offset_v = peak_v-search_size_max_v # Compensate for slice location
        offset_u = peak_u-search_size_max_u
        peak_val = slice[peak_v,peak_u]
        
        logger.debug('Peak # %d - Trial offset U,V %d,%d CORR %f',
                     len(ret_list)+1, 
                     offset_u, offset_v,
                     peak_val)

        if peak_val <= 0:
            logger.debug(
                 'Peak # %d - Correlation value is negative - aborting',
                 len(ret_list)+1)
            break
        
        all_offset_u.append(offset_u)
        all_offset_v.append(offset_v)

        if len(ret_list) < max_offsets-1:
            # Eliminating this peak from future consideration if we're going
            # to be looping again.
            min_u = np.clip(offset_u-peak_margin+slice.shape[1]//2,
                            0,slice.shape[1]-1)
            max_u = np.clip(offset_u+peak_margin+slice.shape[1]//2,
                            0,slice.shape[1]-1)
            min_v = np.clip(offset_v-peak_margin+slice.shape[0]//2,
                            0,slice.shape[0]-1)
            max_v = np.clip(offset_v+peak_margin+slice.shape[0]//2,
                            0,slice.shape[0]-1)
            slice[min_v:max_v+1,min_u:max_u+1] = np.min(slice)           

        if (abs(offset_u) == search_size_max_u or
            abs(offset_v) == search_size_max_v):
            logger.debug('Peak # %d - Offset at edge of search area - BAD',
                         len(ret_list)+1)
            # Go ahead and store a None result. This way we will eventually
            # hit mix_offsets and exit. Otherwise we could be looking for
            # a very long time.
            ret_list.append(None)
            continue

        for i in xrange(len(all_offset_u)):
            if (((offset_u == all_offset_u[i]-peak_margin-1 or
                  offset_u == all_offset_u[i]+peak_margin+1) and
                 offset_v-peak_margin <= all_offset_v[i] <=
                    offset_v+peak_margin) or
                ((offset_v == all_offset_v[i]-peak_margin-1 or
                  offset_v == all_offset_v[i]+peak_margin+1) and
                 offset_u-peak_margin <= all_offset_u[i] <=
                    offset_u+peak_margin)):
                logger.debug(
         'Peak # %d - Offset at edge of previous blackout area - BAD',
                    len(ret_list)+1)
                ret_list.append(None)
                break
        else:
            ret_list.append((offset_u, offset_v, peak_val))
    
    # Now remove all the Nones from the returned list.
    while None in ret_list:
        ret_list.remove(None)
        
    return ret_list 

def find_correlation_and_offset(image, model, search_size_min=0,
                                search_size_max=30,
                                max_offsets=1, peak_margin=3,
                                extend_fov=(0,0),
                                filter=None):
    """Find the offset that best aligns an image and a model.

    Inputs:
        image              The image.
        model              The model to correlate against image.
        search_size_min    The number of pixels from an offset of zero to
        search_size_max    search. If either is a single number, the same
                           search size is used in each dimension. Otherwise it
                           is (search_size_u, search_size_v). The returned
                           offsets will always be within the range [min,max].
        max_offsets        The maximum number of offsets to return.
        peak_margin        The number of correlation pixels around a peak to
                           remove from future consideration before finding
                           the next peak.
        extend_fov         The amount the image and model have been extended
                           on each side (U,V). This is used to search
                           variations where the model 'margins' are shifted
                           onto the image from each side one at a time.
        filter             A filter to apply to the image and each sub-model
                           chosen before running correlation.
        
    Returns:
        List of
            offset_u, offset_v, peak_value
        
        offset_u           The offset in the U direction.
        offset_v           The offset in the V direction.
        peak_value         The correlation value at the peak.    
    """
    logger = logging.getLogger(_LOGGING_NAME+'.find_correlation_and_offset')

    image = image.astype('float')
    model = model.astype('float')

    if np.shape(search_size_min) == ():
        search_size_min_u = search_size_min
        search_size_min_v = search_size_min
    else:
        search_size_min_u, search_size_min_v = search_size_min

    if np.shape(search_size_max) == ():
        search_size_max_u = search_size_max
        search_size_max_v = search_size_max
    else:
        search_size_max_u, search_size_max_v = search_size_max
    
    extend_fov_u, extend_fov_v = extend_fov
    orig_image_size_u = image.shape[1]-extend_fov_u*2
    orig_image_size_v = image.shape[0]-extend_fov_v*2
    
    ret_list = []
    
    # If the image has been extended, try up to nine combinations of
    # sub-models if the model shifted onto the image from each direction.
    # The current implementation falls apart if the extend amount is not
    # the same as the maximum search limit.
    extend_fov_u_list = [0]
    if extend_fov_u:
        extend_fov_u_list = [0, extend_fov_u, 2*extend_fov_u]
        assert search_size_max_u == extend_fov_u
    extend_fov_v_list = [0]
    if extend_fov_v:
        extend_fov_v_list = [0, extend_fov_v, 2*extend_fov_v]
        assert search_size_max_v == extend_fov_v

    # Get the original image and maybe filter it.
    sub_image = image[extend_fov_v:extend_fov_v+orig_image_size_v,
                      extend_fov_u:extend_fov_u+orig_image_size_u]
    if filter is not None:
        sub_image = filter(sub_image)

    new_ret_list = []
        
    # Iterate over each chosen sub-model and correlate it with the image.
    for start_u in extend_fov_u_list:
        for start_v in extend_fov_v_list:
            logger.debug('Model slice U %d:%d V %d:%d', 
                         start_u-extend_fov_u, 
                         start_u-extend_fov_u+orig_image_size_u-1,
                         start_v-extend_fov_v,
                         start_v-extend_fov_v+orig_image_size_v-1)
            sub_model = model[start_v:start_v+orig_image_size_v,
                              start_u:start_u+orig_image_size_u]
            if np.any(sub_model):
                if filter is not None:
                    sub_model = filter(sub_model)
                corr = _correlate2d(sub_image, sub_model,
                                    normalize=True, retile=True)
                ret_list = _find_correlated_offset(corr, search_size_min,
                                                   (search_size_max_u,
                                                    search_size_max_v), 
                                                   max_offsets, peak_margin)

                # Iterate over each returned offset and calculate what the
                # offset actually is based on the model shift amount.
                # Throw away any results that are outside of the given search
                # limits.
                for offset_u, offset_v, peak in ret_list:
                    if offset_u is not None:
                        new_offset_u = offset_u-start_u+extend_fov_u
                        new_offset_v = offset_v-start_v+extend_fov_v
                        if (abs(new_offset_u) <= search_size_max_u and
                            abs(new_offset_v) <= search_size_max_v):
                            logger.debug('Adding possible offset U,V %d,%d',
                                         new_offset_u, new_offset_v)
                            new_ret_list.append((new_offset_u, new_offset_v,
                                                 peak))
                        else:
                            logger.debug(
                                     'Offset beyond search limits U,V %d,%d',
                                     new_offset_u, new_offset_v)

    # Sort the offsets in descending order by correlation peak value.
    # Truncate the (possibly longer) list to the maximum number of requested
    # offsets.
    new_ret_list.sort(key=lambda x: -x[2])
    new_ret_list = new_ret_list[:max_offsets]
    
    if len(new_ret_list) == 0:
        logger.debug('No offsets to return')
    else:
        for i, (offset_u, offset_v, peak) in enumerate(new_ret_list):
            logger.debug('Returning Peak %d offset U,V %d,%d CORR %f',
                         i+1, offset_u, offset_v, peak)
    
        
    return new_ret_list
