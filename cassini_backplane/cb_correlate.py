import cb_logging
import logging

import numpy as np
import scipy.fftpack as fftpack

import matplotlib.pyplot as plt
from imgdisp import ImageDisp
import Tkinter

from cb_util_image import pad_image, unpad_image, filter_local_maximum

LOGGING_NAME = 'cb.' + __name__

DEBUG_CORRELATE_PLOT = False

def _next_power_of_2(n):
    s = bin(n)[2:]
    if s.count('1') == 1: # Already power of 2
        return n
    s = '1' + '0' * len(s) 
    return int(s, 2)
    
def _pad_to_power_of_2(data):
    s1 = _next_power_of_2(data.shape[0])
    s2 = _next_power_of_2(data.shape[1])
    
    if s1 == data.shape[0] and s2 == data.shape[1]:
        return data, (0,0)

    offset1 = (s1-data.shape[0])//2
    offset2 = (s2-data.shape[1])//2
    padding = (offset2,offset1)
    return pad_image(data, padding), padding    
    
def correlate2d(image, model, normalize=False, retile=False):
    """Correlate the image with the model; normalization to [-1,1] is optional.

    Inputs:
        image              The image.
        model              The model to correlation against image.
                           If model is bigger than image, pad image with zeroes
                           and center it.
        normalize          If True, normalize the correlation to 1.
        retile             If True, the resulting correlation matrix is
                           shifted by 1/2 along each dimension so that
                           (0,0) is now in the center.
                       
    Returns:
        The 2-D correlation matrix.
    """
    
    assert image.shape == model.shape
    
    # Padding to a power of 2 makes FFT MUCH MUCH faster
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

#def _fill_correlation(corr, x, y, threshold, min_val):
#    visited = np.zeros(corr.shape, dtype='bool')
#    marked = np.zeros(corr.shape, dtype='bool')
#    
#    x_list = [x]
#    y_list = [y]
#    threshold_reached_list = [False]
#    prev_val_list = [1.]
#    dir_list = [None]
#    
#    while len(x_list):
#        x = x_list.pop(0)
#        y = y_list.pop(0)
#        threshold_reached = threshold_reached_list.pop(0)
#        prev_val = prev_val_list.pop(0)
#        dir = dir_list.pop(0)
#        if visited[y,x]:
#            continue
#        visited[y,x] = True
#        val = corr[y,x]
#        if val < threshold:
#            threshold_reached = True
#        print '%4d %4d %s VAL %8.6f PREV %8.6f TH %8.6f %d' % (x, y, dir, val, prev_val, threshold, threshold_reached),
#        if threshold_reached and dir is not None:
#            count = 0
#            if dir[0] == 'N' and y < corr.shape[0]-1:
#                if corr[y+1,x] < val:
#                    count += 1
#            if dir[0] == 'S' and y > 0:
#                if corr[y-1,x] < val:
#                    count += 1
#            if dir[1] == 'W' and x < corr.shape[0]-1:
#                if corr[y,x+1] < val:
#                    count += 1
#            if dir[1] == 'E' and x > 0:
#                if corr[y,x-1] < val:
#                    count += 1
#            if count > 1:
#                # We're going back up...don't change anything
#                print 'SKIP'
#                continue
#        marked[y,x] = True
#        print 'SET'
#        if ((dir is None or dir[1] == 'W') and
#            x > 0 and not visited[y,x-1]):
#            x_list.append(x-1)
#            y_list.append(y)
#            threshold_reached_list.append(threshold_reached)
#            prev_val_list.append(val)
#            if dir is None:
#                dir_list.append('SW')
#            else:
#                dir_list.append(dir)
#        if ((dir is None or dir[0] == 'N') and
#            y > 0 and not visited[y-1,x]):
#            x_list.append(x)
#            y_list.append(y-1)
#            threshold_reached_list.append(threshold_reached)
#            prev_val_list.append(val)
#            if dir is None:
#                dir_list.append('NW')
#            else:
#                dir_list.append(dir)
#        if ((dir is None or dir[1] == 'E') and
#            x < corr.shape[1]-1 and not visited[y,x+1]):
#            x_list.append(x+1)
#            y_list.append(y)
#            threshold_reached_list.append(threshold_reached)
#            prev_val_list.append(val)
#            if dir is None:
#                dir_list.append('NE')
#            else:
#                dir_list.append(dir)
#        if ((dir is None or dir[0] == 'S') and
#            y < corr.shape[0]-1 and not visited[y+1,x]):
#            x_list.append(x)
#            y_list.append(y+1)
#            threshold_reached_list.append(threshold_reached)
#            prev_val_list.append(val)
#            if dir is None:
#                dir_list.append('SE')
#            else:
#                dir_list.append(dir)
#
#    corr[marked] = min_val
    
def find_correlated_offset(corr, search_size_min=0, search_size_max=30,
                           num_peaks=1, peak_margin=3):
    """Find the offset that best aligns an image and a model given the correlation.

    Inputs:
        corr               A 2-D correlation matrix.
        search_size_min    The number of pixels from an offset of zero to
        search_size_max    search. If either is a single number, the same search
                           size is used in each dimension. Otherwise it is
                           (search_size_u, search_size_v).
    Returns:
        offset_u, offset_v, peak_value
        
        offset_u           The offset in the U direction.
        offset_v           The offset in the V direction.
        peak_value         The correlation value at the peak.
    """
    logger = logging.getLogger(LOGGING_NAME+'.find_correlated_offset')

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
        
    assert 0 <= search_size_min_u <= search_size_max_u
    assert 0 <= search_size_min_v <= search_size_max_v
    assert 0 <= search_size_max_u <= corr.shape[1]//2 
    assert 0 <= search_size_max_v <= corr.shape[0]//2 

#    corr = filter_local_maximum(corr, maximum_boxsize=5,
#                                maximum_blur=peak_margin*2+1)
#                                minimum_boxsize=5, maximum_tolerance=1.2)
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

    ret_offset_u = []
    ret_offset_v = []
    ret_peak = []
    all_offset_u = []
    all_offset_v = []
    
    while len(ret_offset_u) != num_peaks:
        peak = np.where(slice == slice.max())

        if DEBUG_CORRELATE_PLOT: # and search_size_max_u > 22:
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
        
        if False:
            toplevel = Tkinter.Tk()
            frame_toplevel = Tkinter.Frame(toplevel)
            imdisp = ImageDisp([slice], parent=frame_toplevel, canvas_size=(512,512),
                               allow_enlarge=True, enlarge_limit=10, auto_update=True)
            frame_toplevel.pack()
            Tkinter.mainloop()
            
        if len(peak[0]) != 1:
            logger.debug('Peak # %d - Search U %d to %d V %d to %d NO PEAK FOUND',
                         len(ret_offset_u)+1, 
                         search_size_min_u, search_size_max_u,
                         search_size_min_v, search_size_max_v)
            ret_offset_u.append(None)
            ret_offset_v.append(None)
            ret_peak.append(None)
            # This will just loop with Nones being added until we have the 
            # correct return length
            continue

        peak_v = peak[0][0]
        peak_u = peak[1][0]
        offset_v = peak_v-search_size_max_v
        offset_u = peak_u-search_size_max_u
        peak_val = slice[peak_v,peak_u]
        
        logger.debug('Peak # %d - Search U %d to %d V %d to %d Trial offset U,V %d,%d PEAK %f',
                     len(ret_offset_u)+1, 
                     search_size_min_u, search_size_max_u,
                     search_size_min_v, search_size_max_v,
                     offset_u, offset_v,
                     peak_val)

        if offset_u is not None and peak_val > 0:
#            _fill_correlation(slice, 
#                              offset_u+slice.shape[1]//2,
#                              offset_v+slice.shape[0]//2,
#                              peak_val/10,
#                              global_min)


            # We keep going by eliminating this previous peak
            min_u = np.clip(offset_u-peak_margin+slice.shape[1]//2,
                            0,slice.shape[1]-1)
            max_u = np.clip(offset_u+peak_margin+slice.shape[1]//2,
                            0,slice.shape[1]-1)
            min_v = np.clip(offset_v-peak_margin+slice.shape[0]//2,
                            0,slice.shape[0]-1)
            max_v = np.clip(offset_v+peak_margin+slice.shape[0]//2,
                            0,slice.shape[0]-1)
            slice[min_v:max_v+1,min_u:max_u+1] = np.min(slice)                
            all_offset_u.append(offset_u)
            all_offset_v.append(offset_v)

        if (abs(offset_u) == search_size_max_u or
            abs(offset_v) == search_size_max_v):
            logger.debug('Offset at edge of search area - BAD')
            ret_offset_u.append(None)
            ret_offset_v.append(None)
            ret_peak.append(None)
            continue

        found_one = False
        for i in xrange(len(all_offset_u)):
            if (((offset_u == all_offset_u[i]-peak_margin-1 or
                  offset_u == all_offset_u[i]+peak_margin+1) and
                 offset_v-peak_margin <= all_offset_v[i] <= offset_v+peak_margin) or
                ((offset_v == all_offset_v[i]-peak_margin-1 or
                  offset_v == all_offset_v[i]+peak_margin+1) and
                 offset_u-peak_margin <= all_offset_u[i] <= offset_u+peak_margin)):
                logger.debug('Offset at edge of previous blackout area - BAD')
                found_one = True
                break
        if found_one:
            continue

        ret_offset_u.append(offset_u)
        ret_offset_v.append(offset_v)
        ret_peak.append(peak_val)
            
    return ret_offset_u, ret_offset_v, ret_peak 

def find_correlation_and_offset(image, model, search_size_min=0,
                                search_size_max=30,
                                num_peaks=1, peak_margin=3):
    """Find the offset that best aligns an image and a model.

    Inputs:
        image              The image.
        model              The model to correlate against image.
                           If model is bigger than image, pad image with zeroes
                           and center it.
        search_size_min    The number of pixels from an offset of zero to
        search_size_max    search. If either is a single number, the same search
                           size is used in each dimension. Otherwise it is
                           (search_size_u, search_size_v).

    Returns:
        offset_u, offset_v, peak_value
        
        offset_u           The offset in the U direction.
        offset_v           The offset in the V direction.
        peak_value         The correlation value at the peak.    
    """
    image = image.astype('float')
    corr = correlate2d(image, model, normalize=True, retile=True)
    
    return find_correlated_offset(corr, search_size_min, search_size_max, 
                                  num_peaks, peak_margin)
