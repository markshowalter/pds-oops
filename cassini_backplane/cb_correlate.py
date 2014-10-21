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
DEBUG_CORRELATE_IMGDISP = False

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
    
def _correlate2d(image, model, normalize=False, retile=False):
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

    if DEBUG_CORRELATE_IMGDISP:
        toplevel = Tkinter.Tk()
        frame_toplevel = Tkinter.Frame(toplevel)
        imdisp = ImageDisp([image,model], parent=frame_toplevel, canvas_size=(512,512),
                           allow_enlarge=True, enlarge_limit=10, auto_update=True)
        frame_toplevel.pack()
        Tkinter.mainloop()
    
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
    
def _find_correlated_offset(corr, search_size_min=0, search_size_max=30,
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
    logger = logging.getLogger(LOGGING_NAME+'._find_correlated_offset')

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

    logger.debug('Search U %d to %d V %d to %d #PEAKS %d PEAKMARGIN %d',
                 search_size_min_u, search_size_max_u,
                 search_size_min_v, search_size_max_v,
                 num_peaks, peak_margin)
        
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

    ret_list = []
    all_offset_u = []
    all_offset_v = []
    
    while len(ret_list) != num_peaks:
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
        
        if DEBUG_CORRELATE_IMGDISP:
            print slice.shape
            toplevel = Tkinter.Tk()
            frame_toplevel = Tkinter.Frame(toplevel)
            imdisp = ImageDisp([slice], parent=frame_toplevel, canvas_size=(512,512),
                               allow_enlarge=True, enlarge_limit=10, auto_update=True)
            frame_toplevel.pack()
            Tkinter.mainloop()
            
        if len(peak[0]) != 1:
            logger.debug('Peak # %d - NO PEAK FOUND', len(ret_list)+1) 
#            ret_list.append((None,None,None))
            # This will just loop with Nones being added until we have the 
            # correct return length
            break

        peak_v = peak[0][0]
        peak_u = peak[1][0]
        offset_v = peak_v-search_size_max_v
        offset_u = peak_u-search_size_max_u
        peak_val = slice[peak_v,peak_u]
        
        logger.debug('Peak # %d - Trial offset U,V %d,%d CORR %f',
                     len(ret_list)+1, 
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
            ret_list.append((None, None, None))
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

        ret_list.append((offset_u, offset_v, peak_val))
            
    return ret_list 

def find_correlation_and_offset(image, model, search_size_min=0,
                                search_size_max=30,
                                num_peaks=1, peak_margin=3,
                                extend_fov=(0,0),
                                filter=None):
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
    logger = logging.getLogger(LOGGING_NAME+'.find_correlation_and_offset')

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
    image_size_u = image.shape[1]-extend_fov_u*2
    image_size_v = image.shape[0]-extend_fov_v*2
    ret_list = []
    
    extend_fov_u_list = [0]
    if extend_fov_u:
        extend_fov_u_list = [0, extend_fov_u, 2*extend_fov_u]
        assert search_size_max_u == extend_fov_u
    extend_fov_v_list = [0]
    if extend_fov_v:
        extend_fov_v_list = [0, extend_fov_v, 2*extend_fov_v]
        assert search_size_max_v == extend_fov_v

    # Get the original image
    sub_image = image[extend_fov_v:extend_fov_v+image_size_v,
                      extend_fov_u:extend_fov_u+image_size_u]
    if filter is not None:
        sub_image = filter(sub_image)

    new_ret_list = []
         
    for start_u in extend_fov_u_list:
        for start_v in extend_fov_v_list:
            logger.debug('Model slice U %d:%d V %d:%d', 
                         start_u-extend_fov_u, 
                         start_u-extend_fov_u+image_size_u-1,
                         start_v-extend_fov_v,
                         start_v-extend_fov_v+image_size_v-1)
            sub_model = model[start_v:start_v+image_size_v,
                              start_u:start_u+image_size_u]
            if np.any(sub_model):
                if filter is not None:
                    sub_model = filter(sub_model)
                corr = _correlate2d(sub_image, sub_model,
                                    normalize=True, retile=True)
                ret_list = _find_correlated_offset(corr, search_size_min,
                                                   (search_size_max_u,
                                                    search_size_max_v), 
                                                   num_peaks, peak_margin)

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
                            logger.debug('Offset beyond search limits U,V %d,%d',
                                         new_offset_u, new_offset_v)

    new_ret_list.sort(key=lambda x: -x[2])
    
    for i, (offset_u, offset_v, peak) in enumerate(new_ret_list):
        logger.debug('Returning Peak %d offset U,V %d,%d CORR %f',
                     i+1, offset_u, offset_v, peak)
    
    if len(new_ret_list) == 0:
        logger.debug('No offset to return')
        
    return new_ret_list
    


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
