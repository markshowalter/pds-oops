import cb_logging
import logging

import numpy as np
import scipy.fftpack as fftpack

import matplotlib.pyplot as plt

LOGGING_NAME = 'cb.' + __name__

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
    image_fft = fftpack.fft2(image)
    model_fft = fftpack.fft2(model)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))

    if normalize:
        norm_amt = np.sqrt(np.sum(image**2) * np.sum(model**2))
        if norm_amt != 0:
            corr /= norm_amt 
    
    if retile:
        y = corr.shape[0] // 2
        x = corr.shape[1] // 2
        offset_image = np.empty(corr.shape)
        offset_image[ 0:y, 0:x] = corr[-y: ,-x: ]
        offset_image[ 0:y,-x: ] = corr[-y: , 0:x]
        offset_image[-y: , 0:x] = corr[ 0:y,-x: ]
        offset_image[-y: ,-x: ] = corr[ 0:y, 0:x]
        corr = offset_image
    
    return corr

def find_correlated_offset(corr, search_size_min=0, search_size_max=30):
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

    slice = corr[corr.shape[0]//2-search_size_max_v:
                 corr.shape[0]//2+search_size_max_v+1,
                 corr.shape[1]//2-search_size_max_u:
                 corr.shape[1]//2+search_size_max_u+1]

    if search_size_min_u != 0 and search_size_min_v != 0:
        slice[slice.shape[0]//2-search_size_min_v+1:
              slice.shape[0]//2+search_size_min_v,
              slice.shape[1]//2-search_size_min_u+1:
              slice.shape[1]//2+search_size_min_u] = -1.1
    
    peak = np.where(slice == slice.max())
    
    if len(peak[0]) != 1:
        logger.debug('Search U %d to %d V %d to %d NO PEAK FOUND', 
                     search_size_min_u, search_size_max_u,
                     search_size_min_v, search_size_max_v)
        return None, None, None
    
    peak_v = peak[0][0]
    peak_u = peak[1][0]
    offset_v = peak_v-search_size_max_v
    offset_u = peak_u-search_size_max_u
    
    logger.debug('Search U %d to %d V %d to %d Offset U,V %d,%d PEAK %f', 
                 search_size_min_u, search_size_max_u,
                 search_size_min_v, search_size_max_v,
                 offset_u, offset_v,
                 slice[peak_v,peak_u])

    if False:
        plt.jet()
        plt.imshow(slice)
        plt.contour(slice)
        plt.plot((search_size_max_u,search_size_max_u),(0,2*search_size_max_v),'k')
        plt.plot((0,2*search_size_max_u),(search_size_max_v,search_size_max_v),'k')
        plt.plot(peak[1], peak[0], 'ko')
        plt.xticks(range(0,2*search_size_max_u+1,2), 
                   [str(x) for x in range(-search_size_max_u,search_size_max_u+1,2)])
        plt.yticks(range(0,2*search_size_max_v+1,2), 
                   [str(y) for y in range(-search_size_max_v,search_size_max_v+1,2)])
        plt.xlabel('U')
        plt.ylabel('V')
        plt.show()

    return offset_u, offset_v, slice[peak_v,peak_u]

def find_correlation_and_offset(image, model, search_size_min=0,
                                search_size_max=30):
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
    
    return find_correlated_offset(corr, search_size_min, search_size_max)
