import cb_logging
import logging

import numpy as np
import scipy.fftpack as fftpack

import matplotlib.pyplot as plt

LOGGING_NAME = 'cb.' + __name__

def correlate2d(image, model, normalize=False, retile=False):
    """Correlate the image with the model; normalization to [-1,1] is optional.
    
    If retile is True, the resulting correlation matrix is shifted by 1/2 along
    each dimension so that (0,0) is now in the center.
    """
    image_fft = fftpack.fft2(image)
    model_fft = fftpack.fft2(model)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))

    if normalize:
        corr /= np.sqrt(np.sum(image**2) * np.sum(model**2))
    
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

def find_correlated_offset(image, model, search_size=30):
    """Find the offset that best aligns an image and a model.
    
    search_size is the number of pixels from an offset of zero to search.
    If search_size is a single number, the same search size is used in
    each dimension. Otherwise it is (search_size_u, search_size_v).
    
    Returns offset_u, offset_v, peak value
    """
    logger = logging.getLogger(LOGGING_NAME+'.star_list_for_obs')

    image = image.astype('float')
    corr = correlate2d(image, model, retile=True)

    if np.shape(search_size) == ():
        search_size_u = search_size
        search_size_v = search_size
    else:
        search_size_u, search_size_v = search_size
        
    slice = corr[corr.shape[0]//2-search_size_v:
                 corr.shape[1]//2+search_size_v+1,
                 corr.shape[0]//2-search_size_u:
                 corr.shape[1]//2+search_size_u+1]
    peak = np.where(slice == slice.max())
    offset_v = peak[0]-search_size_v
    offset_u = peak[1]-search_size_u
    
    logger.debug('Offset U,V %d,%d PEAK %f', offset_u, offset_v, slice[peak])
    print 'OFFSET', offset_u, offset_v

    #corrpsf = GaussianPSF(sigma=1.)
    #
    #print peak
    #ret = corrpsf.find_position(slice, (11,11), peak, search_limit=(4, 4),
    #                  bkgnd_degree=None, tolerance=1e-20)
    #if ret is not None:
    #    print ret[0:2]

#    plt.jet()
#    plt.imshow(slice)
#    plt.contour(slice)
#    plt.plot((search_size_u,search_size_u),(0,2*search_size_v),'k')
#    plt.plot((0,2*search_size_u),(search_size_v,search_size_v),'k')
#    plt.plot(peak[1], peak[0], 'ko')
#    plt.xticks(range(0,2*search_size_u+1,10), 
#               [str(x) for x in range(-search_size_v,search_size_v+1,10)])
#    plt.yticks(range(0,2*search_size_v+1,10), 
#               [str(y) for y in range(search_size_u,-search_size_u-1,-10)])
#    plt.show()

    return offset_u, offset_v, slice[peak]
