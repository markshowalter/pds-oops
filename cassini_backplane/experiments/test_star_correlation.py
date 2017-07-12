import numpy as np
import scipy.fftpack as fftpack
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt
#import imgdisp
import Tkinter as tk

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

def _correlate2d(image, model, normalize=False, retile=True):
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

    newimage = image
    newmodel = model
    
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

    return corr

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

root = tk.Tk()
root.withdraw()

results = np.load('stars.npz')

data = results['data']
model = results['model']

filtered_data = data.copy()

#filtered_data = filter_local_maximum(
#                         filtered_data, maximum_boxsize=9, median_boxsize=0,
#                         maximum_blur=9, maximum_tolerance=0.,
#                         minimum_boxsize=9, gaussian_blur=0.)

plt.imshow(filtered_data)
plt.show()

corr = _correlate2d(filtered_data, model)
peak = np.where(corr == corr.max())

real_x = -22
real_y = 4

print 'Desired answer:', real_x, real_y
print 'Actual answer:', peak[1][0]-corr.shape[1]//2, peak[0][0]-corr.shape[0]//2

sub_corr = corr[512+real_y-10:512+real_y+11,512+real_x-10:512+real_x+11]
plt.imshow(sub_corr)
plt.figure()
plt.imshow(corr)
plt.show()
