#XXXXXXXX

assert False

import cb_logging
import logging

import numpy as np
import numpy.ma as ma

from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter

import oops
import oops.inst.cassini.iss as iss
from spice_starcat import SPICE_StarCat as StarCat

from PIL import Image

import scipy.fftpack as fftpack

def correlate2d(image, model, normalize=False, retile=False):
    """Correlate the image with the model; normalization to [-1.1] is optional.
    """
    image_fft = fftpack.fft2(image)
    model_fft = fftpack.fft2(model)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))
    if normalize:
        corr /= np.sqrt(np.sum(image**2) * np.sum(model**2))
    if retile:
        y = corr.shape[0] / 2
        x = corr.shape[1] / 2
        offset = np.empty(corr.shape)
        offset[0:y,0:x] = corr[-y:,-x:]
        offset[0:y,-x:] = corr[-y:,0:x]
        offset[-y:,0:x] = corr[0:y,-x:]
        offset[-y:,-x:] = corr[0:y,0:x]
        corr = offset
    return corr

def save_png(image, outfile, lo=-50, hi=300):
    scaled = (image - lo) / (hi - lo)
    bytes = (256.*scaled).clip(0,255).astype("uint8")
    im = Image.fromstring("L", (bytes.shape[1],
                                bytes.shape[0]), bytes)
    im.save(outfile)

def stack(images):
    #
    # Make sure all images are the same size
    count = len(images)
    shape = images[0].shape
    for image in images[1:]:
        assert(image.shape == shape)
    #
    # Create a large buffer to surround each image by zeros
    bigimage = np.zeros((shape[0]*2-1, shape[1]*2-1))
    #
    # Construct a list of square roots of correlations between pairs (a^2,1)
    bigimage[:shape[0],:shape[1]] = 1.
    conj_fft = np.conj(fftpack.fft2(bigimage))
    norms = []
    conj_norms = []
    for image in images:
        bigimage[:shape[0],:shape[1]] = image.astype("float")**2
        fft = fftpack.fft2(bigimage)
        corr = np.real(fftpack.ifft2(fft * conj_fft))
        norm = np.sqrt(corr.clip(1.e-20,1.e99))
        norms.append(norm)
        conj_norms.append(np.roll(np.roll(norm[::-1,::-1],1,0),1,1))
    #
    # Construct the fft of each big image
    ffts = []
    for image in images:
        bigimage[:shape[0],:shape[1]] = image
        ffts.append(fftpack.fft2(bigimage))
    #
    # Construct an array of normalized correlations between pairs (i,j)
    corr_ij = np.empty((count,count), dtype="object")
    for j in range(count):
      conj_fft = np.conj(ffts[j])
      corr = (np.real(fftpack.ifft2(ffts[j] * conj_fft))
              / (norms[j] * conj_norms[j]))
      corr_ij[j,j] = np.roll(np.roll(corr,shape[0],0),shape[1],1)
      #
      for i in range(j+1,count):
        corr = (np.real(fftpack.ifft2(ffts[i] * conj_fft))
                / (norms[i] * conj_norms[j]))
        corr_ij[i,j] = np.roll(np.roll(corr,shape[0],0),shape[1],1)
        corr_ij[j,i] = corr_ij[i,j][-1::-1,-1::-1]
    #
    return corr_ij

####################################
# Image pair alignment
####################################

images = []
raws = []
lats = []
lons = []
for filespec in ('N1677190715_1.IMG', 'N1677191297_1.IMG'):
    obs = iss.from_file(filespec)
    image = obs.data[2:-2,2:-2]
    raws.append(image)
    image = image - gaussian_filter(image.astype('float'), 21, mode='wrap')
    images.append(image.astype('float'))
    #
#     bp = oops.Backplane(obs)
#     lat = bp.latitude('saturn')
#     lon = bp.longitude('saturn')
#     lats.append(lat)
#     lons.append(lon)

pylab.imshow(raws[0].clip(150,190)))
save_png(raws[0].astype('float'), 'Figure_14_raw_image_1.png', 150, 190)
save_png(raws[1].astype('float'), 'Figure_14_raw_image_2.png', 150, 190)

corr = correlate2d(images[1], images[0], normalize=True, retile=True)
#pylab.imshow(corr)

#stacked = stack(images)
#corr = stacked[1][0]

slice = corr[2:103, 512-50:513+50]
pylab.jet()
pylab.imshow(slice)
pylab.contour(slice)

peak = np.where(slice == slice.max())
pylab.plot(peak[1], peak[0], 'ko')

pylab.xticks(range(10,91,20), [str(x) for x in range(-40,41,20)])
pylab.yticks(range(10,91,20), [str(y) for y in range(500,-419,-20)])

pylab.savefig('Figure_15.png', dpi=300)


peak = np.where(corr[:200] == corr[:200].max())

(ds,dl) = (peak[1][0] - 512, 512 - peak[0][0])
print (ds, dl)
# (-2, 452)

aligned1 = np.zeros((1020 + 452, 1020 + 2))
aligned1[:-452,2:] = raws[0]
pylab.imshow(aligned1.clip(150,190))
save_png(aligned1.astype('float'), 'Figure_14r.png', 150, 190)

aligned2 = np.zeros((1020 + 452, 1020 + 2))
aligned2[452:,:-2] = raws[1]
pylab.imshow(aligned2.clip(150,190))
save_png(aligned2.astype('float'), 'Figure_14gb.png', 150, 190)

