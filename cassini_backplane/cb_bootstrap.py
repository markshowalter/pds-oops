###############################################################################
# cb_bootstrap.py
#
# Routines related to bootstrapping.
#
# Exported routines:
#    bootstrap_viable
###############################################################################

import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp

from imgdisp import *
import Tkinter as tk

from cb_bodies import *
from cb_gui_body_mosaic import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__


_BOOTSTRAP_ANGLE_TOLERANCE = 0.5 * oops.RPD

_BOOTSTRAP_KNOWNS = {}
_BOOTSTRAP_CANDIDATES = {}
_BOOTSTRAP_MOSAICS = {}
    

def bootstrap_viable(ref_path, ref_metadata, cand_path, cand_metadata):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_viable')

    if (ref_metadata['filter1'] != cand_metadata['filter1'] or
        ref_metadata['filter2'] != cand_metadata['filter2']):
        logger.debug('Incompatible - different filters')
        return False
    
    if ref_metadata['bootstrap_body'] != cand_metadata['bootstrap_body']:
        logger.debug('Incompatible - different bodies')
        return False
    
    return True

def _bootstrap_mask_overlap(mask1, mask2):
    # Scale the masks along each dimension to be the size of the maximum
    scale1 = float(mask1.shape[0]) / mask2.shape[0]
    scale2 = float(mask1.shape[1]) / mask2.shape[1]
    
    if scale1 < 1. and scale2 < 1.:
        mask1 = ndinterp.zoom(mask1, (1./scale1,1./scale2), order=0)
    elif scale1 > 1. and scale2 > 1.:
        mask2 = ndinterp.zoom(mask2, (scale1,scale2), order=0)
    else:
        if scale1 < 1.:
            mask1 = ndinterp.zoom(mask1, (1./scale1,1), order=0)
        elif scale1 > 1.:
            mask2 = ndinterp.zoom(mask2, (scale1,1), order=0)
        
        if scale2 < 1.:
            mask2 = ndinterp.zoom(mask2, (1,1./scale2), order=0)
        elif scale2 > 1.:
            mask1 = ndinterp.zoom(mask1, (1,scale2), order=0)

    # Deal with roundoff error
    if mask1.shape != mask2.shape:
        if mask1.shape[0] < mask2.shape[0]:
            mask2 = mask2[:mask1.shape[0],:]
        elif mask1.shape[0] > mask2.shape[0]:
            mask1 = mask1[:mask2.shape[0],:]
        if mask1.shape[1] < mask2.shape[1]:
            mask2 = mask2[:,mask1.shape[1]]
        elif mask1.shape[1] > mask2.shape[1]:
            mask1 = mask1[:,mask2.shape[1]]
            
    return np.logical_and(mask1, mask2)
        
def _bootstrap_find_offset(cand_path, cand_metadata, bootstrap_config):
    logger = logging.getLogger(_LOGGING_NAME+'._bootstrap_find_offset')

    _, cand_filename = os.path.split(cand_path)
    logger.info('Bootstrapping candidate %s', cand_filename)

    body_name = cand_metadata['bootstrap_body']

    mosaic_metadata = _BOOTSTRAP_MOSAICS[body_name]
    
    cand_body_metadata = cand_metadata['bodies_metadata'][body_name]
    
    overlap = _bootstrap_mask_overlap(mosaic_metadata['full_mask'],
                                      cand_body_metadata['latlon_mask'])

    if not np.any(overlap):
        logger.debug('No overlap with current mosaic - aborting')
        return None
    
    cand_obs = read_iss_file(cand_path)

    cart_dict = {body_name: mosaic_metadata}
    
    new_metadata = master_find_offset(cand_obs, create_overlay=True,
                                      bodies_cartographic_data=cart_dict,
                                      allow_stars=True) # XXX

    if new_metadata['offset'] is not None:
        logger.debug('Bootstrapping successful - updating mosaic')
        repro_metadata = bodies_reproject(
              cand_obs, body_name,
              offset=new_metadata['offset'],
              latitude_resolution=bootstrap_config['lat_resolution'], 
              longitude_resolution=bootstrap_config['lon_resolution'],
              latlon_type=bootstrap_config['latlon_type'],
              lon_direction=bootstrap_config['lon_direction'])
        
        bodies_mosaic_add(mosaic_metadata, repro_metadata)

        print 'Updated mosaic'
        display_body_mosaic(mosaic_metadata)
        
#    display_offset_data(ref_obs, ref_metadata, show_rings=False, show_bodies=False)
    display_offset_data(cand_obs, new_metadata, show_rings=False, show_bodies=False)

    return new_metadata

def _bootstrap_make_initial_mosaic(body_name, bootstrap_config):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_make_initial_mosaic')

    mosaic_metadata = bodies_mosaic_init(
        body_name,
        latitude_resolution=bootstrap_config['lat_resolution'], 
        longitude_resolution=bootstrap_config['lon_resolution'],
        latlon_type=bootstrap_config['latlon_type'],
        lon_direction=bootstrap_config['lon_direction'])

    for ref_path, ref_metadata in _BOOTSTRAP_KNOWNS[body_name]:
        if ref_metadata['bootstrap_body'] != body_name:
            continue
        ref_obs = read_iss_file(ref_path)
        _, ref_filename = os.path.split(ref_path)
        logger.debug('Adding reference to mosaic %s', ref_filename)

#        display_offset_data(ref_obs, ref_metadata, show_rings=False, show_bodies=False)

        repro_metadata = bodies_reproject(
              ref_obs, body_name,
              offset=ref_metadata['offset'],
              latitude_resolution=bootstrap_config['lat_resolution'], 
              longitude_resolution=bootstrap_config['lon_resolution'],
              latlon_type=bootstrap_config['latlon_type'],
              lon_direction=bootstrap_config['lon_direction'])
        
        bodies_mosaic_add(mosaic_metadata, repro_metadata)
        
    _BOOTSTRAP_MOSAICS[body_name] = mosaic_metadata

    print 'Initial mosaic'    
    display_body_mosaic(mosaic_metadata)
    
def _bootstrap_update_lists(body_name, cand_idx, new_metadata):
    candidates = _BOOTSTRAP_CANDIDATES[body_name]
    cand_path, cand_metadata = candidates[cand_idx]

    del candidates[cand_idx]
    
    _BOOTSTRAP_KNOWNS[body_name].append((cand_path, new_metadata))
        
    _BOOTSTRAP_KNOWNS[body_name].sort(key=lambda x: 
                                            abs(x[1]['midtime']))
    
    
def bootstrap_add_file(image_path, metadata, bootstrap_config=None):
    logger = logging.getLogger(_LOGGING_NAME+'.bootstrap_add_file')

    if bootstrap_config is None:
        bootstrap_config = BOOTSTRAP_DEFAULT_CONFIG
    if metadata is not None:
        _, image_filename = os.path.split(image_path)
        body_name = metadata['bootstrap_body']
        if metadata['offset'] is not None:
            if body_name not in _BOOTSTRAP_KNOWNS:
                _BOOTSTRAP_KNOWNS[body_name] = []
            _BOOTSTRAP_KNOWNS[body_name].append((image_path,metadata))
            _BOOTSTRAP_KNOWNS[body_name].sort(key=lambda x: 
                                                    abs(x[1]['midtime']))
            logger.debug('Known offset %s', image_filename)
        elif metadata['bootstrap_candidate']:
            if body_name not in _BOOTSTRAP_CANDIDATES:
                _BOOTSTRAP_CANDIDATES[body_name] = []
            _BOOTSTRAP_CANDIDATES[body_name].append((image_path,metadata))
            _BOOTSTRAP_CANDIDATES[body_name].sort(key=lambda x: 
                                                        abs(x[1]['midtime']))
            logger.debug('Candidate %s', image_filename)
        else:
            logger.debug('No offset and not a candidate %s', image_filename)

    
    for body_name in BOOTSTRAP_BODY_LIST:
        if body_name not in _BOOTSTRAP_CANDIDATES:
            continue
        candidates = _BOOTSTRAP_CANDIDATES[body_name]
        if metadata is None or False: #_bootstrap_time_expired(body_name):
            _bootstrap_make_initial_mosaic(body_name, bootstrap_config)
            go_again = True
            while go_again:
                go_again = False
                for cand_idx in xrange(len(candidates)):
                    cand_path, cand_metadata = candidates[cand_idx]
                    if cand_metadata['bootstrap_body'] != body_name:
                        continue
                    offset_metadata = _bootstrap_find_offset(
                               cand_path, cand_metadata, bootstrap_config)
                    if (offset_metadata is not None and
                        offset_metadata['offset'] is not None):
                        _bootstrap_update_lists(
                            body_name, cand_idx, offset_metadata)
                        go_again = True
                        break
    
                    

'''
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

http://content.gpwiki.org/index.php/Polygon_Collision
'''


"""
    if ref_metadata['camera'] != cand_metadata['camera']:
        logger.debug('Incompatible - different cameras')
        return False
    if ref_metadata['image_shape'] != cand_metadata['image_shape']:
        logger.debug('Incompatible - different image shapes')
        return False
    if (ref_metadata['filter1'] != cand_metadata['filter1'] or
        ref_metadata['filter2'] != cand_metadata['filter2']):
        logger.debug('Incompatible - different filters')
        return False
    # midtime
    
    if ref_metadata['bootstrap_body'] != cand_metadata['bootstrap_body']:
        logger.debug('Incompatible - different bodies')
        return False
    
    ref_ra_dec = ref_metadata['ra_dec_corner']
    cand_ra_dec = cand_metadata['ra_dec_corner']
    
    # Find the four corners of each
    ref_v1 = (ref_ra_dec[0], ref_ra_dec[2])
    ref_v2 = (ref_ra_dec[1], ref_ra_dec[2])
    ref_v3 = (ref_ra_dec[1], ref_ra_dec[3])
    ref_v4 = (ref_ra_dec[0], ref_ra_dec[2])
    cand_v1 = (cand_ra_dec[0], cand_ra_dec[2])
    cand_v2 = (cand_ra_dec[1], cand_ra_dec[2])
    cand_v3 = (cand_ra_dec[1], cand_ra_dec[3])
    cand_v4 = (cand_ra_dec[0], cand_ra_dec[2])
    
    # Find the rotations of each
    ref_angle = np.arctan2(ref_v1[1], ref_v1[0])
    cand_angle = np.arctan2(cand_v1[1], cand_v1[0])
    
    delta_angle = (ref_angle - cand_angle) % oops.TWOPI
    if delta_angle > _BOOTSTRAP_ANGLE_TOLERANCE:
        logger.debug('Incompatible - Ref angle %.2f Cand angle %.2f', 
                     ref_angle*oops.DPR, cand_angle*oops.DPR)
        return False
    
    print ref_v1, ref_v2, ref_v3, ref_v4
    print cand_v1, cand_v2, cand_v3, cand_v4
    
"""
