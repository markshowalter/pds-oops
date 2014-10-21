import cb_logging
import logging

import numpy as np
import numpy.ma as ma
import scipy.interpolate as interp
import scipy.ndimage.interpolation as ndinterp
import scipy.fftpack as fftpack
import scipy.ndimage.filters as filt
import os
import matplotlib.pyplot as plt
import oops
import oops.inst.cassini.iss as iss
import imgdisp
import Tkinter as tk
import cProfile, pstats, StringIO
from cb_correlate import *
from cb_config import MAX_POINTING_ERROR
from cb_moons import moons_create_model
from cb_util_image import *
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

LONGITUDE_RESOLUTION = 0.1
LATITUDE_RESOLUTION = 0.1

MIN_LAMBERT = 0.2
MAX_EMISSION = 60.

COISS_ROOT = 't:/external/cassini/derived/COISS_2xxx'
MIMAS_FILES = [
#        'COISS_2060/data/1644743986_1644781734/N1644780986_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781164_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781312_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781481_6.IMG',
#        'COISS_2060/data/1644781751_1644850420/N1644782658_1.IMG',
#        'COISS_2060/data/1644781751_1644850420/N1644783429_1.IMG',
#        'COISS_2060/data/1644781751_1644850420/N1644784329_1.IMG'       
               
'COISS_2008/data/1484506648_1484573247/N1484530421_1.IMG',
#'COISS_2008/data/1484506648_1484573247/N1484535522_1.IMG',
#'COISS_2011/data/1492102078_1492217636/N1492217357_1.IMG',
#'COISS_2011/data/1492217706_1492344437/N1492221997_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501630117_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501637285_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501640595_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501640835_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501646143_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501647096_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501647313_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501648088_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501649383_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501649653_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501649933_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501650204_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501650479_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501650761_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501651023_1.IMG',
#'COISS_2014/data/1501647166_1501724619/N1501651303_1.IMG',
#'COISS_2021/data/1521584844_1521609901/N1521594421_1.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558927289_3.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558929550_3.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558938273_3.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558949125_1.IMG',
#'COISS_2033/data/1561668355_1561837358/N1561690178_2.IMG',
#'COISS_2046/data/1593403767_1593531153/N1593516848_1.IMG',
#'COISS_2060/data/1644781751_1644850420/N1644787173_1.IMG',

#'COISS_2011/data/1492102078_1492217636/N1492217357_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501630084_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501630117_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501630150_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501637229_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501637285_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501637345_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501640715_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501645855_1.IMG',
#'COISS_2014/data/1501618408_1501647096/N1501646674_1.IMG',
#'COISS_2027/data/1542749662_1542807100/N1542756630_1.IMG',
#'COISS_2027/data/1542749662_1542807100/N1542758143_1.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558927289_3.IMG',
#'COISS_2032/data/1558910970_1558956168/N1558927322_3.IMG',
#    'COISS_2014/data/1501618408_1501647096/N1501646674_1',
#    'COISS_2014/data/1501618408_1501647096/N1501647096_1',
#    'COISS_2027/data/1542749662_1542807100/N1542756630_1']
]

def test_corr():
    image = np.zeros((1024,1024))
    image[:256,:256] = 1
    model = np.zeros((1024,1024))
    model[:250,:250] = 1
    corr = correlate2d(image, model, normalize=True, retile=True)
    offset_u, offset_v, peak = find_correlated_offset(corr)
    print offset_u, offset_v
    assert False

def _model_filter(image):
    return filter_sub_median(image, median_boxsize=0, gaussian_blur=1.2)

def moons_latitude_longitude_to_pixels(obs, body_name, latitude, longitude):
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    moon_surface = oops.Body.lookup(body_name).surface
    obs_event = oops.Event(obs.midtime, (Vector3.ZERO,Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = moon_surface.photon_to_event_by_coords(obs_event,
                                      (longitude*oops.RPD, latitude*oops.RPD))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals
    
def process_moon_one_file(filename, body_name):
    logger = logging.getLogger(LOGGING_NAME+'process_moon_one_file')

    print 'Processing', filename
    
    latitude_resolution = LATITUDE_RESOLUTION
    longitude_resolution = LONGITUDE_RESOLUTION
    zoom = 2
    
    obs = iss.from_file(filename, fast_distortion=True)

    extend_fov = MAX_POINTING_ERROR[obs.detector]
    search_size_max_u, search_size_max_v = extend_fov

    # Check to see if the moon takes up the whole image - can't handle
    # that case here
    set_obs_ext_corner_bp(obs, extend_fov)
    
    body_mask = obs.ext_corner_bp.where_intercepted(body_name).vals
    if np.all(body_mask):
        print 'Moon takes up whole image - no limb'
        return

    # Extend the data, create a model of the moon, and find the offset
    
    set_obs_ext_bp(obs, extend_fov)
    set_obs_ext_data(obs, extend_fov)

    model = moons_create_model(obs, body_name, lambert=True,
                               extend_fov=extend_fov,
                               use_cartographic=False)

    model_offset_list = find_correlation_and_offset(
                               obs.ext_data,
                               model, search_size_min=0,
                               search_size_max=(search_size_max_u, 
                                                search_size_max_v),
                               extend_fov=extend_fov,
#                               filter=_model_filter
                               )

    if len(model_offset_list) > 0:
        (model_offset_u, model_offset_v,
         peak) = model_offset_list[0]
    else:
        print 'Finding offset failed'
        return
    
    model = shift_image(model, -model_offset_u, -model_offset_v)

    print 'OFFSET U', model_offset_u, 'V', model_offset_v

    im = imgdisp.ImageDisp([obs.ext_data], [model], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()
    
    # Given the offset, create a new offset FOV and compute the image
    # metadata
    obs.fov = oops.fov.OffsetFOV(obs.fov, (model_offset_u, model_offset_v))
    set_obs_bp(obs, force=True)

    body_mask = obs.bp.where_intercepted(body_name).vals
    body_mask = mask_to_array(body_mask, obs.bp.shape)
    body_mask_inv = np.logical_not(body_mask) # Where not intercepted - no data

    lambert = obs.bp.lambert_law(body_name).vals.astype('float')
    
    emission = obs.bp.emission_angle(body_name).vals.astype('float')
    
    # Resolution takes into account the emission angle - the "along sight"
    # projection
    center_resolution = obs.bp.center_resolution(body_name).vals.astype('float')
    resolution = center_resolution / np.cos(emission)

    bp_latitude = obs.bp.latitude(body_name, lat_type='centric')
    bp_latitude = bp_latitude.vals.astype('float') * oops.DPR

    bp_longitude = obs.bp.longitude(body_name, direction='east')
    bp_longitude = bp_longitude.vals.astype('float') * oops.DPR

    #XXX - Test invertibility
#    for u in range(474,646,20):
#        for v in range(446,600,20):
#            lat = bp_latitude[v,u]
#            lon = bp_longitude[v,u]
#            u_pixels, v_pixels = moons_latitude_longitude_to_pixels(obs, body_name, [lat], [lon])
#            print u, u_pixels,
#            print v, v_pixels

    emission_deg = emission * oops.DPR

#    im = imgdisp.ImageDisp([bp_latitude,bp_longitude], canvas_size=(768,768), allow_enlarge=True)
#    tk.mainloop()
    
    # A pixel is OK if it falls on the body, the lambert model is
    # bright enough and the emission angle is large enough
    ok_body_mask_inv = np.logical_or(body_mask_inv, lambert < MIN_LAMBERT)
    ok_body_mask_inv = np.logical_or(ok_body_mask_inv, emission_deg > MAX_EMISSION)
    ok_body_mask = np.logical_not(ok_body_mask_inv)
    
#    im = imgdisp.ImageDisp([obs.data], [lambert], canvas_size=(1024,768), allow_enlarge=True)
#    tk.mainloop()

    # Divide the data by the lambert model in an attempt to account for
    # projected illumination
    lambert[ok_body_mask_inv] = 1e300
    adj_data = obs.data / lambert
    adj_data[ok_body_mask_inv] = 0.
    lambert[ok_body_mask_inv] = 0.

#    im = imgdisp.ImageDisp([adj_data], canvas_size=(1024,768), allow_enlarge=True)
#    tk.mainloop()

    emission[ok_body_mask_inv] = 1e300
    resolution[ok_body_mask_inv] = 1e300

    # The number of pixels in the final reprojection 
    latitude_pixels = int(180. / latitude_resolution)
    longitude_pixels = int(360. / longitude_resolution)

    # Restrict the latitude and longitude ranges for some attempt at efficiency
    min_latitude_pixel = (np.floor(np.min(bp_latitude[ok_body_mask]+90) / 
                                    latitude_resolution)).astype('int')
    min_latitude_pixel = np.clip(min_latitude_pixel, 0, latitude_pixels-1)
    max_latitude_pixel = (np.ceil(np.max(bp_latitude[ok_body_mask]+90) / 
                                   latitude_resolution)).astype('int')
    max_latitude_pixel = np.clip(max_latitude_pixel, 0, latitude_pixels-1)
    num_latitude_pixel = max_latitude_pixel - min_latitude_pixel + 1

    min_longitude_pixel = (np.floor(np.min(bp_longitude[ok_body_mask]) / 
                                    longitude_resolution)).astype('int')
    min_longitude_pixel = np.clip(min_longitude_pixel, 0, longitude_pixels-1)
    max_longitude_pixel = (np.ceil(np.max(bp_longitude[ok_body_mask]) / 
                                   longitude_resolution)).astype('int')
    max_longitude_pixel = np.clip(max_longitude_pixel, 0, longitude_pixels-1)
    num_longitude_pixel = max_longitude_pixel - min_longitude_pixel + 1
    
    # Latitude bin numbers
    lat_bins = np.repeat(np.arange(min_latitude_pixel, max_latitude_pixel+1), 
                         num_longitude_pixel)
    # Actual latitude (deg)
    lat_bins_act = lat_bins * latitude_resolution - 90.

    # Longitude bin numbers
    long_bins = np.tile(np.arange(min_longitude_pixel, max_longitude_pixel+1), 
                        num_latitude_pixel)
    # Actual longitude (deg)
    long_bins_act = long_bins * longitude_resolution

    logger.debug('Latitude range %8.2f %8.2f', np.min(bp_latitude[ok_body_mask]), 
                 np.max(bp_latitude[ok_body_mask]))
    logger.debug('Latitude bin range %8.2f %8.2f', np.min(lat_bins_act), 
                 np.max(lat_bins_act))
    logger.debug('Longitude range %6.2f %6.2f', np.min(bp_longitude[ok_body_mask]), 
                 np.max(bp_longitude[ok_body_mask]))
    logger.debug('Longitude bin range %6.2f %6.2f', np.min(long_bins_act),
                 np.max(long_bins_act))
    logger.debug('Resolution range %7.2f %7.2f', np.min(resolution[ok_body_mask]),
                 np.max(resolution[ok_body_mask]))
    logger.debug('Data range %f %f', np.min(adj_data), np.max(adj_data))

    u_pixels, v_pixels = moons_latitude_longitude_to_pixels(obs, body_name,
                                                            lat_bins_act,
                                                            long_bins_act)
        
    zoom_data = ndinterp.zoom(adj_data, zoom) # XXX Default to order=3 - OK?

    goodumask = np.logical_and(u_pixels >= 0, u_pixels <= obs.data.shape[1]-1)
    goodvmask = np.logical_and(v_pixels >= 0, v_pixels <= obs.data.shape[0]-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    good_u = u_pixels[goodmask]
    good_v = v_pixels[goodmask]
    
    good_lat = lat_bins[goodmask]
    good_long = long_bins[goodmask]
    
    bad_data_mask = (adj_data == 0.)
    # Replicate the mask in each zoom x zoom block
    bad_data_mask_zoom = ndinterp.zoom(bad_data_mask, zoom, order=0)
    zoom_data[bad_data_mask_zoom] = 0.
    
    interp_data = zoom_data[(good_v*zoom).astype('int'), (good_u*zoom).astype('int')]
    
    repro_img = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32)
    repro_img[good_lat,good_long] = interp_data

    good_u = good_u.astype('int')
    good_v = good_v.astype('int')
    
    repro_res = ma.zeros((latitude_pixels, longitude_pixels), dtype=np.float32) + 1e300
    repro_res[good_lat,good_long] = resolution[good_v,good_u]

    im = imgdisp.ImageDisp([repro_img], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

    return repro_img, repro_res

def add_to_mosaic(mosaic, mosaic_resolution, repro_img, repro_resolution):
    better_resolution_mask = (repro_resolution < mosaic_resolution)
    ok_value_mask = np.logical_and(-100 < repro_img, repro_img < 100)
    new_mosaic_mask = np.logical_and(better_resolution_mask, ok_value_mask)
    
    mosaic[new_mosaic_mask] = repro_img[new_mosaic_mask]
    mosaic_resolution[new_mosaic_mask] = repro_resolution[new_mosaic_mask]

    overlay = (repro_img-np.min(repro_img)) / (np.max(repro_img)-np.min(repro_img))
    im = imgdisp.ImageDisp([mosaic], [overlay],
                           canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

def process_moon(filespec_list, body_name):
    nlat = int(np.round(180. / LATITUDE_RESOLUTION))
    nlong = int(np.round(360. / LONGITUDE_RESOLUTION))
    
    mosaic = np.zeros((nlat, nlong))
    mosaic_resolution = np.zeros((nlat, nlong)) + 1e300
    
    for filespec in filespec_list:
        full_filename = os.path.join(COISS_ROOT, filespec)
        full_filename = full_filename[:-4]
        full_filename += '_CALIB.IMG'
        repro_img, repro_res = process_moon_one_file(full_filename, body_name)

        add_to_mosaic(mosaic, mosaic_resolution, repro_img, repro_res)
    
    im = imgdisp.ImageDisp([mosaic], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

process_moon(MIMAS_FILES, 'MIMAS')
