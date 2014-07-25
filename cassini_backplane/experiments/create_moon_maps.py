import numpy as np
import numpy.ma as ma
import scipy.interpolate as interp
import scipy.fftpack as fftpack
import scipy.ndimage.filters as filt
import os
import matplotlib.pyplot as plt
import oops
import oops.inst.cassini.iss as iss
import imgdisp
import Tkinter as tk
import cProfile, pstats, StringIO

LONGITUDE_RES = 0.1
LATITUDE_RES = 0.1

MIN_LAMBERT = 0.2
MAX_EMISSION = 60.

COISS_ROOT = 't:/external/cassini/derived/COISS_2xxx'
MIMAS_FILES = [
#        'COISS_2060/data/1644743986_1644781734/N1644780986_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781164_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781312_1.IMG',
#        'COISS_2060/data/1644743986_1644781734/N1644781481_6.IMG',
#        'COISS_2060/data/1644781751_1644850420/N1644782658_1.IMG',
        'COISS_2060/data/1644781751_1644850420/N1644783429_1.IMG',
#        'COISS_2060/data/1644781751_1644850420/N1644784329_1.IMG'       
               
'COISS_2008/data/1484506648_1484573247/N1484530421_1.IMG',
'COISS_2008/data/1484506648_1484573247/N1484535522_1.IMG',
'COISS_2011/data/1492102078_1492217636/N1492217357_1.IMG',
'COISS_2011/data/1492217706_1492344437/N1492221997_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501630117_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501637285_1.IMG',
'COISS_2014/data/1501618408_1501647096/N1501640595_1.IMG',
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

def correlate2d(image, model, normalize=False, retile=False):
    """Correlate the image with the model; normalization to [-1,1] is optional.

    Inputs:
        image              The image.
        model              The model to correlation against image.
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

def find_correlated_offset(corr, search_size_min=0, search_size_max=40):
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
        return None, None, None
    
    peak_v = peak[0][0]
    peak_u = peak[1][0]
    offset_v = peak_v-search_size_max_v
    offset_u = peak_u-search_size_max_u
    
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

def shift_image(image, offset_u, offset_v):
    """Shift an image by an offset."""
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

def mask_to_array(mask, shape):
    if np.shape(mask) == shape:
        return mask
    
    new_mask = np.empty(shape)
    new_mask[:,:] = mask
    return new_mask

def process_moon_one_file(filename, body_name, mosaic, mosaic_resolution):
    print 'Processing', filename
    
    obs = iss.from_file(filename, fast_distortion=True)
#    meshgrid = oops.Meshgrid.for_fov(obs.fov, origin=0.5-512, undersample=1, oversample=1, limit=(1024+512,1024+512),
#                     swap=True)
#    bp = oops.Backplane(obs, meshgrid=meshgrid)
    pr = cProfile.Profile()
    pr.enable()

    bp = oops.Backplane(obs)
    
    filt_data = obs.data - filt.gaussian_filter(obs.data, 10, mode="reflect")
#    old_filt_data = filt_data
#    filt_data = np.zeros((2048,2048))
#    filt_data[512:1536,512:1536] = old_filt_data

    body_mask = bp.where_intercepted(body_name).vals
    
    body_mask = mask_to_array(body_mask, bp.shape)
    body_mask_inv = np.logical_not(body_mask)

    if (body_mask[ 0, 0] and body_mask[-1, 0] and
        body_mask[ 0,-1] and body_mask[-1,-1]):
        # No limb
        return # XXX
        
    lambert = bp.lambert_law(body_name).vals.astype('float')
    
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    print s.getvalue()
    lambert[body_mask_inv] = 0.

    lambert = lambert - filt.gaussian_filter(lambert, 10, mode="reflect")

    corr = correlate2d(filt_data, lambert, normalize=True, retile=True)
    offset_u, offset_v, peak = find_correlated_offset(corr)
    print 'OFFSET U', offset_u, 'V', offset_v

    im = imgdisp.ImageDisp([filt_data], [lambert], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()
    
#    offset_u = -16
#    offset_v = -20
    
    obs.fov = oops.fov.OffsetFOV(obs.fov, (offset_u, offset_v))
    bp = oops.Backplane(obs)

    body_mask = bp.where_intercepted(body_name).vals
    body_mask = mask_to_array(body_mask, bp.shape)
    body_mask_inv = np.logical_not(body_mask) # Where not intercepted

    if (body_mask[0,0] and body_mask[-1,0] and
        body_mask[0,-1] and body_mask[-1,-1]):
        # No limb
        return # XXX
        
    lambert = bp.lambert_law(body_name).vals.astype('float')
    emission = bp.emission_angle(body_name)
    emission = emission.vals.astype('float') * oops.DPR
    center_resolution = bp.center_resolution(body_name).vals.astype('float')
#    center_resolution = bp.center_distance(body_name).vals.astype('float')**2
    resolution = center_resolution / np.cos(emission * oops.RPD)
    
    ok_body_mask_inv = np.logical_or(body_mask_inv, lambert < MIN_LAMBERT)
    ok_body_mask_inv = np.logical_or(ok_body_mask_inv, emission > MAX_EMISSION)
    ok_body_mask = np.logical_not(ok_body_mask_inv)
    
    im = imgdisp.ImageDisp([obs.data], [lambert], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

    lambert[ok_body_mask_inv] = 1e300
    adj_data = obs.data / lambert
    adj_data[ok_body_mask_inv] = 1e300
    lambert[ok_body_mask_inv] = 0.
    
    latitude = bp.latitude(body_name, lat_type='graphic')
    latitude = latitude.vals.astype('float') * oops.DPR

    longitude = bp.longitude(body_name, direction='west')
    longitude = longitude.vals.astype('float') * oops.DPR

    emission[ok_body_mask_inv] = 1e300
    resolution[ok_body_mask_inv] = 1e300

#    plt.figure()
#    plt.imshow(latitude)
#    plt.figure()
#    plt.imshow(longitude)
#    plt.figure()
#    plt.imshow(adj_data)
#    plt.show()
    
#    valid_latitude = latitude[ok_body_mask].flatten()
#    valid_longitude = longitude[ok_body_mask].flatten()
#    valid_data = adj_data[ok_body_mask].flatten()
#    valid_resolution = resolution[ok_body_mask].flatten()
#    valid_emission = emission[ok_body_mask].flatten()
    valid_latitude = latitude.flatten()
    valid_longitude = longitude.flatten()
    valid_data = adj_data.flatten()
    valid_resolution = resolution.flatten()
    valid_emission = emission.flatten()
    
    lat_bins = np.repeat(np.arange(mosaic.shape[0]),mosaic.shape[1])
    lat_bins_act = (lat_bins+0.5) * LATITUDE_RES - 90.
    
    long_bins = np.tile(np.arange(mosaic.shape[1]),mosaic.shape[0])
    long_bins_act = (long_bins+0.5) * LONGITUDE_RES

    print 'LAT RANGE', np.min(valid_latitude), np.max(valid_latitude)
    print 'LONG RANGE', np.min(valid_longitude), np.max(valid_longitude)
    
    print 'LAT BINS RANGE', np.min(lat_bins_act), np.max(lat_bins_act)
    print 'LONG BINS RANGE', np.min(long_bins_act), np.max(long_bins_act)
    
    latlon_points = np.empty((lat_bins.shape[0], 2))
    latlon_points[:,0] = lat_bins_act
    latlon_points[:,1] = long_bins_act
    
    interp_data = interp.griddata((valid_latitude, valid_longitude), valid_data,
                                  latlon_points, fill_value=1e300)
    interp_res = interp.griddata((valid_latitude, valid_longitude), valid_resolution,
                                 latlon_points, fill_value=1e300)
    interp_em = interp.griddata((valid_latitude, valid_longitude), valid_emission,
                                latlon_points, fill_value=1e300)

    new_mosaic = np.zeros(mosaic.shape)
    new_mosaic[lat_bins,long_bins] = interp_data
    new_resolution = np.zeros(mosaic.shape)+1e300
    new_resolution[lat_bins,long_bins] = interp_res
    new_emission = np.zeros(mosaic.shape)+1e300
    new_emission[lat_bins,long_bins] = interp_em
    
    new_mosaic_mask = np.logical_not(np.isnan(new_mosaic))
    better_resolution_mask = (new_resolution < mosaic_resolution)
    ok_emission_mask = (new_emission < 80)
    ok_value_mask = np.logical_and(-100 < new_mosaic, new_mosaic < 100)
    new_mosaic_mask = np.logical_and(new_mosaic_mask, better_resolution_mask)
    new_mosaic_mask = np.logical_and(new_mosaic_mask, ok_emission_mask)
    new_mosaic_mask = np.logical_and(new_mosaic_mask, ok_value_mask)
    
    mosaic[new_mosaic_mask] = new_mosaic[new_mosaic_mask]
    mosaic_resolution[new_mosaic_mask] = new_resolution[new_mosaic_mask]
    
#    plt.imshow(mosaic)
#    plt.show()
    
    im = imgdisp.ImageDisp([mosaic], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

def process_moon(filespec_list, body_name):
    nlat = int(np.round(180. / LATITUDE_RES))
    nlong = int(np.round(360. / LONGITUDE_RES))
    
    mosaic = np.zeros((nlat, nlong))
    mosaic_resolution = np.zeros((nlat, nlong)) + 1e38
    
    for filespec in filespec_list:
        full_filename = os.path.join(COISS_ROOT, filespec)
        full_filename = full_filename[:-4]
        full_filename += '_CALIB.IMG'
        process_moon_one_file(full_filename, body_name, mosaic, mosaic_resolution)
    im = imgdisp.ImageDisp([mosaic], canvas_size=(1024,768), allow_enlarge=True)
    tk.mainloop()

process_moon(MIMAS_FILES, 'MIMAS')
