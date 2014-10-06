from optparse import OptionParser
import os
import os.path
import oops
import cspice
import pickle
import numpy as np

from cb_util_image import *

oops.spice.load_leap_seconds()

if os.getcwd()[1] == ':':
    # Windows
    PYTHON_EXE = 'c:/Users/rfrench/AppData/Local/Enthought/Canopy/User/python.exe'
    ROOT = 'T:/clumps'
    DATA_ROOT = 'T:/clumps/data'
else:
    # Linux
    assert False
    
PYTHON_RING_REPROJECT = 'fring_ui_reproject.py'
PYTHON_RING_MOSAIC = 'fring_ui_mosaic.py'
PYTHON_RING_BKGND = 'fring_ui_bkgnd.py'

SUFFIX_CALIB = '_CALIB.IMG'

class OffRepData(object):
    """Offset and Reprojection data."""
    def __init__(self):
        self.obsid = None
        self.image_name = None
        self.image_path = None
        self.obs = None

        self.offset_path = None
        self.the_offset = None
        self.manual_offset = None
        self.off_metadata = None
        
        self.repro_path = None
        self.repro_img = None

        self.repro_longitudes = None
        self.repro_phase_angles = None
        self.repro_incidence_angles = None
        self.repro_emission_angles = None
        self.repro_resolutions = None

        
class MosaicData(object):
    """Mosaic metadata."""
    def __init__(self):
        self.img = None

class BkgndData(object):
    """Background metadata."""
    pass

def add_parser_options(parser):
    # For file selection
    parser.add_option('-a', '--all_obs', dest='all_obs', action='store_true', default=False)
    parser.add_option('--radius_inner', dest='radius_inner', type='int', default=-1000)
    parser.add_option('--radius_outer', dest='radius_outer', type='int', default=1000)
    parser.add_option('-r', '--radius_resolution', type='float', dest='radius_resolution', default=5.0)
    parser.add_option('-l', '--longitude_resolution', type='float', dest='longitude_resolution',
                      default=0.02)
    parser.add_option('--verbose', action='store_true', dest='verbose', default=False)    

def enumerate_files(options, args, suffix='', obsid_only=False):
    if options.all_obs:
        dir_list = sorted(os.listdir(DATA_ROOT))
        file_list = []
        for dir in dir_list:
            if os.path.isdir(os.path.join(DATA_ROOT, dir)):
                file_list.append(dir)
    else:
        file_list = args

    for arg in file_list:
        if os.path.exists(arg): # Absolute path
            assert not obsid_only
            path, image_name = os.path.split(arg)
            assert file[0] == 'N' or file[0] == 'W'
            file = file[:11]
            path, obs_id = os.path.split(path)
            yield obs_id, image_name, arg
        else:
            abs_path = os.path.join(DATA_ROOT, arg)
            if os.path.isdir(abs_path): # Observation ID
                if obsid_only:
                    yield arg, None, None
                    continue
                filenames = sorted(os.listdir(abs_path))
                for filename in filenames:
                    full_path = os.path.join(DATA_ROOT, arg, filename)
                    if not os.path.isfile(full_path): continue
                    if filename[-len(suffix):] != suffix: continue
                    image_name = filename[:-len(suffix)]
                    yield arg, image_name, full_path
                    
            else: # Single OBSID/IMAGENAME
                obs_id, image_name = os.path.split(arg)
                abs_path += suffix
                yield obs_id, image_name, abs_path

def offset_path(options, image_path, image_name):
    return image_path + '.FOFFSET'

def repro_path(options, image_path, image_name):
    repro_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
                                                         options.radius_resolution,
                                                         options.longitude_resolution))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_FREPRO')

def repro_path_spec(radius_inner, radius_outer, radius_resolution, longitude_resolution,
                    image_path, image_name):
    repro_res_data = ('_%06d_%06d_%06.3f_%05.3f' % (radius_inner, radius_outer,
                                                         radius_resolution,
                                                         longitude_resolution))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_FREPRO')

def mosaic_paths(options, obsid):
    mosaic_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
                                                          options.radius_resolution,
                                                          options.longitude_resolution))
    data_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-data')
    metadata_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-metadata.pickle')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+obsid+mosaic_res_data+'.png')
    return (data_path, metadata_path, large_png_path)

def mosaic_paths_spec(radius_inner, radius_outer, radius_resolution, longitude_resolution,
                      obsid):
    mosaic_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (radius_inner, radius_outer,
                                                          radius_resolution,
                                                          longitude_resolution))
    data_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-data')
    metadata_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-metadata.pickle')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+obsid+mosaic_res_data+'.png')
    return (data_path, metadata_path, large_png_path)

def bkgnd_paths(options, obsid):
    bkgnd_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
                                                              options.radius_resolution,
                                                              options.longitude_resolution))
    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
                                                obsid+bkgnd_res_data+'-data')
    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                                    obsid+bkgnd_res_data+'-metadata.pickle')
    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
                                       obsid+bkgnd_res_data+'-bkgnd-mask')
    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
                                        obsid+bkgnd_res_data+'-bkgnd-model')
    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                           obsid+bkgnd_res_data+'-bkgnd-metadata.pickle')
    data_path, metadata_path, large_png_path,small_png_path = mosaic_paths(options, obsid)
    return(data_path, metadata_path, bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) 
    
def bkgnd_paths_spec(radius_inner, radius_outer, radius_resolution, longitude_resolution,
                     obsid):
    bkgnd_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (radius_inner, radius_outer,
                                                              radius_resolution,
                                                              longitude_resolution))
    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
                                                obsid+bkgnd_res_data+'-data')
    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                                    obsid+bkgnd_res_data+'-metadata.pickle')
    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
                                       obsid+bkgnd_res_data+'-bkgnd-mask')
    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
                                        obsid+bkgnd_res_data+'-bkgnd-model')
    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                           obsid+bkgnd_res_data+'-bkgnd-metadata.pickle')

    return (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
            bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename)
    
def ew_paths(options, obsid):
    ew_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
                                                           options.radius_resolution,
                                                           options.longitude_resolution))
    ew_data_filename = os.path.join(ROOT, 'ew-data',
                                    obsid+ew_res_data+'-data' + 
                                    '_%06d_%06d' % (options.core_radius_inner, options.core_radius_outer))
    ew_mask_filename = os.path.join(ROOT, 'ew-data',
                                    obsid+ew_res_data+'-mask' +
                                    '_%06d_%06d' % (options.core_radius_inner, options.core_radius_outer))
    return (ew_data_filename, ew_mask_filename)

ROTATING_ET = cspice.utc2et("2007-1-1")
FRING_MEAN_MOTION = 581.964

def ComputeLongitudeShift(img_ET): 
    return - (FRING_MEAN_MOTION * ((img_ET - ROTATING_ET) / 86400.)) % 360.

def InertialToCorotating(longitude, ET):
    return (longitude + ComputeLongitudeShift(ET)) % 360.

def CorotatingToInertial(co_long, ET):
    return (co_long - ComputeLongitudeShift(ET)) % 360.

def CorotatingToTrueAnomaly(co_long, ET):
    return (co_long - ComputeLongitudeShift(ET) - 2.7007*(ET/86400.)) % 360.

OFFSET_FILE_VERSION = 0

def read_offset(offset_path):
    if not os.path.exists(offset_path+'.pickle'):
        return None, None, None
    offset_pickle_fp = open(offset_path+'.pickle', 'rb')
    offset_file_version = pickle.load(offset_pickle_fp)
    
    assert offset_file_version == OFFSET_FILE_VERSION
    the_offset = pickle.load(offset_pickle_fp)
    manual_offset = pickle.load(offset_pickle_fp)
    metadata = pickle.load(offset_pickle_fp)
    offset_pickle_fp.close()

    overlay = np.load(offset_path+'OVER.npy')
    if overlay.shape[0] == 0:
        metadata['overlay'] = None
    else:   
        metadata['overlay'] = uncompress_saturated_overlay(overlay) 
        
    return the_offset, manual_offset, metadata

def write_offset(offset_path, the_offset, manual_offset, metadata):
    offset_pickle_fp = open(offset_path+'.pickle', 'wb')
    pickle.dump(OFFSET_FILE_VERSION, offset_pickle_fp)
    pickle.dump(the_offset, offset_pickle_fp)
    pickle.dump(manual_offset, offset_pickle_fp)
    
    new_metadata = metadata.copy()
    if 'ext_data' in new_metadata:
        del new_metadata['ext_data']
    if 'ext_overlay' in new_metadata:
        del new_metadata['ext_overlay']
    overlay = np.array([])
    if 'overlay' in new_metadata and new_metadata['overlay'] is not None:
        overlay = compress_saturated_overlay(new_metadata['overlay'])
        del new_metadata['overlay']
    pickle.dump(new_metadata, offset_pickle_fp)    
    offset_pickle_fp.close()
    np.save(offset_path+'OVER', overlay)

def read_repro(repro_path):
    if not os.path.exists(repro_path+'.pickle'):
        return None
    repro_pickle_fp = open(repro_path+'.pickle', 'rb')
    repro_data = pickle.load(repro_pickle_fp)
    repro_pickle_fp.close()
        
    return repro_data
    
def write_repro(repro_path, repro_data):
    repro_pickle_fp = open(repro_path+'.pickle', 'wb')
    pickle.dump(repro_data, repro_pickle_fp)
    repro_pickle_fp.close()

def read_mosaic_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return None, None, None, None, None, None
    metadata_pickle_fp = open(metadata_path, 'rb')
#    metadata_

    (mosaicdata.img, mosaicdata.longitudes, mosaicdata.resolutions,
     mosaicdata.image_numbers, mosaicdata.ETs, 
     mosaicdata.emission_angles, mosaicdata.incidence_angles,
     mosaicdata.phase_angles) = result

    # Save metadata
    metadata = result[1:] # Everything except img
    mosaic_metadata_fp = open(mosaicdata.metadata_path, 'wb')
    pickle.dump(metadata, mosaic_metadata_fp)
    pickle.dump(mosaicdata.obsid_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.image_name_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.image_path_list, mosaic_metadata_fp)
    pickle.dump(mosaicdata.repro_path_list, mosaic_metadata_fp)
    mosaic_metadata_fp.close()


#def prometheus_close_approach(min_et, min_et_long):
#    def compute_r(a, e, arg): # Takes argument of pericenter
#        return a*(1-e**2.) / (1+e*np.cos(arg))
#    def compute_r_fring(arg):
#        return compute_r(bosh2002_fring_a, bosh2002_fring_e, arg)
#
#    bosh2002_epoch_et = cspice.utc2et('JD 2451545.0') # J2000
#    bosh2002_fring_a = 140223.7
#    bosh2002_fring_e = 0.00254
#    bosh2002_fring_curly = 24.1 * np.pi/180
#    bosh2002_fring_curly_dot = 2.7001 * np.pi/180 / 86400 # rad/sec
#
#    # Find time for 0 long
#    et_min = min_et - min_et_long / FRING_MEAN_MOTION * 86400.
#    # Find time for 360 long
#    et_max = min_et + 360. / FRING_MEAN_MOTION * 86400 
#    # Find the longitude at the point of closest approach
#    min_dist = 1e38
#    for et in np.arange(et_min, et_max, 60): # Step by minute
#        prometheus_dist, prometheus_longitude = ringimage.saturn_to_prometheus(et)
#        prometheus_longitude = CorotatingToInertial(prometheus_longitude, et)
#        long_peri_fring = (et-bosh2002_epoch_et) * bosh2002_fring_curly_dot + bosh2002_fring_curly
#        fring_r = compute_r_fring(prometheus_longitude-long_peri_fring)
#        if abs(fring_r-prometheus_dist) < min_dist:
#            min_dist = abs(fring_r-prometheus_dist)
#            min_dist_long = prometheus_longitude
#            min_dist_et = et
#    min_dist_long = InertialToCorotating(min_dist_long, min_dist_et)
#    return min_dist, min_dist_long
#
#def compute_mu(e):
#    if type(e) == type([]):
#        e = np.array(e)
#    return np.abs(np.cos(e*np.pi/180.))
#
#def compute_z(mu, mu0, tau, is_transmission):
#    transmission_list = tau*(mu-mu0)/(mu*mu0*(np.exp(-tau/mu)-np.exp(-tau/mu0)))
#    reflection_list = tau*(mu+mu0)/(mu*mu0*(1-np.exp(-tau*(1/mu+1/mu0))))
#    ret = np.where(is_transmission, transmission_list, reflection_list)
#    return ret
#
## This takes EW * mu
#def compute_corrected_ew(ew, emission, incidence, tau=0.034):
#    if type(emission) == type([]):
#        emission = np.array(emission)
#    if type(incidence) == type([]):
#        incidence = np.array(incidence)
#    is_transmission = emission > 90.
#    mu = compute_mu(emission)
#    mu0 = np.abs(np.cos(incidence*np.pi/180))
#    ret = ew * compute_z(mu, mu0, tau, is_transmission)
#    return ret
#
#def clump_phase_curve(phase_angles):
#    coeffs = np.array([6.09918565e-07, -8.81293896e-05, 5.51688159e-03, -3.29583781e-01])
#    
#    return 10**np.polyval(coeffs, phase_angles)
#
##def normalized_phase_curve(alpha):
##    return np.exp((6.56586808e-07*(alpha**3)-9.25559440e-05*(alpha**2)+5.08638514e-03*alpha-2.76364092e-01))
#
#def mu(emission):
#    return np.abs(np.cos(emission*np.pi/180.))
#
#def transmission_function(tau, emission, incidence):
#    assert False
#    #incidence angle is always less than 90, therefore the only case we have to worry about it when Emission Angle changes.
#    #E > 90 = Transmission, E < 90 = Reflection
#    mu0 = np.abs(np.cos(incidence*np.pi/180.))
#    mu = np.abs(np.cos(emission*np.pi/180.))
#
#    if np.mean(emission[np.nonzero(emission)]) > 90:
##        print 'Transmission'
#        return mu * mu0 * (np.exp(-tau/mu)-np.exp(-tau/mu0)) / (tau * (mu-mu0))
#    elif np.mean(emission[np.nonzero(emission)]) < 90:
##        print 'Reflection'
#        return mu * mu0 * (1.-np.exp(-tau*(1/mu+1/mu0))) / (tau * (mu+mu0))
#
#
#def normalized_ew_factor(alpha, emission, incidence):
#    assert False
#    tau_eq = 0.033 #French et al. 2012
#    return mu(emission)/(transmission_function(tau_eq, emission, incidence)*normalized_phase_curve(alpha))
