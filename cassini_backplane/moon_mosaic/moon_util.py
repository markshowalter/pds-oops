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
    ROOT = 'T:/moons'
    DATA_ROOT = 'T:/moons/data'
else:
    # Linux
    assert False
    
PYTHON_MOON_REPROJECT = 'moon_ui_reproject.py'
PYTHON_MOON_MOSAIC = 'moon_ui_mosaic.py'
PYTHON_MOON_BKGND = 'moon_ui_bkgnd.py'

SUFFIX_CALIB = '_CALIB.IMG'

class OffRepData(object):
    """Offset and Reprojection data."""
    def __init__(self):
        self.body_name = None
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
    parser.add_option('-r', '--latitude_resolution', type='float', dest='latitude_resolution', default=0.1)
    parser.add_option('-l', '--longitude_resolution', type='float', dest='longitude_resolution',
                      default=0.1)
    parser.add_option('--verbose', action='store_true', dest='verbose', default=False)    

def enumerate_files(options, args, suffix='', body_name_only=False):
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
            assert not body_name_only
            path, image_name = os.path.split(arg)
            assert file[0] == 'N' or file[0] == 'W'
            file = file[:11]
            path, obs_id = os.path.split(path)
            yield obs_id, image_name, arg
        else:
            abs_path = os.path.join(DATA_ROOT, arg)
            if os.path.isdir(abs_path): # Observation ID
                if body_name_only:
                    yield arg, None, None
                    continue
                filenames = sorted(os.listdir(abs_path))
                for filename in filenames:
                    full_path = os.path.join(DATA_ROOT, arg, filename)
                    if not os.path.isfile(full_path): continue
                    if filename[-len(suffix):] != suffix: continue
                    image_name = filename[:-len(suffix)]
                    yield arg, image_name, full_path
                    
            else: # Single body_name/IMAGENAME
                obs_id, image_name = os.path.split(arg)
                abs_path += suffix
                yield obs_id, image_name, abs_path

def offset_path(options, image_path, image_name):
    return image_path + '.FOFFSET'

def repro_path(options, image_path, image_name):
    repro_res_data = ('_%06.3f_%05.3f' % (options.latitude_resolution,
                                          options.longitude_resolution))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_REPRO')

def repro_path_spec(latitude_resolution, longitude_resolution,
                    image_path, image_name):
    repro_res_data = ('_%06.3f_%05.3f' % (latitude_resolution,
                                          longitude_resolution))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_FREPRO')

def mosaic_paths(options, body_name):
    mosaic_res_data = ('_%06.3f_%05.3f' % (options.latitude_resolution,
                                           options.longitude_resolution))
    data_path = os.path.join(ROOT, 'mosaic-data', body_name+mosaic_res_data+'-data')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+body_name+mosaic_res_data+'.png')
    return (data_path, large_png_path)

def mosaic_paths_spec(latitude_resolution, longitude_resolution,
                      body_name):
    mosaic_res_data = ('_%06.3f_%05.3f' % (latitude_resolution,
                                           longitude_resolution))
    data_path = os.path.join(ROOT, 'mosaic-data', body_name+mosaic_res_data+'-data')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+body_name+mosaic_res_data+'.png')
    return (data_path, large_png_path)

#def bkgnd_paths(options, body_name):
#    bkgnd_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
#                                                              options.radius_resolution,
#                                                              options.longitude_resolution))
#    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
#                                                body_name+bkgnd_res_data+'-data')
#    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
#                                                    body_name+bkgnd_res_data+'-metadata.pickle')
#    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
#                                       body_name+bkgnd_res_data+'-bkgnd-mask')
#    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
#                                        body_name+bkgnd_res_data+'-bkgnd-model')
#    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
#                                           body_name+bkgnd_res_data+'-bkgnd-metadata.pickle')
#    data_path, metadata_path, large_png_path,small_png_path = mosaic_paths(options, body_name)
#    return(data_path, metadata_path, bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) 
#    
#def bkgnd_paths_spec(radius_inner, radius_outer, radius_resolution, longitude_resolution,
#                     body_name):
#    bkgnd_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (radius_inner, radius_outer,
#                                                              radius_resolution,
#                                                              longitude_resolution))
#    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
#                                                body_name+bkgnd_res_data+'-data')
#    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
#                                                    body_name+bkgnd_res_data+'-metadata.pickle')
#    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
#                                       body_name+bkgnd_res_data+'-bkgnd-mask')
#    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
#                                        body_name+bkgnd_res_data+'-bkgnd-model')
#    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
#                                           body_name+bkgnd_res_data+'-bkgnd-metadata.pickle')
#
#    return (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
#            bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename)
#    
#def ew_paths(options, body_name):
#    ew_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (options.radius_inner, options.radius_outer,
#                                                           options.radius_resolution,
#                                                           options.longitude_resolution))
#    ew_data_filename = os.path.join(ROOT, 'ew-data',
#                                    body_name+ew_res_data+'-data' + 
#                                    '_%06d_%06d' % (options.core_radius_inner, options.core_radius_outer))
#    ew_mask_filename = os.path.join(ROOT, 'ew-data',
#                                    body_name+ew_res_data+'-mask' +
#                                    '_%06d_%06d' % (options.core_radius_inner, options.core_radius_outer))
#    return (ew_data_filename, ew_mask_filename)

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
        metadata['overlay'] = overlay #uncompress_saturated_overlay(overlay) 
        
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
        overlay = new_metadata['overlay'] #compress_saturated_overlay(new_metadata['overlay'])
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

#def read_mosaic_metadata(metadata_path):
#    if not os.path.exists(metadata_path):
#        return None, None, None, None, None, None
#    metadata_pickle_fp = open(metadata_path, 'rb')
##    metadata_
#
#    (mosaicdata.img, mosaicdata.longitudes, mosaicdata.resolutions,
#     mosaicdata.image_numbers, mosaicdata.ETs, 
#     mosaicdata.emission_angles, mosaicdata.incidence_angles,
#     mosaicdata.phase_angles) = result
#
#    # Save metadata
#    metadata = result[1:] # Everything except img
#    mosaic_metadata_fp = open(mosaicdata.metadata_path, 'wb')
#    pickle.dump(metadata, mosaic_metadata_fp)
#    pickle.dump(mosaicdata.body_name_list, mosaic_metadata_fp)
#    pickle.dump(mosaicdata.image_name_list, mosaic_metadata_fp)
#    pickle.dump(mosaicdata.image_path_list, mosaic_metadata_fp)
#    pickle.dump(mosaicdata.repro_path_list, mosaic_metadata_fp)
#    mosaic_metadata_fp.close()
