import argparse
import os
import os.path
import oops
import cspice
import numpy as np
import msgpack
import msgpack_numpy

from cb_config import *
from cb_util_file import *

# oops.spice.load_leap_seconds()

RING_FILENAMES = None
   
RING_RESULTS_ROOT = file_clean_join(CB_RESULTS_ROOT, 'ring_mosaic')
RING_SOURCE_ROOT = file_clean_join(CB_SOURCE_ROOT, 'ring_mosaic')

RING_REPROJECT_PY = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_reproject.py')
RING_MOSAIC_PY = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_mosaic.py')
RING_BKGND_PY = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_bkgnd.py')

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
        
        self.image_log_filehandler = None

        
class MosaicData(object):
    """Mosaic metadata."""
    def __init__(self):
        self.img = None

class BkgndData(object):
    """Background metadata."""
    pass

def ring_add_parser_options(parser):
    # For file selection
    parser.add_argument(
        'obsid', action='append', nargs='*',
        help='Specific OBSIDs to process')
    parser.add_argument(
        '--ring-type', default='FMOVIE',
        help='The type of ring mosaics; use to retrieve the file lists')
    parser.add_argument(
        '--corot-type', default='',
        help='The type of co-rotation frame to use')
    parser.add_argument(
        '--all-obsid', action='store_true', default=False,
        help='Process all OBSIDs of the given type')
    parser.add_argument(
        '--ring-radius', type=int, default=0,
        help='The main ring radius; by default loaded from the ring type')
    parser.add_argument(
        '--radius-inner-delta', type=int, default=0,
        help='''The inner delta from the main ring radius; 
                by default loaded from the ring type''')
    parser.add_argument(
        '--radius-outer-delta', type=int, default=0,
        help='''The outer delta from the main ring radius; 
                by default loaded from the ring type''')
    parser.add_argument(
        '--radius-resolution', type=float, default=0.,
        help='The radial resolution for reprojection')
    parser.add_argument(
        '--longitude-resolution', type=float, default=0.,
        help='The longitudinal resolution for reprojection')
    parser.add_argument('--verbose', action='store_true', default=False)    

def ring_init(arguments):
    global RING_FILENAMES
    if RING_FILENAMES is None:
        RING_FILENAMES = {}
        type_dir = file_clean_join(RING_SOURCE_ROOT, 'FILELIST_'+
                                   arguments.ring_type.upper())

        default_filename = file_clean_join(type_dir, 'defaults.txt')
        assert os.path.exists(default_filename)
        default_fp = open(default_filename, 'r')
        default_corot = default_fp.readline().strip()
        default_radius = int(default_fp.readline().strip())
        default_radius_inner = int(default_fp.readline().strip())
        default_radius_outer = int(default_fp.readline().strip())
        default_radius_resolution = float(default_fp.readline().strip())
        default_longitude_resolution = float(default_fp.readline().strip())
        default_fp.close()
        if arguments.corot_type == '':
            arguments.corot_type = default_corot
        if arguments.ring_radius == 0:
            arguments.ring_radius = default_radius
        if arguments.radius_inner_delta == 0:
            arguments.radius_inner_delta = default_radius_inner
        if arguments.radius_outer_delta == 0:
            arguments.radius_outer_delta = default_radius_outer
        if arguments.radius_resolution == 0:
            arguments.radius_resolution = default_radius_resolution
        if arguments.longitude_resolution == 0:
            arguments.longitude_resolution = default_longitude_resolution
            
        for obsid_file in sorted(os.listdir(type_dir)):
            if not obsid_file.endswith('.list'):
                continue
            obsid = obsid_file[:-5]
            fp = open(file_clean_join(type_dir, obsid_file), 'r')
            filenames = fp.read().split()
            fp.close()
            RING_FILENAMES[obsid] = filenames
            
def ring_enumerate_files(arguments):
    if arguments.all_obsid:
        obsid_db = RING_FILENAMES
    else:
        obsid_db = {}
        for arg in arguments.obsid[0]:
            if '/' in arg:
                # OBSID/FILENAME
                obsid, filename = arg.split('/')
                if not obsid in obsid_db:
                    obsid_db[obsid] = []
                obsid_db[obsid].append(filename)
            else:
                # OBSID
                obsid_db[arg] = RING_FILENAMES[arg]
    
        for obsid in sorted(obsid_db.keys()):
            obsid_db[obsid].sort(key=lambda x: x[1:13]+x[0])

    for obsid in sorted(obsid_db.keys()):
        filename_list = obsid_db[obsid]
        for full_path in file_yield_image_filenames(
                                            restrict_list=filename_list):
            _, filename = os.path.split(full_path)
            yield obsid, filename[:13], full_path

def ring_basic_cmd_line(arguments):
    ret = ['--ring-type', arguments.ring_type]
    ret += ['--corot-type', arguments.corot_type]
    ret += ['--ring-radius', str(arguments.ring_radius)]
    ret += ['--radius-inner', str(arguments.radius_inner_delta)]
    ret += ['--radius-outer', str(arguments.radius_outer_delta)]
    ret += ['--radius-resolution', str(arguments.radius_resolution)]
    ret += ['--longitude-resolution', str(arguments.longitude_resolution)]
    
    return ret

def repro_path_spec(ring_radius,
                    radius_inner, radius_outer, radius_resolution, 
                    longitude_resolution, image_path, image_name,
                    make_dirs=False):
    repro_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f-REPRO.dat' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution))
    repro_path = file_results_path(image_path+repro_res_data,
                                   'ring_repro',
                                   root=RING_RESULTS_ROOT, 
                                   make_dirs=make_dirs)
    return repro_path

def repro_path(arguments, image_path, image_name, make_dirs=False):
    return repro_path_spec(arguments.ring_radius,
                           arguments.radius_inner_delta,
                           arguments.radius_outer_delta,
                           arguments.radius_resolution,
                           arguments.longitude_resolution,
                           image_path, image_name,
                           make_dirs=make_dirs)

def mosaic_paths_spec(ring_radius, radius_inner, radius_outer, 
                      radius_resolution, longitude_resolution,
                      obsid, ring_type, make_dirs=False):
    mosaic_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution))
    mosaic_dir = file_clean_join(RING_RESULTS_ROOT, 'mosaic_'+ring_type)
    if make_dirs and not os.path.exists(mosaic_dir):
        os.mkdir(mosaic_dir)
    data_path = file_clean_join(mosaic_dir, obsid+mosaic_res_data+'-MOSAIC')
    metadata_path = file_clean_join(mosaic_dir, 
                                    obsid+mosaic_res_data+'-MOSAIC-METADATA.dat')
    return (data_path, metadata_path)

def mosaic_paths(arguments, obsid, make_dirs=False):
    return mosaic_paths_spec(arguments.ring_radius,
                             arguments.radius_inner_delta,
                             arguments.radius_outer_delta,
                             arguments.radius_resolution,
                             arguments.longitude_resolution,
                             obsid, arguments.ring_type,
                             make_dirs=make_dirs)

def png_path_spec(ring_radius, radius_inner, radius_outer, 
                  radius_resolution, longitude_resolution,
                  obsid, ring_type, png_type, make_dirs=False):
    mosaic_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution))
    png_dir = file_clean_join(RING_RESULTS_ROOT, 
                              'png_'+png_type+'_'+ring_type)
    if make_dirs and not os.path.exists(png_dir):
        os.mkdir(png_dir)
    png_path = file_clean_join(png_dir, 
                               obsid+mosaic_res_data+'-'+png_type+'.png')
    return png_path

def png_path(arguments, obsid, png_type, make_dirs=False):
    return png_path_spec(arguments.ring_radius,
                         arguments.radius_inner_delta,
                         arguments.radius_outer_delta,
                         arguments.radius_resolution,
                         arguments.longitude_resolution,
                         obsid, arguments.ring_type,
                         png_type,
                         make_dirs=make_dirs)

def bkgnd_paths_spec(ring_radius, radius_inner, radius_outer, 
                     radius_resolution, longitude_resolution,
                     obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution))
    bkgnd_dir = file_clean_join(RING_RESULTS_ROOT, 'bkgnd_'+ring_type)
    if make_dirs and not os.path.exists(bkgnd_dir):
        os.mkdir(bkgnd_dir)
    data_path = file_clean_join(bkgnd_dir, obsid+bkgnd_res_data+'-MOSAIC')
    reduced_mosaic_data_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-REDUCED-MOSAIC')
    reduced_mosaic_metadata_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-REDUCED-MOSAIC-METADATA.dat')
    bkgnd_mask_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-BKGND-MASK')
    bkgnd_model_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-BKGND-MODEL')
    bkgnd_metadata_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-BKGND-METADATA.dat')

    return (reduced_mosaic_data_path, reduced_mosaic_metadata_path,
            bkgnd_mask_path, bkgnd_model_path, bkgnd_metadata_path)
    
def bkgnd_paths(arguments, obsid, make_dirs=False):
    return bkgnd_paths_spec(arguments.ring_radius,
                             arguments.radius_inner_delta,
                             arguments.radius_outer_delta,
                             arguments.radius_resolution,
                             arguments.longitude_resolution,
                             obsid, arguments.ring_type,
                             make_dirs=make_dirs)
    
# def ew_paths(arguments, obsid):
#     ew_res_data = ('_%04d_%04d_%06.3f_%05.3f' % (arguments.radius_inner, arguments.radius_outer,
#                                                            arguments.radius_resolution,
#                                                            arguments.longitude_resolution))
#     ew_data_filename = file_clean_join(RING_RESULTS_ROOT, 'ew-data',
#                                     obsid+ew_res_data+'-data' + 
#                                     '_%06d_%06d' % (arguments.core_radius_inner, arguments.core_radius_outer))
#     ew_mask_filename = file_clean_join(RING_RESULTS_ROOT, 'ew-data',
#                                     obsid+ew_res_data+'-mask' +
#                                     '_%06d_%06d' % (arguments.core_radius_inner, arguments.core_radius_outer))
#     return (ew_data_filename, ew_mask_filename)

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

def read_repro(repro_path):
    if not os.path.exists(repro_path):
        return None
    repro_fp = open(repro_path, 'rb')
    repro_data = msgpack.unpackb(repro_fp.read(), 
                                 object_hook=msgpack_numpy.decode)
    repro_fp.close()
        
    return repro_data
    
def write_repro(repro_path, repro_data):
    repro_fp = open(repro_path, 'wb')
    repro_fp.write(msgpack.packb(repro_data, 
                                 default=msgpack_numpy.encode))
    repro_fp.close()

