'''
Created on Sep 19, 2011

@author: rfrench
'''

import argparse
import os
import os.path
import sys
import numpy as np
import oops.inst.cassini.iss as iss
from cb_config import *
from cb_offset import *
from cb_util_file import *

import matplotlib.pyplot as plt

LONGITUDE_RESOLUTION = 0.005
RADIUS_RESOLUTION = 5


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'N1627295812_1'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(description='SAO P124 Backplane Generator')

parser.add_argument(
    '--analyze-b-ring-edge', action='store_true',
    help='Analyze the B ring edge')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

#####################################################################################
#
# FIND THE POINTING OFFSET
#
#####################################################################################

def analayze_b_ring_edge(image_name, obs, off_radii, off_longitudes, 
                         off_resolution, off_emission, off_incidence,
                         off_phase,
                         offset):
    b_ring_edge = obs.bp.border_atop(off_radii.key, 117570.12).vals.astype('bool')
    if not np.any(b_ring_edge):
        min_long = -1000.
        max_long = -1000.
        min_res = -1000.
        max_res = -1000.
        min_em = -1000.
        max_em = -1000.
        min_phase = -1000.
        max_phase = -1000.
    else:
        longitudes = off_longitudes[b_ring_edge].vals.astype('float32')
        min_long = np.min(longitudes)
        max_long = np.max(longitudes)
        resolution = off_resolution[b_ring_edge].vals.astype('float32')
        min_res = np.min(resolution)
        max_res = np.max(resolution)
        emission = off_emission[b_ring_edge].vals.astype('float32')
        min_em = np.min(emission)
        max_em = np.max(emission)
        phase = off_phase[b_ring_edge].vals.astype('float32')
        min_phase = np.min(phase)
        max_phase = np.max(phase)
    data_file_csv = os.path.join(CB_RESULTS_ROOT, 'sao', image_name+'.csv')
    data_file_fp = open(data_file_csv, 'w')
    print >> data_file_fp, '%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d' % (
            image_name, cspice.et2utc(obs.midtime, 'C', 2),
            obs.midtime,
            min_long, max_long, 
            off_incidence,
            min_em, max_em,
            min_phase, max_phase,
            min_res, max_res,
            offset is not None)
    data_file_fp.close()
    
def offset_one_image(image_path):
    print 'Processing', image_path
    obs = file_read_iss_file(image_path)
    offset = None
    metadata = file_read_offset_metadata(image_path, 
                                         bootstrap_pref='prefer', 
                                         overlay=False)
    if metadata is not None:
        offset = metadata['offset']
    else:
        try:
            metadata = master_find_offset(obs,
                                          create_overlay=True)
            offset = metadata['offset'] 
        except:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
            print 'EXCEPTION:'
            print sys.exc_info()
            metadata = None
            offset = None
        
    if offset is None:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    image_name = file_clean_name(image_path)
    
    results = image_name + ' - ' + offset_result_str(metadata)
    print results

    reproject = False
    if not reproject:
        filename = os.path.join(CB_RESULTS_ROOT, 'sao', image_name)
        if offset is not None:
            obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
        set_obs_bp(obs)
        
        off_radii = obs.bp.ring_radius('saturn:ring')
        off_longitudes = obs.bp.ring_longitude('saturn:ring') * oops.DPR
        off_incidence = obs.bp.ring_incidence_angle('saturn:ring',
                                                    pole='north')
        off_incidence = np.mean(off_incidence.mvals) * oops.DPR
        off_resolution = obs.bp.ring_radial_resolution('saturn:ring')
        off_emission = obs.bp.ring_emission_angle('saturn:ring',
                                                  pole='north') * oops.DPR
        off_phase = obs.bp.phase_angle('saturn:ring') * oops.DPR
        if arguments.analyze_b_ring_edge:
            analayze_b_ring_edge(image_name, obs, off_radii, off_longitudes, 
                                 off_resolution, off_emission, 
                                 off_incidence, off_phase,
                                 offset)
            off_resolution = None
            off_emission = None
            off_phase = None
        else:
            off_resolution = off_resolution.vals.astype('float32')
            off_emission = off_emission.vals.astype('float32')
            off_phase = off_phase.vals.astype('float32')
        off_radii = off_radii.vals.astype('float32')
        off_longitudes = off_longitudes.vals.astype('float32')
    else:
        filename = 'j:/Temp/'+image_name+'-repro'
#        ret = rings_reproject(offdata.obs, offset_u=offdata.the_offset[0], offset_v=offdata.the_offset[1],
#                      longitude_resolution=LONGITUDE_RESOLUTION,
#                      radius_resolution=RADIUS_RESOLUTION,
#                      radius_inner=radius_inner,
#                      radius_outer=radius_outer)
#        obs.data = ret['img']
#        radii = rings_generate_radii(radius_inner,radius_outer,radius_resolution=RADIUS_RESOLUTION)
#        off_radii = np.zeros(offdata.obs.data.shape)
#        off_radii[:,:] = radii[:,np.newaxis]
#        longitudes = rings_generate_longitudes(longitude_resolution=LONGITUDE_RESOLUTION)
#        off_longitudes = np.zeros(offdata.obs.data.shape)
#        off_longitudes[:,:] = longitudes[ret['long_mask']]
#        off_resolution = ret['resolution']
#        off_incidence = ret['incidence']
#        off_emission = ret['emission']
#        off_phase = ret['phase']
        
    midtime = cspice.et2utc(obs.midtime,'C',2)
    np.savez(filename, 
             midtime=midtime,
             data=obs.data,
             radii=off_radii,
             longitudes=off_longitudes,
             resolution=off_resolution,
             incidence=off_incidence,
             emission=off_emission,
             phase=off_phase)

#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

for image_path in file_yield_image_filenames_from_arguments(arguments):
    offset_one_image(image_path)
