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
from cb_offset import *
from cb_util_file import *

LONGITUDE_RESOLUTION = 0.005
RADIUS_RESOLUTION = 5


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'N1627295812_1'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(description='SAO P124 Backplane Generator')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

#####################################################################################
#
# FIND THE POINTING OFFSET
#
#####################################################################################
    
def offset_one_image(image_path):
    print 'Processing', image_path
    obs = file_read_iss_file(image_path)
    offset = None
    try:
        metadata = master_find_offset(obs,
                                      create_overlay=True)
        offset = metadata['offset'] 
    except:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
        print 'EXCEPTION:'
        print sys.exc_info()
    
    if offset is None:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
        offset = (0,0)
    
    image_name = file_clean_name(image_path)
    
    results = image_name + ' - ' + offset_result_str(metadata)
    print results

    reproject = False
    if not reproject:
        filename = 'j:/Temp/'+image_name
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
        set_obs_bp(obs)
        
        off_radii = obs.bp.ring_radius('saturn:ring').vals.astype('float')
        off_longitudes = obs.bp.ring_longitude('saturn:ring').vals.astype('float') * oops.DPR
        off_resolution = obs.bp.ring_radial_resolution('saturn:ring').vals.astype('float')
        off_incidence = obs.bp.incidence_angle('saturn:ring').vals.astype('float') * oops.DPR
        off_emission = obs.bp.emission_angle('saturn:ring').vals.astype('float') * oops.DPR
        off_phase = obs.bp.phase_angle('saturn:ring').vals.astype('float') * oops.DPR
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
        
    np.savez(filename, 
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
