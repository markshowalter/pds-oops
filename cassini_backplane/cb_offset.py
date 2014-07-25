import cb_logging
import logging

import numpy as np
import numpy.ma as ma

from cb_config import *
from cb_correlate import *
from cb_moons import *
from cb_rings import *

LOGGING_NAME = 'cb.' + __name__

# These are bodies large enough to be picked up in an image
LARGE_BODY_LIST = ['SATURN', 'ATLAS', 'PROMETHEUS', 'PANDORA',
                   'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                   'TETHYS', 'DIONE', 'RHEA', 'TITAN', 'HYPERION',
                   'IAPETUS', 'PHOEBE']

ELLIPSOIDAL_BODY_LIST = ['SATURN', 'ATLAS', 'PROMETHEUS', 'PANDORA',
                         'EPIMETHEUS', 'JANUS', 'MIMAS', 'ENCELADUS',
                         'TETHYS', 'DIONE', 'RHEA', 'TITAN', 'HYPERION',
                         'IAPETUS', 'PHOEBE']


def _normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_min == data_max:
        return np.zeros(data.shape)
    return (data-data_min) / (data_max-data_min)

def _combine_models(model_list):
    new_model = np.zeros(model_list[0].shape)
    for model in model_list:
        new_model += _normalize(model)

    return new_model
    
def find_offset(obs):

    overlay = np.zeros(obs.data.shape + (3,))
    
    bp = oops.Backplane(obs)

    non_saturn_model_list = []
    saturn_model = None
        
    large_bodies = obs_inventory(obs, LARGE_BODY_LIST)
    
    print 'LARGE BODIES', large_bodies
    
    for body in large_bodies:
#        lambert_model = bp.lambert_law(body_name).vals.astype('float')
        model = moons_create_model(obs, body.body_name, lambert=True,
                                   u_min=body.u_min, u_max=body.u_max,
                                   v_min=body.v_min, v_max=body.v_max,
                                   force_spherical=True)
        if body.body_name == 'SATURN':
            saturn_model = model
        else:
            non_saturn_model_list.append(model)
    
    rings_model = rings_create_model(obs, bp)
    
    if saturn_model is not None:
        overlay[...,0] = _normalize(saturn_model)
    if len(non_saturn_model_list) > 0:
        overlay[...,1] = _normalize(_combine_models(non_saturn_model_list))
    if rings_model is not None:
        overlay[...,2] = _normalize(rings_model)

    body = large_bodies[0]
    model = moons_create_model(obs, body.body_name, lambert=True,
                               u_min=body.u_min, u_max=body.u_max,
                               v_min=body.v_min, v_max=body.v_max,
                               use_cartographic=False)
    non_saturn_model_list = [model]
    
    model_list = non_saturn_model_list
    if saturn_model is not None:
        model_list = model_list + [saturn_model]
    if rings_model is not None:
        model_list = model_list + [rings_model]
    if len(model_list) == 0:
        final_model = np.zeros(obs.data.shape)
    else:
        final_model = _combine_models(model_list)

    search_size_max_u = 140
    search_size_max_v = 140
    
#    overlay[...,0] = final_model
#    overlay[...,1] = 0
#    overlay[...,2] = 0
    
    offset_u, offset_v, peak = find_correlation_and_offset(obs.data, final_model, search_size_min=0,
                                                           search_size_max=(search_size_max_u,
                                                                            search_size_max_v))
    
    return offset_u, offset_v, final_model, overlay

