import cb_logging
import logging

import numpy as np
import numpy.ma as ma

import oops

from pdstable import PdsTable
from tabulation import Tabulation
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

RING_OCCULTATION_TABLE = PdsTable('IS2_P0001_V01_KM002.LBL')
RING_OCCULTATION_DATA = Tabulation(RING_OCCULTATION_TABLE.column_dict['RING_INTERCEPT_RADIUS'],
                                   RING_OCCULTATION_TABLE.column_dict['I_OVER_F'])

RINGS_MIN_RADIUS = oops.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.SATURN_MAIN_RINGS[1]

def rings_create_model(obs, bp):
    logger = logging.getLogger(LOGGING_NAME+'.rings_create_model')

    radii = bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    logger.debug('Radii %.2f to %.2f', min_radius, max_radius)
    
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.debug('No main rings in image - returning null model')
        return None 

    model = RING_OCCULTATION_DATA(radii)

    saturn_shadow = bp.where_inside_shadow('saturn:ring', 'saturn').vals
    model[saturn_shadow] = 0
    
    if not np.any(model):
        logger.debug('Rings are entirely shadowed - returning null model')
        return None
    
    return model

"""
- Profiles at different lighting geometries, including phase, incidence, emission,
and lit/unlit.
- Occultation profiles for transparency.
- When combining models with the rings in front, use the occultation for transparency.
"""
