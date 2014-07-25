import cb_logging
import logging

import numpy as np
import numpy.ma as ma

from pdstable import PdsTable
from tabulation import Tabulation
from cb_util_oops import *

LOGGING_NAME = 'cb.' + __name__

RING_OCCULTATION_TABLE = PdsTable('IS2_P0001_V01_KM002.LBL')
RING_OCCULTATION_DATA = Tabulation(RING_OCCULTATION_TABLE.column_dict['RING_INTERCEPT_RADIUS'],
                                   RING_OCCULTATION_TABLE.column_dict['I_OVER_F'])

def rings_create_model(obs, bp, offset_u=0., offset_v=0.):
    logger = logging.getLogger(LOGGING_NAME+'.rings_create_model')

    radii = bp.ring_radius('saturn:ring').vals.astype('float')
    
    logger.debug('Radii %.2f to %.2f', np.min(radii), np.max(radii)) 

    model = RING_OCCULTATION_DATA(radii)

    saturn_shadow = bp.where_inside_shadow('saturn:ring', 'saturn').vals
    print saturn_shadow
    model[saturn_shadow] = 0
    
    return model

"""
- Profiles at different lighting geometries, including phase, incidence, emission,
and lit/unlit.
- Occultation profiles for transparency.
- When combining models with the rings in front, use the occultation for transparency.
"""