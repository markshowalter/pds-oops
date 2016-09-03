###############################################################################
# cb_rings.py
#
# Routines related to rings.
#
# Exported routines:
#    rings_create_model
#    rings_create_model_from_image
#    rings_sufficient_curvature
#    rings_fiducial_features
#
#    rings_fring_inertial_to_corotating
#    rings_fring_corotating_to_inertial
#    rings_fring_radius_at_longitude
#    rings_fring_longitude_radius
#    rings_fring_pixels
#
#    rings_longitude_radius_to_pixels
#    rings_generate_longitudes
#    rings_generate_radii
#
#    rings_reproject
#    rings_mosaic_init
#    rings_mosaic_add
###############################################################################

import cb_logging
import logging

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.ndimage.interpolation as ndinterp
import scipy.interpolate as sciinterp
from PIL import Image, ImageDraw, ImageFont

import polymath
import oops
import cspice
from pdstable import PdsTable
from tabulation import Tabulation

from cb_config import *
from cb_util_oops import *

_LOGGING_NAME = 'cb.' + __name__


RINGS_MIN_RADIUS = oops.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.SATURN_MAIN_RINGS[1]

_RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION = 0.02 * oops.RPD
_RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION = 5. # KM
_RINGS_DEFAULT_REPRO_ZOOM_AMT = 5
_RINGS_DEFAULT_REPRO_ZOOM_ORDER = 3

_RINGS_LONGITUDE_SLOP = 1e-6 * oops.RPD  # Must be smaller than any longitude 
_RINGS_RADIUS_SLOP    = 1e-6             # or radius resolution we'll be using
_RINGS_MAX_LONGITUDE  = oops.TWOPI-_RINGS_LONGITUDE_SLOP*2

# These are bodies that might cast shadows on the ring plane near equinox
_RINGS_SHADOW_BODY_LIST = ['ATLAS', 'PROMETHEUS', 'PANDORA', 'EPIMETHEUS', 
                           'JANUS', 'MIMAS', 'ENCELADUS', 'TETHYS']

#===============================================================================
#
# FIDUCIAL FEATURE INFORMATION
# 
# We have various ways to analyze the rings. In all cases there need to be a
# certain number of "fiducial features", edges that have predictable orbits.
#
# We use data from Dick French 2016. These include
#        A Ring edge - El Moutamid, et al. 2015, Icarus, TBD
#        C Ring features - Nicholson, et al. 2014, Icarus, 241, 373-396
#            "Noncircular features in Saturn's rings II: The C ring"
#        Cassini Division features - French, et al. 2016, TBD
#            "Noncircular features in Saturn's rings III: The Cassini 
#             Division"
#
#===============================================================================

_RINGS_FIDUCIAL_FEATURES_FRENCH2016_EPOCH = cspice.utc2et('2008-01-01 12:00:00')
_RINGS_FIDUCIAL_FEATURES_FRENCH2016 = [
    # (Feature type, (inner_data, outer_data))
    #    inner/outer_data: (mode, reset)
    #       If mode == 1, data: a, rms, ae, long_peri, rate_peri
    #       If mode  > 1, data: amplitude, phase, pattern speed
    
    # Features are listed in order from closer to Saturn to farther from
    # Saturn
    
    ###########################################################
    ### C RING - Nicholson et al, Icarus 241 (2014) 373-396 ###
    ###########################################################
    
    # No C ring features have inclinations
    
    # Colombo Gap - #487, #43 - 178 km
    ('GAP', 'Colombo',
                ((  1,  77747.89, 0.23,  3.11,  96.90,   22.57346),),  # IEG
                ((  1,  77926.01, 0.27,  4.89, 280.02,   22.57696),)), # OEG

    # Titan Ringlet - #63, #62 - 23 km
    ('RINGLET', 'Titan',
                ((  1,  77867.13, 0.62, 17.39, 270.54,   22.57503),    # IER
                 (  0,                   3.84,  40.93, 1391.16334),
                 ( -5,                   0.45,  60.87, 1692.06574),
                 ( -2,                   0.77,  30.21, 2109.40889)),
                ((  1,  77890.21, 0.94, 27.20, 270.70,   22.57562),    # OER
                 (  2,                   1.55, 172.61,  717.94917),
                 (  3,                   1.54, 110.60,  949.74161),
                 (  4,                   0.90,  80.30, 1065.62338))),

    # Maxwell Gap - #163, #164 - 267 km
    ('GAP', 'Maxwell',
                ((  1,  87342.77, 0.43,  0.00,   0.00,    0.00000),),  # IEG
                ((  1,  87610.12, 0.41,  1.11, 228.54,   14.69150),)), # OEG

    # Maxwell Ringlet - #61, #60 - 59 km
    ('RINGLET', 'Maxwell',
                ((  1,  87480.29, 0.23, 18.93,  55.60,   14.69572),),  # IER
                ((  1,  87539.36, 0.16, 58.02,  57.20,   14.69314),    # OER
                 (  2,                   0.19,  73.26,  599.52336),
                 (  4,                   0.29,  16.55,  891.94002))),

    # Bond Gap - #111, #110 - 37 km
    ('GAP', 'Bond',
                ((  1,  88686.01, 0.76,  0.00,   0.00,    0.00000),),  # IEG 
                ((  1,  88723.04, 0.30,  0.00,   0.00,    0.00000),)), # OEG
     
    # Bond Ringlet - #59, #58 - 17 km
    ('RINGLET', 'Bond',
                ((  1,  88701.89, 0.28,  0.00,   0.00,    0.00000),    # IER
                 (  0,                   0.17,  79.75, 1146.43579),
                 (  3,                   0.16,  88.49,  778.35308)),
                ((  1,  88719.24, 0.32,  0.00,   0.00,    0.00000),    # OER
                 (  2,                   1.08, 105.07,  587.29003),
                 (  3,                   0.55, 107.67,  778.34105),
                 (  4,                   0.47,  40.22,  873.86707),
                 (  5,                   0.30,  52.06,  931.20532),
                 (  6,                   0.30,  30.39,  969.40781),
                 (  7,                   0.41,  14.05,  996.70499))),

    # Dawes Ringlet - Not included because the edge is not sharp

    # Dawes Gap - #56, #112 - 20 km
    ('GAP', 'Dawes',
                ((  1,  90200.38, 0.75,  6.10,  69.24,   13.18027),    # IEG
                 (  2,                   5.27,  62.92,  572.50536),
                 (  3,                   1.46,  41.67,  758.94278),
                 (  5,                   0.89,  71.02,  908.10954)),
                ((  1,  90220.77, 0.32,  2.29, 241.79,   13.17088),    # OEG
                 (  2,                   0.43, 157.21,  572.48458))),

        
    #################################################################
    ### CASSINI DIVISION - French et al Icarus 274 (2016) 131-162 ###
    #################################################################

    #######################################################################                               
    ### B RING OUTER EDGE - Nicholson, et al. Icarus 227 (2014) 152-175 ###
    ### From Dick French                                                ###
    ### ringfit_v1.8.Sa025S-RF-V5697c.out                               ###
    #######################################################################                         

    # B ring outer edge is the same as the Huygens Gap inner edge    
    # Huygens Gap - #20 OEG
    ('GAP', 'Huygens',
                ((  1, 117569.41, 7.56, 22.89,  66.89,    5.08300),    # IEG
                 (  2,                  37.94, 117.29,  382.07925),
                 (  2,                  31.77, 164.83,  381.98729),
                 (  3,                  10.92,  22.06,  507.70626),
                 (  4,                   7.24,   7.52,  570.52601),
                 (  5,                   5.84,  66.45,  608.20820),
                 (  6,                   2.93,  99.17,  633.32360)),
                ((  1, 117930.90, 0.45,  2.20, 248.98,    5.03372),    # OEG
                 ( 91,                   0.44, 245.70,   -4.98425), # Incl
                 (  0,                   1.82, 143.78,  750.25165),
                 (  2,                   1.34,  76.25,  381.98386),
                 ( -4,                   0.35,  45.39,  942.82221),
                 ( -3,                   0.40,  84.88, 1005.34987),
                 ( -1,                   0.82,  67.16, 1505.51205))),

    # Huygens Ringlet - #54, #53 - 18 km
    ('RINGLET', 'Huygens',
                ((  1, 117805.55, 1.30, 27.81, 137.53,    5.02872),    # IER
                 ( 91,                   0.59, 115.57,   -4.98852), # Incl
                 (  2,                   2.09,  81.56,  381.98744),
                 (-10,                   0.74,  14.73,  831.59967),
                 ( -5,                   0.85,  43.95,  906.74404),
                 ( -4,                   1.85,  32.20,  944.31186),
                 ( -3,                   1.20,  84.52, 1006.92419),
                 ( -2,                   1.12, 168.38, 1132.15800)),
                # The first of the two fits
                ((  1, 117823.65, 1.50, 28.03, 141.77,    5.02587),    # OER
                 ( 91,                   0.58,  96.86,   -4.97462), # Incl
                 (  2,                   1.54, 105.43,  380.68870),
                 (  2,                   1.84,  71.33,  381.98878),
                 (  5,                   0.71,  26.40,  606.07903))),
                               
    # Strange Ringlet - #560, #561 - 2 km
    ('RINGLET', 'Strange',
                ((  1, 117907.04, 1.63,  7.63, 153.21,    5.00570),    # IER
                 ( 91,                   7.44, 117.20,   -4.97620), # Incl
                 (  0,                   2.42,   9.73,  750.48932),
                 (  2,                   3.75, 105.79,  380.25154),
                 (  2,                   1.29,  75.92,  381.97024),
                 (  3,                   2.49,  37.53,  505.33539),
                 (  5,                   1.06,  29.00,  605.39489)),
                ((  1, 117908.77, 1.99,  7.40, 153.83,    5.00735),    # OER
                 ( 91,                   7.39, 120.60,   -4.97938), # Incl
                 (  0,                   2.56,  19.83,  750.47956),
                 (  2,                   4.13, 106.14,  380.25204),
                 (  2,                   1.55,  77.43,  381.96666),
                 (  3,                   2.60,  33.24,  505.33449),
                 ( -1,                   1.14, 352.19, 1505.96783))),
                               
    # Herschel Gap - #19, #16 - 95 km
    ('GAP', 'Herschel',
                ((  1, 118188.42, 0.41,  8.27, 347.32,    4.97362),    # IEG
                 ( 91,                   0.34, 279.55,   -4.95092), # Incl
                 (  2,                   1.34,  95.16,  378.89248),
                 (  2,                   0.89,  82.69,  381.98160),
                 (  3,                   0.71,   5.26,  503.53264),
                 (  4,                   0.35,  89.58,  565.85131),
                 (  5,                   0.36,  71.76,  603.24143),
                 (  6,                   0.37,   4.74,  628.16426),
                 (  7,                   0.37,  45.09,  645.97255),
                 (  8,                   0.34,  31.40,  659.32551),
                 ( 10,                   0.25,  24.43,  678.02618)),
                # The first fit with the single m=1 mode
                ((  1, 118283.52, 0.15,  0.24, 127.45,    4.94609),    # OEG
                 ( 91,                   0.25,  57.07,   -4.92430), # Incl
                 (  0,                   1.27, 232.50,  746.91637),
                 (  2,                   0.67,  76.01,  381.98441),
                 ( -3,                   0.11,  96.61, 1000.83562),
                 ( -2,                   0.11, 144.86, 1125.34236),
                 ( -1,                   0.23, 197.18, 1498.79019))),
                               
    # Herschel Ringlet - #18, #17 - 29 km        
    ('RINGLET', 'Herschel',
                ((  1, 118234.30, 0.26,  1.49, 172.81,    4.96229),    # IER
                 ( 91,                   1.49, 274.14,   -4.92970), # Incl
                 (  0,                   0.32, 237.88,  747.36440),
                 (  2,                   0.69,  79.00,  381.98129)),
                ((  1, 118263.25, 0.35,  1.76, 264.77,    4.95659),    # OER
                 ( 91,                   2.12, 294.58,   -4.93101), # Incl
                 (  2,                   0.37,   6.80,  378.53785),
                 (  2,                   0.72,  77.59,  381.98303),
                 (  3,                   0.32,  32.76,  503.04880),
                 (  4,                   0.20,  76.13,  565.30770),
                 (  5,                   0.22,  54.38,  602.66098))),

    # Russell Gap - #123, #13 - 38 km
    ('GAP', 'Russell',
                ((  1, 118589.92, 0.25,  7.60, 236.73,    4.90922),    # IEG
                 (  2,                   0.23, 165.64,  376.94996),
                 (  2,                   0.51,  80.23,  381.98584),
                 (  3,                   0.25,  92.19,  500.95134)),
                ((  1, 118628.40, 0.09,  0.11,  73.68,    4.90829),    # OEG
                 (  2,                   0.47,  78.01,  381.98525))),

    # Jeffreys Gap - #120, #15 - 37 km
    ('GAP', 'Jeffreys',
                ((  1, 118929.63, 0.13,  3.26, 333.51,    4.85753),    # IEG
                 ( 91,                   0.17, 292.09,   -4.82576), # Incl
                 (  2,                   0.44,  75.98,  381.99151)),
                ((  1, 118966.70, 0.12,  0.08, 114.89,    4.80910),    # OEG
                 (  2,                   0.37,  74.90,  381.98629))),

    # Kuiper Gap - #119, #118 - 5 km
    ('GAP', 'Kuiper',
                ((  1, 119401.67, 0.16,  0.93,  19.55,    4.79845),    # IEG
                 ( 91,                   0.18, 226.51,   -4.78025), # Incl
                 (  2,                   0.25,  79.87,  381.98534)),
                ((  1, 119406.30, 0.13,  0.10, 220.24,    4.75654),    # OEG
                 (  2,                   0.29,  79.03,  381.98449))),

    # Laplace Gap - #115, #114 - 239 km
    ('GAP', 'Laplace',
                ((  1, 119844.78, 0.26,  3.25, 310.11,    4.72673),    # IEG
                 ( 91,                   0.25,  10.17,   -4.69233), # Incl
                 (  2,                   0.29,  82.70,  381.98579)),
                ((  1, 120085.65, 0.10,  1.34, 308.79,    4.71705),    # OEG
                 (  2,                   0.22,  76.51,  381.98639))),
                               
    # Laplace Ringlet - #14, #12 - 41 km
    ('RINGLET', 'Laplace',
                ((  1, 120036.53, 0.20,  1.19, 236.12,   4.71250),     # IER
                 (  0,                   2.22, 160.05, 730.64946),
                 (  2,                   0.27,  75.51, 381.98821),
                 ( -4,                   0.13,  19.39, 918.04749),
                 ( -2,                   0.74, 141.40,1100.71164),
                 ( -1,                   0.61, 193.60,1466.03143)),
                ((  1, 120077.75, 0.14,  2.79,  51.49,   4.72457),     # OER
                 (  2,                   0.62,  86.10, 369.88543),
                 (  2,                   0.22,  78.01, 381.98827),
                 (  3,                   0.42,  34.20, 491.60221),
                 (  4,                   0.25,  46.96, 552.46291),
                 (  6,                   0.12,  16.13, 613.31846))),
    
    # Bessel Gap - #127, #11 - 13 km
    ('GAP', 'Bessel',
                ((  1, 120231.17, 0.44,  1.78, 263.16,    4.68450),    # IEG
                 (  2,                   0.29,  73.47,  381.97875),
                 (  8,                   0.36,  10.70,  642.50158)),
                ((  1, 120243.71, 0.23,  0.64, 206.50,    4.68561),    # OEG
                 (  0,                   0.21, 350.29,  728.80170),
                 (  2,                   0.23,  75.99,  381.98988),
                 ( -1,                   0.14, 302.16, 1462.27593))),

    # Barnard Gap - #10, #9 - 12 km
    ('GAP', 'Barnard',
                ((  1, 120303.69, 0.43,  0.44, 200.07,    4.68212),    # IEG
                 (  2,                   0.61,  44.12,  368.82370),
                 (  2,                   0.25,  67.07,  381.99503),
                 (  3,                   1.31, 108.47,  490.20424),
                 (  4,                   1.64,  46.00,  550.89054),
                 (  5,                   1.36,  27.61,  587.28565),
                 (  6,                   0.59,  24.56,  611.58228),
                 (  7,                   0.55,  46.93,  628.91493),
                 (  8,                   0.30,  10.41,  641.92788),
                 (  9,                   0.71,   8.38,  652.04071),
                 ( 10,                   0.42,   1.75,  660.13055),
                 ( 13,                   0.36,  26.84,  676.93190)),
                ((  1, 120316.04, 0.11,  0.23, 166.62,    4.66313),    # OEG
                 (  2,                   0.22,  79.36,  381.98624),
                 (  5,                   0.19,  58.97,  587.28403))),
                               

    #################################################################                               
    ### A RING OUTER EDGE - Moutamid, et al. Icarus (TBD) XXX ###
    #################################################################                               

    # These also include the Keeler OEG to make a solid ringlet
    # Keeler OEG -> A Ring OER
    # See below
    
    # 2005 MAY 1 - 2005 AUG 1
    ('RINGLET_2005-MAY-1_2005-AUG-1', 'Keeler-A Ring OE',
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136767.20, 7.55,  0.00,   0.00,    0.00000),)),

    # 2006 JAN 1 - 2009 JULY 1
    ('RINGLET_2006-JAN-1_2009-JULY-1', 'Keeler-A Ring OE',
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),     
                ((  1, 136770.09, 1.78,  0.00,   0.00,    0.00000),
                 (  3,                   2.28,   8.31,  403.85329),
                 (  4,                   1.80,   6.64,  453.94649),
                 (  5,                   4.85,  60.92,  484.02086),
                 (  6,                   1.92,  17.10,  504.07364),
                 (  7,                  12.91,   4.15,  518.35437),
                 (  8,                   2.77,  23.89,  529.10426),
                 (  9,                   3.12,  30.56,  537.44541),
                 ( 10,                   1.51,  30.95,  544.12491),
                 ( 18,                   1.95,   4.62,  570.85163))),

    # 2010 JAN 1 - 2013 AUG 1
    ('RINGLET_2010-JAN-1_2013-AUG-1', 'Keeler-A Ring OE',
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136772.74, 4.25,  0.00,   0.00,    0.00000),
                 (  9,                   6.14,  27.17,  537.45029),
                 ( 12,                   7.94,  12.84,  554.11853))),

    # Fill in the other dates, when the Keeler gap edge is OK but the A ring edge
    # isn't. The A ring fits are circular from Dick French
    # ringfit_v1.8.Sa025S-RF-V4800circ.out
    ('RINGLET_1990-JAN-1_2005-MAY-1', 'Keeler-A Ring OE', # Before 2005-MAY-1
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136770.41, 10.18, 0.00,   0.00,    0.00000),)),
                                       
    ('RINGLET_2005-AUG-1_2006-JAN-1', 'Keeler-A Ring OE', # Between 2005-AUG-1 and 2006-JAN-1
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136770.41, 10.18, 0.00,   0.00,    0.00000),)),

    ('RINGLET_2009-JULY-1_2010-JAN-1', 'Keeler-A Ring OE', # Between 2009-JULY-1 and 2010-JAN-1
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136770.41, 10.18, 0.00,   0.00,    0.00000),)),

    ('RINGLET_2013-AUG-1_2030-JAN-1', 'Keeler-A Ring OE', # After 2013-AUG-1
                ((  1, 136522.08727, 1.005577, 0.9887854, 322.4368800,   2.9890023),),
                ((  1, 136770.41, 10.18, 0.00,   0.00,    0.00000),)),

    ########################################
    ### OTHER SORT-OF-CIRCULAR FEATURES  ###
    ### ringfit_v1.8.Sa025S-RF-V5351.out ###
    ########################################

    ### C RING ###
            
    # #135 - IEG unpaired
    ('GAP', None,
                ((  1,  74614.73965, 0.164346, 0.0000000,   0.0000000,   0.0000000),),
                None),

    # #144 - OER paired with #143, uncircular
    ('RINGLET', None,
                None,
                ((  1,  75988.65895, 0.142803, 0.0000000,   0.0000000,   0.0000000),
                 (  1,                         0.1345940, 101.5205612,  22.5997842))),

    # #40 - OER unpaired
    ('RINGLET', None,
                None,
                ((  1,  76261.77387, 0.158014, 0.0000000,   0.0000000,   0.0000000),
                 (  1,                         0.2666899,  97.8944271,  22.5677381))),

    # #39 - OER unpaired
    ('RINGLET', None,
                None,
                ((  1,  77162.11501, 0.127343, 0.0000000,   0.0000000,   0.0000000),
                 (  1,                         0.5803149,  95.1273754,  22.5786349))),

    # #38, #37 - 41 km
    ('RINGLET', None,
                ((  1,  79222.04152, 0.113388, 0.0000000,   0.0000000,   0.0000000),
                 (  1,                         0.2341692, 287.4580718,  22.5734575)),
                ((  1,  79262.91082, 0.112969, 0.0000000,   0.0000000,   0.0000000),
                 (  1,                         0.1892659, 284.0750895,  22.5732605))),

    # #35, #34
    ('RINGLET', None,
                ((  1, 84751.77410, 0.159019, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 84947.29467, 0.137759, 0.0000000,   0.0000000,   0.0000000),)),

    # #33, #42
    ('RINGLET', None,
                ((  1, 85661.96178, 0.113055, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 85757.24532, 0.176563, 0.0000000,   0.0000000,   0.0000000),)),

    # #31 - IER unpaired
    ('RINGLET', None,
                ((  1, 85923.69993, 0.104101, 0.0000000,   0.0000000,   0.0000000),),
                None),

    # #30 - IER paired with #29, uncircular
    ('RINGLET', None,
                ((  1, 86373.17361, 0.171814, 0.0000000,   0.0000000,   0.0000000),),
                None),

    # #28 - OER unpaired
    ('RINGLET', None,
                None,
                ((  1, 88592.73625, 0.115294, 0.0000000,   0.0000000,   0.0000000),)),

    # #27, #41 - 103 km
    ('RINGLET', None,
                ((  1, 89190.58600, 0.086866, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 89294.05618, 0.181465, 0.0000000,   0.0000000,   0.0000000),)),

    # #26, #25 - 148 km
    ('RINGLET', None,
                ((  1, 89789.57831, 0.081971, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 89937.78988, 0.141344, 0.0000000,   0.0000000,   0.0000000),)),

    # #24, #23 - 508 km
    ('RINGLET', None,
                ((  1, 90406.12658, 0.094128, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 90613.79856, 0.156829, 0.0000000,   0.0000000,   0.0000000),)),

    ### B RING ###
    
    # #77 - OEG unpaired
    ('GAP', None,
                None,
                ((  1, 100024.40676, 0.117471, 0.0000000,   0.0000000,   0.0000000),)),

    # #74 - OEG paired with #75, uncircular
    ('GAP', None,
                None,
                ((  1, 101743.40857, 0.110368, 0.0000000,   0.0000000,   0.0000000),)),

    # #73 - OER unpaired
    ('RINGLET', None,
                None,
                ((  1, 103008.64563, 0.114512, 0.0000000,   0.0000000,   0.0000000),)),

    # #71 - OEG paired with #72, uncircular
    ('GAP', None,
                None,
                ((  1, 104082.65168, 0.144307, 0.0000000,   0.0000000,   0.0000000),)),



# XXX We don't know what to do with these yet
#    # #272
#    ('RINGLET', ((  1, 99363.03967, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #277
#    ('RINGLET', ((  1, 99576.11351, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #280
#    ('RINGLET', ((  1, 99738.56217, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #284
#    ('RINGLET', ((  1, 99865.92191, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #286
#    ('RINGLET', ((  1, 100420.17808, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #287
#    ('RINGLET', ((  1, 100451.97168, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #288
#    ('RINGLET', ((  1, 101081.94349, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #289
#    ('RINGLET', ((  1, 101190.08197, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #292
#    ('RINGLET', ((  1, 101379.30068, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #293
#    ('RINGLET', ((  1, 101482.86849, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #296
#    ('RINGLET', ((  1, 101879.53499, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #299
#    ('RINGLET', ((  1, 102122.39406, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #302
#    ('RINGLET', ((  1, 102231.65659, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #303
#    ('RINGLET', ((  1, 102245.52922, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #304
#    ('RINGLET', ((  1, 102257.61110, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #305
#    ('RINGLET', ((  1, 102283.10732, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #306
#    ('RINGLET', ((  1, 102291.06343, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #308
#    ('RINGLET', ((  1, 102405.68758, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #309
#    ('RINGLET', ((  1, 102454.79241, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #311
#    ('RINGLET', ((  1, 102578.80565, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #313
#    ('RINGLET', ((  1, 102618.48930, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #314
#    ('RINGLET', ((  1, 102622.17113, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #323
#    ('RINGLET', ((  1, 103260.33884, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #325
#    ('RINGLET', ((  1, 103340.83908, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #326
#    ('RINGLET', ((  1, 103448.48530, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #328
#    ('RINGLET', ((  1, 103452.13918, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #329
#    ('RINGLET', ((  1, 103536.20766, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #331
#    ('RINGLET', ((  1, 103772.49882, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #332
#    ('RINGLET', ((  1, 103778.78296, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #348
#    ('RINGLET', ((  1, 76457.67497, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #352
#    ('RINGLET', ((  1, 102303.00173, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #353
#    ('RINGLET', ((  1, 102305.84060, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#
#    # #140 - ????
#    ('RINGLET', ((  1, 75845.19820, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #145 - ????
#    ('RINGLET', ((  1, 76043.29517, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #153
#    ('RINGLET', ((  1, 77349.11966, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #162
#    ('RINGLET', ((  1, 87291.76260, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #176
#    ('RINGLET', ((  1, 92366.30175, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #177
#    ('RINGLET', ((  1, 92376.71329, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #178
#    ('RINGLET', ((  1, 92395.16613, 0.0000000,   0.0000000,   0.0000000),),
#                None),
#
#    # #179
#    ('RINGLET', ((  1, 92452.82163, 0.0000000,   0.0000000,   0.0000000),),
#                None),


    ########################################
    ### OTHER SORT-OF-CIRCULAR FEATURES  ###
    ### ringfit_v1.8.Sa025S-RF-V5577.out ###
    ########################################

    ### A RING ###
    
    # 1 - Keeler OEG unpaired
    # This is paired with the A ring outer edge above

    # Keeler IEG. Note this would normally be paired with the Keeler OEG above,
    # but it's better to call the area between the OEG and the A ring OER a
    # ringlet so it gets filled in in the model.
    # From Dec 2013 CMF-V4687
    ('GAP', 'Keeler',
                ((  1, 136484.91000, 3.620000, 0.0000000,   0.0000000,   0.0000000),),
                None),
    
    # #4, #3 - Encke IEG/OEG
    ('GAP', 'Encke',
                ((  1, 133423.23793, 0.948856, 0.0000000,   0.0000000,   0.0000000),),
                ((  1, 133744.83759, 0.808255, 0.0000000,   0.0000000,   0.0000000),)),

    # #7 - A ring IER
    ('RINGLET', 'A Ring',
                ((  1, 122050.07651, 1.135272, 0.0000000,   0.0000000,   0.0000000),),
                None),
]



#===============================================================================
#
# PROFILE INFORMATION
# 
# We have various ways to create a model of the rings based on radius and
# longitude. We can use radial I/F scans, radial occultation scans, or
# manufacture a model from edge orbit information.
#
# Which method to use is controlled by the 'model_source' parameter in
# rings_config. The options are:
#
#    'voyager' - The Voyager I/F profile from IS2_P0001_V01_KM002
#    'uvis' - A UVIS stellar occulation
#    'manufacturer' - Manufacture a model using orbit information. This
#        is only valid if using the french1601 fiducial feature list.
#
#===============================================================================

_RINGS_UVIS_OCCULTATION = 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM'


#==============================================================================
# 
# RING MODEL UTILITIES
#
#==============================================================================

### Curvature ###

def rings_sufficient_curvature(obs, extend_fov=(0,0), rings_config=None):
    """Determine if the rings in an image have sufficient curvature."""
    
    logger = logging.getLogger(_LOGGING_NAME+'.rings_sufficient_curvature')

    if rings_config is None:
        rings_config = RINGS_DEFAULT_CONFIG

    set_obs_ext_bp(obs, extend_fov)
        
    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    longitudes = obs.ext_bp.ring_longitude('saturn:ring').vals.astype('float') 
    min_longitude = np.min(longitudes)
    max_longitude = np.max(longitudes)

    logger.debug('Radii %.2f to %.2f Longitudes %.2f to %.2f',
                 min_radius, max_radius, 
                 min_longitude*oops.DPR, max_longitude*oops.DPR)
    
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.debug('No main rings in image - returning bad curvature')
        return False

    min_radius = max(min_radius, RINGS_MIN_RADIUS)
    max_radius = min(max_radius, RINGS_MAX_RADIUS)
    
    # Find the approximate radius with the greatest span of longitudes
    
    best_len = 0
    best_radius = None
    best_longitudes = None

    radius_step = (max_radius-min_radius) / 10.
    longitude_step = (max_longitude-min_longitude) / 100.
    for radius in rings_generate_radii(min_radius, max_radius, 
                                       radius_resolution=radius_step):
        trial_longitudes = rings_generate_longitudes(
                                         min_longitude, max_longitude,
                                         longitude_resolution=longitude_step)
        trial_radius = np.empty(trial_longitudes.shape)
        trial_radius[:] = radius
        (new_longitudes, new_radius,
         u_pixels, v_pixels) = _rings_restrict_longitude_radius_to_obs(
                                         obs, trial_longitudes, trial_radius,
                                         extend_fov=extend_fov)
        if len(new_longitudes) > best_len:
            best_radius = radius
            best_longitudes = new_longitudes
            best_len = len(new_longitudes)
    
    if best_len == 0:
        logger.debug('No valid longitudes - returning bad curvature')
        return False    
    
    logger.debug('Optimal radius %.2f longitude range %.2f to %.2f',
                 best_radius, best_longitudes[0]*oops.DPR,
                 best_longitudes[-1]*oops.DPR)
    
    # Now for this optimal radius, find the pixel values of the minimum
    # and maximum available longitudes as well as a point halfway between.
    # Then see how far it is from the line between the extrema points
    # to the point of closest approach for the halfway point.
    
    line_radius = np.empty(3)
    line_radius[:] = best_radius
    line_longitude = np.empty(3)
    line_longitude[0] = best_longitudes[0]
    line_longitude[2] = best_longitudes[-1]
    line_longitude[1] = (line_longitude[0]+line_longitude[2])/2
    
    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                      obs, line_longitude, line_radius)
    
    logger.debug('Linear pixels %.2f,%.2f / %.2f,%.2f / %.2f,%.2f',
                 u_pixels[0], v_pixels[0], u_pixels[1], v_pixels[1],
                 u_pixels[2], v_pixels[2])
    
    mid_pt_u = (u_pixels[0]+u_pixels[2])/2
    mid_pt_v = (v_pixels[0]+v_pixels[2])/2
    
    dist = np.sqrt((mid_pt_u-u_pixels[1])**2+(mid_pt_v-v_pixels[1])**2)
    
    if dist < rings_config['curvature_threshold']:
        logger.debug('Distance %.2f is too close for curvature', dist)
        return False
    
    logger.debug('Distance %.2f is far enough for curvature', dist)
    return True


### Fiducial Features ###

def _find_resolutions_by_a(obs, extend_fov, a):
    """Find the minimum and maximum resolutions at a given semi-major axis."""
    
    resolutions = (obs.ext_bp.ring_radial_resolution('saturn:ring').vals.
                   astype('float'))
    
    longitudes = rings_generate_longitudes(longitude_resolution=0.1*oops.RPD)

    new_long, new_rad, u, v = _rings_restrict_longitude_radius_to_obs(
                              obs, longitudes, a, extend_fov=extend_fov)

    u = u.astype('int')
    v = v.astype('int')
    feature_res = resolutions[v,u]
    if len(feature_res) == 0:
        return None, None
    min_res = np.min(feature_res)
    max_res = np.max(feature_res)
    
    return min_res, max_res

def _fiducial_is_ok(obs, feature, min_radius, max_radius, rms_gain, blur, 
                    min_sub_radius, max_sub_radius, extend_fov):
    """Decide if a fiducial feature can be used based on its RMS residual.
    
    If the residual is too large, return the amount the image would need to be
    blurred to make it OK.
    """
    if blur is None:
        blur = 1.
    a = feature[0][1]
    rms = feature[0][2]
    out_of_frame = False
    if not min_sub_radius < a < max_sub_radius:
        if not min_radius < a < max_radius:
            # It's not in the extended FOV at all
            return None, None
        # It's not in the extended FOV, but is in the original image
        out_of_frame = True
    min_res, max_res = _find_resolutions_by_a(obs, extend_fov, a)
    if max_res is None:
        return None, None # Something went wrong; we couldn't find a resolution
    if rms*rms_gain/blur > max_res:
        # Additional blurring needed
        return (rms*rms_gain/blur)/max_res, out_of_frame
    return 1., out_of_frame # OK as is
    
def rings_fiducial_features(obs, extend_fov=(0,0), rings_config=None):
    """Return a list of fiducial features in the image.
    
    We first attempt to return at least fiducial_feature_threshold features
    that satisfy resolution requirements. If we can't, then we relax the
    constraints imposed by the RMS residuals until we can and return the 
    amount that we had to relax.
    
    The number of features is based on how many are in the subset of the
    FOV that is guaranteed to be in the image. But the feature list
    returned actually includes all the features that are in the entire
    extended FOV in case they end up being useful.
    """
    
    logger = logging.getLogger(_LOGGING_NAME+'.rings_fiducial_features')

    if rings_config is None:
        rings_config = RINGS_DEFAULT_CONFIG

    set_obs_ext_bp(obs, extend_fov)

    min_features = rings_config['fiducial_feature_threshold']
    min_feature_width = rings_config['fiducial_min_feature_width']

    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    min_sub_radius = np.min(radii[extend_fov[1]*2:-extend_fov[1]*2,
                                  extend_fov[0]*2:-extend_fov[0]*2])
    max_sub_radius = np.max(radii[extend_fov[1]*2:-extend_fov[1]*2,
                                  extend_fov[0]*2:-extend_fov[0]*2])

    logger.debug('Looking for features based on RMS residual vs. resolution')
    logger.debug('Extended FOV radii %.2f to %.2f', min_radius, max_radius)
    logger.debug('Internal FOV radii %.2f to %.2f', min_sub_radius, max_sub_radius) 

    resolutions = (obs.ext_bp.ring_radial_resolution('saturn:ring').vals.
                   astype('float'))

    min_res = np.min(resolutions)
    max_res = np.max(resolutions)
    
    logger.debug('Resolution %.2f to %.2f', min_res, max_res)
    
    rms_gain = rings_config['fiducial_rms_gain']

    # Make a restricted feature list based on the observation date
    restricted_feature_list = []
    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES_FRENCH2016:
        entry_type_str, feature_name, inner, outer = fiducial_feature
        entry_type_list = entry_type_str.split('_')
        entry_type = entry_type_list[0]
        # Eliminate any features that don't match the observation date
        if len(entry_type_list) > 1:
            assert len(entry_type_list) == 3
            start_date = cspice.utc2et(entry_type_list[1])
            end_date = cspice.utc2et(entry_type_list[2])
            if not (start_date < obs.midtime < end_date):
                continue
        restricted_feature_list.append(fiducial_feature)
    
    blur = None
    blur_list = []
    
    # We do two phases - the first one is to determine the amount of blur, if
    # any. If there is no blur needed, we exit the loop early. Otherwise, 
    # during the second phase we apply the blur.
    for allow_blur in [False,True]:
        logger.debug('Allow blur %s', str(allow_blur))
        
        feature_list = []
        best_blur = None
        num_features = 0
        num_features_in_frame = 0
        for fiducial_feature in restricted_feature_list:
            entry_type_str, feature_name, inner, outer = fiducial_feature
            entry_type_list = entry_type_str.split('_')
            entry_type = entry_type_list[0]
            pretty_name = feature_name
            if pretty_name is None:
                pretty_name = 'UNNAMED'
            
            inner_out_of_frame = False
            outer_out_of_frame = False
            ret_inner = None
            ret_outer = None
            if inner is not None:
                ret_inner, inner_out_of_frame = _fiducial_is_ok(
                                            obs, inner, min_radius, max_radius, 
                                            rms_gain, blur, 
                                            min_sub_radius, max_sub_radius,
                                            extend_fov)
                if ret_inner is None or abs(ret_inner-1.) > 0.000001:
                    # If more blurring is required, then it's not a good 
                    # candidate during this phase.
                    inner = None
            if outer is not None:
                if pretty_name == 'Keeler-A Ring OE':
                    pass
                ret_outer, outer_out_of_frame = _fiducial_is_ok(
                                            obs, outer, min_radius, max_radius, 
                                            rms_gain, blur, 
                                            min_sub_radius, max_sub_radius,
                                            extend_fov)
                if ret_outer is None or abs(ret_outer-1.) > 0.000001:
                    # If more blurring is required, then it's not a good 
                    # candidate during this phase
                    outer = None

            # Features aren't THAT big, so just use the resolution for the 
            # inner or outer, whichever we have
            if inner is not None:       
                (min_res_inner_outer, 
                 max_res_inner_outer) = _find_resolutions_by_a(
                                              obs, extend_fov, inner[0][1])
            elif outer is not None:
                (min_res_inner_outer, 
                 max_res_inner_outer) = _find_resolutions_by_a(
                                              obs, extend_fov, outer[0][1])

            # If there is a full gap or ringlet, see if it's big enough to be
            # readily seen.
            force_keep_feature = False  
            if inner is not None and outer is not None:
                # See if the gap or ringlet is too small to be seen in
                # the image
                feature_width = ((outer[0][1]-inner[0][1]) / 
                                 min_res_inner_outer)
                if feature_width < min_feature_width:
                    logger.debug(
                         'Ignoring complete %s %s %.2f to %.2f - '+
                         'feature too narrow (%.2f pixels)',
                         pretty_name, entry_type, inner[0][1], outer[0][1],
                         feature_width)
                    # However, in the special case of the Huygen Gap,
                    # we can go ahead and keep the inner edge because that's
                    # the B ring outer edge and it's really visible.
                    if pretty_name == 'Huygens' and entry_type == 'GAP':
                        outer = None
                        force_keep_feature = True
                    # We do the same thing for the A Ring outer edge
                    elif (pretty_name.startswith('Keeler-A Ring') and 
                          entry_type == 'RINGLET'):
                        inner = None
                        force_keep_feature = True
                    else:
                        continue

            # For single-sided features, see if there is another feature that
            # is too close. If there is, then throw away the current feature
            # because it won't be readily distinguishable in the real image.
            
            if (inner is None) != (outer is None) and not force_keep_feature:
                feature = inner
                if feature is None:
                    feature = outer
                    
                bad = False
                for fiducial_feature2 in restricted_feature_list:
                    (entry_type_str2, feature_pretty_name, inner2, 
                     outer2) = fiducial_feature2
                    if fiducial_feature == fiducial_feature2:
                        continue
                    # If the feature we're going to compare against is itself
                    # too narrow to be seen, then don't bother with the 
                    # comparison
                    if inner2 is not None and outer2 is not None:
                        feature_width = ((outer2[0][1]-inner2[0][1]) / 
                                         min_res_inner_outer)
                        if feature_width < min_feature_width:
                            continue
                    for inner_outer2 in [inner2, outer2]:
                        if inner_outer2 is None:
                            continue
                        feature_dist_pix = abs(inner_outer2[0][1]-
                            feature[0][1]) / min_res_inner_outer
                        if (feature_dist_pix < min_feature_width):
                            in_out_str = 'inner'
                            if inner_outer2 == outer2:
                                in_out_str = 'outer'
                            logger.debug(
                             'Ignoring partial %s %s %s %.2f - '+
                             'feature too close to another '+
                             '(a=%.2f, %.2f pixels)',
                             in_out_str, pretty_name, entry_type,
                             feature[0][1],
                             inner_outer2[0][1], feature_dist_pix)
                            bad = True
                    if bad:
                        break
                if bad:
                    if inner is not None:
                        inner = None
                        ret_inner = None
                    else:
                        outer = None
                        ret_outer = None
                
            # Everything is going well...we can at least use this to determine
            # the blur amount
            if ret_inner is not None and ret_inner != 1.:
                blur_list.append(ret_inner)
            if ret_outer is not None and ret_outer != 1.:
                blur_list.append(ret_outer)

            # And maybe we can even use it as a real feature right now
            if inner is not None or outer is not None:
                feature_list.append((entry_type, feature_name, inner, outer))

            if inner is not None:
                logger.debug(
                     'Keeping inner feature %s %s %.2f (out of frame %s)', 
                     pretty_name, entry_type, inner[0][1],
                     str(inner_out_of_frame))
                num_features += 1
                if not inner_out_of_frame:
                    num_features_in_frame += 1
            if outer is not None:
                logger.debug(
                     'Keeping outer feature %s %s %.2f (out of frame %s)', 
                     pretty_name, entry_type, outer[0][1],
                     str(outer_out_of_frame))
                if outer is not None:
                    num_features += 1
                if not outer_out_of_frame:
                    num_features_in_frame += 1

        if num_features_in_frame >= min_features:
            logger.debug('Found enough (%d vs. %d) features in frame',
                         num_features_in_frame, min_features)
            break

        logger.debug('%d features, %d in frame', num_features,
                     num_features_in_frame)

        if allow_blur or len(blur_list) == 0:
            break

        blur_list.sort()
        
        # We allow this process to continue even if there are too few
        # features because the main offset loop will attempt to use
        # an insufficient number of features at a lower confidence.
        
        blur = blur_list[min(min_features-num_features_in_frame-1,
                             len(blur_list)-1)]

    if blur is not None:
        blur = blur/rms_gain
            
    logger.debug('Returning %d fiducial features, %d in frame, blur %s',
                 num_features, num_features_in_frame, str(blur))

    return num_features_in_frame, feature_list, blur


# Given an I/F or occultation profile, blur out the areas that aren't
# associated with fiducial features.

_RING_VOYAGER_IF_DATA = None
_RING_UVIS_OCC_DATA = None

def _blur_ring_radial_data(tab, resolution):
    min_blur = 10. # km - Don't blur anything closer than this
    max_blur = 100. # km
    blur_copy = 10
    radial_resolution = 1.
    domain = tab.domain()
    start_radius, end_radius = domain
    new_radius = rings_generate_radii(start_radius, end_radius, 
                                      radius_resolution=radial_resolution)
    data = tab(new_radius)
    blurred_data = data.copy()
    
    # Blur 2-sigma over 100 km (50 km each side)
#    _data = filt.gaussian_filter(data, blur_sigma)
#    blurred_data = np.zeros(data.shape)
#    # Now copy in the original data if near a fiducial feature
#    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES:
#        location, resolution = fiducial_feature
#        x = location - domain[0]
#        x1 = max(x-blur_copy, 0)
#        x2 = min(x+blur_copy, domain[1])
#        blurred_data[x1:x2+1] = data[x1:x2+1]

    # Compute the distance to the nearest fiducial feature
    fiducial_dist = np.zeros(new_radius.shape) + 1e38
    for fiducial_feature in _RINGS_FIDUCIAL_FEATURES:
        location, resolution = fiducial_feature
        temp_dist = np.zeros(new_radius.shape) + location
        temp_dist = np.abs(temp_dist-new_radius)
        fiducial_dist = np.minimum(fiducial_dist, temp_dist)
    
    num_blur = int((max_blur-min_blur) / radial_resolution)
    for dist in xrange(num_blur):
        blur_amt = float(dist)
        one_blur = filt.gaussian_filter(data, blur_amt)
        replace_bool = fiducial_dist >= (dist+min_blur)
        blurred_data[replace_bool] = one_blur[replace_bool]
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
##    plt.plot(new_radius, fiducial_dist, '-', color='black')
#    plt.plot(new_radius, data, '-', color='#ff4040')
#    plt.plot(new_radius, blurred_data, '-', color='black')
#    plt.show()
    
    return blurred_data, start_radius, end_radius, radial_resolution

def _compute_ring_radial_data(source, resolution):
    assert source in ('voyager', 'uvis')
    
    if source == 'voyager':
        global _RING_VOYAGER_IF_DATA
        if not _RING_VOYAGER_IF_DATA:
            if_table = PdsTable(os.path.join(CB_SUPPORT_FILES_ROOT,
                                             'IS2_P0001_V01_KM002.LBL'))
            _RING_VOYAGER_IF_DATA = Tabulation(
                   if_table.column_dict['RING_INTERCEPT_RADIUS'],
                   if_table.column_dict['I_OVER_F'])
        tab = _RING_VOYAGER_IF_DATA

    if source == 'uvis':
        global _RING_UVIS_OCC_DATA
        if not _RING_UVIS_OCC_DATA:
            occ_root = os.path.join(COUVIS_8XXX_ROOT, 'COUVIS_8001', 'DATA',
                                    'EASYDATA')
            occ_file = os.path.join(occ_root, _RINGS_UVIS_OCCULTATION+'.LBL')
            label = PdsTable(occ_file)
            data = (1.-np.e**-label.column_dict['NORMAL OPTICAL DEPTH'])
            _RING_UVIS_OCC_DATA = Tabulation(
                   label.column_dict['RING_RADIUS'],
                   data)
        tab = _RING_UVIS_OCC_DATA

    ret = _blur_ring_radial_data(tab, resolution)
    
    return ret


def _make_ephemeris_radii(obs, descr_list):
    orig_radii = obs.ext_bp.ring_radius('saturn:ring')

    last_radii = orig_radii
        
    for descr in descr_list:
        if len(descr) == 6: # First entry
            mode, main_a, rms, amp, long_peri, rate_peri = descr
        else:
            mode, amp, long_peri, rate_peri = descr

        if mode > 90:
            mode -= 90
                    
        last_radii = obs.ext_bp.radial_mode(
                            last_radii.key,
                            mode, _RINGS_FIDUCIAL_FEATURES_FRENCH2016_EPOCH, 
                            amp, long_peri*oops.RPD, 
                            rate_peri*oops.RPD/86400.)

    return last_radii

def _shade_antialias(radii, a, shade_above, resolutions, max=1.):
    # The anti-aliasing shade
    # If we're shading the main object above, then the anti-aliasing
    # is shaded below!
    if shade_above:
        shade_sign = -1.
    else:
        shade_sign = 1.

    shade = 1.-shade_sign*(radii-a)/resolutions
    # Since the radius is actually the middle of a pixel, when radii==a the
    # shading should be 0.5! Only if a is 0.5*resolution beyond the radius
    # should it be 1.
    shade -= 0.5
    
    shade[shade < 0.] = 0.
    shade[shade > 1.] = 0.
    shade *= max
    
    return shade
    
def _shade_model(model, radii, a, shade_above, radius_width_km, resolutions,
                 feature_list_by_a):
    # shade_above == True means shade towards larger a
    # The primary shade - the hard edge and shade away from it
    logger = logging.getLogger(_LOGGING_NAME+'._shade_model')

    if shade_above:
        shade_sign = 1.
    else:
        shade_sign = -1.
        
    # Check to see if there is another feature that's going to be in the way
    for other_a, other_type in feature_list_by_a:
        if (shade_sign*(other_a-a) > 0 and 
            shade_sign*(other_a-a) < radius_width_km):
            # Change the width of the feature so it doesn't conflict
            radius_width_km = abs(other_a-a)/2
            logger.debug(
                 'Fixing conflicting feature at %.2f vs. %.2f, new width %.2f', 
                 a, other_a, radius_width_km)
    shade = 1.-shade_sign*(radii-a)/radius_width_km
    shade[shade < 0.] = 0.
    shade[shade > 1.] = 0.
    
    shade_anti = _shade_antialias(radii, a, shade_above, resolutions)
    
    return model + shade + shade_anti

def _compute_model_ephemeris(obs, feature_list, label_avoid_mask, 
                             in_front_mask, extend_fov, rings_config):
    logger = logging.getLogger(_LOGGING_NAME+'._compute_model_ephemeris')

    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    longitudes = obs.ext_bp.ring_longitude('saturn:ring').vals.astype('float')
    resolutions = (obs.ext_bp.ring_radial_resolution('saturn:ring').vals.
                   astype('float'))
    
    min_radius = np.min(radii)
    max_radius = np.max(radii)

    min_res = np.min(resolutions)
    
    radius_width_pix = rings_config['fiducial_ephemeris_width']
    radius_width_km = radius_width_pix * min_res
    
    model = np.zeros((obs.data.shape[0]+extend_fov[1]*2,
                      obs.data.shape[1]+extend_fov[0]*2),
                     dtype=np.float32)
    model_text = np.zeros(model.shape, dtype=np.bool)
    text_im = Image.frombuffer('L', (model_text.shape[1], 
                                     model_text.shape[0]), 
                               model_text, 'raw', 'L', 0, 1)
    text_draw = ImageDraw.Draw(text_im)
    
    if label_avoid_mask is not None:
        label_avoid_mask = label_avoid_mask.copy()
    else:
        label_avoid_mask = np.zeros(model.shape, dtype=np.bool)

    # Mask out all the areas around the edge that might be off the image
    # once the offset is applied
#     if extend_fov[0] != 0:
#         label_avoid_mask[:,:extend_fov[0]*2] = True
#         label_avoid_mask[:,-extend_fov[0]*2:] = True
#     if extend_fov[1] != 0:
#         label_avoid_mask[:extend_fov[1]*2,:] = True
#         label_avoid_mask[-extend_fov[1]*2:,:] = True
        
    # This gives a margin around bodies and also solves a problem with
    # border_atop while masked with bodies
    in_front_mask = filt.maximum_filter(in_front_mask, 5)

    # Create an array of distances from the center pixel - always try to put
    # text labels as close to the center as possible to minimize the chance
    # of going off the edge.
    axis1 = np.tile(np.arange(float(model.shape[1]))-model.shape[1]//2,
                    model.shape[0]).reshape(model.shape)
    axis2 = np.repeat(np.arange(float(model.shape[0]))-model.shape[0]//2,
                      model.shape[1]).reshape(model.shape)
    dist_from_center = axis1**2+axis2**2
    # Don't do anything too close to the top or bottom edges because text
    # has height
    dist_from_center[:5,:] = 1e38
    dist_from_center[-5:,:] = 1e38
        
    feature_list_by_a = []
    for entry_type, feature_name, inner, outer in feature_list:
        if inner is not None:
            feature_list_by_a.append((inner[0][1], 
                                      'IEG' if entry_type == 'GAP' else 'IER'))
        if outer is not None:
            feature_list_by_a.append((outer[0][1], 
                                      'OEG' if entry_type == 'GAP' else 'OER'))
    feature_list_by_a.sort(key=lambda x: x[0], reverse=True)

    # Do gaps first, because gaps might actually have ringlets inside of them.
    # Then go back and add the ringlets, which might fill in the gaps.
    for do_type in ['GAP', 'RINGLET']:
        for entry_type, feature_name, inner, outer in feature_list:
            pretty_name = feature_name
            if pretty_name is None:
                pretty_name = 'UNNAMED'
            if entry_type != do_type:
                continue
            inner_radii = None
            inner_radii_bp = None
            outer_radii = None  
            outer_radii_bp = None
            text_name_list = []
            intersect_list = []
            if (inner is not None and outer is not None and
                entry_type == 'RINGLET'):
                # If we have a full ringlet, find both edges even if one
                # of them is off the image so we can fill it in solid
                # later
                inner_a = inner[0][1]
                outer_a = outer[0][1]
                if (min_radius < inner_a < max_radius or
                    min_radius < outer_a < max_radius):
                    inner_radii_bp = _make_ephemeris_radii(obs, inner)
                    outer_radii_bp = _make_ephemeris_radii(obs, outer)
            else:
                if inner is not None:
                    inner_a = inner[0][1]
                    if min_radius < inner_a < max_radius:
                        inner_radii_bp = _make_ephemeris_radii(obs, inner)
                if outer is not None:
                    outer_a = outer[0][1]
                    if min_radius < outer_a < max_radius:
                        outer_radii_bp = _make_ephemeris_radii(obs, outer)
            if inner_radii_bp is not None:
                inner_radii = inner_radii_bp.vals.astype('float')
                inner_radii[in_front_mask] = 0.
            if outer_radii_bp is not None:
                outer_radii = outer_radii_bp.vals.astype('float')
                outer_radii[in_front_mask] = 0.
            if (inner_radii is not None and outer_radii is not None
                and entry_type == 'RINGLET'):
                # We have both edges for a ringlet - just make it solid
                logger.debug('Adding %s %s a=%.2f to %.2f', 
                             pretty_name, entry_type, inner_a, outer_a)
                inner_above = inner_radii >= inner_a
                outer_below = outer_radii <= outer_a
                intersect = np.logical_and(inner_above, outer_below)
                intersect[in_front_mask] = False
                model[intersect] += 1.
                shade = _shade_antialias(inner_radii, inner_a, False,
                                         resolutions, max=0.5)
                model += shade
                shade = _shade_antialias(outer_radii, outer_a, True,
                                         resolutions, max=0.5)
                model += shade
                if (feature_name and 
                    feature_name.upper().startswith('KEELER-A RING')):
                    # We need to fake this one out - it's really the Keeler
                    # OEG and the A Ring outer edge
                    intersect = obs.ext_bp.border_atop(
                                          inner_radii_bp.key,
                                          inner_a).vals.astype('bool')
                    intersect[in_front_mask] = False
                    intersect_list.append(intersect)
                    text_name_list.append('Keeler OEG')
                    intersect = obs.ext_bp.border_atop(
                                          outer_radii_bp.key,
                                          outer_a).vals.astype('bool')
                    intersect[in_front_mask] = False
                    intersect_list.append(intersect)
                    text_name_list.append('A Ring Outer Edge')                        
                else:
                    intersect_list.append(intersect)
                    if feature_name:
                        text_name_list.append(feature_name + ' ' + 
                                              entry_type.capitalize())
                    else:
                        text_name_list.append('a=%.2f-%.2f %s' % 
                              (inner_a, outer_a, entry_type.capitalize()))
            else:
                named_full_gap = False
                if (inner_radii is not None and outer_radii is not None and
                    entry_type == 'GAP'):
                    if (feature_name and 
                        feature_name.upper().startswith('HUYGENS')):
                        # We need to fake this one out - it's really the B Ring
                        # OE and the Huygens Gap outer edge
                        intersect = obs.ext_bp.border_atop(
                                              inner_radii_bp.key,
                                              inner_a).vals.astype('bool')
                        intersect[in_front_mask] = False
                        intersect_list.append(intersect)
                        text_name_list.append('B Ring Outer Edge')
                        intersect = obs.ext_bp.border_atop(
                                              outer_radii_bp.key,
                                              outer_a).vals.astype('bool')
                        intersect[in_front_mask] = False
                        intersect_list.append(intersect)
                        text_name_list.append('Huygens OEG')                        
                        named_full_gap = True
                    else:
                        # Go ahead and name the full gap here even though we're 
                        # going to shade the edges separately later
                        inner_above = inner_radii >= inner_a
                        outer_below = outer_radii <= outer_a
                        intersect = np.logical_and(inner_above, outer_below)
                        intersect[in_front_mask] = False
                        intersect_list.append(intersect)
                        if feature_name:
                            text_name_list.append(feature_name + ' Gap')
                        else:
                            text_name_list.append('a=%.2f-%.2f Gap' % (inner_a, 
                                                                       outer_a))
                        named_full_gap = True
                    # Fall through to...
                if inner_radii is not None:
                    # Isolated inner ringlet edge or isolated/full gap edge
                    shade_above = entry_type == 'RINGLET'
                    logger.debug('Adding %s inner %s a=%.2f shade_above %d',
                                 pretty_name, entry_type, inner_a, shade_above)
                    model = _shade_model(model, inner_radii, inner_a, 
                                         shade_above, radius_width_km, 
                                         resolutions, feature_list_by_a)
                    if not named_full_gap:
                        intersect = obs.ext_bp.border_atop(
                                              inner_radii_bp.key,
                                              inner_a).vals.astype('bool')
                        intersect[in_front_mask] = False
                        intersect_list.append(intersect)
                        if (feature_name and 
                            feature_name.upper().startswith('KEELER-A RING')):
                            text_name_list.append('Keeler OEG')
                        else:
                            feature_name_sfx = ('IER' if entry_type == 'RINGLET' 
                                                      else 'IEG')
                            if feature_name:
                                text_name_list.append(feature_name + ' ' +
                                                      feature_name_sfx) 
                            else:
                                text_name_list.append(('a=%.2f' % inner_a) + 
                                                      ' ' + feature_name_sfx)                     
                if outer_radii is not None:
                    # Isolated outer ringlet edge or isolated/full gap edge
                    shade_above = entry_type == 'GAP'
                    logger.debug('Adding %s outer %s a=%.2f shade_above %d',
                                 pretty_name, entry_type, outer_a, shade_above)
                    model = _shade_model(model, outer_radii, outer_a, 
                                         shade_above, radius_width_km, 
                                         resolutions, feature_list_by_a)
                    if not named_full_gap:
                        intersect = obs.ext_bp.border_atop(
                                              outer_radii_bp.key,
                                              outer_a).vals.astype('bool')
                        intersect[in_front_mask] = False
                        intersect_list.append(intersect)
                        if (feature_name and 
                            feature_name.upper().startswith('KEELER-A RING')):
                            text_name_list.append('A Ring Outer Edge')
                        else:
                            feature_name_sfx = ('OER' if entry_type == 'RINGLET' 
                                                      else 'OEG')
                            if feature_name:
                                text_name_list.append(feature_name + ' ' +
                                                      feature_name_sfx) 
                            else:
                                text_name_list.append(('a=%.2f' % outer_a) + 
                                                      ' ' + feature_name_sfx)                     

            # Now find a good place for each label that doesn't overlap other
            # labels and doesn't overlap text from previous model steps
            for text_name, intersect in zip(text_name_list, intersect_list):
                text_size = text_draw.textsize(text_name+'->')
                dist = dist_from_center.copy()
                dist[np.logical_not(intersect)] = 1e38
                first_u = None
                first_v = None
                first_text = None
                u = None
                v = None
                while np.any(dist != 1e38):
                    # Find the closest good pixel to the image center
                    v,u = np.unravel_index(np.argmin(dist), dist.shape)
                    raw_v, raw_u = v,u
                    if (text_size[1]/2 <= v <
                        model_text.shape[0]-text_size[1]/2):
                        v -= text_size[1]/2
                        if u > model_text.shape[1]/2:
                            u = u-text_size[0]
                            text = text_name + '->'
                        else:
                            text = '<-' + text_name
                        if first_u is None:
                            first_u = u
                            first_v = v
                            first_text = text
                        # Check for conflicts with existing labels
                        if (not np.any(
                          label_avoid_mask[
                                   max(v-3,0):
                                   min(v+text_size[1]+3, 
                                       label_avoid_mask.shape[0]),
                                   max(u-3,0):
                                   min(u+text_size[0]+3, 
                                       label_avoid_mask.shape[1])])):
                            break
                    # We're overlapping with existing text! Or V is too close
                    # to the edge.
                    dist[max(raw_v-3,0):
                         min(raw_v+3,model_text.shape[0]),
                         max(raw_u-3,0):
                         min(raw_u+3,model_text.shape[1])] = 1e38
                    u = None
                    
                if u is None:
                    # We give up - just use the first one we found
                    u = first_u
                    v = first_v
                    text = first_text
                
                if u is not None:
                    text_draw.text((u,v), text, fill=1)
                    label_avoid_mask[v:v+text_size[1],
                                     u:u+text_size[0]] = True

    model_text = (np.array(text_im.getdata()).astype('bool').
                  reshape(model_text.shape))

    return model, model_text

#===============================================================================
# 
# CREATE THE MODEL
#
#===============================================================================

def rings_create_model(obs, extend_fov=(0,0), always_create_model=False,
                       label_avoid_mask=None,
                       bodies_model_list=None,
                       rings_config=None):
    """Create a model for the rings.

    If there are no rings in the image or they are entirely in shadow,
    return None. Also return None if the curvature is insufficient or there
    aren't enough fiducial features in view and always_create_model is False.
    
    Inputs:
        obs                    The Observation.
        extend_fov             The amount beyond the image in the (U,V)
                               dimension to model rings.
        always_create_model    True to always return a model even if the 
                               curvature is insufficient or there aren't
                               enough fiducial features.
        label_avoid_mask       A mask giving places where text labels should
                               not be placed (i.e. labels from another
                               model are already there). None if no mask.
        bodies_model_list      A list of (body_model, body_metadata, body_text)
                               that is used to remove parts of the ring model 
                               that are behind a body. None if this is not
                               desired.
        rings_config           Configuration parameters. None uses the default.

    Returns:
        metadata           A dictionary containing information about the
                           offset result:
            'shadow_bodies'         The list of Bodies that shadow the rings.
                                    Only populated if rings_config allows
                                    shadow removal. 
            'curvature_ok'          True if the curvature is sufficient for 
                                    correlation.
            'emission_ok'           True if the emission angle is sufficient 
                                    for correlation.
            'num_good_fiducial_features'
                                    The number of fiducial features that are
                                    fully contained within the sub-image
                                    guaranteed to be visible even with
                                    extended_fov.
            'fiducial_features'     The list of fiducial features in the 
                                    extended FOV.
            'fiducial_blur'         The amount the RMS residual of the features
                                    had to be reduced in order to get enough.
                                    1 means nothing was reduced. Otherwise
                                    a number < 1.
            'fiducial_features_ok'  True if the number of fidcual features
                                    is greater than the threshold.
            'start_time'            The time (s) when rings_create_model
                                    was called.
            'end_time'              The time (s) when rings_create_model
                                    returned.

    """
    start_time = time.time()
    
    logger = logging.getLogger(_LOGGING_NAME+'.rings_create_model')

    if rings_config is None:
        rings_config = RINGS_DEFAULT_CONFIG
        
    assert rings_config['model_source'] in ('uvis', 'voyager', 'ephemeris')
    
    metadata = {}
    metadata['shadow_bodies'] = []
    metadata['curvature_ok'] = False
    metadata['emission_ok'] = False
    metadata['num_good_fiducial_features'] = 0
    metadata['fiducial_features'] = []
    metadata['fiducial_features_ok'] = False
    metadata['fiducial_blur'] = None
    metadata['start_time'] = start_time
    
    set_obs_ext_bp(obs, extend_fov)

    radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')

    min_radius = np.min(radii)
    max_radius = np.max(radii)
        
    logger.info('Radii %.2f to %.2f', min_radius, max_radius)
        
    if max_radius < RINGS_MIN_RADIUS or min_radius > RINGS_MAX_RADIUS:
        logger.info('No main rings in image - aborting')
        metadata['end_time'] = time.time()
        return None, metadata, None
    
    emission = obs.ext_bp.emission_angle('saturn:ring').vals.astype('float')
    good_mask = np.logical_and(radii >= RINGS_MIN_RADIUS,
                               radii <= RINGS_MAX_RADIUS)
    min_emission = np.min(np.abs(emission[good_mask]*oops.DPR-90))
    if min_emission < rings_config['emission_threshold']:
        logger.info('Minimum emission angle %.2f from 90 too close to ring '+
                    'plane - aborting', min_emission)
        return None, metadata, None

    logger.debug('Minimum emission angle %.2f from 90 OK', min_emission)
    metadata['emission_ok'] = True

    if not rings_sufficient_curvature(obs, extend_fov=extend_fov, 
                                      rings_config=rings_config):
        logger.info('Too little curvature')
        if not always_create_model:
            metadata['end_time'] = time.time()
            return None, metadata, None
    else:
        logger.info('Curvature OK')
        metadata['curvature_ok'] = True     
   
    num_features, fiducial_features, fiducial_blur = rings_fiducial_features(
                                         obs, extend_fov, rings_config)
    metadata['num_good_fiducial_features'] = num_features
    metadata['fiducial_features'] = fiducial_features
    metadata['fiducial_blur'] = fiducial_blur
    
    fiducial_features_ok = (num_features >=
                            rings_config['fiducial_feature_threshold'])
    metadata['fiducial_features_ok'] = fiducial_features_ok
    
    if not fiducial_features_ok:
        logger.info('Insufficient number (%d) of in-frame fiducial features', 
                    num_features)
        if not always_create_model:
            metadata['end_time'] = time.time()
            return None, metadata, None
    else:
        logger.info('Enough fiducial features (%d)', num_features)

    shadow_body_list = []

    # Compute the shadows before the model so we can use them as
    # areas to avoid for labels
    shadows = None    
    if rings_config['remove_saturn_shadow']:
        saturn_shadow = obs.ext_bp.where_inside_shadow('saturn:ring',
                                                       'saturn').vals
        if np.any(saturn_shadow):
            shadow_body_list.append('SATURN')
            shadows = saturn_shadow
            if np.all(shadows):
                logger.info('Rings completely shadowed by SATURN - aborting')
                metadata['end_time'] = time.time()
                return None, metadata, None
            logger.info('Rings at least partially shadowed by SATURN')

    if rings_config['remove_body_shadows']:
        for body_name in _RINGS_SHADOW_BODY_LIST:
            body_shadow = obs.ext_bp.where_inside_shadow('saturn:ring',
                                                         body_name).vals
            if np.any(body_shadow):
                shadow_body_list.append(body_name)
                if shadows is None:
                    shadows = body_shadow
                else:
                    shadows = np.logical_or(shadows, body_shadow)
                if np.all(body_shadow):
                    logger.info('Rings completely shadowed by %s - aborting', 
                                body_name)
                    metadata['end_time'] = time.time()
                    return None, metadata, None
                logger.info('Rings at least partially shadowed by %s', body_name)
    
    metadata['shadow_bodies'] = shadow_body_list
    
    model_source = rings_config['model_source']
    if model_source != 'ephemeris':
        assert False # Not currently tested
        radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
        radii = radii.copy()
        
        radii[radii < RINGS_MIN_RADIUS] = 0
        radii[radii > RINGS_MAX_RADIUS] = 0
        
        ret = _compute_ring_radial_data(model_source, 0.) # XXX
        radial_data, start_radius, end_radius, radial_resolution = ret
    
        radii[radii < start_radius] = 0
        radii[radii > end_radius] = 0
    
        radial_index = np.round((radii-start_radius)/radial_resolution)
        radial_index = np.clip(radial_index, 0, radial_data.shape[0]-1)
        radial_index = radial_index.astype('int')
        model = radial_data[radial_index]
        model_text = np.zeros(model.shape, dtype=np.uint8)
    else:
        in_front_mask = np.zeros(radii.shape, dtype=np.bool)
        if bodies_model_list is not None:
            # Erase from the model any location where the rings are further from
            # the observer than a body
            rings_dist = obs.ext_bp.distance('saturn:ring').vals.astype('float')
            for body_model, body_metadata, body_text in bodies_model_list:
                body_dist = body_metadata['inventory']['range']
                in_front_mask[((rings_dist>body_dist) & 
                               (body_model != 0))] = True
    
        if shadows is not None:
            label_avoid_mask = np.logical_or(label_avoid_mask, shadows)
        model, model_text = _compute_model_ephemeris(obs, fiducial_features,
                                                     label_avoid_mask,
                                                     in_front_mask,
                                                     extend_fov, rings_config)
    
    if not np.any(model):
        logger.info('Model is empty - aborting')
        metadata['end_time'] = time.time()
        return None, metadata, None

    if shadows is not None:
        model[shadows] = 0
            
        if not np.any(model):
            logger.info('Rings are entirely shadowed - returning null model')
            metadata['end_time'] = time.time()
            return None, metadata, None
    
    metadata['end_time'] = time.time()
    return model, metadata, model_text

def rings_create_model_from_image(obs, extend_fov=(0,0), data=None):
    """Create a model for the rings from a radial scan of this image.
    
    Inputs:
        obs                    The Observation.

    Returns:
        The model.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.rings_create_model_from_image')

    if data is None:
        data = obs.data
    
    # We really want to use the normal BP, but we use the extended one because
    # the BP will be computed elsewhere and we will save time by only
    # computing it once. We can extract the normal radius BP by taking
    # a subarray of the extended one.
    set_obs_ext_bp(obs, extend_fov)

    bp_radii = obs.ext_bp.ring_radius('saturn:ring').vals.astype('float')
    bp_radii = bp_radii[extend_fov[1]:extend_fov[1]+data.shape[0],
                        extend_fov[0]:extend_fov[0]+data.shape[1]]
    min_radius = np.clip(np.min(bp_radii), RINGS_MIN_RADIUS, RINGS_MAX_RADIUS)
    max_radius = np.clip(np.max(bp_radii), RINGS_MIN_RADIUS, RINGS_MAX_RADIUS)

    diag = np.sqrt(data.shape[0]*data.shape[1])
    
    radius_resolution = (max_radius-min_radius) / diag

    # XXX This will have a problem with wrap-around
    bp_longitude = obs.ext_bp.ring_longitude('saturn:ring').vals.astype('float')
    bp_longitude = bp_longitude[extend_fov[1]:extend_fov[1]+data.shape[0],
                                extend_fov[0]:extend_fov[0]+data.shape[1]]
    min_longitude = np.min(bp_longitude)
    max_longitude = np.max(bp_longitude)
    longitude_resolution = (max_longitude-min_longitude) / diag

    logger.info('Image radii %.2f to %.2f res %.3f / '+
                'longitudes %.2f to %.2f, res %.5f', 
                min_radius, max_radius, radius_resolution,
                min_longitude*oops.DPR, max_longitude*oops.DPR,
                longitude_resolution*oops.DPR)

    if min_radius == max_radius:
        return None
    
    reproj = rings_reproject(
            obs, data=data,
            radius_range=(min_radius,max_radius),
            radius_resolution=radius_resolution,
            longitude_range=(min_longitude,max_longitude),
            longitude_resolution=longitude_resolution,
#            zoom_amt=1,
            omit_saturns_shadow=False,
            image_only=True,
            mask_fill_value=None)
    
    reproj_img = reproj['img']
    radial_scan = ma.median(reproj_img, axis=1)
    
    radii = rings_generate_radii(min_radius, max_radius, 
                                 radius_resolution=radius_resolution)
    
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    bp_radii[bp_radii < min_radius] = min_radius
    bp_radii[bp_radii > max_radius] = max_radius

    interp = sciinterp.interp1d(radii, radial_scan)
    model = interp(bp_radii)
    model[bp_radii > max_radius] = 0.
    
    return model

#==============================================================================
# 
# RING REPROJECTION UTILITIES
#
#==============================================================================

##
# Non-F-ring-specific routines
##


def rings_generate_longitudes(longitude_start=0.,
                              longitude_end=_RINGS_MAX_LONGITUDE,
                              longitude_resolution=
                                    _RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION):
    """Generate a list of longitudes.
    
    The list will be on longitude_resolution boundaries and is guaranteed to
    not contain a longitude less than longitude_start or greater than
    longitude_end."""
    longitude_start = (np.ceil(longitude_start/longitude_resolution) *
                       longitude_resolution)
    longitude_end = (np.floor(longitude_end/longitude_resolution) *
                     longitude_resolution)
    return np.arange(longitude_start, longitude_end+_RINGS_LONGITUDE_SLOP,
                     longitude_resolution)

def rings_generate_radii(radius_inner, radius_outer,
                         radius_resolution=
                             _RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION):
    """Generate a list of radii (km)."""
    return np.arange(radius_inner, radius_outer+_RINGS_RADIUS_SLOP,
                     radius_resolution)

def _rings_restrict_longitude_radius_to_obs(obs, longitude, radius,
                                            offset=None, extend_fov=None):
    """Restrict the list of longitude and radius to those present
    in the image. Also return the U,V coordinate of each longitude,radius
    pair."""
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    if radius.shape == ():
        new_radius = np.empty(longitude.shape)
        new_radius[:] = radius
        radius = new_radius 
    
    offset_u = 0
    offset_v = 0
    if offset is not None:
        offset_u, offset_v = offset
        
    set_obs_ext_bp(obs, extend_fov)
    bp = obs.ext_bp
    u_min = 0
    v_min = 0
    u_max = obs.data.shape[1]-1 + obs.extend_fov[0]*2
    v_max = obs.data.shape[0]-1 + obs.extend_fov[1]*2

    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float')
    
    min_bp_radius = np.min(bp_radius)
    max_bp_radius = np.max(bp_radius)
    min_bp_longitude = np.min(bp_longitude)
    max_bp_longitude = np.max(bp_longitude)
    
    # First pass restriction so rings_longitude_radius_to_pixels will run
    # faster
    goodr = np.logical_and(radius >= min_bp_radius, radius <= max_bp_radius)
    goodl = np.logical_and(longitude >= min_bp_longitude,
                           longitude <= max_bp_longitude)
    good = np.logical_and(goodr, goodl)
    
    radius = radius[good]
    longitude = longitude[good]

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                                  obs, longitude, radius)

    u_pixels += offset_u
    v_pixels += offset_v
        
    # Catch the cases that fell outside the image boundaries
    goodumask = np.logical_and(u_pixels >= u_min, u_pixels <= u_max)
    goodvmask = np.logical_and(v_pixels >= v_min, v_pixels <= v_max)
    good = np.logical_and(goodumask, goodvmask)
    
    radius = radius[good]
    longitude = longitude[good]
    u_pixels = u_pixels[good]
    v_pixels = v_pixels[good]
    
    return longitude, radius, u_pixels, v_pixels
    
def rings_longitude_radius_to_pixels(obs, longitude, radius, corotating=None):
    """Convert longitude,radius pairs to U,V."""
    assert corotating in (None, 'F')
    longitude = np.asarray(longitude)
    radius = np.asarray(radius)
    
    if corotating == 'F':
        longitude = rings_fring_corotating_to_inertial(longitude, obs.midtime)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    ring_surface = oops.Body.lookup('SATURN_RING_PLANE').surface
    obs_event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                         polymath.Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = ring_surface.photon_to_event_by_coords(obs_event,
                                                          (radius,longitude))

    uv = obs.fov.uv_from_los(obs_event.neg_arr_ap)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals

##
# F ring routines
##

FRING_ROTATING_ET = None
FRING_MEAN_MOTION = 581.964 * oops.RPD # rad/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 24.2 * oops.RPD # deg
FRING_DW = 2.70025 * oops.RPD # deg/day                

def _compute_fring_longitude_shift(et):
    global FRING_ROTATING_ET
    if FRING_ROTATING_ET is None:
        FRING_ROTATING_ET = cspice.utc2et("2007-1-1")
 
    return - (FRING_MEAN_MOTION * 
              ((et - FRING_ROTATING_ET) / 86400.)) % oops.TWOPI

def rings_fring_inertial_to_corotating(longitude, et):
    """Convert inertial longitude to corotating."""
    return (longitude + _compute_fring_longitude_shift(et)) % oops.TWOPI

def rings_fring_corotating_to_inertial(co_long, et):
    """Convert corotating longitude (deg) to inertial."""
    return (co_long - _compute_fring_longitude_shift(et)) % oops.TWOPI

def rings_fring_radius_at_longitude(obs, longitude):
    """Return the radius (km) of the F ring core at a given inertial longitude
    (deg)."""
    curly_w = FRING_W0 + FRING_DW*obs.midtime/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(longitude-curly_w)))

    return radius
    
def rings_fring_longitude_radius(obs, longitude_step=0.01*oops.RPD):
    """Return  a set of longitude,radius pairs for the F ring core."""
    num_longitudes = int(oops.TWOPI / longitude_step)
    longitudes = np.arange(num_longitudes) * longitude_step
    radius = rings_fring_radius_at_longitude(obs, longitudes)
    
    return longitudes, radius

def rings_fring_pixels(obs, offset=None, longitude_step=0.01*oops.RPD):
    """Return a set of U,V pairs for the F ring in an image."""
    longitude, radius = rings_fring_longitude_radius(
                                     obs,
                                     longitude_step=longitude_step)
    
    (longitude, radius,
     u_pixels, v_pixels) = _rings_restrict_longitude_radius_to_obs(
                                     obs, longitude, radius,
                                     offset=offset)
    
    return u_pixels, v_pixels


#==============================================================================
# 
# RING REPROJECTION MAIN ROUTINES
#
#==============================================================================

def rings_reproject(
            obs, data=None, offset=None,
            longitude_resolution=_RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
            longitude_range=None,
            radius_resolution=_RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION,
            radius_range=None,
            zoom_amt=_RINGS_DEFAULT_REPRO_ZOOM_AMT,
            zoom_order=_RINGS_DEFAULT_REPRO_ZOOM_ORDER,
            corotating=None,
            uv_range=None,
            compress_longitude=True,
            mask_fill_value=0.,
            omit_saturns_shadow=True,
            image_only=False):
    """Reproject the rings in an image into a rectangular longitude/radius
    space.
    
    Inputs:
        obs                      The Observation.
        data                     The image data to use for the reprojection. If
                                 None, use obs.data.
        offset                   The offsets in (U,V) to apply to the image
                                 when computing the longitude and radius
                                 values.
        longitude_resolution     The longitude resolution of the new image
                                 (rad/pix).
        longitude_range          None, or a tuple (start,end) specifying the
                                 longitude limits to reproject.
        radius_resolution        The radius resolution of the new image
                                 (km/pix).
        radius_range             None, or a tuple (inner,outer) specifying the
                                 radius limits to reproject.
        zoom                     The amount to magnify the original image for
                                 pixel value interpolation.
        corotating               The name of the ring to use to compute
                                 co-rotating longitude. None if inertial
                                 longitude should be used.
        uv_range                 None, or a tuple (start_u,end_u,start_v,end_v)
                                 that defines the part of the image to be
                                 reprojected.
        compress_longitude       True to compress the returned image to contain
                                 only valid longitudes. False to return the
                                 entire range 0-2PI or as specified by
                                 longitude_range.
        mask_fill_value          What to replace masked values with. None means
                                 leave the values masked.
        omit_saturns_shadow      True to mask out pixels that are in Saturn's
                                 shadow.
        image_only               True to only include image pixel data. False 
                                 to also include incidence, emission, phase,
                                 and resolution.
                                 
    Returns:
        A dictionary containing
        
        'long_mask'            The mask of longitudes from the full 2PI-radian
                               set that contain reprojected data. This can be
                               used to recreate the list of actual longitudes
                               present.
        'time'                 The midtime of the observation (TDB).
        'radius_resolution'    The radius resolution.
        'longitude_resolution' The longitude resolution.
                           
            The following only contain longitudes with mask values of True
            above. All angles are in degrees.
        'img'                  The reprojected image [radius,longitude].
        
            If data_only is False:
        
        'resolution'           The radial resolution [radius,longitude].
        'phase'                The phase angle [radius,longitude].
        'emission'             The emission angle [radius,longitude].
        'incidence'            The incidence angle. Note this is a scalar
                               because it doesn't change over the ring plane.
        'mean_resolution'      The radial resolution averaged over all radii
                               [longitude].
        'mean_phase'           The phase angle averaged over all radii 
                               [longitude].
        'mean_emission'        The emission angle averaged over all radii
                               [longitude].
        
        The image data is taken from the zoomed, interpolated image,
        while the incidence, emission, phase, and resolution are taken from
        the original non-interpolated data and thus will be slightly more
        coarse-grained.
    """
    logger = logging.getLogger(_LOGGING_NAME+'.rings_reproject')
    
    assert corotating in (None, 'F')

    if data is None:
        data = obs.data
    
    # Radius defaults to the entire main rings
    # Longitude defaults to the entire circle
    if radius_range is None:
        radius_inner = RINGS_MIN_RADIUS
        radius_outer = RINGS_MAX_RADIUS
    else:
        radius_inner, radius_outer = radius_range
        
    if longitude_range is None:
        longitude_start = 0.
        longitude_end = _RINGS_MAX_LONGITUDE
    else:
        longitude_start, longitude_end = longitude_range

    # Offset the image if we can
    orig_fov = None
    if offset is not None and offset != (0,0):
        # We need to be careful not to use obs.bp from this point forward
        # because it will disagree with our current OffsetFOV
        orig_fov = obs.fov
        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    
    # Get all the info for each pixel, restricted to uv_range if provided
    meshgrid = None
    start_u = 0
    end_u = data.shape[1]-1
    start_v = 0
    end_v = data.shape[0]-1
    if uv_range is not None:
        start_u, end_u, start_v, end_v = uv_range
        meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                     origin=(start_u+.5, start_v+.5), 
                                     limit=(end_u+.5, end_v+.5), swap=True)

    if orig_fov is None and meshgrid is None:
        # No offset and no uv_range means it's safe to use the normal Backplane
        # We would prefer to do this for performance since it might already be
        # cached
        set_obs_bp(obs)
        bp = obs.bp
    else:
        bp = oops.Backplane(obs, meshgrid)
        
    bp_radius = bp.ring_radius('saturn:ring').vals.astype('float')
    bp_longitude = bp.ring_longitude('saturn:ring').vals.astype('float')
    
    if not image_only: 
        bp_resolution = (bp.ring_radial_resolution('saturn:ring')
                         .vals.astype('float'))
        bp_phase = bp.phase_angle('saturn:ring').vals.astype('float')
        bp_emission = bp.emission_angle('saturn:ring').vals.astype('float') 
        bp_incidence = bp.incidence_angle('saturn:ring').vals.astype('float')

    # Clear out any pixels in Saturn's shadow
    if omit_saturns_shadow:
        saturn_shadow = bp.where_inside_shadow('saturn:ring','saturn').vals
        data = data.copy()
        data[saturn_shadow] = 0

    # Deal with co-rotating longitudes
    if corotating == 'F':
        bp_longitude = rings_fring_inertial_to_corotating(bp_longitude, 
                                                          obs.midtime)
    
    # The number of pixels in the final reprojection in the radial direction
    radius_pixels = int(np.ceil((radius_outer-radius_inner+
                                 _RINGS_RADIUS_SLOP) / radius_resolution))

    # The total number of pixels in the longitude direction if the whole ring
    # (or at least the ring longitudes specified by the caller) is visible
    full_min_longitude_pixel = (np.floor(longitude_start /
                                         longitude_resolution)).astype('int')
    full_max_longitude_pixel = (np.floor(longitude_end / 
                                         longitude_resolution)).astype('int')
    full_longitude_pixels = (full_max_longitude_pixel - 
                             full_min_longitude_pixel + 1)
    
    # The longitudes restricted to the user-supplied range, if any
    restr_bp_longitude = bp_longitude[
              np.logical_and(bp_longitude >= longitude_start,
                             bp_longitude <= longitude_end)]
    
    # The bin numbers for each longitude in the image
    bp_longitude_binned = np.floor((restr_bp_longitude-longitude_start) / 
                                   longitude_resolution).astype('int')
                                   
    # Mark which longitude bins are going to be used - True if that longitude
    # exists in the image
    full_good_long_bins_mask = np.zeros(full_longitude_pixels, dtype=np.bool)
    full_good_long_bins_mask[bp_longitude_binned] = True

    # Longitude bin numbers before we limit to what's actually in the image
    full_long_bins = np.arange(full_longitude_pixels)
     
    # The actual set of longitude bins we're going to use
    long_bins_restr = full_long_bins[full_good_long_bins_mask]
    longitude_pixels = len(long_bins_restr)
    
    # The bin numbers go from 0->N-1, but the actual bin values
    # are taken sparesly form the complete set 0->2PI based on what longitudes
    # are in the image
    long_bins = np.tile(np.arange(longitude_pixels), radius_pixels) 
    long_bins_act = (np.tile(long_bins_restr, radius_pixels) * 
                     longitude_resolution + longitude_start)
        
    # Radius bin numbers
    rad_bins = np.repeat(np.arange(radius_pixels), longitude_pixels)
    # Actual radius for each bin (km)
    if corotating == 'F':
        rad_bins_offset = rings_fring_radius_at_longitude(obs,
                              rings_fring_corotating_to_inertial(long_bins_act,
                                                                 obs.midtime))        
        rad_bins_act = (rad_bins * radius_resolution + radius_inner +
                        rad_bins_offset)
        logger.debug('Radius offset range %.2f %.2f',
                     np.min(rad_bins_offset),
                     np.max(rad_bins_offset))
    else:
        rad_bins_act = rad_bins * radius_resolution + radius_inner

    logger.info('Radius range %.2f %.2f', np.min(bp_radius), 
                 np.max(bp_radius))
    logger.debug('Radius bin range %.2f %.2f', np.min(rad_bins_act), 
                 np.max(rad_bins_act))
    logger.debug('Longitude bin range %.2f %.2f', 
                 np.min(long_bins_act)*oops.DPR,
                 np.max(long_bins_act)*oops.DPR)
    logger.debug('Number of longitude bins %d', longitude_pixels)
    if not image_only:
        logger.info('Resolution range %.2f %.2f', np.min(bp_resolution),
                     np.max(bp_resolution))
    logger.debug('Data range %f %f', np.min(data), np.max(data))

    u_pixels, v_pixels = rings_longitude_radius_to_pixels(
                                                  obs, long_bins_act,
                                                  rad_bins_act,
                                                  corotating=corotating)
    
    # Zoom the data and restrict the bins and pixels to ones actually in the
    # final reprojection.
    if zoom_amt == 1:
        zoom_data = data
    else:
        zoom_data = ndinterp.zoom(data, zoom_amt, order=zoom_order)

    u_zoom = (u_pixels*zoom_amt).astype('int')
    v_zoom = (v_pixels*zoom_amt).astype('int')
    
    goodumask = np.logical_and(u_pixels >= start_u, 
                               u_zoom <= (end_u+1)*zoom_amt-1)
    goodvmask = np.logical_and(v_pixels >= start_v, 
                               v_zoom <= (end_v+1)*zoom_amt-1)
    goodmask = np.logical_and(goodumask, goodvmask)
    
    u_pixels = u_pixels[goodmask].astype('int') - start_u
    v_pixels = v_pixels[goodmask].astype('int') - start_v
    u_zoom = u_zoom[goodmask]
    v_zoom = v_zoom[goodmask]
    good_rad_bins = rad_bins[goodmask]
    good_long_bins = long_bins[goodmask]
    
    interp_data = zoom_data[v_zoom, u_zoom]
    
    # Create the reprojected results.
    repro_img = ma.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    repro_img.mask = True
    repro_img[good_rad_bins,good_long_bins] = interp_data

    # Mean will mask if ALL radii are masked at a particular longitude
    # We should do this more efficiently using operations on the mask 
    # itself
    good_long_bins_mask = np.logical_not(ma.getmaskarray(ma.mean(repro_img,
                                                                 axis=0)))
    
    repro_img = repro_img[:,good_long_bins_mask]

    if not image_only:
        repro_res = ma.zeros((radius_pixels, longitude_pixels), 
                             dtype=np.float32)
        repro_res.mask = True
        repro_res[good_rad_bins,good_long_bins] = bp_resolution[v_pixels,
                                                                u_pixels]

        repro_res = repro_res[:,good_long_bins_mask]
        repro_mean_res = ma.mean(repro_res, axis=0)

        repro_phase = ma.zeros((radius_pixels, longitude_pixels), 
                               dtype=np.float32)
        repro_phase.mask = True
        repro_phase[good_rad_bins,good_long_bins] = bp_phase[v_pixels,u_pixels]
        repro_phase = repro_phase[:,good_long_bins_mask] 
        repro_mean_phase = ma.mean(repro_phase, axis=0)
    
        repro_emission = ma.zeros((radius_pixels, longitude_pixels), 
                                  dtype=np.float32)
        repro_emission.mask = True
        repro_emission[good_rad_bins,good_long_bins] = bp_emission[v_pixels,
                                                                   u_pixels]
        repro_emission = repro_emission[:,good_long_bins_mask]
        repro_mean_emission = ma.mean(repro_emission, axis=0)
    
        repro_incidence = ma.mean(bp_incidence[v_pixels,u_pixels])

    new_full_good_long_bins_mask = np.zeros(full_longitude_pixels, dtype=np.bool)
    new_full_good_long_bins_mask[long_bins_restr[good_long_bins_mask]] = True

    if mask_fill_value is not None:
        repro_img = ma.filled(repro_img, mask_fill_value)
        if not image_only:
            repro_res = ma.filled(repro_res, mask_fill_value)
            repro_phase = ma.filled(repro_phase, mask_fill_value)
            repro_emission = ma.filled(repro_emission, mask_fill_value)
            repro_incidence = ma.filled(repro_incidence, mask_fill_value)

    if orig_fov is not None:   
        obs.fov = orig_fov

    ret = {}    
    ret['time'] = obs.midtime
    ret['long_mask'] = new_full_good_long_bins_mask
    ret['img'] = repro_img
    if not image_only:
        ret['resolution'] = repro_res
        ret['phase'] = repro_phase
        ret['emission'] = repro_emission
        ret['incidence'] = repro_incidence
        ret['mean_resolution'] = repro_mean_res
        ret['mean_phase'] = repro_mean_phase
        ret['mean_emission'] = repro_mean_emission
    
    return ret

def rings_mosaic_init(
        radius_range,
        longitude_resolution=_RINGS_DEFAULT_REPRO_LONGITUDE_RESOLUTION,
        radius_resolution=_RINGS_DEFAULT_REPRO_RADIUS_RESOLUTION):
    """Create the data structure for a ring mosaic.

    Inputs:
        longitude_resolution     The longitude resolution of the new image
                                 (rad/pix).
        radius_resolution        The radius resolution of the new image
                                 (km/pix).
        radius_range             None, or a tuple (inner,outer) specifying the
                                 radius limits of the new image.
                                 
    Returns:
        A dictionary containing an empty mosaic

        'img'                    The full mosaic image.
        'radius_resolution'      The radius resolution.
        'longitude_resolution'   The longitude resolution.
        'long_mask'              The valid-longitude mask (all False).
        'mean_resolution'        The per-longitude mean resolution.
        'mean_phase'             The per-longitude mean phase angle.
        'mean_emission'          The per-longitude mean emission angle.
        'mean_incidence'         The scalar mean incidence angle.
        'image_number'           The per-longitude image number giving the image
                                 used to fill the data for each longitude.
        'time'                   The per-longitude time (TDB).
    """
    if radius_range is None:
        radius_inner = _RINGS
    radius_inner, radius_outer = radius_range
    radius_pixels = int(np.ceil((radius_outer-radius_inner+_RINGS_RADIUS_SLOP) / 
                                radius_resolution))
    longitude_pixels = int(oops.TWOPI / longitude_resolution)
    
    ret = {}
    ret['img'] = np.zeros((radius_pixels, longitude_pixels), dtype=np.float32)
    ret['long_mask'] = np.zeros(longitude_pixels, dtype=np.bool)
    ret['mean_resolution'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_phase'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_emission'] = np.zeros(longitude_pixels, dtype=np.float32)
    ret['mean_incidence'] = 0.
    ret['image_number'] = np.zeros(longitude_pixels, dtype=np.int32)
    ret['time'] = np.zeros(longitude_pixels, dtype=np.float32)
    
    return ret

def rings_mosaic_add(mosaic_metadata, repro_metadata, image_number):
    """Add a reprojected image to an existing mosaic.
    
    For each valid longitude in the reprojected image, it is copied to the
    mosaic if it has more valid radial data, or the same amount of radial
    data but the resolution is better.
    """
    mosaic_metadata['mean_incidence'] = repro_metadata['incidence']
    
    radius_pixels = mosaic_metadata['img'].shape[0]
    repro_good_long = repro_metadata['long_mask']
    mosaic_good_long = mosaic_metadata['long_mask']
        
    # Create full-size versions of all the longitude-compressed reprojection
    # data.
    mosaic_img = mosaic_metadata['img']
    repro_img = np.zeros(mosaic_img.shape) 
    repro_img[:,repro_good_long] = repro_metadata['img']
    mosaic_res = mosaic_metadata['mean_resolution']
    repro_res = np.zeros(mosaic_res.shape) + 1e300
    repro_res[repro_good_long] = repro_metadata['mean_resolution']
    mosaic_phase = mosaic_metadata['mean_phase']
    repro_phase = np.zeros(mosaic_phase.shape)
    repro_phase[repro_good_long] = repro_metadata['mean_phase']
    mosaic_emission = mosaic_metadata['mean_emission']
    repro_emission = np.zeros(mosaic_emission.shape)
    repro_emission[repro_good_long] = repro_metadata['mean_emission']
    mosaic_image_number = mosaic_metadata['image_number']
    image_time = repro_metadata['time']
    mosaic_time = mosaic_metadata['time']
    
    # Calculate the number of good entries and where the number is larger than
    # in the existing mosaic.
    mosaic_valid_radius_count = radius_pixels-np.sum(mosaic_img == 0., axis=0)
    new_valid_radius_count = radius_pixels-np.sum(repro_img == 0., axis=0)
    valid_radius_count_better_mask = (new_valid_radius_count > 
                                      mosaic_valid_radius_count)
    valid_radius_count_equal_mask = (new_valid_radius_count == 
                                     mosaic_valid_radius_count)
    
    # Calculate where the new resolution is better
    better_resolution_mask = repro_res < mosaic_res
    
    # Make the final mask for which columns to replace mosaic values in.
    good_longitude_mask = np.logical_or(valid_radius_count_better_mask,
        np.logical_and(valid_radius_count_equal_mask, better_resolution_mask))

    mosaic_good_long[:] = np.logical_or(mosaic_good_long, good_longitude_mask)
    mosaic_img[:,good_longitude_mask] = repro_img[:,good_longitude_mask]
    mosaic_res[good_longitude_mask] = repro_res[good_longitude_mask] 
    mosaic_phase[good_longitude_mask] = repro_phase[good_longitude_mask] 
    mosaic_emission[good_longitude_mask] = repro_emission[good_longitude_mask] 
    mosaic_image_number[good_longitude_mask] = image_number
    mosaic_time[good_longitude_mask] = image_time

