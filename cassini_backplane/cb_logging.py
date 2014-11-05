###############################################################################
# cb_logging.py
#
# Routines related to logging.
###############################################################################

import logging

#LOG_DEFAULT_LEVEL = logging.DEBUG
LOG_DEFAULT_LEVEL = logging.ERROR
LOG_OVERRIDE_CORRELATE = None #logging.ERROR
LOG_OVERRIDE_MOONS     = None #logging.ERROR
LOG_OVERRIDE_RINGS     = None #logging.ERROR
LOG_OVERRIDE_STARS     = None #logging.ERROR
LOG_OVERRIDE_UTIL_FLUX = None#logging.ERROR
LOG_OVERRIDE_UTIL_OOPS = None

root_logger = logging.getLogger('cb')
root_logger.setLevel(LOG_DEFAULT_LEVEL)

#fh = logging.FileHandler('spam.log')
#fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                              datefmt='%m/%d/%Y %I:%M:%S %p')
formatter = logging.Formatter('%(name)s - %(message)s')
ch.setFormatter(formatter)

#fh.setFormatter(formatter)
# add the handlers to logger
root_logger.addHandler(ch)

if LOG_OVERRIDE_CORRELATE:
    logger = logging.getLogger('cb.cb_correlate')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_CORRELATE)

if LOG_OVERRIDE_MOONS:
    logger = logging.getLogger('cb.cb_moons')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_MOONS)
    
if LOG_OVERRIDE_RINGS:
    logger = logging.getLogger('cb.cb_rings')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_RINGS)
    
if LOG_OVERRIDE_STARS:
    logger = logging.getLogger('cb.cb_stars')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_STARS)
    
if LOG_OVERRIDE_UTIL_FLUX:
    logger = logging.getLogger('cb.cb_util_flux')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_UTIL_FLUX)

if LOG_OVERRIDE_UTIL_OOPS:
    logger = logging.getLogger('cb.cb_util_oops')
    logger.propagate = False
    logger.setLevel(LOG_OVERRIDE_UTIL_OOPS)
