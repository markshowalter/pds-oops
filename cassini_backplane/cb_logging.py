###############################################################################
# cb_logging.py
#
# Routines related to logging.
###############################################################################

import logging

_LOG_DEFAULT_LEVEL = logging.INFO
_LOG_OVERRIDE_BODIES    = None #logging.ERROR
_LOG_OVERRIDE_CORRELATE = None #logging.ERROR
_LOG_OVERRIDE_OFFSET    = None #logging.ERROR
_LOG_OVERRIDE_RINGS     = None #logging.ERROR
_LOG_OVERRIDE_STARS     = None #logging.ERROR
_LOG_OVERRIDE_UTIL_FILE = None #logging.ERROR
_LOG_OVERRIDE_UTIL_FLUX = logging.ERROR
_LOG_OVERRIDE_UTIL_OOPS = None

def log_set_default_level(level=_LOG_DEFAULT_LEVEL):
    root_logger = logging.getLogger('cb')
    root_logger.setLevel(level)

def log_set_bodies_level(level=_LOG_OVERRIDE_BODIES):
    if level:
        logger = logging.getLogger('cb.cb_bodies')
        logger.propagate = False
        logger.setLevel(level)
    
def log_set_correlate_level(level=_LOG_OVERRIDE_CORRELATE):
    if level:
        logger = logging.getLogger('cb.cb_correlate')
        logger.propagate = False
        logger.setLevel(level)

def log_set_offset_level(level=_LOG_OVERRIDE_OFFSET):
    if level:
        logger = logging.getLogger('cb.cb_offset')
        logger.propagate = False
        logger.setLevel(level)

def log_set_rings_level(level=_LOG_OVERRIDE_RINGS):
    if level:
        logger = logging.getLogger('cb.cb_rings')
        logger.propagate = False
        logger.setLevel(level)
    
def log_set_stars_level(level=_LOG_OVERRIDE_STARS):
    if level:
        logger = logging.getLogger('cb.cb_stars')
        logger.propagate = False
        logger.setLevel(level)
    
def log_set_util_file_level(level=_LOG_OVERRIDE_UTIL_FILE):
    if level:
        logger = logging.getLogger('cb.cb_util_file')
        logger.propagate = False
        logger.setLevel(level)
    
def log_set_util_flux_level(level=_LOG_OVERRIDE_UTIL_FLUX):
    if level:
        logger = logging.getLogger('cb.cb_util_flux')
        logger.propagate = False
        logger.setLevel(level)
    
def log_set_util_oops_level(level=_LOG_OVERRIDE_UTIL_OOPS):
    if level:
        logger = logging.getLogger('cb.cb_util_oops')
        logger.propagate = False
        logger.setLevel(level)

def log_set_format(full=True):
    if full:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'+
                                      ' - %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        formatter = logging.Formatter('%(name)s - %(message)s')
    _LOG_STREAMHANDLER.setFormatter(formatter)

_LOG_STREAMHANDLER = logging.StreamHandler()
_LOG_STREAMHANDLER.setLevel(logging.DEBUG)
root_logger = logging.getLogger('cb')
root_logger.addHandler(_LOG_STREAMHANDLER)
log_set_format()

log_set_default_level()
log_set_bodies_level()
log_set_correlate_level()
log_set_rings_level()
log_set_stars_level()
log_set_util_file_level()
log_set_util_flux_level()
log_set_util_oops_level()

