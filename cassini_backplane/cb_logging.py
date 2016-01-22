###############################################################################
# cb_logging.py
#
# Routines related to logging.
###############################################################################

import datetime
import logging
import os

from cb_config import *

_LOG_DEFAULT_LEVEL = logging.INFO
_LOG_OVERRIDE_BODIES     = None #logging.ERROR
_LOG_OVERRIDE_BOOTSTRAP  = None #logging.ERROR
_LOG_OVERRIDE_CORRELATE  = None #logging.ERROR
_LOG_OVERRIDE_OFFSET     = None #logging.ERROR
_LOG_OVERRIDE_RINGS      = None #logging.ERROR
_LOG_OVERRIDE_STARS      = None #logging.ERROR
_LOG_OVERRIDE_UTIL_FILE  = None #logging.ERROR
_LOG_OVERRIDE_UTIL_FLUX  = logging.ERROR
_LOG_OVERRIDE_UTIL_IMAGE = None #logging.ERROR
_LOG_OVERRIDE_UTIL_OOPS  = None #logging.ERROR

LOGGING_SUPERCRITICAL = 60

# Functions to set the main level and per-module overrides

def log_set_default_level(level=_LOG_DEFAULT_LEVEL):
    root_logger = logging.getLogger('cb')
    root_logger.setLevel(level)

def log_set_bodies_level(level=_LOG_OVERRIDE_BODIES):
    if level:
        logger = logging.getLogger('cb.cb_bodies')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_bootstrap_level(level=_LOG_OVERRIDE_BOOTSTRAP):
    if level:
        logger = logging.getLogger('cb.cb_bootstrap')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_correlate_level(level=_LOG_OVERRIDE_CORRELATE):
    if level:
        logger = logging.getLogger('cb.cb_correlate')
        logger.propagate = True
        logger.setLevel(level)

def log_set_offset_level(level=_LOG_OVERRIDE_OFFSET):
    if level:
        logger = logging.getLogger('cb.cb_offset')
        logger.propagate = True
        logger.setLevel(level)

def log_set_rings_level(level=_LOG_OVERRIDE_RINGS):
    if level:
        logger = logging.getLogger('cb.cb_rings')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_stars_level(level=_LOG_OVERRIDE_STARS):
    if level:
        logger = logging.getLogger('cb.cb_stars')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_util_file_level(level=_LOG_OVERRIDE_UTIL_FILE):
    if level:
        logger = logging.getLogger('cb.cb_util_file')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_util_flux_level(level=_LOG_OVERRIDE_UTIL_FLUX):
    if level:
        logger = logging.getLogger('cb.cb_util_flux')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_util_image_level(level=_LOG_OVERRIDE_UTIL_IMAGE):
    if level:
        logger = logging.getLogger('cb.cb_util_image')
        logger.propagate = True
        logger.setLevel(level)
    
def log_set_util_oops_level(level=_LOG_OVERRIDE_UTIL_OOPS):
    if level:
        logger = logging.getLogger('cb.cb_util_oops')
        logger.propagate = True
        logger.setLevel(level)

# Set up the console handler

_LOG_CONSOLE_HANDLER = None

def log_set_console_format(full=True):
    if full:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'+
                                      ' - %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
    else:
        formatter = logging.Formatter('%(name)s - %(message)s')
    _LOG_CONSOLE_HANDLER.setFormatter(formatter)

def log_add_console_handler(level=logging.DEBUG):
    global _LOG_CONSOLE_HANDLER
    _LOG_CONSOLE_HANDLER = logging.StreamHandler()
    _LOG_CONSOLE_HANDLER.setLevel(level)
    root_logger = logging.getLogger('cb')
    root_logger.addHandler(_LOG_CONSOLE_HANDLER)
    log_set_console_format()

def log_remove_console_handler():
    root_logger = logging.getLogger('cb')
    root_logger.removeHandler(_LOG_CONSOLE_HANDLER)

log_add_console_handler()


# Set up file handlers

def log_add_file_handler(filename, level=logging.DEBUG):
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'+
                                  ' - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fh.setFormatter(formatter)
    root_logger = logging.getLogger('cb')
    root_logger.addHandler(fh)
    return fh

def log_remove_file_handler(fh):
    root_logger = logging.getLogger('cb')
    if fh is not None:
        root_logger.removeHandler(fh)


# Set the overrides, if any

log_set_default_level()
log_set_bodies_level()
log_set_correlate_level()
log_set_rings_level()
log_set_stars_level()
log_set_util_file_level()
log_set_util_flux_level()
log_set_util_image_level()
log_set_util_oops_level()


# General utility routines

def log_decode_level(s):
    if s.upper() == 'NONE':
        return LOGGING_SUPERCRITICAL
    return getattr(logging, s.upper())

def log_min_level(level1, level2):
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
                  logging.CRITICAL, LOGGING_SUPERCRITICAL]:
        if level1 == level or level2 == level:
            return level
    return LOGGING_SUPERCRITICAL


def log_setup_main_logging(module_name, main_logfile_level, main_console_level, 
                           main_logfile, image_logfile_level, 
                           image_console_level):
    # Set up main loop logging
    main_logfile_level = log_decode_level(main_logfile_level)
    main_console_level = log_decode_level(main_console_level)
    
    # Note the main loop logger is not part of the cb.* name hierarchy.
    main_logger = logging.getLogger(module_name)
    main_logger.setLevel(log_min_level(main_logfile_level,
                                       main_console_level))
    
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - '+
                                       '%(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
    
    if main_logfile_level is not LOGGING_SUPERCRITICAL:
        # Only create the logfile is we're actually going to log to it
        if main_logfile is not None:
            main_log_path = main_logfile
        else:
            main_log_path = os.path.join(RESULTS_ROOT, 'logs')
            if not os.path.exists(main_log_path):
                os.mkdir(main_log_path)
            main_log_path = os.path.join(main_log_path, module_name)
            if not os.path.exists(main_log_path):
                os.mkdir(main_log_path)
            main_log_datetime = datetime.datetime.now().isoformat()[:-7]
            main_log_datetime = main_log_datetime.replace(':','-')
            main_log_path = os.path.join(main_log_path, 
                                         main_log_datetime+'.log')
        
        main_log_file_handler = logging.FileHandler(main_log_path)
        main_log_file_handler.setLevel(main_logfile_level)
        main_log_file_handler.setFormatter(main_formatter)
        main_logger.addHandler(main_log_file_handler)
    
    # Always create a console logger so we don't get a 'no handler' error
    main_log_console_handler = logging.StreamHandler()
    main_log_console_handler.setLevel(main_console_level)
    main_log_console_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_log_console_handler)
    
    # Set up per-image logging
    _LOGGING_NAME = 'cb.' + module_name
    image_logger = logging.getLogger(_LOGGING_NAME)
    
    image_logfile_level = log_decode_level(image_logfile_level)
    image_log_console_level = log_decode_level(image_console_level)
    
    log_set_default_level(log_min_level(image_logfile_level,
                                        image_console_level))
    log_set_util_flux_level(logging.CRITICAL)
    
    log_remove_console_handler()
    log_add_console_handler(image_log_console_level)

    return main_logger, image_logger
