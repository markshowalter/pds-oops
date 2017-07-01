###############################################################################
# cb_util_web.py
#
# Routines from downloading files from the web.
#
# Exported routines:
###############################################################################

import cb_logging
import logging

import urllib2

from cb_config import *
from cb_util_file import *

import oops

_LOGGING_NAME = 'cb.' + __name__

def web_read_url(url):
    try:
        url_fp = urllib2.urlopen(url)
        ret = url_fp.read()
        url_fp.close()
        return ret
    except urllib2.HTTPError, e:
        return None
    except urllib2.URLError, e:
        return None

def web_copy_url_to_file(url, file_path):
    try:
        file_fp = open(file_path, 'wb')
        url_fp = urllib2.urlopen(url)
        file_fp.write(url_fp.read())
        url_fp.close()
        file_fp.close()
    except urllib2.HTTPError, e:
        return 'Failed to retrieve %s: %s' % (url, e)
    except urllib2.URLError, e:
        return 'Failed to retrieve %s: %s' % (url, e)
    return None

def web_update_index_files_from_pds(logger):
    logger.info('Downloading PDS index files')
    index_no = 2001
    while True:
        index_file = file_clean_join(COISS_2XXX_DERIVED_ROOT,
                                     'COISS_%04d-index.tab' % index_no)
        if not os.path.exists(index_file):
            url = PDS_RINGS_VOLUMES_ROOT + (
                            'COISS_%04d/index/index.tab' % index_no)
            logger.debug('Copying %s', url)
            err = web_copy_url_to_file(url, index_file)
            if err:
                logger.info(err)
                return
        index_no += 1

def web_retrieve_image_from_pds(image_path, main_logger=None, 
                                image_logger=None):
    short_path = file_img_to_short_img_path(image_path)
    if short_path.find('_CALIB') == -1:
        short_path = short_path.replace('.IMG', '_CALIB.IMG')
    url = PDS_RINGS_CALIB_ROOT + short_path
    image_name = file_clean_name(image_path)
    image_path_local = file_clean_join(TMP_DIR, image_name+'.IMG')
    if main_logger is not None:
        main_logger.debug('Retrieving %s to %s', url, image_path_local)
    if image_logger is not None:
        image_logger.info('Retrieving %s to %s', url, image_path_local)
    err = web_copy_url_to_file(url, image_path_local)
    return err, image_path_local
