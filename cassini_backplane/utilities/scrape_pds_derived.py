###############################################################################
# scrape_pds_derived.py
#
# Download files in bulk from pds-rings.seti.org.
###############################################################################

import argparse
import os
import sys
import traceback
import urllib2

from cb_util_file import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = ''

    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Scrape Derived Files from PDS-RINGS',
    epilog='''Default behavior is to scrape all files''')


file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


def read_url(url):
    try:
        url_fp = urllib2.urlopen(url)
        ret = url_fp.read()
        url_fp.close()
        return ret
    except urllib2.HTTPError, e:
        return 'READ FAILURE'
    except urllib2.URLError, e:
        return 'READ FAILURE'


def copy_one_image(image_path):
    short_path = file_img_to_short_img_path(image_path)
    if short_path.find('_CALIB') == -1:
        short_path = short_path.replace('.IMG', '_CALIB.IMG')
    url = PDS_RINGS_CALIB_ROOT + short_path
    image_path_local = file_clean_join(COISS_2XXX_DERIVED_ROOT, short_path)
    print image_path_local
    image_path_dir, _ = os.path.split(image_path_local)
    if not os.path.exists(image_path_dir):
        os.makedirs(image_path_dir)
    local_fp = open(image_path_local, 'wb')
    url_fp = urllib2.urlopen(url)
    local_fp.write(url_fp.read())
    url_fp.close()
    local_fp.close()

    short_path = short_path.replace('.IMG', '.LBL')
    url = PDS_RINGS_CALIB_ROOT + short_path
    image_path_local = file_clean_join(COISS_2XXX_DERIVED_ROOT, short_path)
    print image_path_local
    image_path_dir, _ = os.path.split(image_path_local)
    if not os.path.exists(image_path_dir):
        os.makedirs(image_path_dir)
    local_fp = open(image_path_local, 'wb')
    url_fp = urllib2.urlopen(url)
    local_fp.write(url_fp.read())
    url_fp.close()
    local_fp.close()
        

#===============================================================================
# 
#===============================================================================


print 'Downloading PDS index files'
index_no = 2001
while True:
    index_file = file_clean_join(COISS_2XXX_DERIVED_ROOT,
                                 'COISS_%04d-index.tab' % index_no)
    if not os.path.exists(index_file):
        url = PDS_RINGS_VOLUMES_ROOT + (
                        'COISS_%04d/index/index.tab' % index_no)
        print 'Trying %s' % url
        try:
            url_fp = urllib2.urlopen(url)
            index_fp = open(index_file, 'w')
            index_fp.write(url_fp.read())
            index_fp.close()
            url_fp.close()
        except urllib2.HTTPError, e:
            print 'Failed to retrieve %s: %s' % (url, e)
            break
        except urllib2.URLError, e:
            print 'Failed to retrieve %s: %s' % (url, e)
            break
    index_no += 1
        
for image_path in file_yield_image_filenames_from_arguments(
                                                arguments, True):
    copy_one_image(image_path)
