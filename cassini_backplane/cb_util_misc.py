###############################################################################
# cb_util_misc.py
#
# Miscellaneous routines.
#
# Exported routines:
###############################################################################

import cb_logging
import logging

import numpy as np
import subprocess
import urllib2

from cb_util_file import *

import oops

_LOGGING_NAME = 'cb.' + __name__


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n-1:] /= n
    ret[:n-1] /= np.arange(1., n)
    return ret

def shift_1d(a, offset):
    if offset == 0:
        return a
    a = np.roll(a, offset)
    if offset < 0:
        a[offset:] = 0
    else:
        a[:offset] = 0
    return a

def find_shift_1d(a, b, n):
    best_amt = None
    best_rms = 1e38
    if a.shape[0] < b.shape[0]:
        pad_a = np.zeros(b.shape[0])
        pad_a[:a.shape[0]] = a
    else:
        pad_a = a
    pad_b = np.zeros(b.shape[0]+2*n)
    pad_b[n:b.shape[0]+n] = b
    for amt in xrange(-n, n+1):
        b2 = pad_b[amt+n:amt+n+b.shape[0]]
        rms = np.sum((pad_a-b2)**2)
        if rms < best_rms:
            best_rms = rms
            best_amt = amt
    return -best_amt

def _simple_filter_name_helper(filter1, filter2, consolidate_pol):
    if consolidate_pol:
        if filter1 == 'P0' or filter1 == 'P60' or filter1 == 'P120':
            filter1 = 'P'

    if filter1 == 'CL1' and filter2 == 'CL2':
        return 'CLEAR'

    if filter1 == 'CL1':
        return filter2
    
    if filter2 == 'CL2':
        return filter1
        
    return filter1 + '+' + filter2

def simple_filter_name(obs, consolidate_pol=False):
    return _simple_filter_name_helper(obs.filter1, obs.filter2,
                                      consolidate_pol=consolidate_pol)

def simple_filter_name_metadata(metadata, consolidate_pol=False):
    return _simple_filter_name_helper(metadata['filter1'], metadata['filter2'],
                                      consolidate_pol=consolidate_pol)

def current_git_version():
    try:
        return subprocess.check_output(['git', 'describe', '--long', '--dirty', 
                                    '--abbrev=40', '--tags']).strip()
    except:
        return 'GIT DESCRIBE FAILED'
