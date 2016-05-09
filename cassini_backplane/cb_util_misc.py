###############################################################################
# cb_util_misc.py
#
# Routines related to image manipulation.
#
# Exported routines:
###############################################################################

import cb_logging
import logging

import numpy as np

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
    pad_b = np.zeros(b.shape[0]+2*n)
    pad_b[n:b.shape[0]+n] = b
    for amt in xrange(-n, n+1):
        b2 = pad_b[amt+n:amt+n+b.shape[0]]
        rms = np.sum((a-b2)**2)
        if rms < best_rms:
            best_rms = rms
            best_amt = amt
    return -best_amt

