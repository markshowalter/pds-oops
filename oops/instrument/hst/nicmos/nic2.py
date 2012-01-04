################################################################################
# oops.instrument.hst.nicmos.NIC2
################################################################################

import pyfits
import oops

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/NICMOS/NIC1."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the NIC2 class
    this = NIC2()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is NICMOS
    if this.instrument_name(hst_file) != "NICMOS":
        raise IOError("not an HST/NICMOS file: " + this.filespec(hst_file))

    # Confirm that the detector is NIC2
    if this.detector_name(hst_file) != "IR":
        raise IOError("not an HST/NICMOS/NIC2 file: " + this.filespec(hst_file))

    return from_opened_fitsfile(hst_file, parameters)

def from_opened_fitsfile(hst_file, parameters={}):
    """A general, static method to return an Observation object based on an hst
    data file generated by HST/NICMOS/NIC2."""

    return NIC2().construct_snapshot(hst_file, parameters)

################################################################################
# Class NIC2
################################################################################

class NIC2(oops.instrument.hst.nicmos.NICMOS):
    """This class defines functions and properties unique to the NIC2 detector.
    Everything else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    # Used by select_syn_files, defined in NICMOS.py

    DETECTOR_SYN_FILES = ["NICMOS/nic2_bend_???_syn.fits",
                          "NICMOS/nic2_cmask_???_syn.fits",
                          "NICMOS/nic2_dewar_???_syn.fits",
                          "NICMOS/nic2_image_???_syn.fits",
                          "NICMOS/nic2_para1_???_syn.fits",
                          "NICMOS/nic2_para2_???_syn.fits"]

    FILTER_SYN_FILE_PARTS = ["NICMOS/nic2_", "_???_syn.fits"]

################################################################################
