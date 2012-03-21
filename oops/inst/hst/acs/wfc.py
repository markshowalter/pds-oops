################################################################################
# oops/inst/hst/acs/wfc.py: HST/ACS subclass WFC
################################################################################

import pyfits
from oops.inst.hst.acs import ACS

################################################################################
# Classless method
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/ACS/WFC."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the WFC class
    this = WFC()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is ACS
    if this.instrument_name(hst_file) != "ACS":
        raise IOError("not an HST/ACS file: " + this.filespec(hst_file))

    # Confirm that the detector is WFC
    if this.detector_name(hst_file) != "WFC":
        raise IOError("not an HST/ACS/WFC file: " + this.filespec(hst_file))

    return WFC.from_opened_fitsfile(hst_file, parameters)

################################################################################
# Class WFC
################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ["OTA/hst_ota_???_syn.fits",
                     "ACS/acs_wfc_im123_???_syn.fits"]

CCD_SYN_FILE_PARTS    = ["ACS/acs_wfc_ccd", "_???_syn.fits"]
FILTER_SYN_FILE_PARTS = ["ACS/acs_", "_???_syn.fits"]

class WFC(ACS):
    """This class defines unique detector properties of the ACS/WFC. Other
    properties are inherited from higher levels in the class hierarcy. All
    functions are static so this class requires no instances."""

    def define_fov(self, hst_file, parameters={}):
        """Returns an FOV object defining the field of view of the given image
        file.
        """

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hst_file, ("DETCHIP", "FILTER1",
                                                                "FILTER2"))

        # Determine the layer of the FITS file to read
        try:
            layer = parameters["layer"]
            assert hst_file[layer].header["EXTTYPE"] == "SCI"
        except KeyError:
            layer = 1

        # Define the key into the dictionary
        idc_key = (hst_file[layer].header["CCDCHIP"],
                   hst_file[0].header["FILTER1"],
                   hst_file[0].header["FILTER2"])

        # Use the default function defined at the HST level for completing the
        # definition of the FOV
        return self.construct_fov(IDC_DICT[idc_key], hst_file)

    def select_syn_files(self, hst_file, parameters={}):
        """Returns the list of SYN files containing profiles that are to be
        multiplied together to obtain the throughput of the given instrument,
        detector and filter combination."""

        global GENERAL_SYN_FILES, CCD_SYN_FILE_PARTS, FILTER_SYN_FILE_PARTS

        # Copy all the standard file names
        syn_filenames = []
        for filename in GENERAL_SYN_FILES:
            syn_filenames.append(filename)

        # Add the filter file names
        for filter_name in (hst_file[0].header["FILTER1"],
                            hst_file[0].header["FILTER2"]):

            if filter_name[0:3] == "POL":
                if filter_name[-2:] == "UV":
                    filter_name = "POL_UV"
                else:
                    filter_name = "POL_V"

            if filter_name[0:5] != "CLEAR":
                syn_filenames.append(FILTER_SYN_FILE_PARTS[0] +
                                     filter_name.lower() +
                                     FILTER_SYN_FILE_PARTS[1])

        # Determine the layer of the FITS file to read
        try:
            layer = parameters["layer"]
            assert hst_file[layer].header["EXTTYPE"] == "SCI"
        except KeyError:
            layer = 1

        # Add the CCD file name
        syn_filenames.append(CCD_SYN_FILE_PARTS[0] +
                             str(hst_file[layer].header["CCDCHIP"]) +
                             CCD_SYN_FILE_PARTS[1])

        return syn_filenames

    @staticmethod
    def from_opened_fitsfile(hst_file, parameters={}):
        """A general class method to return an Observation object based on an
        HST data file generated by HST/ACS/WFC."""

        return WFC().construct_snapshot(hst_file, parameters)

################################################################################
