################################################################################
# oops_/inst/hst/acs/hrc.py: HST/ACS subclass HRC
################################################################################

import pyfits
from oops_.inst.hst.acs import ACS

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, parameters={}):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/ACS/HRC."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the HRC class
    this = HRC()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is ACS
    if this.instrument_name(hst_file) != "ACS":
        raise IOError("not an HST/ACS file: " + this.filespec(hst_file))

    # Confirm that the detector is HRC
    if this.detector_name(hst_file) != "HRC":
        raise IOError("not an HST/ACS/HRC file: " + this.filespec(hst_file))

    return HRC.from_opened_fitsfile(hst_file, parameters)

################################################################################
# Class HRC
################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ["OTA/hst_ota_???_syn.fits",
                     "ACS/acs_hrc_win_???_syn.fits",
                     "ACS/acs_hrc_m12_???_syn.fits",
                     "ACS/acs_hrc_m3_???_syn.fits",
                     "ACS/acs_hrc_ccd_mjd_???_syn.fits"]

CORONOGRAPH_SYN_FILE = "ACS/acs_hrc_coron_???_syn.fits"

FILTER_SYN_FILE = ["ACS/acs_", "_???_syn.fits"]

class HRC(ACS):
    """This class defines functions and properties unique to the NIC1 detector.
    Everything else is inherited from higher levels in the class hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    def define_fov(self, hst_file, parameters={}):
        """Returns an FOV object defining the field of view of the given image
        file.
        """

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hst_file, ("FILTER1", "FILTER2"))

        # Define the key into the dictionary
        idc_key = (hst_file[0].header["FILTER1"], hst_file[0].header["FILTER2"])

        # Use the default function defined at the HST level for completing the
        # definition of the FOV
        return self.construct_fov(IDC_DICT[idc_key], hst_file)

    def select_syn_files(self, hst_file):
        """Returns the list of SYN files containing profiles that are to be
        multiplied together to obtain the throughput of the given instrument,
        detector and filter combination."""

        global GENERAL_SYN_FILES, CORONOGRAPH_SYN_FILE, FILTER_SYN_FILE

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
                syn_filenames.append(FILTER_SYN_FILE[0] +
                                     filter_name.lower() +
                                     FILTER_SYN_FILE[1])

        # Add the coronograph file name if necessary
        if hst_file[0].header["APERTURE"][0:9] == "HRC-CORON":
            syn_filenames.append(CORONOGRAPH_SYN_FILE)

        return syn_filenames

    @staticmethod
    def from_opened_fitsfile(hst_file, parameters={}):
        """A general class method to return an Observation object based on an
        HST data file generated by HST/ACS/HRC."""

        return HRC().construct_snapshot(hst_file, parameters)

################################################################################
