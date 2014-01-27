################################################################################
# oops/instrument/hst/wfpc2/__init__.py: HST subclass WFPC2
#
# 6/7/12 MRS - revised super().register_frame() so that it can support WFPC2;
#   local method modified to call super() version.
# 8/4/12 MRS - repaired bugs in quality_mask(); fixed name for data_array().
################################################################################

import os.path
import pyfits
import oops
from oops.inst.hst import HST

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, **parameters):
    """A general, static method to return an Observation object based on a given
    data file generated by HST/WFPC2."""

    # Open the file
    hst_file = pyfits.open(filespec)

    # Make an instance of the NICMOS class
    this = WFPC2()

    # Confirm that the telescope is HST
    if this.telescope_name(hst_file) != "HST":
        raise IOError("not an HST file: " + this.filespec(hst_file))

    # Confirm that the instrument is WFPC2
    if this.instrument_name(hst_file) != "WFPC2":
        raise IOError("not an HST/WFPC2 file: " + this.filespec(hst_file))

    return WFPC2.from_opened_fitsfile(hst_file, **parameters)

################################################################################
# Class WFPC2
################################################################################

IDC_DICT = None

GENERAL_SYN_FILES = ["OTA/hst_ota_???_syn.fits",
                     "WFPC2/wfpc2_optics_???_syn.fits"]

FILTER_SYN_FILE_PARTS = ["WFPC2/wfpc2_", "_???_syn.fits"]

class WFPC2(HST):
    """This class defines functions and properties unique to the WFPC2
    instrument. Everything else is inherited from higher levels in the class
    hierarchy.

    Objects of this class are empty; they only exist to support inheritance.
    """

    CHIPS = ["", "PC1", "WF2", "WF3", "WF4"]
    QUALITY_MASK_LOOKUP = {"d0":"q0", "D0":"Q0", "c0":"c1", "C0":"C1"}

    # WFPC2 is treated as a single detector
    def detector_name(self, hst_file, **parameters):
        """Returns the name of the detector on the HST instrument that was used
        to obtain this file. Always blank for WFPC2."""

        return ""

    # WFPC2 has two filter wheels. Names are identified by FITS parameters
    # FILTNAM1 and FILTNAM2 in the first header.
    def filter_name(self, hst_file):
        """Returns the name of the filter for this particular ACS detector.
        Overlapped filters are separated by a plus sign."""

        name1 = hst_file[0].header["FILTNAM1"].strip()
        name2 = hst_file[0].header["FILTNAM2"].strip()

        if name1 == "":
            if name2[0:5] == "":
                return "CLEAR"
            else:
                return name2
        else:
            if name2 == "":
                return name1
            else:
                return name1 + "+" + name2

    def data_array(self, hst_file, **parameters):
        """Returns an array containing the data. Use layer=1-4 for PC1, WF2,
        WF3 and WF4."""

        # Figure out the layer; otherwise use PC1
        if "layer" in parameters.keys():
            layer = parameters["layer"]
        else:
            layer = 1

        return hst_file[layer].data

    def quality_mask(self, hst_file, **parameters):
        """Returns an array containing the quality mask. Use parameter
        "layer"=1-4 for PC1, WF2, WF3 and WF4. Use "mask"="omit" to omit mask;
        "require" to raise an IOError if the mask is unavailable, "optional" to
        include it if available, otherwise return None. Default is "optional".
        """

        # Unlike the other HST instruments, WFPC2 masks are saved in a separate
        # file. We attempt to open the mask file under the assumption that it is
        # in the same directory as the data file.

        # Quality masks for raw files ("*d0m.fits") end in "q0m.fits".
        # Quality masks for calibrated files ("*c0m.fits") end in "c1m.fits".

        if "mask" in parameters.keys():
            mask_option = parameters["mask"]
        else:
            mask_option = "optional"

        if mask_option == "omit": return

        if mask_option not in {"require", "required", "optional"}:
            raise ValueError("Illegal value for 'mask' parameter: '" +
                             mask_option + "'")

        # Get the full path to the image file
        data_filespec = self.filespec(hst_file)

        # Extract the extension
        (head,tail) = os.path.splitext(data_filespec)

        # Attempt to define the mask filespec
        data_tag = head[-3:-1]
        try:
            mask_tag = WFPC2.QUALITY_MASK_LOOKUP[data_tag]
        except KeyError:
            if mask_option == "optional":
                return None
            else:
                raise IOError("Unable to identify mask file for " +
                              data_filespec)

        # Figure out the layer; otherwise use PC1
        if "layer" in parameters.keys():
            layer = parameters["layer"]
        else:
            layer = 1

        # Attempt to load and return the mask
        mask_filespec = head[:-3] + mask_tag + head[-1] + tail
        try:
            f = pyfits.open(mask_filespec)
            mask_array = f[layer].data
            f.close()

            return mask_array

        except IOError:
            if mask_option == "optional":
                return None
            else:
                raise IOError("WFPC2 mask file not found: " +
                              mask_filespec)

    def register_frame(self, hst_file, fov, **parameters):
        """Returns the Tracker frame that rotates from J2000 coordinates into
        the frame of the HST observation.
        """

        global CHIPS

        # Figure out the layer; otherwise use PC1
        if "layer" in parameters.keys():
            layer = parameters["layer"]
        else:
            layer = 1

        return super(WFPC2,self).register_frame(hst_file, fov,
                                            index = layer,
                                            suffix = "_" + WFPC2.CHIPS[layer],
                                            **parameters)

    # The IDC dictionaries for WFPC2 are keyed by (FILTNAM1, FILTNAM2, DETCHIP)
    def define_fov(self, hst_file, **parameters):
        """Returns an FOV object defining the field of view of the given image
        file and layer, where layers 1-4 refer to PC1, WF2, WF3 and WF4.
        """

        global IDC_DICT

        # Load the dictionary of IDC parameters if necessary
        if IDC_DICT is None:
            IDC_DICT = self.load_idc_dict(hst_file, ("FILTER1",
                                                     "FILTER2", "DETCHIP"))

            # The IDC_DICT parameters for WFPC2 need to be re-scaled!
            for entry in IDC_DICT.keys():
                dict = IDC_DICT[entry]
                scale = dict["SCALE"]
                for key in dict.keys():
                    if key[0:2] in ("CX", "CY") and len(key) == 4:
                        dict[key] *= scale

        # Figure out the layer; otherwise use PC1
        if "layer" in parameters.keys():
            layer = parameters["layer"]
        else:
            layer = 1

        # Define the key into the dictionary
        filtnam1 = hst_file[0].header["FILTNAM1"].strip()
        filtnam2 = hst_file[0].header["FILTNAM2"].strip()
        if filtnam1 == "": filtnam1 = "CLEAR"
        if filtnam2 == "": filtnam2 = "CLEAR"
        idc_key = (filtnam1, filtnam2, layer)

        # Define the FOV
        fov = super(WFPC2,self).construct_fov(IDC_DICT[idc_key], hst_file)

        # Handle AREA mode
        if hst_file[0].header["MODE"] == "AREA":
            fov = oops.fov.Subsampled(fov, 2)

        return fov

    def select_syn_files(self, hst_file, **parameters):
        """Returns the list of SYN files containing profiles that are to be
        multiplied together to obtain the throughput of the given instrument,
        detector and filter combination."""

        global GENERAL_SYN_FILES, FILTER_SYN_FILE_PARTS

        # Copy all the standard file names
        syn_filenames = []
        for filename in GENERAL_SYN_FILES:
            syn_filenames.append(filename)

        # Add the filter file names
        for filter_name in (hst_file[0].header["FILTNAM1"],
                            hst_file[0].header["FILTNAM2"]):
            filter_name = filter_name.strip()
            if filter_name != "":
                syn_filenames.append(FILTER_SYN_FILE_PARTS[0] +
                                     filter_name.lower() +
                                     FILTER_SYN_FILE_PARTS[1])

        return syn_filenames

    def dn_per_sec_factor(self, hst_file):
        """Returns a factor that converts a pixel value to DN per second.
        
        Input:
            hst_file        the object returned by pyfits.open()            
        
        Return              the factor to multiply a pixel value by to get DN/sec
        """
        return 1. / hst_file[0].header["EXPTIME"]

    @staticmethod
    def from_opened_fitsfile(hst_file, **parameters):
        """A general, static method to return an Observation object based on an
        HST data file generated by HST/WFPC2."""

        # Make an instance of the WFPC2 class
        this = WFPC2()

        # Figure out the layer; otherwise use PC1
        if "layer" in parameters.keys():
            layer = parameters["layer"]
        else:
            layer = 1

        obs = this.construct_snapshot(hst_file, **parameters)

        # Insert subfields common to all NICMOS images
        obs.insert_subfield("layer", layer)
        obs.insert_subfield("chip", WFPC2.CHIPS[layer])

        return obs

################################################################################
