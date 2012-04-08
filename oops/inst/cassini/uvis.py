################################################################################
# oops/inst/cassini/uvis.py
################################################################################

import numpy as np
import pylab
import os
import julian
import pdstable
import pdsparser
import cspice
import oops

from oops.inst.cassini.cassini_ import Cassini


################################################################################
# Standard class methods
################################################################################

def uvis_from_file(filename):
    
    # Read file
    lines = pdsparser.PdsLabel.load_file(filename)
    
    # Deal with corrupt syntax
    for i in range(len(lines)):
        line = lines[i]
        if "CORE_UNIT" in line:
            if '"COUNTS/BIN"' not in line:
                lines[i] = re.sub( "COUNTS/BIN", '"COUNTS/BIN"', line)
    
    # Get dictionary
    this = pdsparser.PdsLabel.from_string(lines)
    this.filename = filename
    
    return this

def from_file(filespec, parameters={}):
    """A general, static method to return an Snapshot object based on a given
        Cassini UVIS image file."""
    
    UVIS.initialize()    # Define everything the first time through
    
    # Load the VICAR file
    label = uvis_from_file(filespec)
    qube = label["QUBE"]
    
    # Get key information from the header
    time = label["START_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb0 = cspice.str2et(time)
    
    time = label["STOP_TIME"].value
    if time[-1] == "Z": time = time[:-1]
    tdb1 = cspice.str2et(time)
    
    # Make sure the SPICE kernels are loaded
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)

    detector = "SOL_OFF"
    detectors = ["SOL_OFF", "HSP", "FUV", "EUV", "SOLAR"]
    if "HDAC" in filespec:
        # don't know how to handle the fov for this yet
        return None
    
    for d in detectors:
        if d in filespec:
            detector = d
            break

    mode = str(label["SLIT_STATE"])

    uvis_ss = oops.obs.Snapshot((tdb0, tdb1),               # time
                                UVIS.fovs[detector, mode],  # fov
                                "CASSINI",                  # path_id
                                "CASSINI_UVIS_" + detector)
    return uvis_ss

################################################################################

def uvis_from_index(filespec):
    lines = pdsparser.PdsLabel.load_file(filespec)
    
    # Deal with corrupt syntax
    newlines = []
    first_obj = "None"
    for line in lines:
        if "RECORD_TYPE" in line:
            #words = line.split("=")
            newline = "RECORD_TYPE             = FIXED_LENGTH"
            print "RECORD_TYPE line:"
            print newline
            newlines.append(newline)
        elif "^TABLE" in line:
            words = line.split("=")
            newlines.append("^INDEX_TABLE = " + words[-1])
            """newline = ""
            for word in words:
                sword = word.rstrip()
                if sword == "^TABLE":
                    newline = "^INDEX_TABLE"
                else:
                    newline += " " + sword
            newlines.append(newline)"""
        elif first_obj == "None":
            if "OBJECT" in line:
                words = line.split("=")
                first_obj = words[-1].strip()
            newlines.append(line)
        elif "END_OBJECT" in line and "=" not in line:
            newline = line + " = " + first_obj
            newlines.append(newline)
        else:
            newlines.append(line)
    
    f = open("/Users/bwells/lsrc/pds-tools/test.txt", "w")
    for line in newlines:
        f.write("%s\n" % line)
    f.close()
    table = pdstable.PdsTable(filespec, ["START_TIME", "STOP_TIME"], newlines)
    return table

def from_index(filespec, parameters={}):
    """A static method to return a list of Snapshot objects, one for each row
        in an UVIS index file. The filespec refers to the label of the index file.
        """
    
    UVIS.initialize()    # Define everything the first time through
    
    # Read the index file
    table = uvis_from_index(filespec)
    row_dicts = table.dicts_by_row()
    
    # Create a list of Snapshot objects
    snapshots = []
    for dict in row_dicts:
        sub_label_name = os.path.dirname(filespec) + dict["FILE_NAME"]
        snapshots.append(from_file(sub_label_name))
        
        
    
    # Make sure all the SPICE kernels are loaded
    tdb0 = row_dicts[0]["START_TIME"]
    tdb1 = row_dicts[-1]["STOP_TIME"]
    
    Cassini.load_cks( tdb0, tdb1)
    Cassini.load_spks(tdb0, tdb1)
    
    return snapshots

################################################################################


class UVIS(object):
    """A instance-free class to hold Cassini UVIS instrument parameters."""
    
    instrument_kernel = None
    fovs = {}
    initialized = False
    
    @staticmethod
    def initialize():
        """Fills in key information about the WAC and NAC. Must be called first."""
        
        # Quick exit after first call
        if UVIS.initialized: return
        
        Cassini.initialize()
        Cassini.load_instruments()
        
        # Load the instrument kernel
        UVIS.instrument_kernel = Cassini.spice_instrument_kernel("UVIS")[0]
        
        # Construct a flat FOV for each camera
        #for detector in ["SOL_OFF", "HSP", "FUV", "EUV", "SOLAR", "HDAC"]:
        for detector in ["SOL_OFF", "HSP", "FUV", "EUV", "SOLAR"]:
            info = UVIS.instrument_kernel["INS"]["CASSINI_UVIS_" + detector]
            
            # Full field of view
            lines = info["PIXEL_LINES"]
            samples = info["PIXEL_SAMPLES"]
            
            xfov = info["FOV_REF_ANGLE"]
            yfov = info["FOV_CROSS_ANGLE"]
            #assert info["FOV_ANGLE_UNITS"] == "DEGREES"
            conversion = 1.
            if info["FOV_ANGLE_UNITS"] == "DEGREES":
                conversion = np.pi/180.
            
            uscale = np.arctan(np.tan(xfov * conversion) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * conversion) / (lines/2.))
            
            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.Flat((uscale,vscale), (samples,lines))
            
            # Load the dictionary, include the subsampling modes
            if detector == "EUV":
                UVIS.fovs[detector, "HIGH_RESOLUTION"] = full_fov
                UVIS.fovs[detector, "LOW_RESOLUTION"] = oops.fov.Subsampled(full_fov,
                                                                       (1, 2))
                UVIS.fovs[detector, "OCCULTATION"] = oops.fov.Subsampled(full_fov,
                                                                    (1, 8))
            elif detector == "FUV":
                UVIS.fovs[detector, "HIGH_RESOLUTION"] = full_fov
                UVIS.fovs[detector, "LOW_RESOLUTION"] = oops.fov.Subsampled(full_fov,
                                                                       (1, 2))
                UVIS.fovs[detector, "OCCULTATION"] = oops.fov.Subsampled(full_fov,
                                                                    (1, 8./.75))
            else:
                #NULL shows up in non EUV/FUV data
                UVIS.fovs[detector, "NULL"] = full_fov
        
        # Construct a SpiceFrame for each camera
        # Deal with the fact that the instrument's internal coordinate system is
        # rotated 180 degrees
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOL_OFF",
                                       id="CASSINI_UVIS_SOL_OFF_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HSP",
                                       id="CASSINI_UVIS_HSP_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_FUV",
                                       id="CASSINI_UVIS_FUV_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_EUV",
                                       id="CASSINI_UVIS_EUV_FLIPPED")
        ignore = oops.frame.SpiceFrame("CASSINI_UVIS_SOLAR",
                                       id="CASSINI_UVIS_SOLAR_FLIPPED")
        #ignore = oops.frame.SpiceFrame("CASSINI_UVIS_HDAC",
        #                               id="CASSINI_UVIS_HDAC_FLIPPED")
        
        rot180 = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_SOL_OFF",
                                    "CASSINI_UVIS_SOL_OFF_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_HSP",
                                    "CASSINI_UVIS_HSP_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_FUV",
                                    "CASSINI_UVIS_FUV_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_EUV",
                                    "CASSINI_UVIS_EUV_FLIPPED")
        ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_SOLAR",
                                    "CASSINI_UVIS_SOLAR_FLIPPED")
        #ignore = oops.frame.Cmatrix(rot180, "CASSINI_UVIS_HDAC",
        #                            "CASSINI_UVIS_HDAC_FLIPPED")
        UVIS.initialized = True
            
        @staticmethod
        def reset():
            """Resets the internal Cassini UVIS parameters. Can be useful for
                debugging."""
            
            UVIS.instrument_kernel = None
            UVIS.fovs = {}
            UVIS.initialized = False
            
            Cassini.reset()

################################################################################
# UNIT TESTS
################################################################################

import unittest


class Test_Cassini_UVIS(unittest.TestCase):
    
    def runTest(self):

        #from_file("test_data/cassini/UVIS/EUV1999_007_17_05.DAT")
        #from_file("test_data/cassini/UVIS/EUV1999_007_17_05.LBL")
        ob = from_file("test_data/cassini/UVIS/COUVIS_0034/DATA/D2011_090/EUV2011_090_23_13.LBL")
        print "observation time:"
        print ob.time
        #from_index("test_data/cassini/UVIS/INDEX.LBL")

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

