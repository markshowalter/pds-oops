import oops
import numpy as np
import sys
import csv
import datetime
import os

import pylab
import oops.inst.cassini.iss as cassini_iss
import oops.inst.cassini.vims as cassini_vims
import oops_.surface.ansa as ansa_
from math import log, ceil
from oops_.meshgrid import Meshgrid
import vicar


ISS_TYPE = "ISS"
VIMS_TYPE = "VIMS"
UVIS_TYPE = "UVIS"

################################################################################
# Hanlde command line arguments for generate_tables.py
################################################################################
list_file = 'geometry_list.csv'
radii_file = 'geometry_ring_ranges.csv'
grid_resolution = 8.
start_file_index = 0
stop_file_index = -1
write_frequency = 50
do_output_opus1 = True
do_output_opus2 = True

nArguments = len(sys.argv)
for i in range(nArguments):
    if sys.argv[i] == '-lf':
        if i < (nArguments - 1):
            list_file = sys.argv[i+1]
            i += 1
    elif sys.argv[i] == '-rf':
        if i < (nArguments - 1):
            radii_file = sys.argv[i+1]
            i += 1
    elif sys.argv[i] == '-res':
        if i < (nArguments - 1):
            grid_resolution = float(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-start':
        if i < (nArguments - 1):
            start_file_index = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-stop':
        if i < (nArguments - 1):
            stop_file_index = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-wfreq':
        if i < (nArguments - 1):
            write_frequency = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-opus1':
        if i < (nArguments - 1):
            do_output_opus1 = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-opus2':
        if i < (nArguments - 1):
            do_output_opus2 = int(sys.argv[i+1])
            i += 1
    elif sys.argv[i] == '-h':
        print "usage: python %s [-lf list_file] [-rf radii_file] [-res grid_resolution] [-start start_file_index] [-stop stop_file_index] [-wfreq write_frequency] [-opus1 do_output_opus1] [-opus2 do_output_opus2]" % sys.argv[0]
        print "default values:"
        print "\tlist_file: ", list_file
        print "\tradii_file: ", radii_file
        print "\tgrid_resolution: ", grid_resolution
        print "\tstart_file_index: ", start_file_index
        print "\tstop_file_index (-1 means do all): ", stop_file_index
        print "\twrite_frequency: ", write_frequency
        print "\tdo_output_opus1: ", int(do_output_opus1)
        print "\tdo_output_opus2: ", int(do_output_opus2)
        sys.exit()


def image_code_name(ob, ndx):
    file_name = ob.index_dict['PRODUCT_ID']
    #number_code = file_name.split('.')[0][2:]
    if ndx == 1:
        wave = 'IR'
    else:
        wave = 'VIS'
    image_code = 'S/CUBE/CO/VIMS/' + file_name + '/' + wave
    return image_code

################################################################################

class FileGeometry(object):
    
    def __init__(self, parent=None):
        if parent is None:
            self.clear()
        else:
            self.copy_data(parent)
    
    def clear(self):
        self.image_name = ""
        self.ra = None
        self.dec = None
        self.ring_radius = None
        self.radial_resolution = None
        self.longitude_j2000 = None
        self.longitude_obs = None
        self.longitude_sha = None
        self.phase = None
        self.incidence = None
        self.emission = None
        self.range_to_rings = None
        self.shadow_of_planet = None
        self.planet_behind_rings = None
        self.backlit_rings = None
        self.ansa_radius = None
        self.ansa_elevation = None
        self.ansa_longitude_j2000 = None
        self.ansa_longitude_sha = None
        self.ansa_range = None
        self.ansa_resolution = None
        self.az_sun = None
        self.el_sun = None
        self.az_obs = None
        self.el_obs = None
        self.sub_solar_long = None
        self.sub_solar_lat = None
        self.solar_dist = None
        self.sub_obs_long = None
        self.sub_obs_lat = None
        self.obs_dist = None
    
    def copy_data(self, parent):
        self.image_name = parent.image_name
        self.ra = parent.ra.copy()
        self.dec = parent.dec.copy()
        self.ring_radius = parent.ring_radius.copy()
        self.radial_resolution = parent.radial_resolution.copy()
        self.longitude_j2000 = parent.longitude_j2000.copy()
        self.longitude_obs = parent.longitude_obs.copy()
        self.longitude_sha = parent.longitude_sha.copy()
        self.phase = parent.phase.copy()
        self.incidence = parent.incidence.copy()
        self.emission = parent.emission.copy()
        self.range_to_rings = parent.range_to_rings.copy()
        self.shadow_of_planet = parent.shadow_of_planet.copy()
        self.planet_behind_rings = parent.planet_behind_rings.copy()
        self.backlit_rings = parent.backlit_rings.copy()
		# we don't use ansa data in OPUS 2, which is when we use copy_data()
        """
            self.ansa_radius = parent.ansa_radius.copy()
            self.ansa_elevation = parent.ansa_elevation.copy()
            self.ansa_longitude_j2000 = parent.ansa_longitude_j2000.copy()
            self.ansa_longitude_sha = parent.ansa_longitude_sha.copy()
            self.ansa_range = parent.ansa_range.copy()
            self.ansa_resolution = parent.ansa_resolution.copy()
            """
        self.az_sun = parent.az_sun.copy()
        self.el_sun = parent.el_sun.copy()
        self.az_obs = parent.az_obs.copy()
        self.el_obs = parent.el_obs.copy()
    
    def or_mask(self, mask):
        self.ra.mask |= mask
        self.dec.mask |= mask
        self.ring_radius.mask |= mask
        self.radial_resolution.mask |= mask
        self.longitude_j2000.mask |= mask
        self.longitude_obs.mask |= mask
        self.longitude_sha.mask |= mask
        self.phase.mask |= mask
        self.incidence.mask |= mask
        self.emission.mask |= mask
        self.range_to_rings.mask |= mask
        self.shadow_of_planet.mask |= mask
        self.planet_behind_rings.mask |= mask
        self.backlit_rings.mask |= mask
		# we don't use ansa data in OPUS 2, which is when we use or_mask()
        """
            self.ansa_radius.mask |= mask
            self.ansa_elevation.mask |= mask
            self.ansa_longitude_j2000.mask |= mask
            self.ansa_longitude_sha.mask |= mask
            self.ansa_range.mask |= mask
            self.ansa_resolution.mask |= mask
            """
        self.az_sun.mask |= mask
        self.el_sun.mask |= mask
        self.az_obs.mask |= mask
        self.el_obs.mask |= mask
    
    def set_image_camera(self, obs):
        file_name = obs.index_dict['PRODUCT_ID']
        #number_code = file_name.split('.')[0][2:]
        if "_IR" in obs.path_id:
            wave = 'IR'
        else:
            wave = 'VIS'
        self.image_name = 'S/CUBE/CO/VIMS/' + file_name + '/' + wave
    
    def add_single_minmax_info(self, array, minmax=None):
        if minmax is not None:
            output_buf = ", " + str(minmax[0]) + ", " + str(minmax[1])
        elif array is None:
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        elif np.all(array.mask):
            output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        else:
            output_buf = ", %.15f, %.15f" % (array.min(), array.max())
        return output_buf
    
    def angle_single_limits(self, angles):
        
        #find the histogram for the angles
        bins = np.arange(361)
        m_arr = angles.mvals
        h,b = np.histogram(m_arr[~m_arr.mask], bins)    # hist doesn't support masks
        
        #find all of the sections without values (where the histogram is zero)
        empty_sections = []
        first_non_zero = -1
        i = 0
        in_zero = False
        num_zero = 0
        min_zero = 5
        current_value_start = 0.
        current_value_stop = 0.
        sorted_angles = m_arr.copy().reshape(m_arr.size)    # to get actual values
        sorted_angles.sort()
        samples = 0
        while i < 360:
            if h[i] == 0 and (not in_zero):
                num_zero += 1
                if num_zero == min_zero:
                    current_section_start = i + 1 - num_zero
                    if i != 0:
                        if samples == 0:
                            current_value_start = 0.
                        else:
                            current_value_start = sorted_angles[samples-1]
                    in_zero = True
            elif h[i] != 0:
                num_zero = 0
                if in_zero:
                    current_section = (current_section_start,
                                       i - current_section_start,
                                       current_value_start,
                                       sorted_angles[samples])
                    empty_sections.append(current_section)
                    in_zero = False
                if first_non_zero == -1:
                    first_non_zero = i
            if i == 359:
                if first_non_zero == -1:
                    return (angles.min(), angles.max())   # we have no angles
                if in_zero:
                    current_section = (current_section_start,
                                       i + 1 - current_section_start,
                                       current_value_start, 0.)
                    empty_sections.append(current_section)
            samples += h[i]
            i += 1
        
        # if we have a a gap that starts at zero and ends at 360, we need to combine
        # them
        first_section = empty_sections[0]
        if first_section[0] == 0:
            last_section = empty_sections[-1]
            if (last_section[0] + last_section[1]) == 360:
                new_section = (last_section[0],
                               first_section[1] + last_section[1],
                               last_section[2],
                               first_section[3])
                empty_sections[0] = new_section
                empty_sections.pop()
        
        #find the largest of those
        item = max(empty_sections, key=lambda x: x[1])
        return (item[3], item[2])
    
    def angle_coverage(self, angles):
        """return a list of min/max pairs defining sections of angles that are
            covered in angles.
            """
        #find the histogram for the angles
        bins = np.arange(361)
        m_arr = angles.mvals
        h,b = np.histogram(m_arr[~m_arr.mask], bins)    # hist doesn't support masks
        
        #find all of the sections without values (where the histogram is zero)
        empty_sections = []
        first_non_zero = -1
        i = 0
        in_zero = False
        num_zero = 0
        min_zero = 5
        current_value_start = 0.
        current_value_stop = 0.
        sorted_angles = m_arr.copy().reshape(m_arr.size)    # to get actual values
        sorted_angles.sort()
        samples = 0
        while i < 360:
            if h[i] == 0 and (not in_zero):
                num_zero += 1
                if num_zero == min_zero:
                    current_section_start = i + 1 - num_zero
                    if i != 0:
                        if samples == 0:
                            current_value_start = 0.
                        else:
                            current_value_start = sorted_angles[samples-1]
                    in_zero = True
            elif h[i] != 0:
                num_zero = 0
                if in_zero:
                    current_section = (current_section_start,
                                       i - current_section_start,
                                       current_value_start,
                                       sorted_angles[samples])
                    empty_sections.append(current_section)
                    in_zero = False
                if first_non_zero == -1:
                    first_non_zero = i
            if i == 359:
                if first_non_zero == -1:
                    return (angles.min(), angles.max())   # we have no angles
                if in_zero:
                    current_section = (current_section_start,
                                       i + 1 - current_section_start,
                                       current_value_start, 0.)
                    empty_sections.append(current_section)
            samples += h[i]
            i += 1
        
        # if we have a a gap that starts at zero and ends at 360, we need to combine
        # them
        first_section = empty_sections[0]
        if first_section[0] == 0:
            last_section = empty_sections[-1]
            if (last_section[0] + last_section[1]) == 360:
                new_section = (last_section[0],
                               first_section[1] + last_section[1],
                               last_section[2],
                               first_section[3])
                empty_sections[0] = new_section
                empty_sections.pop()
        
        angle_masks = []
        spans = []
        for i in range(len(empty_sections)):
            min_angle = empty_sections[i][2] / oops.DPR
            max_angle = empty_sections[i][3] / oops.DPR
            span = (max_angle, min_angle)
            spans.append(span)
            a = angles.vals > min_angle
            b = angles.vals < max_angle
            if min_angle > max_angle:
                angle_mask = a | b
            else:
                #print "empty_section: ", empty_sections[i]
                #print "min_angle: ", min_angle
                #print "a: ", a
                #print "b: ", b
                angle_mask = a & b
            angle_masks.append(angle_mask)
        return (angle_masks, spans)
    
    def output_opus2(self, radii_ranges):
        """output to a string lines in the OPUS2 format representing one image.
            
            Input:
            distance_ranges     list of distances for which we have a separate
            line in the output file. For N values in
            radii_ranges, we have N distance ranges to
            evaluate. 0 should always be the first value,
            so that range i includes the range
            [radii_ranges[i], radii_ranges[i+1]) where
            radii_ranges[N] is taken to be infinite.
            """
        l = len(radii_ranges)
        lines = ""
        for i in (range(l-1)):
            start = radii_ranges[i]
            if i < l:
                end = radii_ranges[i+1]
            else:
                end = sys.float_info.max
            a = self.ring_radius.vals < start
            b = self.ring_radius.vals >= end
            radius_mask = a | b
            
            # now for each radius range we have a line for each angle span
            # ACTUALLY, it makes no sense to print a separate line for each
            # span as what is filled in for the other values other than this
            # particular angle?  What about other angles than have multiple
            # min/max spans?
            (angle_masks, spans) = self.angle_coverage(self.longitude_j2000 * oops.DPR)
            j = 0
            for angle_mask in angle_masks:
                sub_mask = radius_mask & angle_mask
                sub_geom = FileGeometry(self)
                sub_geom.or_mask(sub_mask)
                lines += sub_geom.output_single_line(start, end, j+1, spans[j])
                j += 1
        
        return lines
    
    
    def output_single_line(self, start=None, end=None, index=None, span=None):
        """output to a string line in the OPUS1 format"""
        line = self.image_name
        if (start is not None) and (end is not None) and (index is not None):
            # for Opus2 format
            line += ", %f, %f, %d" % (start, end, index)
        line += self.add_single_minmax_info(self.ra * oops.DPR)
        line += self.add_single_minmax_info(self.dec * oops.DPR)
        line += self.add_single_minmax_info(self.ring_radius)
        line += self.add_single_minmax_info(self.radial_resolution)
        #print "self.longitude_j2000.mask: ", self.longitude_j2000.mask
        if np.all(self.longitude_j2000.mask):
            line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
            line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
            line += ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        elif span is not None:
            line += ", %f, %f" % (span[0], span[1])
            line += self.add_single_minmax_info(self.longitude_obs)
            line += self.add_single_minmax_info(self.longitude_sha)
        else:
            limits = self.angle_single_limits(self.longitude_j2000 * oops.DPR)
            line += add_info(self.longitude_j2000, limits)
            limits = self.angle_single_limits(self.longitude_obs * oops.DPR)
            line += self.add_single_minmax_info(self.longitude_obs, limits)
            limits = self.angle_single_limits(self.longitude_sha * oops.DPR)
            line += self.add_single_minmax_info(self.longitude_sha, limits)
        #line += self.add_single_minmax_info(self.longitude_j2000)
        #line += self.add_single_minmax_info(self.longitude_obs)
        #line += self.add_single_minmax_info(self.longitude_sha)
        line += self.add_single_minmax_info(self.phase * oops.DPR)
        line += self.add_single_minmax_info(self.incidence * oops.DPR)
        line += self.add_single_minmax_info(self.emission * oops.DPR)
        line += self.add_single_minmax_info(self.range_to_rings)
        line += " %f," % np.float(np.all(self.shadow_of_planet.vals))
        line += " %f," % np.float(np.any(self.shadow_of_planet.vals))
        #line += self.add_single_minmax_info(self.shadow_of_planet)
        line += " %f," % np.float(np.all(self.planet_behind_rings.vals))
        line += " %f," % np.float(np.any(self.planet_behind_rings.vals))
        #line += self.add_single_minmax_info(self.planet_behind_rings)
        line += " %f," % np.float(np.all(self.backlit_rings.vals))
        line += " %f," % np.float(np.any(self.backlit_rings.vals))
        #line += self.add_single_minmax_info(self.backlit_rings)
        if start is None:
            line += self.add_single_minmax_info(self.ansa_radius)
            line += self.add_single_minmax_info(self.ansa_elevation)
            limits = self.angle_single_limits(self.ansa_longitude_j2000 * oops.DPR)
            line += self.add_single_minmax_info(self.ansa_longitude_j2000, limits)
            limits = self.angle_single_limits(self.ansa_longitude_sha * oops.DPR)
            line += self.add_single_minmax_info(self.ansa_longitude_sha, limits)
            line += self.add_single_minmax_info(self.ansa_range)
            line += self.add_single_minmax_info(self.ansa_resolution)
        
        # new backplanes
        line += self.add_single_minmax_info(self.az_sun * oops.DPR)
        line += self.add_single_minmax_info(self.el_sun * oops.DPR)
        line += self.add_single_minmax_info(self.az_obs * oops.DPR)
        line += self.add_single_minmax_info(self.el_obs * oops.DPR)
        
        """if start is None:
            line += " %f," % self.sub_solar_long.vals
            line += " %f," % self.sub_solar_lat.vals
            line += " %f," % self.solar_dist.vals
            line += " %f," % self.sub_obs_long.vals
            line += " %f," % self.sub_obs_lat.vals
            line += " %f," % self.obs_dist.vals"""
        line += '\n'
        return line




spacer = '    ,   '
radii_ranges = []
geom_file_name = ""

PRINT = False
DISPLAY = False
OBJC_DISPLAY = False

def add_info(array, minmax=None):
    if PRINT:
        print "array min/max:"
    if minmax is not None:
        if PRINT:
            print ", %f, %f" % (minmax[0], minmax[1])
        output_buf = ", " + str(minmax[0]) + ", " + str(minmax[1])
    elif np.all(array.mask):
        if PRINT:
            print ", -0.1000000000000000E+31, -0.1000000000000000E+31"
        output_buf = ", -0.1000000000000000E+31, -0.1000000000000000E+31"
    else:
        if PRINT:
            print array.min()
            print array.max()
        output_buf = ", %.15f, %.15f" % (array.min(), array.max())
    return output_buf

def show_info(title, array):
    """Internal method to print summary information and display images as
        desired."""
    
    print "into show_info with array: ", array
    global PRINT, DISPLAY
    if not PRINT:
        if OBJC_DISPLAY:
            print "printing imsave area 0.3"
            if isinstance(array, np.ndarray):
                print "printing imsave area 0.5"
                if array.dtype == np.dtype("bool"):
                    print "printing imsave area 1"
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array, vmin=0, vmax=1, cmap=pylab.cm.gray)
                else:
                    minval = np.min(array)
                    maxval = np.max(array)
                    if minval != maxval:
                        print "printing imsave area 2"
                        pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                     array, cmap=pylab.cm.gray)
            elif isinstance(array, oops.Array):
                print "printing imsave area 2.5"
                if np.any(array.mask):
                    print "printing imsave area 3"
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array.vals)
                else:
                    minval = np.min(array.vals)
                    maxval = np.max(array.vals)
                    if minval != maxval:
                        print "printing imsave area 4"
                        pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                     array.vals, cmap=pylab.cm.gray)
        return
    
    print ""
    print title
    
    print "printing imsave area 5"
    if isinstance(array, np.ndarray):
        print "printing imsave area 5.3"
        if array.dtype == np.dtype("bool"):
            print "printing imsave area 5.4"
            count = np.sum(array)
            total = np.size(array)
            percent = int(count / float(total) * 100. + 0.5)
            print "   ", (count, total-count),
            print (percent, 100-percent), "(True, False pixels)"
            if DISPLAY:
                ignore = pylab.imshow(array, norm=None, vmin=0, vmax=1)
                ignore = raw_input(title + ": ")
            elif OBJC_DISPLAY:
                pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                             array, vmin=0, vmax=1, cmap=pylab.cm.gray)
        
        else:
            print "printing imsave area 5.5"
            minval = np.min(array)
            maxval = np.max(array)
            print "minval = ", minval
            print "maxval = ", maxval
            if minval == maxval:
                print "    ", minval
            else:
                print "    ", (minval, maxval), "(min, max)"
                
                if DISPLAY:
                    ignore = pylab.imshow(array, cmap=pylab.cm.gray)
                    ignore = raw_input(title + ": ")
                elif OBJC_DISPLAY:
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array, cmap=pylab.cm.gray)
    
    elif isinstance(array, oops.Array):
        print "printing imsave area 6"
        if np.any(array.mask):
            print "    ", (np.min(array.vals),
                           np.max(array.vals)), "(unmasked min, max)"
            print "    ", (array.min(),
                           array.max()), "(masked min, max)"
            masked = np.sum(array.mask)
            total = np.size(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print "    ", (masked, total-masked),
            print         (percent, 100-percent), "(masked, unmasked pixels)"
            
            if DISPLAY:
                ignore = pylab.imshow(array.vals)
                ignore = raw_input(title + ": ")
                background = np.zeros(array.shape, dtype="uint8")
                background[0::2,0::2] = 1
                background[1::2,1::2] = 1
                ignore = pylab.imshow(background)
                ignore = pylab.imshow(array.mvals)
                ignore = raw_input(title + ": ")
            elif OBJC_DISPLAY:
                pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                             array.vals, cmap=pylab.cm.gray)
        
        else:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print "    ", minval
            else:
                print "    ", (minval, maxval), "(min, max)"
                
                if DISPLAY:
                    ignore = pylab.imshow(array.vals)
                    ignore = raw_input(title + ": ")
                elif OBJC_DISPLAY:
                    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png",
                                 array.vals, cmap=pylab.cm.gray)
    
    else:
        print "    ", array

def is_power_of_2(n):
    return log(n, 2) % 1.0 == 0.0

def next_power_of_2(n):
    return (2 ** ceil(log(n, 2)))

def number_of_digits(ivalue, ndigits=0):
    ivalue_10 = ivalue / 10
    if ivalue_10 > 0:
        return number_of_digits(ivalue_10, ndigits + 1)
    return ndigits + 1

def width_str(width, value):
    ivalue = int(value)
    ndigits = number_of_digits(ivalue)
    n_sig_digits = width - ndigits
    fmt_str = '%%%d.%df%%s' % (width, n_sig_digits)
    ret_str = fmt_str % (value, spacer)
    return ret_str

def argmax(items, func, cmp=cmp):
    '''argmax(items, func) -> max_item, max_value'''
    it = iter(items)
    # Initial values
    try:
        item = it.next()
    except StopIteration:
        raise ValueError("can't run over empty sequence")
    val = func(item)
    
    for i in it:
        v = func(i)
        if cmp(v, val) == 1:
            item = i
            val = v
    return item, val

def argmin(items, func, cmp=cmp):
    '''argmin(items, func) -> min_item, min_value'''
    return argmax(items, func, lambda x,y : -cmp(x,y))

def create_overlay_image(overlay_mask, resolution, underlay_file_name,
                         overlay_data=None):
    rel_path = "./test_data/cassini/ISS/" + underlay_file_name
    vic = vicar.VicarImage.from_file(rel_path)
    vic_res = vic.data[0].shape
    ov_res = np.empty([vic_res[0], vic_res[1]])
    ir = int(resolution)
    mask_val = vic.data[0].max()
    for j in range(0,vic_res[1]):
        for i in range(0,vic_res[0]):
            d = overlay_mask.vals[i/ir][j/ir]
            if d == 0.:
                ov_res[i][j] = vic.data[0][i][j]
            elif overlay_data is None:
                ov_res[i][j] = mask_val
            else:
                ov_res[i][j] = overlay_data[i/ir][j/ir]
    pylab.imsave("/Users/bwells/lsrc/pds-tools/tempImage.png", ov_res,
                 cmap=pylab.cm.gray)

def get_error_buffer_size(ob):
    # deal with possible pointing errors - up to 3 pixels in any
    # direction for WAC and 30 pixels for NAC
    error_buffer = 1    # don't know what it should be for VIMS
    #error_buffer = 60
    #if 'NORMAL' not in ob.index_dict['VIS_SAMPLING_MODE_ID']:
    #    error_buffer = 6
    return error_buffer

def snapshot_has_partial_intercept(snapshot, resolution):
    error_buffer = get_error_buffer_size(snapshot)
    limit = snapshot.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                        error_buffer]))
    
    meshgrid = Meshgrid.for_fov(snapshot.fov, undersample=resolution,
                                limit=limit, swap=True)
    bp = oops.Backplane(snapshot, meshgrid)
    
    intercepted = bp.where_intercepted("saturn_rings")
    
    if np.all(intercepted.vals):
        return 1
    elif np.any(intercepted.vals):
        return 0
    else:
        return -1


def generate_metadata(obs, resolution, ndx):
    
    #print the camera field
    geometry = FileGeometry()
    geometry.image_name = image_code_name(obs, ndx)
    #geometry.set_image_camera(snapshot)
    
    # deal with possible pointing errors - up to 3 pixels in any
    # direction for WAC and 30 pixels for NAC
    error_buffer = get_error_buffer_size(obs)
    #print "snapshot.fov.uv_shape: ", snapshot.fov.uv_shape
    limit = obs.fov.uv_shape + oops.Pair(np.array([error_buffer,
                                                   error_buffer]))
    (meshgrid, times) = cassini_vims.meshgrid_and_times(obs)
    #(meshgrid, times) = cassini_vims.meshgrid_and_times(obs,
    #                                                    undersample=resolution)
    bp = oops.Backplane(obs, meshgrid, times)
    try:
        intercepted = bp.where_intercepted("saturn_rings")
    except:
        print "generate_metadata() returning None"
        return None
    #show_info("Intercepted:", intercepted.vals)
    
    saturn_in_front = bp.where_in_front("saturn", "saturn_rings")
    saturn_intercepted = bp.where_intercepted("saturn")
    """
        show_info("saturn_intercepted:", saturn_intercepted.vals)
        if np.all(saturn_intercepted.vals):
        print "SATURN INTEREPTED EVERYWHERE"
        elif np.any(saturn_intercepted.vals):
        print "SATURN INTERCEPTED SOME PLACES"
        else:
        print "SATURN NOT INTERCEPTED"
        """
    rings_blocked = saturn_in_front & saturn_intercepted
    rings_in_view = intercepted & (~rings_blocked)
    """
        if np.all(rings_in_view.vals):
        print "ONLY RINGS CLAIM IN VIEW"
        else:
        print "RINGS CLAIM PARTIAL VIEW"
        """
    rings_not_in_shadow = bp.where_outside_shadow("saturn_rings", "saturn")
    vis_rings_not_shadow = rings_in_view & rings_not_in_shadow
	#show_info("Visible Rings not in shadow:", vis_rings_not_shadow.vals)
    
    test = bp.right_ascension()
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ra = test.copy()
    
	#print "Doing declination:"
    test = bp.declination()
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.dec = test.copy()
    
	#print "Doing ring radius:"
    test = bp.ring_radius("saturn_ring_plane")
    """
        print "###################################################################"
        if np.all(test.mask):
        print "ring_radius mask entirely true"
        elif np.any(test.mask):
        print "ring_radius mask partially true"
        else:
        print "ring_radius mask entirely false"
        print "ring radius: ", test
        print "###################################################################"
        """
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.ring_radius = test.copy()
    
    test = bp.ring_radial_resolution("saturn_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.radial_resolution = test.copy()
    
    test = bp.ring_longitude("saturn_rings",reference="j2000")
	#show_info("Ring longitude:", test.vals)
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_j2000 = test.copy()
    
    test = bp.ring_longitude("saturn_rings", reference="obs")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_obs = test.copy()
    
    test = bp.ring_longitude("saturn_rings", reference="sha")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.longitude_sha = test.copy()
    
    test = bp.phase_angle("saturn_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.phase = test.copy()
    
    test = bp.incidence_angle("saturn_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.incidence = test.copy()
    
    test = bp.emission_angle("saturn_rings")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.emission = test.copy()
    
    test = bp.distance("saturn_rings")
	#print "distance: ", test
    test.mask |= ~vis_rings_not_shadow.vals
    range_test = test.rebroadcast(test.mask.shape)
    geometry.range_to_rings = range_test.copy()
    
    geometry.shadow_of_planet = rings_not_in_shadow.copy()
    
    geometry.planet_behind_rings = rings_in_view.copy()
    
    test = bp.where_sunward("saturn_rings")
    geometry.backlit_rings = test.copy()
    
    #####################################################
    # ANSA surface values
    #####################################################
    saturn_intercepted = bp.where_intercepted("saturn:ansa")
    #show_info("ansa intercepted:", saturn_intercepted)
    
    test = bp.ansa_radius("saturn:ansa")
    test.mask |= ~saturn_intercepted.vals
    geometry.ansa_radius = test.copy()
    
    test = bp.ansa_altitude("saturn:ansa")
    test.mask |= ~saturn_intercepted.vals
    geometry.ansa_elevation = test.copy()
    
    test = bp.ansa_longitude("saturn:ansa",reference="j2000")
    #print "test of ansa_longitude before mask: ", test
    test.mask |= ~saturn_intercepted.vals
    #print "test of ansa_longitude after mask: ", test
    geometry.ansa_longitude_j2000 = test.copy()
    
    test = bp.ansa_longitude("saturn:ansa", reference="sha")
    test.mask |= ~saturn_intercepted.vals
    geometry.ansa_longitude_sha = test.copy()
    
    test = bp.distance("saturn:ansa")
    test.mask |= ~saturn_intercepted.vals
    geometry.ansa_range = test.copy()
    
    test = bp.ansa_radial_resolution("saturn:ansa")
    test.mask |= ~saturn_intercepted.vals
    geometry.ansa_resolution = test.copy()
    
	#####################################################
	# new backplanes and other values
	#####################################################
    test = bp.ring_azimuth("saturn_rings",reference="sun")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.az_sun = test.copy()
    
    test = bp.ring_elevation("saturn_rings",reference="sun")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.el_sun = test.copy()
    
    test = bp.ring_azimuth("saturn_rings",reference="obs")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.az_obs = test.copy()
    
    test = bp.ring_elevation("saturn_rings",reference="obs")
    test.mask |= ~vis_rings_not_shadow.vals
    geometry.el_obs = test.copy()
    
    geometry.sub_solar_long = bp.sub_solar_longitude("saturn_rings")
    geometry.sub_solar_lat = bp.sub_solar_latitude("saturn_rings")
    #geometry.solar_dist = bp.solar_distance_to_center("saturn_rings")
    geometry.solar_dist = bp.center_distance("saturn_rings", "sun")
    geometry.sub_obs_long = bp.sub_observer_longitude("saturn_rings")
    geometry.sub_obs_lat = bp.sub_observer_latitude("saturn_rings")
    #geometry.obs_dist = bp.observer_distance_to_center("saturn_rings")
    geometry.obs_dist = bp.center_distance("saturn_rings")
    
    return geometry

def output_opus1_file(file_name, geometries):
    f = open(file_name, 'w')
    output_buf = ""
    for geometry in geometries:
        output_buf += geometry.output_single_line()
    f.write(output_buf)
    f.close()

def output_opus2_file(file_name, geometries, radii_ranges):
    f = open(file_name, 'w')
    output_buf = ""
    for geometry in geometries:
        output_buf += geometry.output_opus2(radii_ranges)
    f.write(output_buf)
    f.close()

def append_opus1_file(file_name, geometries, start, stop):
    if start < write_frequency:
        f = open(file_name, 'w')
    else:
        f = open(file_name, 'a')
    output_buf = ""
    for i in range(start, stop):
        output_buf += geometries[i].output_single_line()
    f.write(output_buf)
    f.close()

def append_opus2_file(file_name, geometries, radii_ranges, start, stop):
    if start < write_frequency:
        f = open(file_name, 'w')
    else:
        f = open(file_name, 'a')
    output_buf = ""
    for i in range(start, stop):
        output_buf += geometries[i-start].output_opus2(radii_ranges)
    f.write(output_buf)
    f.close()


def look_for_proper_rotation(file_name, omit_range, fortran_list, j):
    snapshots = cassini_vims.different_rots_from_index(index_file_name, start_file_index, j)
    
    nSnapshots = len(snapshots)
    if stop_file_index > 0 and stop_file_index < nSnapshots:
        nSnapshots = stop_file_index
    
    snapshot = snapshots[0]
    if snapshot_has_partial_intercept(snapshot, grid_resolution):
        print "snapshot %d does not have partial intercept" % j
    else:
        print "snapshot %d has partial intercept!!!!!!!!!!!!!!!!!" % j

def load_vims_observations(valid_path):
    obs = []
    i = 0
    break_the_loop = False
    for root, dirs, files in os.walk(valid_path):
        for name in files:
            if ("DATA" in root) and (".LBL" in name):
                fname = os.path.join(root,name)
                print "fname:", fname
                ob = cassini_vims.from_file(fname)
                obs.append(ob)
                i += 1
                if i >= stop_file_index:
                    return obs
    return obs

def generate_table_for_index(file_name):
    """Create the geometry objects for all the image files in the list of files
        within the index file file_name.
        
        Input:
        file_name       index file for this volume.
        omit_range      list of images, by index, to omit that are not
        omitted in the FORTRAN code.
        fortran_list    list of files processed in the FORTRAN code (so we
        can avoid further segmnentation faults.
        """
    output_buf = ''
    obs = load_vims_observations(file_name)
    nObs = len(obs)
    if stop_file_index > 0 and stop_file_index < nObs:
        nObs = stop_file_index
    i = 0
    then = datetime.datetime.now()
    start = start_file_index
    geometries = []
    info_str = ""
    actual_i = start
    zero_time = then - then
    write_time = zero_time
    nIOs = 1
    if write_frequency > 0:
        nIOs += (nObs - start) / write_frequency
    iIOs = 0
    previous_write_end = start
    for i in range(start, nObs):
        info_len = len(info_str)
        i1 = i + 1
        init_info_str = "    " + str(i+1) + " of " + str(nObs)
        for j in range(2):
            ob = obs[i][j]
            #image_code = image_code_name(ob)
            try:
                geometry = generate_metadata(ob, grid_resolution, j)
            except:
                continue
            if geometry is not None:
                geometries.append(geometry)
                actual_i += 1
        
        # write time and image number status to stdout
        now = datetime.datetime.now()
        time_so_far = now - then
        time_left = time_so_far * (nObs - start) / (i + 1 - start) - time_so_far
        if write_time != zero_time:
            time_left += (nIOs - iIOs) * write_time
        init_info_str += ", time rem: " + str(time_left)
        info_str = init_info_str.split('.')[0]
        for item in range(0, info_len):
            sys.stdout.write('\b')
        sys.stdout.write(info_str)
        sys.stdout.flush()
        try:
            if write_frequency > 0 and (actual_i % write_frequency) == 0:
                if write_time == zero_time:
                    write_then = datetime.datetime.now()
                i0 = actual_i - write_frequency
                if do_output_opus1:
                    info_str = "appending rows %d to %d to OPUS 1 file %s" % (previous_write_end, actual_i, geom_file_name)
                    for item in range(0, info_len):
                        sys.stdout.write('\b')
                    sys.stdout.write(info_str)
                    sys.stdout.flush()
                    info_len = len(info_str)
                    
                    #print "\nappending rows %d to %d to OPUS 1 file %s" % (i0,
                    #                                                       actual_i,
                    #                                                       geom_file_name)
                    append_opus1_file(geom_file_name, geometries,
                                      previous_write_end, actual_i)
                    previous_write_end = actual_i
                if do_output_opus2:
                    geom_file_name2 = geom_file_name.replace("summary", "detailed")
                    info_str = "appending rows %d to %d to OPUS 2 file %s" % (previous_write_end, actual_i, geom_file_name2)
                    for item in range(0, info_len):
                        sys.stdout.write('\b')
                    sys.stdout.write(info_str)
                    sys.stdout.flush()
                    info_len = len(info_str)
                    #print "appending rows %d to %d to OPUS 2 file %s" % (i0,
                    #                                                     actual_i,
                    #                                                     geom_file_name2)
                    append_opus2_file(geom_file_name2, geometries, radii_ranges,
                                      previous_write_end, actual_i)
                    previous_write_end = actual_i
                    for item in range(0, info_len):
                        sys.stdout.write('\b')
                    for item in range(0, info_len):
                        sys.stdout.write(' ')
                if write_time == zero_time:
                    write_time = datetime.datetime.now() - write_then
                iIOs += 1
        except:
            print "skipping area 2:", i
            continue
    
    #i += 1
    
    if write_frequency > 0 and (actual_i % write_frequency) != 0:
        if do_output_opus1:
            print "\nappending rows %d to %d to OPUS 1 file %s" % (previous_write_end,
                                                                   actual_i,
                                                                   geom_file_name)
            append_opus1_file(geom_file_name, geometries, previous_write_end,
                              actual_i)
        if do_output_opus2:
            geom_file_name2 = geom_file_name.replace("summary", "detailed")
            print "appending rows %d to %d to OPUS 2 file %s" % (previous_write_end,
                                                                 actual_i,
                                                                 geom_file_name2)
            append_opus2_file(geom_file_name2, geometries, radii_ranges,
                              previous_write_end, actual_i)
    
    sys.stdout.write('\n')
    #return output_buf
    return geometries

radiiReader = csv.reader(open(radii_file, 'rU'), delimiter=';')
for row in radiiReader:
    for i in range(len(row)):
        radii_ranges.append(float(row[i]))

volumeReader = csv.reader(open(list_file, 'rU'), delimiter=';')
for row in volumeReader:
    index_file_name = str(row[0])
    geom_file_name = str(row[1])
    omit_range = []
    len_of_row = len(row)
    print "Generating geometry table for file: ", index_file_name
    geometries = generate_table_for_index(index_file_name)
    if write_frequency < 0:
        if do_output_opus1:
            print "writing OPUS 1 file ", geom_file_name
            output_opus1_file(geom_file_name, geometries)
        if do_output_opus2:
            geom_file_name2 = geom_file_name.replace("summary", "detailed")
            print "writing OPUS 2 file ", geom_file_name2
            output_opus2_file(geom_file_name2, geometries, radii_ranges)
print "Done."
