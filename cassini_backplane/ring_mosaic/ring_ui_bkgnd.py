'''
Created on Sep 23, 2011

@author: rfrench
'''

from optparse import OptionParser
import ringutil
import mosaic
import modelbkgnd
import pickle
import os.path
import numpy as np
import numpy.ma as ma
import sys
import gc
import cspice
from imgdisp import ImageDisp
from Tkinter import *
from PIL import Image

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    #fake_cmd_line = ['ISS_036RF_FMOVIE001_VIMS']
#    cmd_line = ['--verbose', '--no-bkgnd', 'ISS_029RF_FMOVIE001_VIMS', '--display-bkgnd']
#    cmd_line = ['--verbose', 'ISS_085RF_FMOVIE003_PRIME_1', '--recompute-bkgnd']
    cmd_line = ['ISS_082RI_FMONITOR003_PRIME','--mosaic-reduction-factor', '2', '--display-bkgnd', '--verbose']
#    cmd_line = ['-a', '--verbose', '--mosaic-reduction-factor', '2', '--no-update-bkgnd']

parser = OptionParser()

parser.add_option('--no-bkgnd', dest='no_bkgnd',
                  action='store_true', default=False,
                  help="Don't compute the background even if we don't have one")
parser.add_option('--no-update-bkgnd', dest='no_update_bkgnd',
                  action='store_true', default=False,
                  help="Don't compute the background unless we don't have one")
parser.add_option('--recompute-bkgnd', dest='recompute_bkgnd', action='store_true', default=False)
parser.add_option('--display-bkgnd', dest='display_bkgnd', action='store_true', default=False)
parser.add_option('--use-default-params', dest='use_default_params', action='store_true', default=False)

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

class BkgndDispData:
    def __init__(self):
        self.toplevel = None
        self.mask_overlay = None
        self.imdisp = None
        self.radial_profile_canvas = None
        self.label_longitude = None
        self.label_radius = None
        self.label_phase = None
        self.label_incidence = None
        self.label_emission = None
        self.label_resolution = None
        self.label_image = None
        self.label_obsid = None
        self.label_date = None
        self.label_ew = None
        self.label_ewmu = None
        self.label_ew_stats = None
        self.label_ewmu_stats = None
        self.var_row_cutoff = None
        self.scale_row_cutoff = None
        self.var_column_cutoff = None
        self.scale_column_cutoff = None
        self.var_ignore_fraction = None
        self.scale_ignore_fraction = None
        self.var_row_blur = None
        self.scale_row_blur = None
        self.var_ring_lower_limit = None
        self.scale_ring_lower_limit = None
        self.var_ring_upper_limit = None
        self.scale_ring_upper_limit = None
        self.var_inside_background_pixels = None
        self.var_outside_background_pixels = None
        self.scale_inside_background_pixels = None
        self.scale_outside_background_pixels = None
        self.var_degree = None
        self.scale_degree = None
        self.radial_profile_last_x = None
        self.radial_profile_last_y = None

#####################################################################################
#
# COMPUTE ONE BKGND
#
#####################################################################################

#
# Read in mosaic if we don't already have it; create a reduced mosaic if we don't already have one
#
def read_mosaic(bkgnddata):
    if bkgnddata.mosaic_img == None:
        create_reduced = False
        if (os.path.exists(bkgnddata.reduced_mosaic_metadata_filename) and
            os.path.exists(bkgnddata.reduced_mosaic_data_filename+'.npy') and
            os.path.exists(bkgnddata.mosaic_data_filename+'.npy') and
            os.stat(bkgnddata.reduced_mosaic_metadata_filename).st_mtime > os.stat(bkgnddata.mosaic_data_filename+'.npy').st_mtime and
            os.stat(bkgnddata.reduced_mosaic_data_filename+'.npy').st_mtime > os.stat(bkgnddata.mosaic_data_filename+'.npy').st_mtime):
            # If we already have a reduced file, read it in
            metadata_filename = bkgnddata.reduced_mosaic_metadata_filename
            data_filename = bkgnddata.reduced_mosaic_data_filename
        else:
            # We don't have a reduced file, so read the normal full mosaic
            metadata_filename = bkgnddata.mosaic_metadata_filename
            data_filename = bkgnddata.mosaic_data_filename
            create_reduced = True # Create the reduced file later
            
        mosaic_metadata_fp = open(metadata_filename, 'rb')
        meta_data = pickle.load(mosaic_metadata_fp)
        bkgnddata.obsid_list = pickle.load(mosaic_metadata_fp)
        bkgnddata.image_name_list = pickle.load(mosaic_metadata_fp)
        bkgnddata.full_filename_list = pickle.load(mosaic_metadata_fp)
        mosaic_metadata_fp.close()
        (bkgnddata.longitudes, bkgnddata.resolutions, bkgnddata.image_numbers,
         bkgnddata.ETs, bkgnddata.emission_angles, bkgnddata.incidence_angles,
         bkgnddata.phase_angles) = meta_data
    
        bkgnddata.mosaic_img = np.load(data_filename+'.npy')
    
        # If we need to, reduce the mosaic and save the new version for future use
        if options.mosaic_reduction_factor != 1 and create_reduced:
            bkgnddata.mosaic_img = bkgnddata.mosaic_img[:,::options.mosaic_reduction_factor]
            bkgnddata.longitudes = bkgnddata.longitudes[::options.mosaic_reduction_factor]
            bkgnddata.resolutions = bkgnddata.resolutions[::options.mosaic_reduction_factor]
            bkgnddata.image_numbers = bkgnddata.image_numbers[::options.mosaic_reduction_factor]
            bkgnddata.ETs = bkgnddata.ETs[::options.mosaic_reduction_factor]
            bkgnddata.emission_angles = bkgnddata.emission_angles[::options.mosaic_reduction_factor]
            bkgnddata.incidence_angles = bkgnddata.incidence_angles[::options.mosaic_reduction_factor]
            bkgnddata.phase_angles = bkgnddata.phase_angles[::options.mosaic_reduction_factor]
            reduced_metadata = (bkgnddata.longitudes, bkgnddata.resolutions,
                                bkgnddata.image_numbers, bkgnddata.ETs, 
                                bkgnddata.emission_angles, bkgnddata.incidence_angles,
                                bkgnddata.phase_angles)
            if create_reduced:
                np.save(bkgnddata.reduced_mosaic_data_filename, bkgnddata.mosaic_img)
                reduced_mosaic_metadata_fp = open(bkgnddata.reduced_mosaic_metadata_filename, 'wb')
                pickle.dump(reduced_metadata, reduced_mosaic_metadata_fp)
                pickle.dump(bkgnddata.obsid_list, reduced_mosaic_metadata_fp)
                pickle.dump(bkgnddata.image_name_list, reduced_mosaic_metadata_fp)
                pickle.dump(bkgnddata.full_filename_list, reduced_mosaic_metadata_fp)
                reduced_mosaic_metadata_fp.close()

    bkgnddata.bkgnd_model_mask = None # Start with an empty mask
    
#
# Check if all the background files already exist
#
def all_bkgnd_files_exist(bkgnddata):
    return ((options.mosaic_reduction_factor == 1 or os.path.exists(bkgnddata.reduced_mosaic_data_filename+'.npy')) and
            (options.mosaic_reduction_factor == 1 or os.path.exists(bkgnddata.reduced_mosaic_metadata_filename)) and
            os.path.exists(bkgnddata.bkgnd_mask_filename+'.npy') and
            os.path.exists(bkgnddata.bkgnd_model_filename+'.npy') and
            os.path.exists(bkgnddata.bkgnd_metadata_filename))

#
# Commit the current results
#
def save_bkgnd_results(bkgnddata):
    np.save(bkgnddata.bkgnd_model_filename, bkgnddata.bkgnd_model.data)
    np.save(bkgnddata.bkgnd_mask_filename, ma.getmaskarray(bkgnddata.bkgnd_model))
    
    bkgnd_metadata_fp = open(bkgnddata.bkgnd_metadata_filename, 'wb')
    bkgnd_data = (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction, bkgnddata.row_blur,
                  bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit, bkgnddata.column_cutoff_sigmas,
                  bkgnddata.column_inside_background_pixels, bkgnddata.column_outside_background_pixels,
                  bkgnddata.degree)
    bkgnd_data = pickle.dump(bkgnd_data, bkgnd_metadata_fp)
    bkgnd_metadata_fp.close()

#
# Compute background for one mosaic, if we don't already have one (based on options)
#
def make_bkgnd(bkgnddata, option_no, option_no_update, option_recompute, save_results=True, override_defaults=False):
    if options.verbose:
        print '** Compute background', bkgnddata.obsid
            
    if option_no:  # Just don't do anything - we hope you know what you're doing!
        if options.verbose:
            print 'Ignored because of --no-bkgnd'
        return

    if not option_recompute:
        if all_bkgnd_files_exist(bkgnddata):
            if option_no_update:
                if options.verbose:
                    print 'Not doing anything because output files already exist and --no-update-bkgnd'
                return  # Mosaic file already exists, don't update

            max_mosaic_mtime = max(os.stat(bkgnddata.mosaic_data_filename+'.npy').st_mtime,
                                   os.stat(bkgnddata.mosaic_metadata_filename).st_mtime)
            if ((options.mosaic_reduction_factor == 1 or os.stat(bkgnddata.reduced_mosaic_data_filename+'.npy').st_mtime > max_mosaic_mtime) and
                ((options.mosaic_reduction_factor == 1 or os.stat(bkgnddata.reduced_mosaic_metadata_filename).st_mtime > max_mosaic_mtime) and
                os.stat(bkgnddata.bkgnd_mask_filename+'.npy').st_mtime > max_mosaic_mtime and
                os.stat(bkgnddata.bkgnd_model_filename+'.npy').st_mtime > max_mosaic_mtime and
                os.stat(bkgnddata.bkgnd_metadata_filename).st_mtime > max_mosaic_mtime)):
                # The mosaic file exists and is more recent than the reprojected images, and we're not forcing a recompute
                if options.verbose:
                    print 'Not doing anything because output files already exist and are current'
                return
    
    read_mosaic(bkgnddata)

    if os.path.exists(bkgnddata.bkgnd_metadata_filename) and not options.use_default_params and not override_defaults:
        read_bkgnd_metadata(bkgnddata)

    mosaic_masked = None
    bkgnddata.bkgnd_model = None
    if options.verbose:
        print 'Computing background model'

    mosaic_masked = modelbkgnd.MaskImage(bkgnddata.mosaic_img, cutoff_sigmas=bkgnddata.row_cutoff_sigmas,
                                         ignore_fraction=bkgnddata.row_ignore_fraction,
                                         row_blur=bkgnddata.row_blur, debug = True)
    bkgnddata.bkgnd_model = modelbkgnd.ModelBackground(mosaic_masked,
                                                       ring_rows=(bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit),
                                                       cutoff_sigmas=bkgnddata.column_cutoff_sigmas,
                                                       background_pixels=(bkgnddata.column_inside_background_pixels,
                                                                          bkgnddata.column_outside_background_pixels),
                                                       degree=bkgnddata.degree, debug=True, debug_col=10)
    del mosaic_masked
    mosaic_masked = None

    bkgnddata.bkgnd_model_mask = ma.getmaskarray(bkgnddata.bkgnd_model) 

    bkgnddata.corrected_mosaic_img = bkgnddata.mosaic_img - bkgnddata.bkgnd_model
    
    if save_results:
        save_bkgnd_results(bkgnddata)


#####################################################################################
#
# DISPLAY ONE BKGND
#
#####################################################################################

# "Recalculate background" button pressed
def command_recalc_bkgnd(bkgnddata, bkgnddispdata):
    bkgnddata.row_cutoff_sigmas = bkgnddispdata.var_row_cutoff.get() 
    bkgnddata.row_ignore_fraction = bkgnddispdata.var_ignore_fraction.get()
    bkgnddata.row_blur = bkgnddispdata.var_row_blur.get()
    bkgnddata.ring_lower_limit = bkgnddispdata.var_ring_lower_limit.get()
    bkgnddata.ring_upper_limit = bkgnddispdata.var_ring_upper_limit.get()
    bkgnddata.column_cutoff_sigmas = bkgnddispdata.var_column_cutoff.get()
    bkgnddata.column_inside_background_pixels = bkgnddispdata.var_inside_background_pixels.get()
    bkgnddata.column_outside_background_pixels = bkgnddispdata.var_outside_background_pixels.get()
    bkgnddata.degree = bkgnddispdata.var_degree.get()
    make_bkgnd(bkgnddata, False, False, True, save_results=False, override_defaults=True)
    update_ew_stats(bkgnddata)
    del bkgnddispdata.mask_overlay
    bkgnddispdata.mask_overlay = np.zeros((bkgnddata.mosaic_img.shape[0],
                                           bkgnddata.mosaic_img.shape[1], 3))
    if bkgnddata.bkgnd_model_mask != None:
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 0] = 1
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 1] = 0
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 2] = 0

    bkgnddispdata.imdisp.set_overlay(0, bkgnddispdata.mask_overlay)    
    radial_profile_update(bkgnddata, bkgnddispdata)

# "Commit changes" button pressed
def command_commit_changes(offrepdata, offrepdispdata):
    save_bkgnd_results(bkgnddata)


#
# The mouse move callback - plot the radial profile
#
def callback_radial(x, y, bkgnddata, bkgnddispdata):
    longitude_num = x
    if longitude_num < 0 or longitude_num > bkgnddata.mosaic_img.shape[1]:
        return
    if bkgnddata.bkgnd_model == None:
        return
    bkgnddispdata.radial_profile_last_x = x
    bkgnddispdata.radial_profile_last_y = y
    radial_profile_update(bkgnddata, bkgnddispdata)
    
def radial_profile_update(bkgnddata, bkgnddispdata):
    if bkgnddispdata.radial_profile_last_x == None:
        return
    
    longitude_num = bkgnddispdata.radial_profile_last_x
    
    mosaic_img_min = ma.min(bkgnddata.mosaic_img)
    mosaic_img_max = ma.max(bkgnddata.mosaic_img)
    radial_profile = bkgnddata.mosaic_img[:,longitude_num] - mosaic_img_min
    radial_profile_mask = bkgnddata.bkgnd_model_mask[:,longitude_num]
    bkgnd_profile = bkgnddata.bkgnd_model.data[:,longitude_num] - mosaic_img_min
    bkgnddispdata.radial_profile_canvas.delete('line')
    xsize = float(bkgnddispdata.radial_profile_canvas.cget('width'))
    ysize = float(bkgnddispdata.radial_profile_canvas.cget('height'))
    xscale = float(xsize) / float(radial_profile.shape[0])
    yscale = float(ysize) / float(mosaic_img_max-mosaic_img_min)

    # Plot lower and upper ring limits
    bkgnddispdata.radial_profile_canvas.create_line([bkgnddispdata.var_ring_lower_limit.get() * xscale, 0,
                                                     bkgnddispdata.var_ring_lower_limit.get() * xscale, (mosaic_img_max-mosaic_img_min)*yscale],
                                                    fill='green', tags='line')
    bkgnddispdata.radial_profile_canvas.create_line([bkgnddispdata.var_ring_upper_limit.get() * xscale, 0,
                                                     bkgnddispdata.var_ring_upper_limit.get() * xscale, (mosaic_img_max-mosaic_img_min)*yscale],
                                                    fill='green', tags='line')

    # Plot lower and upper background pixel limits
    ring_center = (bkgnddispdata.var_ring_lower_limit.get() + bkgnddispdata.var_ring_upper_limit.get())/2    
    bkgnddispdata.radial_profile_canvas.create_line([(bkgnddispdata.var_ring_lower_limit.get() -
                                                      bkgnddispdata.var_inside_background_pixels.get()) * xscale, 0,
                                                     (bkgnddispdata.var_ring_lower_limit.get() -
                                                      bkgnddispdata.var_inside_background_pixels.get()) * xscale, (mosaic_img_max-mosaic_img_min)*yscale],
                                                    fill='blue', tags='line')
    bkgnddispdata.radial_profile_canvas.create_line([(bkgnddispdata.var_ring_upper_limit.get() +
                                                      bkgnddispdata.var_outside_background_pixels.get()) * xscale, 0,
                                                     (bkgnddispdata.var_ring_upper_limit.get() +
                                                      bkgnddispdata.var_outside_background_pixels.get()) * xscale, (mosaic_img_max-mosaic_img_min)*yscale],
                                                    fill='blue', tags='line')

    # Plot current mouse Y position
    bkgnddispdata.radial_profile_canvas.create_line([bkgnddispdata.radial_profile_last_y * xscale, 0,
                                                     bkgnddispdata.radial_profile_last_y * xscale, (mosaic_img_max-mosaic_img_min)*yscale],
                                                    fill='cyan', tags='line')
    #Update the radius
    bkgnddispdata.radius = mosaic.IndexToRadius(bkgnddispdata.radial_profile_last_y, options.radius_resolution)
    
    # Plot background model profile
    c_list = []
    for radius in range(bkgnd_profile.shape[0]):
        val = bkgnd_profile[radius]
        cx = float(radius) * xscale
        cy = ysize-1 - val * yscale
        c_list.append(cx)
        c_list.append(cy)
    bkgnddispdata.radial_profile_canvas.create_line(*c_list, fill='red', tags='line')

    # Plot radial profile
    c_list = []
    for radius in range(radial_profile.shape[0]): # 0 = closest to Saturn
        if radial_profile_mask[radius]:
            if len(c_list) >= 4:
                bkgnddispdata.radial_profile_canvas.create_line(*c_list, fill='black', tags='line', width=2)
            c_list = []                
        val = radial_profile[radius]
        cx = float(radius) * xscale
        cy = ysize-1 - val * yscale
        c_list.append(cx)
        c_list.append(cy)
    if len(c_list) >= 4:
        bkgnddispdata.radial_profile_canvas.create_line(*c_list, fill='black', tags='line', width=2)

    # Update text data
    if bkgnddata.longitudes[longitude_num] < 0:  # Invalid longitude
        bkgnddispdata.label_longitude.config(text='')
        bkgnddispdata.label_radius.config(text='')
        bkgnddispdata.label_phase.config(text='')
        bkgnddispdata.label_incidence.config(text='')
        bkgnddispdata.label_emission.config(text='')
        bkgnddispdata.label_resolution.config(text='')
        bkgnddispdata.label_image.config(text='')
        bkgnddispdata.label_obsid.config(text='')
        bkgnddispdata.label_date.config(text='')
        bkgnddispdata.label_ew.config(text='')
        bkgnddispdata.label_ewmu.config(text='')
    else:
        bkgnddispdata.label_longitude.config(text=('%7.3f'%bkgnddata.longitudes[longitude_num]))
        bkgnddispdata.label_radius.config(text=('%7.3f'%bkgnddispdata.radius))
        bkgnddispdata.label_phase.config(text=('%7.3f'%bkgnddata.phase_angles[longitude_num]))
        bkgnddispdata.label_incidence.config(text=('%7.3f'%bkgnddata.incidence_angles[longitude_num]))
        bkgnddispdata.label_emission.config(text=('%7.3f'%bkgnddata.emission_angles[longitude_num]))
        bkgnddispdata.label_resolution.config(text=('%7.3f'%bkgnddata.resolutions[longitude_num]))
        bkgnddispdata.label_image.config(text=bkgnddata.image_name_list[bkgnddata.image_numbers[longitude_num]])
        bkgnddispdata.label_obsid.config(text=bkgnddata.obsid_list[bkgnddata.image_numbers[longitude_num]])
        bkgnddispdata.label_date.config(text=cspice.et2utc(bkgnddata.ETs[longitude_num], 'C', 0))
        if np.sum(~bkgnddata.corrected_mosaic_img.mask[:,longitude_num]) == 0: # Fully masked?
            bkgnddispdata.label_ew.config(text='Masked')
            bkgnddispdata.label_ewmu.config(text='Masked')
        else:
            ew = np.sum(ma.compressed(bkgnddata.corrected_mosaic_img[bkgnddispdata.var_ring_lower_limit.get():
                                                                     bkgnddispdata.var_ring_upper_limit.get(),
                                                                     longitude_num])) * options.radius_resolution
            ewmu = ew*np.abs(np.cos(bkgnddata.emission_angles[longitude_num]*np.pi/180))
            bkgnddispdata.label_ew.config(text=('%.5f'%ew))
            bkgnddispdata.label_ewmu.config(text=('%.5f'%ewmu))

#
# Update EW stats
#
def update_ew_stats(bkgnddata):
    if bkgnddata.bkgnd_model == None:
        bkgnddispdata.label_ew_stats.config(text='N/A')
        bkgnddispdata.label_ewmu_stats.config(text='N/A')
        return
    ew_list = []
    ewmu_list = []
    for idx in range(len(bkgnddata.longitudes)):
        if bkgnddata.longitudes[idx] >= 0:
            if np.sum(~bkgnddata.corrected_mosaic_img.mask[:,idx]) == 0: # Fully masked?
                continue
            ew = np.sum(ma.compressed(bkgnddata.corrected_mosaic_img[:,idx])) * options.radius_resolution
            ewmu = ew*np.abs(np.cos(bkgnddata.emission_angles[idx]*np.pi/180))
            ew_list.append(ew)
            ewmu_list.append(ewmu)
    bkgnddata.ew_mean = np.mean(ew_list)
    bkgnddata.ew_std = np.std(ew_list)
    bkgnddata.ewmu_mean = np.mean(ewmu_list)
    bkgnddata.ewmu_std = np.std(ewmu_list)
    bkgnddispdata.label_ew_stats.config(text = ('%.5f +/- %.5f'%(bkgnddata.ew_mean, bkgnddata.ew_std)))
    bkgnddispdata.label_ewmu_stats.config(text = ('%.5f +/- %.5f'%(bkgnddata.ewmu_mean, bkgnddata.ewmu_std)))

bkgnddispdata = BkgndDispData()

def setup_bkgnd_window(bkgnddata, bkgnddispdata):
    bkgnddispdata.toplevel = Tk()
    bkgnddispdata.toplevel.title(bkgnddata.obsid)
    frame_toplevel = Frame(bkgnddispdata.toplevel)  # Master frame

    # Background canvas
    bkgnddispdata.mask_overlay = np.zeros((bkgnddata.mosaic_img.shape[0],
                                           bkgnddata.mosaic_img.shape[1], 3))
    if bkgnddata.bkgnd_model_mask != None:
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 0] = 1
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 1] = 0
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 2] = 0
    
    bkgnddispdata.imdisp = ImageDisp([bkgnddata.mosaic_img], overlay_list=[bkgnddispdata.mask_overlay],
                                     parent=frame_toplevel, canvas_size=(512,512), one_zoom=False,
                                     flip_y=True)
    bkgnddispdata.imdisp.grid(row=0, column=0)

    frame_rightside = Frame(frame_toplevel)  # Frame for radial profile and mouse-point info
    
    bkgnddispdata.radial_profile_canvas = Canvas(frame_rightside, width=512, height=512, bg='white')
    bkgnddispdata.radial_profile_canvas.grid(row=0, column=0)

    bkgnddispdata.imdisp.bind_mousemove(0, lambda x, y, bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
                                        callback_radial(x, y, bkgnddata, bkgnddispdata))

    ######################################################
    # The control/data pane of the radial profile window #
    ######################################################

    gridrow = 0
    gridcolumn = 0
    
    radial_control_frame = Frame(frame_rightside)
    
    label = Label(radial_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_longitude = Label(radial_control_frame, text='')
    bkgnddispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_radius = Label(radial_control_frame, text='')
    bkgnddispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_phase = Label(radial_control_frame, text='')
    bkgnddispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_incidence = Label(radial_control_frame, text='')
    bkgnddispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_emission = Label(radial_control_frame, text='')
    bkgnddispdata.label_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_resolution = Label(radial_control_frame, text='')
    bkgnddispdata.label_resolution.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='Image:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_image = Label(radial_control_frame, text='')
    bkgnddispdata.label_image.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='OBSID:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_obsid = Label(radial_control_frame, text='')
    bkgnddispdata.label_obsid.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    
    label = Label(radial_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_date = Label(radial_control_frame, text='')
    bkgnddispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ew = Label(radial_control_frame, text='')
    bkgnddispdata.label_ew.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW*mu:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ewmu = Label(radial_control_frame, text='')
    bkgnddispdata.label_ewmu.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW stats:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ew_stats = Label(radial_control_frame, text='')
    bkgnddispdata.label_ew_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW*mu stats:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ewmu_stats = Label(radial_control_frame, text='')
    bkgnddispdata.label_ewmu_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    radial_control_frame.grid(row=1, column=0, sticky=NW)

    ##################################################
    # The control/data pane of the background window #
    ##################################################

    gridrow = 0
    gridcolumn = 0

    addon_control_frame = bkgnddispdata.imdisp.addon_control_frame

    label = Label(addon_control_frame, text='Row cutoff sigmas')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_row_cutoff = DoubleVar()
    bkgnddispdata.scale_row_cutoff = Scale(addon_control_frame, orient=HORIZONTAL, resolution=0.1,
                                           from_=0., to=40., variable=bkgnddispdata.var_row_cutoff)
    bkgnddispdata.var_row_cutoff.set(bkgnddata.row_cutoff_sigmas)
    bkgnddispdata.scale_row_cutoff.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Column cutoff sigmas')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_column_cutoff = DoubleVar()
    bkgnddispdata.scale_column_cutoff = Scale(addon_control_frame, orient=HORIZONTAL, resolution=0.1,
                                              from_=0., to=20., variable=bkgnddispdata.var_column_cutoff)
    bkgnddispdata.var_column_cutoff.set(bkgnddata.column_cutoff_sigmas)
    bkgnddispdata.scale_column_cutoff.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Row ignore fraction')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ignore_fraction = DoubleVar()
    bkgnddispdata.scale_ignore_fraction = Scale(addon_control_frame, orient=HORIZONTAL, resolution=0.001,
                                                from_=0., to_=0.1, variable=bkgnddispdata.var_ignore_fraction)
    bkgnddispdata.var_ignore_fraction.set(bkgnddata.row_ignore_fraction)
    bkgnddispdata.scale_ignore_fraction.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Row blur')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_row_blur = IntVar()
    bkgnddispdata.scale_row_blur = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                         from_=0, to=20, variable=bkgnddispdata.var_row_blur)
    bkgnddispdata.var_row_blur.set(bkgnddata.row_blur)
    bkgnddispdata.scale_row_blur.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    ring_limit_command = (lambda val, bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
                          radial_profile_update(bkgnddata, bkgnddispdata))

    label = Label(addon_control_frame, text='Ring lower limit')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ring_lower_limit = IntVar()
    bkgnddispdata.scale_ring_lower_limit = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                                 from_=0, to=600, variable=bkgnddispdata.var_ring_lower_limit,
                                                 command=ring_limit_command)
    bkgnddispdata.var_ring_lower_limit.set(bkgnddata.ring_lower_limit)
    bkgnddispdata.scale_ring_lower_limit.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Ring upper limit')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ring_upper_limit = IntVar()
    bkgnddispdata.scale_ring_upper_limit = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                                 from_=400, to=1000, variable=bkgnddispdata.var_ring_upper_limit,
                                                 command=ring_limit_command)
    bkgnddispdata.var_ring_upper_limit.set(bkgnddata.ring_upper_limit)
    bkgnddispdata.scale_ring_upper_limit.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Min inside background pixels')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_inside_background_pixels = IntVar()
    bkgnddispdata.scale_inside_background_pixels = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                                         from_=0., to=500, variable=bkgnddispdata.var_inside_background_pixels,
                                                         command=ring_limit_command)
    bkgnddispdata.var_inside_background_pixels.set(bkgnddata.column_inside_background_pixels)
    bkgnddispdata.scale_inside_background_pixels.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Min outside background pixels')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_outside_background_pixels = IntVar()
    bkgnddispdata.scale_outside_background_pixels = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                                          from_=0., to=500, variable=bkgnddispdata.var_outside_background_pixels,
                                                          command=ring_limit_command)
    bkgnddispdata.var_outside_background_pixels.set(bkgnddata.column_outside_background_pixels)
    bkgnddispdata.scale_outside_background_pixels.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    label = Label(addon_control_frame, text='Polynomial degree')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_degree = IntVar()
    bkgnddispdata.scale_degree = Scale(addon_control_frame, orient=HORIZONTAL, resolution=1,
                                       from_=1, to=2, variable=bkgnddispdata.var_degree)
    bkgnddispdata.var_degree.set(bkgnddata.degree)
    bkgnddispdata.scale_degree.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1
    
    # Recalculate background
    button_recalc_bkgnd_command = (lambda bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
                                   command_recalc_bkgnd(bkgnddata, bkgnddispdata))
    button_recalc_bkgnd = Button(addon_control_frame, text='Recalculate Background',
                                 command=button_recalc_bkgnd_command)
    button_recalc_bkgnd.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1
    
    # Commit changes button
    button_commit_changes_command = (lambda bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
                                     command_commit_changes(bkgnddata, bkgnddispdata))
    button_commit_changes = Button(addon_control_frame, text='Commit Changes',
                                   command=button_commit_changes_command)
    button_commit_changes.grid(row=gridrow, column=gridcolumn+1)

    frame_rightside.grid(row=0, column=1, sticky=N)
    frame_toplevel.pack()

def read_bkgnd_metadata(bkgnddata):
    bkgnd_metadata_fp = open(bkgnddata.bkgnd_metadata_filename, 'rb')
    bkgnd_data = pickle.load(bkgnd_metadata_fp)
    if len(bkgnd_data) == 7: # Really old version (no polynomial degree)
        (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction, bkgnddata.row_blur,
         bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit, bkgnddata.column_cutoff_sigmas,
         background_pixels) = bkgnd_data
        bkgnddata.degree = 2
        bkgnddata.column_inside_background_pixels = background_pixels - ((bkgnddata.ring_upper_limit+bkgnddata.ring_lower_limit)/2-
                                                                         bkgnddata.ring_lower_limit)
        bkgnddata.column_outside_background_pixels = bkgnddata.column_inside_background_pixels
    elif len(bkgnd_data) == 8: # Next oldest version - single background pixels in old style
        (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction, bkgnddata.row_blur,
         bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit, bkgnddata.column_cutoff_sigmas,
         background_pixels, bkgnddata.degree) = bkgnd_data
        bkgnddata.column_inside_background_pixels = background_pixels - ((bkgnddata.ring_upper_limit+bkgnddata.ring_lower_limit)/2-
                                                                         bkgnddata.ring_lower_limit)
        bkgnddata.column_outside_background_pixels = bkgnddata.column_inside_background_pixels
    else: # Current style
        (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction, bkgnddata.row_blur,
         bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit, bkgnddata.column_cutoff_sigmas,
         bkgnddata.column_inside_background_pixels, bkgnddata.column_outside_background_pixels, bkgnddata.degree) = bkgnd_data
    bkgnd_metadata_fp.close()
    
#
# Display the mosaic and background
#
def display_bkgnd(bkgnddata, bkgnddispdata):
    if options.verbose:
        print '** Display background', bkgnddata.obsid

    if bkgnddata.mosaic_img == None:
        read_mosaic(bkgnddata)
        
    if all_bkgnd_files_exist(bkgnddata):
        bkgnddata.bkgnd_model = np.load(bkgnddata.bkgnd_model_filename+'.npy')
        bkgnddata.bkgnd_model = bkgnddata.bkgnd_model.view(ma.MaskedArray)
        bkgnddata.bkgnd_model.mask = np.load(bkgnddata.bkgnd_mask_filename+'.npy')
        bkgnddata.bkgnd_model_mask = ma.getmaskarray(bkgnddata.bkgnd_model)
        bkgnddata.corrected_mosaic_img = bkgnddata.mosaic_img - bkgnddata.bkgnd_model
            
        read_bkgnd_metadata(bkgnddata)
        
    else:
        bkgnddata.bkgnd_model = None
        # Background params already set in main loop            
            
    setup_bkgnd_window(bkgnddata, bkgnddispdata)
    update_ew_stats(bkgnddata)
        
    mainloop()

#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################
    
for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
    bkgnddata = ringutil.BkgndData()

    bkgnddata.row_cutoff_sigmas = 7
    bkgnddata.row_ignore_fraction = 0.01
    bkgnddata.row_blur = 5
    bkgnddata.ring_lower_limit = 300
    bkgnddata.ring_upper_limit = 700
    bkgnddata.column_cutoff_sigmas = 4
    bkgnddata.column_inside_background_pixels = 50
    bkgnddata.column_outside_background_pixels = 50
    bkgnddata.degree = 2
    
    bkgnddata.obsid = obs_id

    (bkgnddata.mosaic_data_filename, bkgnddata.mosaic_metadata_filename,
     trash1, trash2) = ringutil.mosaic_paths(options, obs_id)
     
    (bkgnddata.reduced_mosaic_data_filename, bkgnddata.reduced_mosaic_metadata_filename,
     bkgnddata.bkgnd_mask_filename, bkgnddata.bkgnd_model_filename,
     bkgnddata.bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
     
    make_bkgnd(bkgnddata, options.no_bkgnd, options.no_update_bkgnd, options.recompute_bkgnd)

    if options.display_bkgnd:
        display_bkgnd(bkgnddata, bkgnddispdata)
