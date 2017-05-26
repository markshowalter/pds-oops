'''
Created on Oct 9, 2011
Creates EW profiles of all cassini mosaics.
Run this to create Voyager EW profiles and create metadata files.
@author: rfrench
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import ringutil
import sys
import os.path
import matplotlib.pyplot as plt
import cspice
#fringobs.Init()

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
#    cmd_line = ['-a', '--mosaic-reduction-factor', '2', '--ignore-voyager']
    cmd_line = ['--voyager', '--mosaic-reduction-factor', '1']

parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

if options.voyager:
    # Phase Angle, Emission Angle, Incidence Angle, Days from Encounter*Secs/Day, Encounter Date
    #Closest encounter dates courtesy of: http://nssdc.gsfc.nasa.gov/nmc/SpacecraftQuery.jsp
    # Emission angles are weighted from Table 1 of Showalter 2004
    # Phase Angles are from Figure of Showalter 2004
    # Incidence angles 
    voyager_data_dict = {}
    voyager_data_dict['V1I'] = (15.0, 81.36, 86.1, -2.0*86400, '1980-11-12T23:46:30')
    voyager_data_dict['V1O'] = (125.0, 64.5, 86.1, 2.5*86400, '1980-11-12T23:46:30')
    voyager_data_dict['V2I'] = (10.5, 79.0, 82.1, -4.7*86400, '1981-08-26T03:24:57')
    voyager_data_dict['V2O'] = (98.0, 109.71, 82.1, 4.5*86400, '1981-08-26T03:24:57')
    
    for root, dirs, files in os.walk(ringutil.VOYAGER_PATH):
    
        for file in files:
            if '.STACK' in file:
                filepath = os.path.join(root, file)
                f = open(filepath)
                longitudes = []
                EWs = []
                trash = []
                filename = file[:-6]
                
                for line in f:
                    vars = line.split()
                    long, ew, trash = vars
                    longitudes.append(float(long))
                    EWs.append(float(ew))
                
                longitudes = np.array(longitudes)
                EWs = np.array(EWs)/1000
                
                #fake data (Except emission and phase angles)
                et = cspice.utc2et(voyager_data_dict[filename][4]) + voyager_data_dict[filename][3]
                print cspice.et2utc(et, 'C', 0)
                ETs = np.zeros(len(EWs)) + et
                resolutions = np.zeros(len(EWs)) - 9999
                image_numbers = np.zeros(len(EWs)) - 9999
                emission_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][1]
                phase_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][0]
                incidence_angles = np.zeros(len(EWs)) + voyager_data_dict[filename][2]
                
                #make metadata for voyager
                data_path, metadata_path, large_png_path, small_png_path = ringutil.mosaic_paths(options, filename)
                metadata = (longitudes, resolutions, image_numbers,
                            ETs, emission_angles, incidence_angles,
                            phase_angles)
                metadata_fp = open(metadata_path, 'wb')
                pickle.dump(metadata, metadata_fp)
                metadata_fp.close()
                
                #fake bkgnd data for voyager
                (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                     bkgnd_mask_filename, bkgnd_model_filename,
                     bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, filename)
                
                np.save(bkgnd_model_filename, [])     #just need the blank files
                np.save(bkgnd_mask_filename, [])
                bkgnd_data = (None, None, None, None, None, None, None, None, None)
                bkgnd_metadata_fp = open(bkgnd_metadata_filename, 'wb')
                pickle.dump(bkgnd_data, bkgnd_metadata_fp)
                bkgnd_metadata_fp.close()
                
                reduced_mosaic_metadata_fp = open(reduced_mosaic_metadata_filename, 'wb')
                pickle.dump(metadata, reduced_mosaic_metadata_fp)
                pickle.dump([filename], reduced_mosaic_metadata_fp)
                pickle.dump([None], reduced_mosaic_metadata_fp)
                pickle.dump([None], reduced_mosaic_metadata_fp)
                reduced_mosaic_metadata_fp.close()
                
                fake_reduced_mosaic = np.zeros((10,10))+999
                reduced_mosaic_data_fp = open(reduced_mosaic_data_filename, 'wb')
                np.save(reduced_mosaic_data_fp, fake_reduced_mosaic)
                reduced_mosaic_data_fp.close()
                
                EWs /= np.abs(np.cos(emission_angles[0]*np.pi/180.)) # Convert normal to raw I/F
                EWs = EWs.view(ma.MaskedArray)
                EWs.mask = False
                EWs[np.where(EWs == 0.)] = ma.masked

                (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, filename)
                
                np.save(ew_data_filename, EWs.data)
                np.save(ew_mask_filename, ma.getmask(EWs))
else:    
    for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
    
        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
    
        if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(bkgnd_mask_filename+'.npy')):
            print 'NO FILE', reduced_mosaic_metadata_filename, 'OR', bkgnd_mask_filename+'.npy'
            continue
        
        bkgnd_metadata_fp = open(bkgnd_metadata_filename, 'rb')
        bkgnd_data = pickle.load(bkgnd_metadata_fp)
        (row_cutoff_sigmas, row_ignore_fraction, row_blur,
         ring_lower_limit, ring_upper_limit, column_cutoff_sigmas,
         column_inside_background_pixels, column_outside_background_pixels, degree) = bkgnd_data
        bkgnd_metadata_fp.close()
        
        reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(reduced_metadata_fp)
        obsid_list = pickle.load(reduced_metadata_fp)
        image_name_list = pickle.load(reduced_metadata_fp)
        full_filename_list = pickle.load(reduced_metadata_fp)
        reduced_metadata_fp.close()
    
        (longitudes, resolutions, image_numbers,
         ETs, emission_angles, incidence_angles,
         phase_angles) = mosaic_data
    
        mosaic_img = np.load(reduced_mosaic_data_filename+'.npy')
    
        mosaic_img = mosaic_img.view(ma.MaskedArray)
        mosaic_img.mask = np.load(bkgnd_mask_filename+'.npy')
        bkgnd_model = np.load(bkgnd_model_filename+'.npy')
        bkgnd_model = bkgnd_model.view(ma.MaskedArray)
        bkgnd_model.mask = mosaic_img.mask
    
        corrected_mosaic_img = mosaic_img - bkgnd_model
        
        percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100
        
        mean_incidence = ma.mean(incidence_angles)
        
        ew_data = np.zeros(len(longitudes))
        ew_data = ew_data.view(ma.MaskedArray)
        movie_phase_angles = []
        for idx in range(len(longitudes)):
            if longitudes[idx] < 0 or np.sum(~corrected_mosaic_img.mask[:,idx]) == 0: # Fully masked?
                ew_data[idx] = ma.masked
            else:
                column = corrected_mosaic_img[:,idx][ring_lower_limit:ring_upper_limit+1]
                # Sometimes there is a reprojection problem at the edge
                # If the very edge is masked, then find the first non-masked value and mask it, too
                if column.mask[-1]:
                    colidx = len(column)-1
                    while column.mask[colidx]:
                        colidx -= 1
                    column[colidx] = ma.masked
                    column[colidx-1] = ma.masked
                if column.mask[0]:
                    colidx = 0
                    while column.mask[colidx]:
                        colidx += 1
                    column[colidx] = ma.masked
                    column[colidx+1] = ma.masked
                ew = np.sum(ma.compressed(column)) * options.radius_resolution
                mu_ew = ew*np.abs(np.cos(emission_angles[idx]*np.pi/180))

                corr_ew = ringutil.compute_corrected_ew(mu_ew, emission_angles[idx], mean_incidence)
                ratio = ringutil.clump_phase_curve(0) / ringutil.clump_phase_curve(phase_angles[idx])
                phase_ew = corr_ew * ratio
                
                if phase_ew < 0.2 or phase_ew > 3.5:
                    ew_data[idx] = ma.masked
                else:
                    ew_data[idx] = ew
                movie_phase_angles.append(phase_angles[idx])
                emission_angle = emission_angles[idx]
                incidence_angle = incidence_angles[idx]
    
        # When we find a masked column, mask one column on either side because this data tends to be bad too
        old_mask = ma.getmaskarray(ew_data)
        new_mask = old_mask.copy()
        for idx in range(len(new_mask)):
            if old_mask[idx]:
                if idx > 0:
                    new_mask[idx-1] = True
                if idx > 1:
                    new_mask[idx-2] = True
                if idx < len(new_mask)-1:
                    new_mask[idx+1] = True
                if idx < len(new_mask)-2:
                    new_mask[idx+2] = True
        ew_data.mask = new_mask
    
        np.save(ew_data_filename, ew_data.data)
        np.save(ew_mask_filename, ma.getmask(ew_data))
                
        print '%-30s %3d%% P %7.3f E %7.3f I %7.3f EW %8.5f +/- %8.5f' % (obs_id, percentage_ok,
            np.mean(movie_phase_angles), 180-emission_angle, 180-incidence_angle, ma.mean(ew_data), np.std(ew_data))

#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        plt.plot(longitudes, ew_data)
#        plt.show()
