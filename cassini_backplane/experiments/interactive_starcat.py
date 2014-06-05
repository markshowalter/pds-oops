'''
Created on Apr 11, 2014

@author: rfrench
'''

import numpy as np
import numpy.ma as ma
import oops.inst.cassini.iss as iss
import oops.inst.nh.lorri as lorri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cspice
from polymath import *
from imgdisp import *
import Tkinter as tk
from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter
from psfmodel.gaussian import GaussianPSF
from cb_correlate import *
from cb_stars import *

from starcat import UCAC4StarCatalog
star_catalog = UCAC4StarCatalog('t:/external/ucac4')
#from starcat import SpiceStarCatalog
#star_catalog = SpiceStarCatalog('hipparcos')

#plot_johnson_filter_transmission()
#plot_planck_vs_solar_flux()

PSF_SIZE = 9

do_astrometry = False
do_roll = True
do_contour = False

#plot_cassini_filter_transmission()


#obs = iss.from_file('J:/AppData/SugarSync/test_data/cassini/ISS/N1649465323_1.IMG')
#obs = iss.from_file('J:/Temp/W1528616654_1_CALIB.IMG') # DOESN'T WORK AT ALL

# 1024x1024, 1.0 sec, CL1/CL2
#obs = iss.from_file('P:/SETI/CDAPS-components/N1555565413_1.IMG')
#obs = iss.from_file('T:/external/cassini/derived/COISS_2xxx/COISS_2031/data/1555449244_1555593613/N1555565413_1_CALIB.IMG')

# 512x512, 220.0 sec, CL1/CL2, Stars vis to 14 = 
#obs = iss.from_file('J:/Temp/W1711408650_1_CALIB.IMG') # 512x512

#obs = iss.from_file('J:/Temp/W1611007104_1_CALIB.IMG') # Wraparound - there has to be something wrong with this image

# 1024x1024, 220.0 sec, CL1/CL2, Stars vis to 14 = 
obs = iss.from_file('J:/Temp/W1636550840_1_CALIB.IMG')

# 512x512, 56.0 sec, UV1/CL2 NO STARS
#obs = iss.from_file('J:/Temp/N1468429796_1.IMG') # UV1+CL1
#obs = iss.from_file('J:/Temp/N1468429796_1_CALIB.IMG') # UV1+CL1
#obs = iss.from_file('J:/Temp/N1468429796_1_FLUX.IMG') # UV1+CL1

# 1024x1024, 1.5 sec, CL1/CL2, Stars vis to 14.4 = 13,000
#obs = iss.from_file('J:/Temp/N1593959052_1.IMG') # CL1+CL2
#obs = iss.from_file('J:/Temp/N1593959052_1_CALIB.IMG') # CL1+CL2
#obs = iss.from_file('J:/Temp/N1593959052_1_FLUX.IMG') # CL1+CL2

# 1024x1024, 5.6 sec, IR2/IR1 NO STARS
#obs = iss.from_file('J:/Temp/W1477666816_1.IMG') # IR2+IR1
#obs = iss.from_file('J:/Temp/W1477666816_1_CALIB.IMG') # IR2+IR1
#obs = iss.from_file('J:/Temp/W1477666816_1_FLUX.IMG') # IR2+IR1

#obs = iss.from_file('T:/clumps/data/ISS_029RF_FMOVIE001_VIMS/N1538218132_1_CALIB.IMG')
#obs = iss.from_file('T:/clumps/data/ISS_029RF_FMOVIE001_VIMS/N1538179934_1_CALIB.IMG')
#obs = iss.from_file('T:/clumps/data/ISS_030RF_FMOVIE001_VIMS/N1539680794_1_CALIB.IMG')
#obs = iss.from_file('T:/clumps/data/ISS_032RF_FMOVIE001_VIMS/N1542095211_1_CALIB.IMG')


data = obs.data
data = calibrate_iof_image_as_dn(obs)
print np.min(data), np.max(data)
print 'DATA SIZE', data.shape, 'TEXP', obs.texp, 'FILTERS', obs.filter1, obs.filter2

best_peak = 0.

best_offset_u = None
best_offset_v = None

for star_offset_u in [0]:#np.arange(-1., 1.1, 0.25):
    for star_offset_v in [0]:#np.arange(-1., 1.1, 0.25):

        star_list = star_list_for_obs(star_catalog, obs, num_stars=30)
        assert len(star_list) > 0
        
        model = star_create_model(obs, star_list,
                return_overlay=True,
                star_offset_u=star_offset_u, star_offset_v=star_offset_v,
                verbose=True)
        
        offset_u, offset_v, peak_val = find_correlated_offset(data, model)

        if peak_val > best_peak:
            best_offset_u = offset_u + star_offset_u
            best_offset_v = offset_v + star_offset_v
            best_peak = peak_val

        print offset_u + star_offset_u, offset_v + star_offset_u, peak_val

offset_u = best_offset_u
offset_v = best_offset_v

#offset_u = 8
#offset_v = -6

print
print 'BEST OFFSET', offset_u, offset_v

if do_roll:
    data = shift_image(data, offset_u, offset_v)

photometry_size = 3
believe_ratio = 2.

good_stars = 0

overlay = np.zeros(obs.data.shape+(3,))

for star in star_list:
    u = int(np.round(star.u))
    v = int(np.round(star.v))
    if (u < photometry_size or u > data.shape[1]-photometry_size-1 or
        v < photometry_size or v > data.shape[0]-photometry_size-1):
        continue
    
    subimage = data[v-photometry_size:v+photometry_size+1,
                    u-photometry_size:u+photometry_size+1]
    subimage = subimage.view(ma.MaskedArray)
    subimage[1:-1, 1:-1] = ma.masked
    
    bkgnd = ma.mean(subimage)
    
    subimage.mask = ~subimage.mask
    
    integrated_dn = np.sum(subimage-bkgnd)
    
    print '%4d %4d %7.3f %7.3f' % (u, v, star.dn, integrated_dn),
    
    if (integrated_dn < 0 or
        star.dn  / integrated_dn > believe_ratio or
        integrated_dn / star.dn > believe_ratio):
        print 'BAD'
        draw_circle(overlay, u, v, 1, (1,0,0), 3)
    else:
        print 'GOOD'
        good_stars += 1
        draw_circle(overlay, u, v, 1, (0,1,0), 3)
        
print
print 'NUM GOOD', good_stars
    
if do_astrometry:
    u_list = []
    v_list = []
    udelta_list = []
    vdelta_list = []
    mag_list = []
    scale_list = []
    
    scale_mag_coeffs = [-0.3101978, 5.01754699]
    
    gausspsf = GaussianPSF(sigma=NAC_SIGMA)
    for star, u, v in star_data:
        if (v <= -offset_v+PSF_SIZE//2 or
            v >= 1023-offset_v-PSF_SIZE//2 or
            u <= -offset_u+PSF_SIZE//2 or
            u >= 1023-offset_u-PSF_SIZE//2):
            continue
        star_mag = star.vmag
        ret = gausspsf.find_position(data, (PSF_SIZE,PSF_SIZE), (v,u), search_limit=(2.5, 2.5),
                          bkgnd_degree=2, bkgnd_ignore_center=(2,2),
                          tolerance=1e-4, num_sigma=5, bkgnd_num_sigma=5)
        if ret is None:
            print u, v, "FAILED"
            continue
        
        new_v, new_u, metadata = ret
        scale = metadata['scale']
        print u, v, new_u, new_v, u-new_u, v-new_v, scale
    
        if scale < 0:
            print 'NEGATIVE SCALE'
            continue
        
        pred_scale = np.polyval(scale_mag_coeffs, star_mag)
        print np.log10(scale), pred_scale
    #    if np.log10(scale) < pred_scale * .8:
    #        print 'SCALE TOO LOW'
    #        continue
        
        u_list.append(u)
        v_list.append(v)
        udelta_list.append(u-new_u)
        vdelta_list.append(v-new_v)
        mag_list.append(star_mag)
        scale_list.append(metadata['scale'])
        
        plot3d = False    
        if plot3d:
            print metadata
        
            psf = metadata['scaled_psf']
            subimg = metadata['subimg-gradient']
            
            xindices = np.repeat(np.arange(psf.shape[1])-psf.shape[1]//2,
                                 psf.shape[0])
            yindices = np.tile(np.arange(psf.shape[0])-psf.shape[0]//2,
                               psf.shape[1])
            xindices = xindices.reshape((psf.shape[1], psf.shape[0]))
            yindices = yindices.reshape((psf.shape[1], psf.shape[0]))
        
            fig = plt.figure()
        
            ax = Axes3D(fig)
            ax.plot_surface(xindices, yindices, psf,
                            rstride=1, cstride=1, color='red', alpha=0.3)
            ax.plot_surface(xindices, yindices, subimg,
                            rstride=1, cstride=1, color='blue', alpha=0.3)
            plt.title('Best Fit PSF and Original')
        
            plt.show()
    
    print 'UPDATED OFFSET', offset_u+np.mean(udelta_list), offset_v+np.mean(vdelta_list)
    
    coeffs = np.polyfit(mag_list, np.log10(scale_list), 1)
    print coeffs
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.08,bottom=0.06,right=0.96,top=0.96)
    ax = fig.add_subplot(321)
    for u, udelta, mag in zip(u_list, udelta_list, mag_list):
        plt.plot(u, udelta, 'o', mfc='none', mec='black', mew=1, ms=(star_mag_max-mag)*2.5+3)
    plt.xlabel('U')
    plt.ylabel('Delta U')
    plt.xlim(-100,data.shape[1]+100)
    plt.ylim(-3,3)
    plt.title('U Position Error vs. U Position')
    
    ax = fig.add_subplot(322)
    for v, vdelta, mag in zip(v_list, vdelta_list, mag_list):
        plt.plot(v, vdelta, 'o', mfc='none', mec='black', mew=1, ms=(star_mag_max-mag)*2.5+3)
    plt.xlabel('V')
    plt.ylabel('Delta V')
    plt.xlim(-100,data.shape[0]+100)
    plt.ylim(-3,3)
    plt.title('V Position Error vs. V Position')
    
    ax = fig.add_subplot(323)
    for u, vdelta, mag in zip(u_list, vdelta_list, mag_list):
        plt.plot(u, vdelta, 'o', mfc='none', mec='black', mew=1, ms=(star_mag_max-mag)*2.5+3)
    plt.xlabel('U')
    plt.ylabel('Delta V')
    plt.xlim(-100,data.shape[0]+100)
    plt.ylim(-3,3)
    plt.title('V Position Error vs. U Position')
    
    ax = fig.add_subplot(324)
    for v, udelta, mag in zip(v_list, udelta_list, mag_list):
        plt.plot(v, udelta, 'o', mfc='none', mec='black', mew=1, ms=(star_mag_max-mag)*2.5+3)
    plt.xlabel('V')
    plt.ylabel('Delta U')
    plt.xlim(-100,data.shape[1]+100)
    plt.ylim(-3,3)
    plt.title('U Position Error vs. V Position')
    
    ax = fig.add_subplot(325)
    for udelta, vdelta, mag in zip(udelta_list, vdelta_list, mag_list):
        plt.plot(udelta, vdelta, 'o', mfc='none', mec='black', mew=1, ms=(star_mag_max-mag)*2.5+3)
    plt.xlabel('Delta U')
    plt.ylabel('Delta V')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('Relative error in predicted position')
    
    ax = fig.add_subplot(326)
    plt.plot(mag_list, np.log10(scale_list), 'o', mfc='none', mec='black', mew=1, ms=7)
    plt.xlabel('Visual Magnitude')
    plt.ylabel('log PSF Scale')
    plt.title('Magnitude vs. Scale')
    
    plt.show()

imgdisp = ImageDisp([data], [overlay], canvas_size=(1024,768), allow_enlarge=True)

tk.mainloop()
