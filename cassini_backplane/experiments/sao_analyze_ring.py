'''
Created on Sep 19, 2011

@author: rfrench
'''

import argparse
import math
import os
import os.path
import sys
import numpy as np
import scipy.optimize as sciopt
import oops.inst.cassini.iss as iss
import gravity
from cb_config import *
from cb_offset import *
from cb_util_file import *

import matplotlib.pyplot as plt

#DEST_ROOT = os.path.join(CB_RESULTS_ROOT, 'sao')
DEST_ROOT = '/home/rfrench/Dropbox-SETI/Shared/Shared-John-Daicopoulos/170425-redo'
# DEST_ROOT = '/tmp'

LONGITUDE_RESOLUTION = 0.002
RADIUS_RESOLUTION = 2.5
RADIUS_INNER = 117570. - 100 - 1000
RADIUS_OUTER = 117570. + 100 + 100

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'N1627295812_1'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(description='SAO P124 Backplane Generator')

parser.add_argument(
    '--analyze-b-ring-edge', action='store_true',
    help='Analyze the B ring edge')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

#####################################################################################
#
# FIND THE LOCATION OF MIMAS WHEN ANALYZING THE B RING
#
#####################################################################################

TWOPI = 2*np.pi

# A nicer version of arctan2
def pos_arctan2(y, x):
    return np.arctan2(y, x) % TWOPI 

# Take the geometric osculating elements and create frequencies
# Returns n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq
# From Renner & Sicardy (2006)  EQ 14-21

def orb_geom_to_freq(gm, j2j4rp, a, e, inc):
    j2 = j2j4rp[0]/a**2.
    j4 = j2j4rp[1]/a**4.
    
    n = np.sqrt(gm / a**3.) * (1. + 3./4.*j2 - 15./16.*j4 -
                                     9./32.*j2**2. + 45./64.*j2*j4 +
                                     27./128.*j2**3. +
                                     3.*j2*e**2. - 12.*j2*inc**2.)
    
    kappa = np.sqrt(gm / a**3.) * (1. - 3./4.*j2 + 45./16.*j4 -
                                         9./32.*j2**2. + 135./64.*j2*j4 -
                                         27./128.*j2**3. - 9.*j2*inc**2.)

    nu = np.sqrt(gm / a**3.) * (1. + 9./4.*j2 - 75./16.*j4 -
                                      81./32.*j2**2. + 675./64.*j2*j4 +
                                      729./128.*j2**3. +
                                      6.*j2*e**2. - 51./4.*j2*inc**2.)
    
    eta2 = gm / a**3. * (1. - 2.*j2 + 75./8.*j4)
    
    chi2 = gm / a**3. * (1. + 15./2.*j2 - 175./8.*j4)
    
    alpha1 = 1./3. * (2.*nu + kappa)
    alpha2 = 2.*nu - kappa
    alphasq = alpha1 * alpha2
    
    return (n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq)

# Take the frequencies and convert them to cylindrical coordinates
# Returns a, e, inc, long_peri, long_node, lam, rc, Lc, zc, rdotc, Ldotc, zdotc
# From Renner & Sicardy (2006) EQ 36-41

def orb_freq_to_geom(r, L, z, rdot, Ldot, zdot, rc, Lc, zc, rdotc, Ldotc, 
                     zdotc, n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq):
    kappa2 = kappa**2.
    n2 = n**2.
    
    # EQ 42-47
    a = (r-rc) / (1.-(Ldot-Ldotc-n)/(2.*n))
    
    e = np.sqrt(((Ldot-Ldotc-n)/(2.*n))**2. + ((rdot-rdotc)/(a*kappa))**2.)
    
    inc = np.sqrt(((z-zc)/a)**2. + ((zdot-zdotc)/(a*nu))**2.)
    
    lam = L - Lc - 2.*n/kappa*(rdot-rdotc)/(a*kappa)
    
    long_peri = (lam - pos_arctan2(rdot-rdotc, a*kappa*(1.-(r-rc)/a))) % TWOPI    
    
    long_node = (lam - pos_arctan2(nu*(z-zc), zdot-zdotc)) % TWOPI
    
    # EQ 36-41
    rc = (a * e**2. * (3./2.*eta2/kappa2 - 1. - 
                       eta2/2./kappa2*np.cos(2.*(lam-long_peri))) +
          a * inc**2. * (3./4.*chi2/kappa2 - 1. + 
                         chi2/4./alphasq*np.cos(2.*(lam-long_node))))
    
    Lc = (e**2.*(3./4. + eta2/2./kappa2)*n/kappa*np.sin(2.*(lam-long_peri)) - 
          inc**2.*chi2/4./alphasq*n/nu*np.sin(2.*(lam-long_node)))
    
    zc = a*inc*e*(chi2/2./kappa/alpha1*np.sin(2*lam-long_peri-long_node) - 
                  3./2.*chi2/kappa/alpha2*np.sin(long_peri-long_node))
    
    rdotc = (a*e**2.*eta2/kappa*np.sin(2.*(lam-long_peri)) - 
             a*inc**2*chi2/2./alphasq*nu*np.sin(2.*(lam-long_node)))
    
    Ldotc = (e**2.*n*(7./2. - 3.*eta2/kappa2 - kappa2/2./n2 + 
                      (3./2. + eta2/kappa2)*np.cos(2.*(lam-long_peri))) +
             inc**2.*n*(2. - kappa2/2./n2 - 3./2.*chi2/kappa2 - 
                        chi2/2./alphasq*np.cos(2.*(lam-long_node))))
    
    zdotc = a*inc*e*(chi2*(kappa+nu)/2./kappa/
                        alpha1*np.cos(2*lam-long_peri-long_node) + 
             3./2.*chi2*(kappa-nu)/kappa/alpha2*np.cos(long_peri-long_node))
    
    # EQ 30-35
#    r = a*(1. - e*np.cos(lam-long_peri)) + rc
#    
#    L = lam + 2*e*n/kappa*np.sin(lam-long_peri) + Lc
#    
#    z = a*inc*np.sin(lam-long_node) + zc
#    
#    rdot = a*e*kappa*np.sin(lam-long_peri) + rdotc
#    
#    Ldot = n*(1. + 2.*e*np.cos(lam-long_peri)) + Ldotc
#    
#    zdot = a*inc*nu*np.cos(lam-long_node) + zdotc
    
    return (a, e, inc, long_peri, long_node, lam, 
            rc, Lc, zc, rdotc, Ldotc, zdotc)

# Given the state vector x,y,z,vx,vy,vz retrieve the geometric elements
# Returns: a, e, inc, long_peri, long_node, mean_anomaly
# From Renner and Sicardy (2006) EQ 22-47

def orb_xyz_to_geom(gm, j2j4rp, x, y, z, vx, vy, vz, tol=1e-6, quiet=False):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vz = np.asarray(vz)

    # EQ 22-25
    r = np.sqrt(x**2. + y**2.)
    L = pos_arctan2(y, x)
    rdot = vx*np.cos(L) + vy*np.sin(L)
    Ldot = (vy*np.cos(L)-vx*np.sin(L))/r
    
    # Initial conditions
    a = r
    e = 0.
    inc = 0.
    rc = 0.
    Lc = 0.
    zc = 0.
    rdotc = 0.
    Ldotc = 0.
    zdotc = 0.
    
    old_diffmax = 1e38
    old_diff = None
    idx_to_use = np.where(x!=-1e38,True,False) # All True
    announced = False
    while True:
        (n, kappa, nu, eta2, chi2, 
         alpha1, alpha2, alphasq) = orb_geom_to_freq(gm, j2j4rp, a, e, inc)
        ret = orb_freq_to_geom(r, L, z, rdot, Ldot, vz, rc, Lc, zc, rdotc, 
                               Ldotc, zdotc, n, kappa, nu, eta2, chi2,
                               alpha1, alpha2, alphasq)
        old_a = a
        (a, e, inc, long_peri, long_node, lam, 
         rc, Lc, zc, rdotc, Ldotc, zdotc) = ret
        diff = np.abs(a-old_a)
        diffmax = np.max(diff[idx_to_use])
        if diffmax < tol:
            break
        if diffmax > old_diffmax:
            idx_to_use = np.where(diff > old_diff,False,True) & idx_to_use
            if not idx_to_use.any(): break
            if not announced:
                if not quiet:
                    print ('WARNING: orb_xyz_to_geom started diverging!  '+
                           'Met tolerance %e') % diffmax
                announced = True
            if not quiet:
                diff_of_diff = diff - old_diff
                bad_idx = diff_of_diff.argmax()
                print 'Bad index', bad_idx
                print 'X', x[bad_idx]
                print 'Y', y[bad_idx]
                print 'Z', z[bad_idx]
                print 'VX', vx[bad_idx]
                print 'VY', vy[bad_idx]
                print 'VZ', vz[bad_idx]
        old_diffmax = diffmax
        old_diff = diff
        
    mean_anomaly = (lam-long_peri) % TWOPI

    return (a, e, inc, long_peri, long_node, mean_anomaly)

def keplers_equation_resid(p, M, e):
    return np.sqrt((M - (p[0]-e*np.sin(p[0])))**2)

# Mean anomaly M = n(t-tau)
# Find E, the eccentric anomaly, the angle from the center of the ellipse and the pericenter to a
# circumscribed circle at the place where the orbit is projected vertically.
def find_E(M, e):
    result = sciopt.fmin(keplers_equation_resid, (0.,), args=(M, e), disp=False, xtol=1e-20, ftol=1e-20)
    return result[0]

# Find r, the radial distance from the focus, and f, the true anomaly, the angle from the focus and the
# pericenter to the orbit.
def rf_from_E(a, e, E):
    f = np.arccos((np.cos(E)-e)/(1-e*np.cos(E)))
    if E > np.pi:
        f = 2*np.pi - f
    return a*(1-e*np.cos(E)), f   

def saturn_to_mimas(et):
    '''
    Return Saturn->Mimas vector
    '''
    SATURN_ID     = cspice.bodn2c("SATURN")
    MIMAS_ID      = cspice.bodn2c("MIMAS")
    
    # Reference time
    REFERENCE_DATE = "1 JANUARY 2007"       # This is the date Doug used. It is only
                                            # used to define the instantaneous pole.
    REFERENCE_ET = cspice.utc2et(REFERENCE_DATE)
    
    # Coordinate frame:
    #   Z-axis is Saturn's pole;
    #   X-axis is the ring plane ascending node on J2000
    j2000_to_iau_saturn = cspice.pxform("J2000", "IAU_SATURN", REFERENCE_ET)
    
    saturn_z_axis_in_j2000 = cspice.mtxv(j2000_to_iau_saturn, (0,0,1))
    saturn_x_axis_in_j2000 = cspice.ucrss((0,0,1), saturn_z_axis_in_j2000)
    
    J2000_TO_SATURN = cspice.twovec(saturn_z_axis_in_j2000, 3,
                                    saturn_x_axis_in_j2000, 1)
    
    SATURN_TO_J2000 = J2000_TO_SATURN.transpose()
    
    (mimas_j2000, lt) = cspice.spkez(MIMAS_ID, et, "J2000", "LT+S", SATURN_ID)
    mimas_sat = np.dot(J2000_TO_SATURN, mimas_j2000[0:3])
    mimas_vel = np.dot(J2000_TO_SATURN, mimas_j2000[3:])
    mimas_sat = np.append(mimas_sat, mimas_vel)
    dist = np.sqrt(mimas_sat[0]**2.+mimas_sat[1]**2.+mimas_sat[2]**2.)
    longitude = math.atan2(mimas_sat[1], mimas_sat[0]) * 180./np.pi
    if longitude < 0:
        longitude += 360
    return (dist, longitude, mimas_sat)

def analayze_b_ring_edge(image_name, obs, off_radii, off_longitudes, 
                         off_resolution, off_emission, off_incidence,
                         off_phase,
                         offset):
    b_ring_edge = obs.bp.border_atop(off_radii.key, 117570.12).vals.astype('bool')
    off_sha = obs.bp.ring_longitude('saturn:ring', reference='sha') * oops.DPR
    if not np.any(b_ring_edge):
        min_long = -1000.
        max_long = -1000.
        min_res = -1000.
        max_res = -1000.
        min_em = -1000.
        max_em = -1000.
        min_phase = -1000.
        max_phase = -1000.
        min_sha = -1000.
        max_sha = -1000.
    else:
        longitudes = off_longitudes[b_ring_edge].vals.astype('float32')
        min_long = np.min(longitudes)
        max_long = np.max(longitudes)
        resolution = off_resolution[b_ring_edge].vals.astype('float32')
        min_res = np.min(resolution)
        max_res = np.max(resolution)
        emission = off_emission[b_ring_edge].vals.astype('float32')
        min_em = np.min(emission)
        max_em = np.max(emission)
        phase = off_phase[b_ring_edge].vals.astype('float32')
        min_phase = np.min(phase)
        max_phase = np.max(phase)
        sha = off_sha[b_ring_edge].vals.astype('float32')
        min_sha = np.min(sha)
        max_sha = np.max(sha)
        
    mimas_dist, mimas_long = saturn_to_mimas(obs.midtime)[:2]
    print 'MIMAS DIST', mimas_dist
    print 'LONG %.2f %.2f MIMAS %.2f SHA %.2f %.2f' % (
                   min_long, max_long, mimas_long, min_sha, max_sha)
    
    avg_long = (min_long+max_long)/2
    
    bring_mean_motion = gravity.SATURN.n(117570.12) * 180/np.pi
    mimas_mean_motion = gravity.SATURN.n(185539) * 180/np.pi
    
    delta_lon = avg_long - mimas_long
    if delta_lon < 0: delta_lon += 360
    
    delta_t = -delta_lon / (bring_mean_motion-mimas_mean_motion)
    
    conj_et = obs.midtime + delta_t
    print 'PREV CONJUNCTION', cspice.et2utc(conj_et, 'C', 2)
    
    conj_mimas_dist, conj_mimas_long, conj_mimas_state = saturn_to_mimas(conj_et)
    print 'CONJUNCTION STATE VECTOR:'
    print conj_mimas_state

    orb_el = orb_xyz_to_geom(gravity.SATURN.gm, 
                             gravity.SATURN.jn[0:2],
                             conj_mimas_state[0], conj_mimas_state[1], conj_mimas_state[2],
                             conj_mimas_state[3], conj_mimas_state[4], conj_mimas_state[5])

    print 'CONJ MIMAS DIST FROM STATE:', conj_mimas_dist
    
    print 'DERIVED MIMAS A', orb_el[0], 'e', orb_el[1], 'i', orb_el[2]*180/np.pi
    
    long_peri, long_node, mean_anomaly = orb_el[3:]
    long_peri *= 180/np.pi
    long_node *= 180/np.pi
    mean_anomaly *= 180/np.pi
    
    print 'MIMAS LPERI', long_peri, 'LNODE', long_node, 'MEAN ANOM', mean_anomaly
    
    E = find_E(mean_anomaly*np.pi/180, orb_el[1])
    r, f = rf_from_E(orb_el[0], orb_el[1], E)
    f *= 180/np.pi
    arg_peri = (long_peri - long_node) %360
    print 'CONJ MIMAS RAD DIST', r, 'TRUE ANOMALY', f, 'ARG PERI', arg_peri
    
    print 'Z DIST', orb_el[0]*np.sin((arg_peri+f)*np.pi/180)*np.sin(orb_el[2])
    
    data_file_csv = os.path.join(DEST_ROOT, image_name+'.csv')
    data_file_fp = open(data_file_csv, 'w')
    print >> data_file_fp, '%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%.5f,%.5f,%d' % (
            image_name, cspice.et2utc(obs.midtime, 'C', 2),
            obs.midtime,
            min_long, max_long, 
            off_incidence,
            min_em, max_em,
            min_phase, max_phase,
            min_res, max_res,
            min_sha, max_sha,
            mimas_long,
            cspice.et2utc(conj_et, 'C', 2),
            f, (arg_peri+f)%360,
            offset is not None)
    data_file_fp.close()


#####################################################################################
#
# FIND THE POINTING OFFSET, IF NECESSARY; GENERATE BACKPLANES
#
#####################################################################################
    
def offset_one_image(image_path):
    print 'Processing', image_path
    obs = file_read_iss_file(image_path)
    offset = None
    metadata = file_read_offset_metadata(image_path, 
                                         bootstrap_pref='prefer', 
                                         overlay=False)
    if metadata is not None:
        offset = metadata['offset']
    else:
        try:
            metadata = master_find_offset(obs,
                                          create_overlay=True)
            offset = metadata['offset'] 
        except:
            print 'COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR'
            print 'EXCEPTION:'
            print sys.exc_info()
            metadata = None
            offset = None
        
    if offset is None:
        print 'COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE'
    
    image_name = file_clean_name(image_path)
    
    results = image_name + ' - ' + offset_result_str(metadata)
    print results

    for reproject in [False, True]:
        if not reproject:
            filename = os.path.join(DEST_ROOT, image_name)
            if os.path.exists(filename+'.npz'):
                return
            if offset is not None:
                orig_fov = obs.fov
                obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
            set_obs_bp(obs)
            
            off_radii = obs.bp.ring_radius('saturn:ring')
            off_longitudes = obs.bp.ring_longitude('saturn:ring') * oops.DPR
            off_incidence = obs.bp.ring_incidence_angle('saturn:ring',
                                                        pole='north')
            off_incidence = np.mean(off_incidence.mvals) * oops.DPR
            off_resolution = obs.bp.ring_radial_resolution('saturn:ring')
            off_emission = obs.bp.ring_emission_angle('saturn:ring',
                                                      pole='north') * oops.DPR
            off_phase = obs.bp.phase_angle('saturn:ring') * oops.DPR
            if arguments.analyze_b_ring_edge:
                analayze_b_ring_edge(image_name, obs, off_radii, off_longitudes, 
                                     off_resolution, off_emission, 
                                     off_incidence, off_phase,
                                     offset)
                # This is just to save disk space
                off_resolution = np.zeros((1,1))
                off_emission = np.zeros((1,1))
                off_phase = np.zeros((1,1))
            else:
                off_resolution = off_resolution.vals.astype('float32')
                off_emission = off_emission.vals.astype('float32')
                off_phase = off_phase.vals.astype('float32')
            off_radii = off_radii.vals.astype('float32')
            off_longitudes = off_longitudes.vals.astype('float32')
            if offset is not None:
                obs.fov = orig_fov
                set_obs_bp(obs, force=True)
        else:
            filename = os.path.join(DEST_ROOT, image_name+'-repro')
            ret = rings_reproject(obs, offset=offset,
                          longitude_resolution=LONGITUDE_RESOLUTION*oops.RPD,
                          radius_resolution=RADIUS_RESOLUTION,
                          radius_range=(RADIUS_INNER,RADIUS_OUTER))
            obs.data = ret['img']
            radii = rings_generate_radii(RADIUS_INNER,RADIUS_OUTER,radius_resolution=RADIUS_RESOLUTION)
            off_radii = np.zeros(obs.data.shape)
            off_radii[:,:] = radii[:,np.newaxis]
            longitudes = rings_generate_longitudes(longitude_resolution=LONGITUDE_RESOLUTION*oops.RPD)
            decimate = max(obs.data.shape[1] // 1024, 1)
            obs.data = obs.data[:,::decimate]
            off_longitudes = np.zeros(obs.data.shape)
            off_longitudes[:,:] = (longitudes[ret['long_mask']])[::decimate] * oops.DPR
            off_resolution = ret['resolution'][:,:decimate]
            off_incidence = ret['incidence'] * oops.DPR
            off_emission = ret['emission'][:,::decimate] * oops.DPR
            off_phase = ret['phase'][:,::decimate] * oops.DPR
            # This is just to save disk space
            off_resolution = np.zeros((1,1))
            off_emission = np.zeros((1,1))
            off_phase = np.zeros((1,1))
            
        midtime = cspice.et2utc(obs.midtime,'C',2)
        np.savez(filename, 
                 midtime=midtime,
                 data=obs.data,
                 radii=off_radii,
                 longitudes=off_longitudes,
                 resolution=off_resolution,
                 incidence=off_incidence,
                 emission=off_emission,
                 phase=off_phase)

#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

for image_path in file_yield_image_filenames_from_arguments(arguments):
    offset_one_image(image_path)
