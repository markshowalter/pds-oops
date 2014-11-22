import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle

from pdstable import PdsTable
from tabulation import Tabulation
import oops.inst.cassini.iss as iss
from imgdisp import *
import Tkinter as tk

from cb_config import *
from cb_offset import *
from cb_rings import *

OCC_ROOT = os.path.join(COUVIS_8XXX_ROOT, 'COUVIS_8001', 'DATA',
                        'EASYDATA')

RING_MIN = 74600
RING_MAX = 137000
RING_INCREMENT = 10

PROFILE_DB = 't:/external/cb_support/ring_profiles.pickle'
 
def read_occultation(filename):
    label = PdsTable(filename)
    print label.info.label['RING_OPEN_ANGLE']
    occ_tab = Tabulation(
           label.column_dict['RING_RADIUS'],
           label.column_dict['NORMAL OPTICAL DEPTH'])
    return occ_tab

def plot_all_occultations():
    radius_range = np.arange(RING_MIN, RING_MAX)
#    filenames = sorted(os.listdir(OCC_ROOT))
    filenames = [
    'UVIS_HSP_2007_015_DELPER_I_TAU_01KM.LBL',
    'UVIS_HSP_2007_078_THEARA_E_TAU_01KM.LBL',
    'UVIS_HSP_2007_082_DELPER_I_TAU_01KM.LBL',
    'UVIS_HSP_2007_129_LAMSCO_I_TAU_01KM.LBL',
    'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL']
    for filename in filenames:
        if not filename.endswith('01KM.LBL'):
            continue
        print filename
        full_path = os.path.join(OCC_ROOT, filename)
        if os.path.isfile(full_path):
            occ_tab = read_occultation(full_path)
            domain = occ_tab.domain()
            if domain[0] > RING_MIN or domain[1] < RING_MAX:
                print 'Incomplete range'
                continue
            results = occ_tab(radius_range)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(radius_range, results, '-')
    plt.show()
    
def read_voyager():
    if_table = PdsTable(os.path.join(SUPPORT_FILES_ROOT,
                                 'IS2_P0001_V01_KM002.LBL'))
    if_data = Tabulation(
           if_table.column_dict['RING_INTERCEPT_RADIUS'],
           if_table.column_dict['I_OVER_F'])

    return if_data

def plot_occ_vs_voyager():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    full_path = os.path.join(OCC_ROOT, 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL')
    occ_tab = read_occultation(full_path)
    occ_results = occ_tab(radius_range)
    occ_results = np.e ** occ_results
    occ_results = (1-1/occ_results)
    voy_tab = read_voyager()
    voy_results = voy_tab(radius_range)
    occ_mean = np.mean(occ_results)
    voy_mean = np.mean(voy_results)
    voy_results *= occ_mean / voy_mean
    print np.corrcoef(occ_results, voy_results)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(radius_range, occ_results, '-', color='black', alpha=0.7)
    plt.plot(radius_range, voy_results, '-', color='red', alpha=0.7)
    plt.show()

def plot_occ_vs_cassini():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    full_path = os.path.join(OCC_ROOT, 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL')
    occ_tab = read_occultation(full_path)
    occ_results = occ_tab(radius_range)
    occ_results = np.e ** occ_results
    occ_results = (1-1/occ_results)
    # Lit side
#    cas_tab = get_ring_radial_profile('COISS_2086/data/1760115351_1760272825/W1760118133_1_CALIB.IMG')

    # Unlit side
    cas_tab = get_ring_radial_profile('COISS_2086/data/1764684406_1765019615/W1764709644_1_CALIB.IMG')
    
    cas_results = cas_tab(radius_range)
    occ_mean = np.mean(occ_results)
    cas_mean = np.mean(cas_results)
    cas_results *= occ_mean / cas_mean
    print np.corrcoef(occ_results, cas_results)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(radius_range, occ_results, '-', color='black', alpha=0.7)
    plt.plot(radius_range, cas_results, '-', color='red', alpha=0.7)
    plt.show()

def plot_lit_vs_unlit():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    # Lit side
    lit_tab = get_ring_radial_profile('COISS_2086/data/1760115351_1760272825/W1760118133_1_CALIB.IMG')

    # Unlit side
    unlit_tab = get_ring_radial_profile('COISS_2086/data/1764684406_1765019615/W1764709644_1_CALIB.IMG')
    
    lit_results = lit_tab(radius_range)
    unlit_results = unlit_tab(radius_range)
    lit_mean = np.mean(lit_results)
    unlit_mean = np.mean(unlit_results)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(radius_range, lit_results, '-', color='black', alpha=0.7)
    plt.plot(radius_range, unlit_results, '-', color='red', alpha=0.7)
    plt.show()

def find_corr_sections_voyager():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    full_path = os.path.join(OCC_ROOT, 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL')
    occ_tab = read_occultation(full_path)
    occ_results = occ_tab(radius_range)
    occ_results = np.e ** occ_results
    occ_results = (1-1/occ_results)
    voy_tab = read_voyager()
    voy_results = voy_tab(radius_range)
    corr_res = []
    window = 50
    for start in xrange(len(radius_range)-window):
        end = start+window
        corr = np.corrcoef(occ_results[start:end], voy_results[start:end])[0,1]
        corr_res.append(corr)
        
    plt.plot(corr_res)
    plt.show()
    
def find_corr_sections_cassini():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    full_path = os.path.join(OCC_ROOT, 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL')
    occ_tab = read_occultation(full_path)
    occ_results = occ_tab(radius_range)
    occ_results = np.e ** occ_results
    occ_results = (1-1/occ_results)
    cas_tab = get_ring_radial_profile('t:/external/cassini/derived/COISS_2xxx/COISS_2086/data/1760115351_1760272825/W1760118133_1_CALIB.IMG')

    cas_results = cas_tab(radius_range)
    corr_res = []
    window = 20
    for start in xrange(len(radius_range)-window):
        end = start+window
        corr = np.corrcoef(occ_results[start:end], cas_results[start:end])[0,1]
        corr_res.append(corr)
        
    plt.plot(corr_res)
    plt.show()
    
#===============================================================================
# 
#===============================================================================

def get_ring_radial_profile(filename, start_long_idx=None, end_long_idx=None,
                            interactive=True):
    filename = os.path.join(COISS_2XXX_DERIVED_ROOT, filename)
        
    obs = iss.from_file(filename)
    print filename
    print 'DATA SIZE', obs.data.shape, 'TEXP', obs.texp, 'FILTERS', 
    print obs.filter1, obs.filter2

    offset_u, offset_v, metadata = master_find_offset(obs, create_overlay=True,
                                                      allow_moons=False,
                                                      allow_rings=True,
                                                      allow_saturn=False,
                                                      allow_stars=False)

    if offset_u is None:
        print 'OFFSET FAILED'
        return None
    
    radius_inner = 74000
    radius_outer = 139000
    radius_resolution = 25.
    longitude_resolution = 0.05
    
    reproj = rings_reproject(obs,
                             offset_u=offset_u, offset_v=offset_v,
                             longitude_resolution=longitude_resolution,
                             radius_resolution=radius_resolution,
                             radius_inner=radius_inner, radius_outer=radius_outer)

    reproj_img = reproj['img']
    reproj_incidence = reproj['incidence']
    reproj_emission = reproj['emission']
    reproj_phase = reproj['phase']
    
    if interactive:
        toplevel = Tk()
        toplevel.title(filename)
        frame_toplevel = Frame(toplevel)
    
        imgdisp = ImageDisp([reproj_img], canvas_size=(1024,768), 
                            parent=frame_toplevel, allow_enlarge=True,
                            auto_update=True)
    
        frame_toplevel.pack()
    
        tk.mainloop()

    if start_long_idx is None:
        start_long_idx = 0
    if end_long_idx is None:
        end_long_idx = reproj_img.shape[1]-1
        
    radius = rings_generate_radii(radius_inner, radius_outer,
                                  radius_resolution=radius_resolution)

    reproj_incidence = ma.masked_equal(reproj_incidence, 0.)
    reproj_emission = ma.masked_equal(reproj_emission, 0.)
    reproj_phase  = ma.masked_equal(reproj_phase, 0.)

    mean_incidence = np.mean(reproj_incidence[:,start_long_idx:end_long_idx+1])
    mean_emission = np.mean(reproj_emission[:,start_long_idx:end_long_idx+1])
    mean_phase = np.mean(reproj_phase[:,start_long_idx:end_long_idx+1])
    
    reproj_img = ma.masked_equal(reproj_img, 0.)
    
    return (Tabulation(radius,
                       ma.median(reproj_img[:,start_long_idx:end_long_idx+1], axis=1)),
            mean_incidence, mean_emission, mean_phase)

def add_radial_profile(filename, start_long_idx=None, end_long_idx=None):
    tab, incidence, emission, phase = get_ring_radial_profile(filename, start_long_idx=None, end_long_idx=None)
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)
    prof = tab(radius_range)
    
    try:
        fp = open(PROFILE_DB, 'rb')
        dict = pickle.load(fp)
        fp.close()
    except IOError:
        dict = {}
    
    dict[filename] = (incidence, emission, phase, prof)
    
    fp = open(PROFILE_DB, 'wb')
    pickle.dump(dict, fp)
    fp.close()

    print radius_range
    print prof
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(radius_range, prof, '-', color='black')
    plt.show()

def moving_average(values, window):
    ret = np.empty(values.shape)
    for i in xrange(ret.shape[0]):
        i1 = np.clip(i-window//2, 0, ret.shape[0]-1)
        i2 = np.clip(i+window//2, 0, ret.shape[0]-1)
        ret[i] = np.mean(values[i1:i2+1])
    return ret

def plot_radial_profiles():
    radius_range = np.arange(RING_MIN, RING_MAX+1, RING_INCREMENT)

    full_path = os.path.join(OCC_ROOT, 'UVIS_HSP_2008_231_BETCEN_I_TAU_01KM.LBL')
    occ_tab = read_occultation(full_path)
    occ_results = occ_tab(radius_range)
#    occ_results = np.e ** occ_results
#    occ_results = (1-1/occ_results)

    occ_results = moving_average(occ_results, 100)
    
    fp = open(PROFILE_DB, 'rb')
    dict = pickle.load(fp)
    fp.close()

    for lit in [False, True]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        plt.plot(radius_range, occ_results/np.mean(occ_results), '-', color='black', lw=3)
        
        for key in sorted(dict.keys()):
            (incidence, emission, phase, prof) = dict[key]
            print key, incidence, emission, phase
            if lit and emission > 90 or not lit and emission < 90:
                continue
            label = key[-15:-12] + ' ('
            label += 'i=%4.1f e=%3d p=%3d' % (incidence, emission, phase)
            label += ')'
            if phase > 120:
                ls = ':'
            elif phase > 90:
                ls = '-'
            lw = 1
    #        prof = prof * np.abs(np.cos(emission*oops.RPD))
    #        prof = prof / np.abs(np.cos(incidence*oops.RPD))
            prof = prof / np.mean(prof)
            plt.plot(radius_range, prof, ls, lw=lw, label=label) #, color=color)
    
        plt.legend()
    plt.show() 
    
    
def kill_entry():
    fp = open(PROFILE_DB, 'rb')
    dict = pickle.load(fp)
    fp.close()

    del dict['COISS_2053/data/1614298008_1614456510/W1614311192_1_CALIB.IMG']

    fp = open(PROFILE_DB, 'wb')
    pickle.dump(dict, fp)
    fp.close()

#kill_entry()
    
#find_corr_sections_cassini()

#plot_lit_vs_unlit()

#plot_occ_vs_cassini()
#plot_occ_vs_voyager()
#plot_all_occultations()


#get_ring_radial_profile('t:/external/cassini/derived/COISS_2xxx/COISS_2086/data/1760115351_1760272825/W1760118133_1_CALIB.IMG')

plot_radial_profiles()

# High phase
#add_radial_profile('COISS_2053/data/1614298008_1614456510/W1614319532_1_CALIB.IMG')
# BAD add_radial_profile('COISS_2053/data/1614298008_1614456510/W1614311192_1_CALIB.IMG')
#add_radial_profile('COISS_2030/data/1553210008_1553330871/W1553243403_1_CALIB.IMG', 0, 1593)
#add_radial_profile('COISS_2027/data/1543428718_1543604268/W1543458274_1_CALIB.IMG', 1060, 1790)
#add_radial_profile('COISS_2026/data/1539755921_1540366583/W1539812139_1_CALIB.IMG', 1100, None)

# Mid phase
#add_radial_profile('COISS_2053/data/1614298008_1614456510/W1614455971_1_CALIB.IMG')
#add_radial_profile('COISS_2029/data/1550433865_1550493832/W1550453281_1_CALIB.IMG')
#add_radial_profile('COISS_2062/data/1653888591_1654131882/W1653991810_1_CALIB.IMG', 1140, 1300)
#add_radial_profile('COISS_2063/data/1656533083_1656606224/W1656595136_1_CALIB.IMG', 1200, 1400)
#add_radial_profile('COISS_2026/data/1540725997_1540867420/W1540757533_1_CALIB.IMG')
#add_radial_profile('COISS_2030/data/1552434197_1552731724/W1552688371_1_CALIB.IMG')
#add_radial_profile('COISS_2053/data/1613001873_1613171522/W1613065241_1_CALIB.IMG')
#add_radial_profile('COISS_2056/data/1627523072_1627597226/W1627591691_1_CALIB.IMG')
#add_radial_profile('COISS_2081/data/1741299057_1741467421/W1741302783_1_CALIB.IMG')


