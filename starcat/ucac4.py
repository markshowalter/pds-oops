################################################################################
# starcat/ucac4.py
################################################################################

from starcatalog import *
import numpy as np
import os.path
import struct

OBJ_TYPE_CLEAN = 0
OBJ_TYPE_NEAR_OVEREXPOSED = 1
OBJ_TYPE_STREAK = 2
OBJ_TYPE_HPM = 3
OBJ_TYPE_EXT_HPM = 4
OBJ_TYPE_POOR_PM = 5
OBJ_TYPE_SUBST_ASTROMETRY = 6
OBJ_TYPE_SUPPL = 7
OBJ_TYPE_HPM_NOT_MATCHED = 8
OBJ_TYPE_HPM_DISCREPANT = 9

CATMATCH_TYCHO = 0
CATMATCH_AC2000 = 1
CATMATCH_AGK2_BONN = 2
CATMATCH_AKG2_HAMBURG = 3
CATMATCH_ZONE_ASTROGRAPH = 14
CATMATCH_BLACK_BIRCH = 5
CATMATCH_LICK_ASTROGRAPH = 6
CATMATCH_NPM_LICK = 7
CATMATCH_SPM_YSJ1 = 8

class UCAC4Star(Star):
    """A holder for star attributes.
    
    This class includes attributes unique to the UCAC4 catalog."""
    
    def __init__(self):
        # Initialize the standard fields
        Star.__init__(self)
        
        # Initialize the UCAC4-specific fields
        self.vmag_model = None
        self.obj_type = None
        self.double_star_flag = None
        self.double_star_type = None
        self.galaxy_match = None
        self.extended_source = None
        self.pm_rac = None
        self.pm_rac_sigma = None
        self.rac_sigma = None
        self.num_img_total = None
        self.num_img_used = None
        self.num_cat_pm = None
        self.ra_mean_epoch = None
        self.dec_mean_epoch = None
        self.cat_match = None
        self.id_str = None
        

#col byte item   fmt unit       explanation                            notes
#---------------------------------------------------------------------------
# 1  1- 4 ra     I*4 mas        right ascension at  epoch J2000.0 (ICRS) (1)
# 2  5- 8 spd    I*4 mas        south pole distance epoch J2000.0 (ICRS) (1)
# 3  9-10 magm   I*2 millimag   UCAC fit model magnitude                 (2)
# 4 11-12 maga   I*2 millimag   UCAC aperture  magnitude                 (2)
# 5 13    sigmag I*1 1/100 mag  error of UCAC magnitude                  (3)
# 6 14    objt   I*1            object type                              (4)
# 7 15    cdf    I*1            combined double star flag                (5)
#         15 bytes
# 8 16    sigra  I*1 mas        s.e. at central epoch in RA (*cos Dec)   (6)
# 9 17    sigdc  I*1 mas        s.e. at central epoch in Dec             (6)
#10 18    na1    I*1            total # of CCD images of this star
#11 19    nu1    I*1            # of CCD images used for this star       (7)
#12 20    cu1    I*1            # catalogs (epochs) used for proper motions
#          5 bytes
#13 21-22 cepra  I*2 0.01 yr    central epoch for mean RA, minus 1900
#14 23-24 cepdc  I*2 0.01 yr    central epoch for mean Dec,minus 1900
#15 25-26 pmrac  I*2 0.1 mas/yr proper motion in RA*cos(Dec)             (8)
#16 27-28 pmdc   I*2 0.1 mas/yr proper motion in Dec
#17 29    sigpmr I*1 0.1 mas/yr s.e. of pmRA * cos Dec                   (9)
#18 30    sigpmd I*1 0.1 mas/yr s.e. of pmDec                            (9)
#         10 bytes
#19 31-34 pts_key I*4           2MASS unique star identifier            (10)
#20 35-36 j_m    I*2 millimag   2MASS J  magnitude
#21 37-38 h_m    I*2 millimag   2MASS H  magnitude
#22 39-40 k_m    I*2 millimag   2MASS K_s magnitude
#23 41    icqflg I*1            2MASS cc_flg*10 + ph_qual flag for J    (11)
#24 42     (2)   I*1            2MASS cc_flg*10 + ph_qual flag for H    (11)
#25 43     (3)   I*1            2MASS cc_flg*10 + ph_qual flag for K_s  (11)
#26 44    e2mpho I*1 1/100 mag  error 2MASS J   magnitude               (12)
#27 45     (2)   I*1 1/100 mag  error 2MASS H   magnitude               (12)
#28 46     (3)   I*1 1/100 mag  error 2MASS K_s magnitude               (12)
#         16 bytes
#29 47-48 apasm  I*2 millimag   B magnitude from APASS                  (13)
#30 49-50  (2)   I*2 millimag   V magnitude from APASS                  (13)
#31 51-52  (3)   I*2 millimag   g magnitude from APASS                  (13)
#32 53-54  (4)   I*2 millimag   r magnitude from APASS                  (13)
#33 55-56  (5)   I*2 millimag   i magnitude from APASS                  (13)
#34 57    apase  I*1 1/100 mag  error of B magnitude from APASS         (14)
#35 58     (2)   I*1 1/100 mag  error of V magnitude from APASS         (14)
#36 59     (3)   I*1 1/100 mag  error of g magnitude from APASS         (14)
#37 60     (4)   I*1 1/100 mag  error of r magnitude from APASS         (14)
#38 61     (5)   I*1 1/100 mag  error of i magnitude from APASS         (14)
#39 62    gcflg  I*1            Yale SPM g-flag*10  c-flag              (15)
#         16 bytes
#40 63-66 icf(1) I*4            FK6-Hipparcos-Tycho source flag         (16)
#41       icf(2) ..             AC2000       catalog match flag         (17)
#42       icf(3) ..             AGK2 Bonn    catalog match flag         (17)
#43       icf(4) ..             AKG2 Hamburg catalog match flag         (17)
#44       icf(5) ..             Zone Astrog. catalog match flag         (17)
#45       icf(6) ..             Black Birch  catalog match flag         (17)
#46       icf(7) ..             Lick Astrog. catalog match flag         (17)
#47       icf(8) ..             NPM  Lick    catalog match flag         (17)
#48       icf(9) ..             SPM  YSJ1    catalog match flag         (17)
#          4 bytes
#49 67    leda   I*1            LEDA galaxy match flag                  (18)
#50 68    x2m    I*1            2MASS extend.source flag                (19)
#51 69-72 rnm    I*4            unique star identification number       (20)
#52 73-74 zn2    I*2            zone number of UCAC2 (0 = no match)     (21)
#53 75-78 rn2    I*4            running record number along UCAC2 zone  (21)

UCAC4_FMT = '=iihhbbbbbbbbhhhhbbihhhbbbbbbhhhhhbbbbbbibbihi'
UCAC4_RECORD_SIZE = 78
assert struct.calcsize(UCAC4_FMT) == UCAC4_RECORD_SIZE

UCAC4_FMT_RA = '=i'
UCAC4_RECORD_SIZE_RA = 4
assert struct.calcsize(UCAC4_FMT_RA) == UCAC4_RECORD_SIZE_RA

class UCAC4StarCatalog(StarCatalog):
    def __init__(self, dir=None):
        if dir is None:
            self.dirname = os.environ["UCAC4_DIR"]
        else:
            self.dirname = dir
        self.debug = False
        
    def _find_stars(self, ra_min, ra_max, dec_min, dec_max, **kwargs):
        """Yield the results for all zones in the DEC range.
        
        Optional arguments:      DEFAULT
            ra_min, ra_max       0, 2PI    RA range in radians
            dec_min, dec_max     -PI, PI   DEC range in radians
            vmag_min, vmag_max     ALL       Magnitude range
            require_clean        True      Only return stars that are clean
                                           detections
            allow_double         False     Allow double stars
            allow_galaxy         False     Allow galaxies and extended objects
            require_pm           True      Only return stars with known proper
                                           motion
            full_result          True      Populate all fields
                                           False: Stop after fields required
                                           for matching criteria
            optimize_ra          True      Use a binary search to find the
                                           starting RA
        """
        
        start_znum = int(max(np.floor((dec_min*DPR+90)*5)+1, 1))
        end_znum = int(min(np.floor((dec_max*DPR+90-1e-15)*5)+1, 900))
        
        for znum in xrange(start_znum, end_znum+1):
            fn = self._zone_filename(znum)
            with open(fn, 'rb') as fp:
                for star in self._find_stars_one_file(znum, fp,
                                                      ra_min, ra_max,
                                                      dec_min, dec_max,
                                                      **kwargs):
                    yield star
    
    def _find_stars_one_file(self, znum, fp, ra_min, ra_max, dec_min, dec_max,
                             **kwargs):
        """Yield the results for a single zone."""
        full_result = kwargs.get('full_result', True)
        vmag_min = kwargs.get('vmag_min', None)
        vmag_max = kwargs.get('vmag_max', None)
        require_clean = kwargs.get('require_clean', True)
        allow_double = kwargs.get('allow_double', False)
        allow_galaxy = kwargs.get('allow_galaxy', False)
        require_pm = kwargs.get('require_pm', True)
        optimize_ra = kwargs.get('optimize_ra', True)
        
        if optimize_ra:
            rec_num = self._find_starting_ra(fp, ra_min) # 0-based
        else:
            rec_num = 0
            
        while True:
            record = fp.read(UCAC4_RECORD_SIZE)
            if len(record) != UCAC4_RECORD_SIZE:
                break
            rec_num += 1 # This makes rec_num 1-based
            parsed = struct.unpack(UCAC4_FMT, record)
            star = UCAC4Star()
            
            if self.debug:
                print 'EXAMINING UNIQUE ID', parsed[42], '...',
            
            ###########
            # RA, DEC #
            ###########
            
# 1  1- 4 ra     I*4 mas        right ascension at  epoch J2000.0 (ICRS) (1)
# 2  5- 8 spd    I*4 mas        south pole distance epoch J2000.0 (ICRS) (1)
#    Note  (1):  Positions are on the International Celestial Reference
#    System (ICRS) as represented by the Hipparcos / Tycho-2  catalogs.
#    The epoch for the positions of all stars is J2000.0; the weighted
#    mean catalog position was updated using the provided proper
#    motions.  The observational UCAC position is but one of several
#    going into these values and is not given in the UCAC4; thus the
#    original UCAC observation cannot be recovered from these data. 
#    The declination is given in south pole distance (spd) and can be
#    converted back to a true declination by subtracting 324000000 mas.
            star.ra = parsed[0] * MAS_TO_DEG * RPD
            star.dec = (parsed[1] * MAS_TO_DEG - 90) * RPD
            if star.ra >= ra_max:
                # RA is in ascending order in the file
                if self.debug:
                    print 'SKIPPED RA AND REST OF FILE', star.ra
                break
            if (star.ra < ra_min or
                star.dec < dec_min or star.dec >= dec_max):
                if self.debug:
                    print 'SKIPPED RA/DEC', star.ra, star.dec
                continue

            #############
            # MAGNITUDE #
            #############

# 3  9-10 magm   I*2 millimag   UCAC fit model magnitude                 (2)
# 4 11-12 maga   I*2 millimag   UCAC aperture  magnitude                 (2)
# 5 13    sigmag I*1 1/100 mag  error of UCAC magnitude                  (3)
#    Note  (2):  Unknown, or unrealistic photometric results are set to 
#    magnitude = 20 (20000 mmag entry in catalog).  Systematic errors
#    are expected to be below 0.1 mag for magm,maga photometric results
#    obtained from the UCAC CCD pixel data.  The aperture photometry
#    is considered more robust, particularly for "odd" cases, while 
#    the model fit magnitude is expected to be more accurate for 
#    "well behaved" stars.
#
#    Note  (3):  A value of 99 for error in magnitude means "no data".
#    For many stars a photometric error based on the scatter from 
#    individual observations of that star on different CCD frames
#    could be obtained.  A model error was also attempted to be
#    assigned, based on the S/N ratio.  The error quoted here is
#    the larger of the 2.  If that error exceeds 0.9 mag the error
#    was set to 0.9 mag (= value 90 in catalog data, unit = 10 mmag).
            if parsed[2] == 20000:
                star.vmag_model = None
            else:
                star.vmag_model = parsed[2] / 1000.
            if parsed[3] == 20000:
                star.vmag = None
            else:
                star.vmag = parsed[3] / 1000.
            if parsed[4] == 99:
                star.vmag_sigma = None
            else:
                star.vmag_sigma = parsed[4] / 100.
            if vmag_min is not None:
                if star.vmag is None or star.vmag < vmag_min:
                    if self.debug:
                        print 'SKIPPED MODEL MAG', star.vmag_model
                    continue
            if vmag_max is not None:
                if star.vmag is None or star.vmag > vmag_max:
                    if self.debug:
                        print 'SKIPPED MODEL MAG', star.vmag_model
                    continue
            
            ###############
            # OBJECT TYPE #
            ###############
            
# 6 14    objt   I*1            object type                              (4)
#    Note  (4):  The object type flag is used to identify possible problems 
#    with a star or the source of data.  Of the individual image flags
#    the one with the largest value (worst problem case) is propagated
#    into this object type flag, unless it is superseded by an overriding
#    flag at the combined image stage.
#    The object type flag has the following meaning:
#
#    0 = good, clean star (from MPOS), no known problem
#    1 = largest flag of any image = near overexposed star (from MPOS)
#    2 = largest flag of any image = possible streak object (from MPOS)
#    3 = high proper motion (HPM) star, match with external PM file (MPOS)
#    4 = actually use external HPM data instead of UCAC4 observ.data
#          (accuracy of positions varies between catalogs)
#    5 = poor proper motion solution, report only CCD epoch position
#    6 = substitute poor astrometric results by FK6/Hip/Tycho-2 data
#    7 = added supplement star (no CCD data) from FK6/Hip/Tycho-2 data,
#         and 2 stars added from high proper motion surveys
#    8 = high proper motion solution in UCAC4, star not matched with PPMXL
#    9 = high proper motion solution in UCAC4, discrepant PM to PPMXL
#     (see discussion of flags 8,9 in redcution section 2e above)
            star.obj_type = parsed[5]
            if (require_clean and
                (star.obj_type == OBJ_TYPE_STREAK or
                 star.obj_type == OBJ_TYPE_HPM_NOT_MATCHED or
                 star.obj_type == OBJ_TYPE_HPM_DISCREPANT)):
                # Use with extreme caution
                if self.debug:
                    print 'SKIPPED NOT CLEAN', star.obj_type
                continue

            #############################
            # COMBINED DOUBLE STAR FLAG #
            #############################
            
# 7 15    cdf    I*1            combined double star flag                (5)
#    Note  (5):  The cdf flag is a combined double star flag used to indicate 
#    the type/quality of double star fit.  It is a combination of 2 flags,
#    cdf = 10 * dsf + dst  with the following meaning:
#
#    dsf = double star flag = overall classification
#      0 = single star
#      1 = component #1 of "good" double star
#      2 = component #2 of "good" double star
#      3 = blended image
#
#    dst = double star type, from pixel data image profile fits,
#           largest value of all images used for this star
#      0 = no double star, not sufficient #pixels or elongation
#           to even call double star fit subroutine
#      1 = elongated image but no more than 1 peak detected
#      2 = 2 separate peaks detected -> try double star fit
#      3 = secondary peak found on each side of primary
#      4 = case 1 after successful double fit (small separ. blended image)
#      5 = case 2 after successful double fit (most likely real double)
#      6 = case 3 after successful double fit (brighter secondary picked)
#
#    A word of caution: often a dsf= 1 or 2 image is paired with a dsf= 3.
#    If for a star any of the several images reveals a "blended image",
#    that higher dsf=3 flag is carried into the output file.  This can
#    happen for a regular double star with unique components 1 and 2.
#    A flag dsf=3 means this could be component 1 or 2 but at least on
#    one CCD frame a blended image was detected.  This blend could be
#    with the other component, or a spurious image or artifact.
#    The double star flags need to be interpreted with caution; anything
#    but a zero means "likely some double star component or blended image".
            cdf = parsed[6]
            if not allow_double and cdf != 0:
                if self.debug:
                    print 'SKIPPED DOUBLE', star.cdf
                continue
            star.double_star_flag = cdf // 10
            star.double_star_type = cdf % 10

            #################################
            # GALAXIES AND EXTENDED SOURCES #
            #################################
            
#(41)49 67    leda   I*1            LEDA galaxy match flag                  (18)
#(42)50 68    x2m    I*1            2MASS extend.source flag                (19)
#    Note (18):  This flag is either 0 (no match) or contains the log10 of
#    the apparent total diameter for I-band (object size) information
#    (unit = 0.1 arcmin) copied from the LEDA catalog (galaxies).  
#    A size value of less than 1 has been rounded up to 1.
#
#    Note (19):  This flag is either 0 (no match) or contains the length of
#    the semi-major axis of the fiducial ellipse at the K-band 
#    (object size) information copied from the 2MASS extended source
#    catalog. 
            star.galaxy_match = parsed[40]
            star.extended_source = parsed[41] # XXX What units is this in?
            if not allow_galaxy and (star.galaxy_match or star.extended_source):
                if self.debug:
                    print 'SKIPPED GALAXY/EXTENDED', star.galaxy_match, star.extended_source
                continue
            if star.galaxy_match:
                star.galaxy_match = 10.**star.galaxy_match / 10. / 60. # Degrees

            #################
            # PROPER MOTION #
            #################
            
#15 25-26 pmrac  I*2 0.1 mas/yr proper motion in RA*cos(Dec)             (8)
#16 27-28 pmdc   I*2 0.1 mas/yr proper motion in Dec
#17 29    sigpmr I*1 0.1 mas/yr s.e. of pmRA * cos Dec                   (9)
#18 30    sigpmd I*1 0.1 mas/yr s.e. of pmDec                            (9)
#    Note  (8):  A value of 32767 for either proper motion component means
#    the real PM of that star is larger and found in the extra table
#    file  u4hpm.dat (32 stars, ASCII).  The cross reference is established
#    by the unique, modified MPOS number (column 51 of main data file),
#    which is also given on the HPM supplement stars file.
#    For stars without valid proper motion the proper motion data are set
#    to 0.  However, valid proper motions can also be 0.  The "no data"
#    case is indicated by the sigma proper motion columns (see below).
#
#    Note  (9):  Values in the binary data files are represented as signed,
#    1-byte integer (range -128 to 127).  Add 128 to bring those values
#    to the range of 0 to 255, which is the error in proper motion in
#    unit of 0.1 mas/yr, with the following exception.
#    Data entries above 250 indicate larger errors as follows:
#    251 --> 275 = 27.5 mas/yr
#    252 --> 325 = 32.5 mas/yr
#    253 --> 375 = 37.5 mas/yr
#    254 --> 450 = 45.0 mas/yr
#    255 --> "no data" = set to 500 for output tables
#
#    For astrometric data copied from the FK6, Hipparcos and Tycho-2
#    catalogs a mean error in positions was adopted depending on input
#    catalog and the brightness of the star rather than giving the 
#    individual star's error quoted in those catalogs.
    
            star.pm_rac = parsed[14] * 0.1 * MAS_TO_DEG # RA*COS(DEC), DEG/YR
            star.pm_dec = parsed[15] * 0.1 * MAS_TO_DEG # DEG/YR
            if parsed[14] == 32767 or parsed[15] == 32767:
                # PM is too large and needs to be looked up in another
                # table, which we don't support yet. XXX
                star.pm_rac = None
                star.pm_dec = None

            prse = parsed[16]+128
            pdse = parsed[17]+128
            star.pm_rac_sigma = prse * 0.1 * MAS_TO_DEG # RA*COS(DEC), DEG/YR
            star.pm_dec_sigma = pdse * 0.1 * MAS_TO_DEG # DEG/YR
            if prse == 251:
                star.pm_rac_sigma = 27.5 * MAS_TO_DEG
            elif prse == 252:
                star.pm_rac_sigma = 32.5 * MAS_TO_DEG
            elif prse == 253:
                star.pm_rac_sigma = 37.5 * MAS_TO_DEG
            elif prse == 254:
                star.pm_rac_sigma = 45.0 * MAS_TO_DEG
            elif prse == 255:
                star.pm_rac_sigma = None
                if star.pm_rac == 0:
                    star.pm_rac = None
            if pdse == 251:
                star.pm_dec_sigma = 27.5 * MAS_TO_DEG
            elif pdse == 252:
                star.pm_dec_sigma = 32.5 * MAS_TO_DEG
            elif pdse == 253:
                star.pm_dec_sigma = 37.5 * MAS_TO_DEG
            elif pdse == 254:
                star.pm_dec_sigma = 45.0 * MAS_TO_DEG
            elif pdse == 255:
                star.pm_dec_sigma = None
                if star.pm_dec == 0:
                    star.pm_dec = None

            if require_pm and (star.pm_rac is None or star.pm_dec is None):
                if self.debug:
                    print 'SKIPPED NO PM', parsed[14:18]
                continue
            
            #################################
            #################################
            ### END OF SELECTION CRITERIA ###
            #################################
            #################################
            
            if not full_result:
                if self.debug:
                    print 'OK!'
                yield star
                continue
            
            ###########################
            # RA/DEC SYSTEMATIC ERROR #
            ###########################
            
# 8 16    sigra  I*1 mas        s.e. at central epoch in RA (*cos Dec)   (6)
# 9 17    sigdc  I*1 mas        s.e. at central epoch in Dec             (6)
#    Note  (6):  The range of values here is 1 to 255 which is represented
#    as a signed 1-byte integer (range -127 to 127); thus add 128 to the
#    integer number found in the data file.  There is no 0 mas value;
#    data less than 1 mas have been set to 1 mas.  Original data larger
#    than 255 mas have been set to 255.  
#    If the astrometric data for a star was substituted from an external
#    catalog like Hipparcos, Tycho or high proper motion data, a mean
#    error in position and proper motion depending on the catalog and
#    magnitude of the star was adopted.
            star.rac_sigma = (parsed[7]+128.) * MAS_TO_DEG # RA * COS(DEC)
            star.dec_sigma = (parsed[8]+128.) * MAS_TO_DEG

            ##############
            # IMAGE INFO #
            ##############
            
#10 18    na1    I*1            total # of CCD images of this star
#11 19    nu1    I*1            # of CCD images used for this star       (7)
#12 20    cu1    I*1            # catalogs (epochs) used for proper motions
#    Note  (7):  A zero for the number of used images indicates that all images
#    have some "problem" (such as overexposure). In that case an unweighted
#    mean over all available images (na) is taken to derive the mean
#    position, while normally a weighted mean was calculated based on
#    the "good" images, excluding possible problem images (nu <= na).
            star.num_img_total = parsed[9]
            star.num_img_used = parsed[10]
            star.num_cat_pm = parsed[11]
            
            #########
            # EPOCH #
            #########
            
#13 21-22 cepra  I*2 0.01 yr    central epoch for mean RA, minus 1900
#14 23-24 cepdc  I*2 0.01 yr    central epoch for mean Dec,minus 1900
            star.ra_mean_epoch = parsed[12] * 0.01 + 1900
            star.dec_mean_epoch = parsed[13] * 0.01 + 1900
            
            ##############                
            # 2MASS DATA #
            ##############
            
#19 31-34 pts_key I*4           2MASS unique star identifier            (10)
#20 35-36 j_m    I*2 millimag   2MASS J  magnitude
#21 37-38 h_m    I*2 millimag   2MASS H  magnitude
#22 39-40 k_m    I*2 millimag   2MASS K_s magnitude
#23 41    icqflg I*1            2MASS cc_flg*10 + ph_qual flag for J    (11)
#24 42     (2)   I*1            2MASS cc_flg*10 + ph_qual flag for H    (11)
#25 43     (3)   I*1            2MASS cc_flg*10 + ph_qual flag for K_s  (11)
#26 44    e2mpho I*1 1/100 mag  error 2MASS J   magnitude               (12)
#27 45     (2)   I*1 1/100 mag  error 2MASS H   magnitude               (12)
#28 46     (3)   I*1 1/100 mag  error 2MASS K_s magnitude               (12)
#    Note (10):  The 2MASS items copied into UCAC4 are described at
#    pegasus.astro.umass.edu/ipac_wget/releases/allsky/doc/sec2_2a.html
#
#    Note (11):  For each 2MASS bandpass a combined flag was created
#    (cc_flg*10 + ph_qual) consisting of the contamination flag (0 to 5) 
#    and the photometric quality flag (0 to 8).  
#
#    0 =  cc_flg  2MASS 0, no artifacts or contamination
#    1 =  cc_flg  2MASS p, source may be contaminated by a latent image
#    2 =  cc_flg  2MASS c, photometric confusion
#    3 =  cc_flg  2MASS d, diffraction spike confusion
#    4 =  cc_flg  2MASS s, electronic stripe
#    5 =  cc_flg  2MASS b, bandmerge confusion
#
#    0 =  no ph_qual flag
#    1 =  ph_qual 2MASS X, no valid brightness estimate
#    2 =  ph_qual 2MASS U, upper limit on magnitude
#    3 =  ph_qual 2MASS F, no reliable estimate of the photometric error
#    4 =  ph_qual 2MASS E, goodness-of-fit quality of profile-fit poor
#    5 =  ph_qual 2MASS A, valid measurement, [jhk]snr>10 AND [jhk]cmsig<0.10857
#    6 =  ph_qual 2MASS B, valid measurement, [jhk]snr> 7 AND [jhk]cmsig<0.15510
#    7 =  ph_qual 2MASS C, valid measurement, [jhk]snr> 5 AND [jhk]cmsig<0.21714
#    8 =  ph_qual 2MASS D, valid measurement, no [jhk]snr OR [jhk]cmsig req.
#
#    For example icqflg = 05 is decoded to be cc_flg=0, and ph_qual=5, meaning
#    no artifacts or contamination from cc_flg and 2MASS qual flag = "A" .
#
#    Note (12):  The photometric errors from 2MASS were rounded by 1 digit
#    here to fit into fewer bytes (1/100 mag instead of millimag).
#    These data were taken from the j_msigcom, h_msigcom, and k_msigcom columns
#    of the 2MASS point source catalog.  See note (10).

            ##############
            # APASS DATA #
            ##############

#29 47-48 apasm  I*2 millimag   B magnitude from APASS                  (13)
#30 49-50  (2)   I*2 millimag   V magnitude from APASS                  (13)
#31 51-52  (3)   I*2 millimag   g magnitude from APASS                  (13)
#32 53-54  (4)   I*2 millimag   r magnitude from APASS                  (13)
#33 55-56  (5)   I*2 millimag   i magnitude from APASS                  (13)
#34 57    apase  I*1 1/100 mag  error of B magnitude from APASS         (14)
#35 58     (2)   I*1 1/100 mag  error of V magnitude from APASS         (14)
#36 59     (3)   I*1 1/100 mag  error of g magnitude from APASS         (14)
#37 60     (4)   I*1 1/100 mag  error of r magnitude from APASS         (14)
#38 61     (5)   I*1 1/100 mag  error of i magnitude from APASS         (14)
#39 62    gcflg  I*1            Yale SPM g-flag*10  c-flag              (15)
#    Note (13):  Data are from the AAVSO Photometric all-sky survey (APASS)
#    DR6 plus single observation stars kindly provided by A.Henden.
#    A magnitude entry of 20000 indicates "no data".  For bright stars
#    the  apasm(1) = B mag and apasm(2) = V mag columns contain the
#    Hipparcos/Tycho Bt and Vt mags respectively, whenever there is no
#    APASS B or V available and valid Bt or Vt mags were found.
#    For the bright supplement stars the same was done.  All thses cases
#    are identified by apasm(1) < 20000 and apase(1) = 0  for B mags,
#    and similarly for apasm(2) < 20000 and apase(2) = 0  for V mags.
#    For over 10,000 stars no Vt mag was available and the V mag from Tycho
#    was used instead.  
#
#    Note (14):  Positive errors are from the official release data error
#    estimates (at least 2 observations per star).  Formal, S/N estimated
#    errors for single observations are multiplied by -1 for this column.
#    The valid range for each APASS magnitude error is +-90 = +-0.90 mag.
#    For "no data" (i.e. magnitude = 20000 = 20.0 mag) the error is set to 99.
#
#    Note (15):  The g-flag from the Yale San Juan first epoch Southern
#    Proper Motion data (YSJ1, SPM) has the following meaning:
#
#     0 = no info
#     1 = matched with 2MASS extended source list
#     2 = LEDA  galaxy
#     3 = known QSO
#
#    The c-flag from the Yale San Juan first epoch Southern
#    Proper Motion data (YSJ1, SPM) indicates which input catalog
#    has been used to identify stars for pipeline processing:
#
#     1 = Hipparcos
#     2 = Tycho2
#     3 = UCAC2
#     4 = 2MASS psc
#     5 = 2MASS xsc (extended sources, largely (but not all!) galaxies)
#     6 = LEDA  (confirmed galaxies, Paturel et al. 2005)
#     7 = QSO   (Veron-Cetty & Veron 2006)
     
            #######################
            # CATALOG MATCH FLAGS #
            #######################
            
#40 63-66 icf(1) I*4            FK6-Hipparcos-Tycho source flag         (16)
#41       icf(2) ..             AC2000       catalog match flag         (17)
#42       icf(3) ..             AGK2 Bonn    catalog match flag         (17)
#43       icf(4) ..             AKG2 Hamburg catalog match flag         (17)
#44       icf(5) ..             Zone Astrog. catalog match flag         (17)
#45       icf(6) ..             Black Birch  catalog match flag         (17)
#46       icf(7) ..             Lick Astrog. catalog match flag         (17)
#47       icf(8) ..             NPM  Lick    catalog match flag         (17)
#48       icf(9) ..             SPM  YSJ1    catalog match flag         (17)
#    Note (16, 17) binary data:  a single 4-byte integer is used to store
#    the 10 flags of "icf".  That 4-byte integer has the value:
#       icf = icf(1)*10^8 + icf(2)*10^7 + ...  + icf(8)*10 + icf(9)
#
#     Note (16):  The FK6-Hipparcos-Tycho-source-flag has the following meaning:
#            (= icf(1))
#     0 = not a Hip. or Tycho star
#     1 = Hipparcos 1997 version main catalog (not in UCAC4 data files)
#     2 = Hipparcos double star annex
#     3 = Tycho-2
#     4 = Tycho annex 1
#     5 = Tycho annex 2
#     6 = FK6 position and proper motion (instead of Hipparcos data)
#     7 = Hippparcos 2007 solution position and proper motion
#     8 = FK6      only PM substit. (not in UCAC4 data)
#     9 = Hipparcos 2007, only proper motion substituted
#
#     Note (17):  The catflg match flag is provided for major catalogs used
#     in the computation of the proper motions.  Each match is analyzed
#     for multiple matches of entries of the 1st catalog to 2nd catalog 
#     entries, and the other way around.  Matches are also classified
#     by separation and difference in magnitude to arrive at a confidence
#     level group.  The flag has the following meaning: 
#
#     0 = star not matched with this catalog
#     1 = unique-unique match,  not involving a double star
#     2 =  ... same, but involving a flagged double star
#     3 = multiple match but unique in high confidence level group, no double
#     4 =  ... same, but involving a flagged double star
#     5 = closest match, not involving a double, likely o.k. 
#     6 =  ... same, but involving a flagged double star
#     7 = maybe o.k. smallest sep. match in both directions, no double
#     8 =  ... same, but involving a flagged double star
            star.cat_match = [int(x) for x in str(parsed[39])]

            ###################
            # UCAC4 UNIQUE ID #
            ###################
            
#51 69-72 rnm    I*4            unique star identification number       (20)
#    Note (20):  This unique star identification number is between 200001
#    and  321640 for Hipparcos stars, and between 1 and 9430 for non-
#    Hipparcos stars supplemented to the UCAC4 catalog (no CCD observ.).
#    For all other stars this unique star identification number is the
#    internal mean-position-file (MPOS) number + 1 million.
#    For both the Hipparcos and the supplement stars there is an entry
#    on the u4supl.dat file providing more information, including the
#    original Hipparcos star number.  Note, there are several thousand
#    cases where different UCAC4 stars link to the same Hipparcos star
#    number due to resolved binary stars with each component being a 
#    separate star entry in UCAC4.
            star.unique_number = parsed[42]

            # SKIP UCAC2 REFERENCES
#52 73-74 zn2    I*2            zone number of UCAC2 (0 = no match)     (21)
#53 75-78 rn2    I*4            running record number along UCAC2 zone  (21)

            ###################################
            # SET UCAC4 IDENTIFICATION NUMBER #
            ###################################
            
            star.id_str = 'UCAC4-%03d-%06d' % (znum, rec_num)
            
            if self.debug:
                print 'OK!'
            yield star            

#############################################################################

    def _zone_filename(self, znum):
        """Convert a UCAC4 zone number to an absolute pathspec."""
        fn = os.path.join(self.dirname, 'u4b', 'z%03d'%znum)
        return fn

    def _find_starting_ra(self, fp, ra_min):
        """Efficiently find the first record >= RA_MIN."""
        if ra_min <= 0.:
            # No point in searching!
            return 0
        
        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()
        num_rec = file_size // UCAC4_RECORD_SIZE
        assert num_rec * UCAC4_RECORD_SIZE == file_size

        lo = 0
        hi = num_rec-1
        while lo <= hi:
            mid = (lo+hi)//2
            fp.seek(mid*UCAC4_RECORD_SIZE, os.SEEK_SET)
            record = fp.read(UCAC4_RECORD_SIZE_RA)
            parsed = struct.unpack(UCAC4_FMT_RA, record)
            midval = parsed[0] * MAS_TO_DEG
            if midval < ra_min:
                lo = mid+1
            elif midval > ra_min: 
                hi = mid-1
            else:
                # Exact match!
                fp.seek(mid*UCAC4_RECORD_SIZE, os.SEEK_SET)
                return mid // UCAC4_RECORD_SIZE

        # At this point lo is the best we can do
        fp.seek(lo*UCAC4_RECORD_SIZE, os.SEEK_SET)
        return lo // UCAC4_RECORD_SIZE


#===============================================================================
# UNIT TESTS 
#===============================================================================

import unittest

class Test_UCAC4StarCatalog(unittest.TestCase):

    def runTest(self):
        cat = UCAC4StarCatalog('t:/external/ucac4')

        # Zone 1
        num_pm = cat.count_stars(require_clean=False, allow_double=True,
                                 allow_galaxy=True, require_pm=True,
                                 dec_min=-90*RPD, dec_max=-89.8*RPD)
        num_all = cat.count_stars(require_clean=False, allow_double=True,
                                  allow_galaxy=True, require_pm=False,
                                  dec_min=-90*RPD, dec_max=-89.8*RPD)
        self.assertEqual(num_all, 206)
        self.assertEqual(num_all-num_pm, 5)

        # Zone 451
        num_pm = cat.count_stars(require_clean=False, allow_double=True,
                                 allow_galaxy=True, require_pm=True,
                                 dec_min=0., dec_max=0.2*RPD)
        num_all = cat.count_stars(require_clean=False, allow_double=True,
                                  allow_galaxy=True, require_pm=False,
                                  dec_min=0., dec_max=0.2*RPD)
        self.assertEqual(num_all, 133410)
        self.assertEqual(num_all-num_pm, 6509) # zone_stats says 6394??

        # Zone 900
        num_pm = cat.count_stars(require_clean=False, allow_double=True,
                                 allow_galaxy=True, require_pm=True,
                                 dec_min=89.8*RPD, dec_max=90.*RPD)
        num_all = cat.count_stars(require_clean=False, allow_double=True,
                                  allow_galaxy=True, require_pm=False,
                                  dec_min=89.8*RPD, dec_max=90.*RPD)
        self.assertEqual(num_all, 171)
        self.assertEqual(num_all-num_pm, 10) # zone_stats says 9??

        # Compare slicing directions
        num_dec = 0
        for idec in xrange(20):
            num_dec += cat.count_stars(dec_min=0.2*idec*RPD,
                                       dec_max=0.2*(idec+1)*RPD,
                                       ra_min=60*RPD, ra_max=70*RPD)
        num_ra = 0
        for ira in xrange(10):
            num_ra += cat.count_stars(dec_min=0., dec_max=4.*RPD,
                                      ra_min=(ira+60)*RPD, ra_max=((ira+1)+60)*RPD)
        self.assertEqual(num_dec, num_ra)
        
        # Compare optimized RA search with non-optimized
        for dec_idx in xrange(10):
            dec_min = (dec_idx*10-90.)*RPD
            dec_max = (dec_idx*10-89.8)*RPD
            for ra_min_idx in xrange(10):
                ra_min = ra_min_idx * 10 * RPD
                ra_max = ra_min + 10*RPD
                num_opt = cat.count_stars(dec_min=dec_min, dec_max=dec_max,
                                          ra_min=ra_min, ra_max=ra_max)
                num_no_opt = cat.count_stars(dec_min=dec_min, dec_max=dec_max,
                                             ra_min=ra_min, ra_max=ra_max,
                                             optimize_ra=False)
                self.assertEqual(num_opt, num_no_opt)
            
if __name__ == '__main__':
    unittest.main(verbosity=2)
    