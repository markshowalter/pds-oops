################################################################################
# starcat/starcatalog.py
################################################################################

import numpy as np

DPR = 180/np.pi
RPD = np.pi/180
AS_TO_DEG = 1/3600.
MAS_TO_DEG = AS_TO_DEG / 1000.
MAS_TO_RAD = MAS_TO_DEG * RPD
YEAR_TO_SEC = 1/365.25/86400.

TWOPI = 2*np.pi
HALFPI = np.pi/2

class Star(object):
    """A holder for star attributes.
    
    This is the base class that defines attributes common to all
    star catalogs."""
    
    def __init__(self):
        """Constructor for Star superclass."""
        
        self.ra = None
        """Right ascension at J2000 epoch (radians)"""
        
        self.ra_sigma = None
        """Right ascension error (radians)"""
        
        self.dec = None
        """Declination at J2000 epoch (radians)"""
        
        self.dec_sigma = None
        """Declination error (radians)"""
        
        self.vmag = None
        """Visual magnitude"""
        
        self.vmag_sigma = None
        """Visual magnitude error"""
        
        self.pm_ra = None
        """Proper motion in RA (radians/sec)"""
        
        self.pm_ra_sigma = None
        """Proper motion in RA error (radians/sec)"""
        
        self.pm_dec = None
        """Proper motion in DEC (radians/sec)"""
        
        self.pm_dec_sigma = None
        """Proper motion in DEC error (radians/sec)"""
        
        self.unique_number = None
        """Unique catalog number"""

    def __str__(self):
        ret = 'UNIQUE ID %d' % (self.unique_number)
        
        ra_deg = self.ra*DPR/15 # In hours
        hh = int(ra_deg)
        mm = int((ra_deg-hh)*60)
        ss = (ra_deg-hh-mm/60.)*3600
        ret += '   RA %02dh%02dm%05.3fs' % (hh,mm,ss)

        if self.ra_sigma is not None:
            ret += ' +/- %.4fs' % (self.ra_sigma*DPR*3600)

        dec_deg = self.dec*DPR # In degrees
        neg = '+'
        if dec_deg < 0.:
            neg = '-'
            dec_deg = -dec_deg
        dd = int(dec_deg)
        mm = int((dec_deg-dd)*60)
        ss = (dec_deg-dd-mm/60.)*3600
        ret += ' DEC %s%03dd%02dm%05.3fs' % (neg,dd,mm,ss)
        
        if self.dec_sigma is not None:
            ret += ' +/- %.4fs' % (self.dec_sigma*DPR*3600)
            
        ret += '\n'
        
        if self.vmag is not None:
            ret += 'VMAG %6.3f ' % (self.vmag)
            if self.vmag_sigma is not None:
                ret += '+/- %6.3f ' % (self.vmag_sigma)
         
        if self.pm_ra is not None:
            ret += 'PM RA %.3f mas/yr ' % (self.pm_ra/MAS_TO_RAD/YEAR_TO_SEC)
            if self.pm_ra_sigma:
                ret += '+/- %.3f ' % (self.pm_ra_sigma/MAS_TO_RAD/YEAR_TO_SEC)

        if self.pm_dec is not None:
            ret += 'PM DEC %.3f mas/yr ' % (self.pm_dec/MAS_TO_RAD/YEAR_TO_SEC)
            if self.pm_dec_sigma:
                ret += '+/- %.3f ' % (self.pm_dec_sigma/MAS_TO_RAD/YEAR_TO_SEC)
        
        ret += '\n'
        
        return ret
                     
    def ra_dec_with_pm(self, tdb):
        """Return the star's RA and DEC adjusted for proper motion.
        
        If no proper motion is available, the original RA and DEC are returned.
        
        Input:
            tdb        time since the J2000 epoch in seconds
        """
        
        if self.pm_ra is None or self.pm_dec is None:
            return (self.ra, self.dec)
        
        return (self.ra + tdb*self.pm_ra, self.dec + tdb*self.pm_dec)
        

class StarCatalog(object):
    def __init__(self):
        pass
    
    def count_stars(self, **kwargs):
        """Count the stars that match the given search criteria."""
        count = 0
        for result in self.find_stars(full_result=False, **kwargs):
            count += 1
        return count
    
    def find_stars(self, **kwargs):
        """Find the stars that match the given search criteria.
        
        Optional arguments:      DEFAULT
            ra_min, ra_max       0, 2PI    RA range in radians
            dec_min, dec_max     -PI, PI   DEC range in radians
            vmag_min, vmag_max     ALL     Magnitude range
        """

        kwargs = kwargs.copy() # Private copy so pop doesn't mutate
        ra_min = np.clip(kwargs.pop('ra_min', 0), 0., TWOPI)
        ra_max = np.clip(kwargs.pop('ra_max', TWOPI), 0., TWOPI)
        dec_min = np.clip(kwargs.pop('dec_min', -HALFPI), -HALFPI, HALFPI)
        dec_max = np.clip(kwargs.pop('dec_max', HALFPI), -HALFPI, HALFPI)
        
        if ra_min > ra_max:
            if dec_min > dec_max:
                # Split into four searches
                for star in self._find_stars(0., ra_max, -HALFPI, dec_max, 
                                             **kwargs):
                    yield star
                for star in self._find_stars(ra_min, TWOPI, -HALFPI, dec_max, 
                                             **kwargs):
                    yield star
                for star in self._find_stars(0., ra_max, dec_min, HALFPI, 
                                             **kwargs):
                    yield star
                for star in self._find_stars(ra_min, TWOPI, dec_min, HALFPI, 
                                             **kwargs):
                    yield star
            else:
                # Split into two searches - RA
                for star in self._find_stars(0., ra_max, dec_min, dec_max, 
                                             **kwargs):
                    yield star
                for star in self._find_stars(ra_min, TWOPI, dec_min, dec_max, 
                                             **kwargs):
                    yield star
        else:
            if dec_min > dec_max:
                # Split into two searches - DEC
                for star in self._find_stars(ra_min, ra_max, -HALFPI, dec_max, 
                                             **kwargs):
                    yield star
                for star in self._find_stars(ra_min, ra_max, dec_min, HALFPI, 
                                             **kwargs):
                    yield star
            else:
                # No need to split at all
                for star in self._find_stars(ra_min, ra_max,
                                             dec_min, dec_max, **kwargs):
                    yield star
                
    def _find_stars(self, **kwargs):
        assert False
        
