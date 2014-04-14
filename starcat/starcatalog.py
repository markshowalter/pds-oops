################################################################################
# starcat/starcatalog.py
################################################################################

import numpy as np

DPR = 180/np.pi
RPD = np.pi/180
AS_TO_DEG = 1/3600.
MAS_TO_DEG = AS_TO_DEG / 1000.
TWOPI = 2*np.pi
HALFPI = np.pi/2

class Star(object):
    """A holder for star attributes.
    
    This is the base class that defines attributes common to all
    star catalogs."""
    
    def __init__(self):
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
        """Proper motion in RA (radians/yr)"""
        
        self.pm_ra_sigma = None
        """Proper motion in RA error (radians/yr)"""
        
        self.pm_dec = None
        """Proper motion in DEC (radians/yr)"""
        
        self.pm_dec_sigma = None
        """Proper motion in DEC error (radians/yr)"""
        
        self.unique_number = None
        """Unique catalog number"""

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
        
