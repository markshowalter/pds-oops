from cb_config import *
from starcat import UCAC4StarCatalog, YBSCStarCatalog

ucac4_cat = UCAC4StarCatalog(os.path.join(STAR_CATALOG_ROOT, 'UCAC4'))
ybsc_cat = YBSCStarCatalog(os.path.join(STAR_CATALOG_ROOT, 'YBSC'))

ybsc_stars = list(ybsc_cat.find_stars())
ybsc_stars.sort(key=lambda x:-x.vmag)
DEL = 1e-4

for star in ybsc_stars:
    if star.vmag < 7.1:
        break
    ucac4_star_list = list(ucac4_cat.find_stars(ra_min=star.ra-DEL,
                                                ra_max=star.ra+DEL,
                                                dec_min=star.dec-DEL,
                                                dec_max=star.dec+DEL,
                                                return_everything=True))
    if len(ucac4_star_list) == 0:
        continue
    
    first = True
    for ustar in ucac4_star_list:
        if abs(star.vmag-ustar.vmag) > 0.5:
            continue
        if first:
            print '*' * 80
            print '************** YBSC:'
            print star
            print '************** UCAC4'
            first = False
        print ustar
