from cb_logging import *
import logging

import matplotlib.pyplot as plt

import oops.inst.nh.lorri as lorri
import oops

from cb_config import *
from cb_stars import *

log_set_stars_level(level=logging.DEBUG)

stars_config = {
    # True if data is already calibrated as I/F and needs to be converted back
    # to raw DN.
    'calibrated_data': False,

    # Allow non-integer offsets; these use astrometry to refine the mean
    # star offset.
    'allow_fractional_offsets': True,
    
    # The order of multipliers to use to gradually expand the search area.
    'search_multipliers': [1.],

    # Maximum number of stars to use.
    'max_stars': 30,
    
    # Verify offset with photometry?
    'perform_photometry': False,
    
    # Minimum number of stars that must photometrically match for an offset
    # to be considered acceptable and the corresponding confidence.
    # Also the minimum number of stars that must match to give a confidence 
    # of 1.0.
    'min_stars_low_confidence': (3, 0.75),
    'min_stars_high_confidence': (6, 1.0),

    # The minimum photometry confidence allowed for a star to be considered
    # valid.
    'min_confidence': 0.9,

    # Minimum PSF size for modeling a star (must be odd). The PSF is square.
    # This will be added to the smearing in each dimension to create a final
    # possibly-rectangular PSF.
    'min_psf_size': 9,
    
    # The maximum number of steps to use when smearing a PSF. This is really 
    # only a suggestion, as the number will be clipped at either extreme to
    # guarantee a good smear.
    'max_movement_steps': 50,
    
    # The default star class when none is available in the star catalog.
    'default_star_class': 'G0',
    
    # The minimum DN that is guaranteed to be visible in the image.
    'min_brightness_guaranteed_vis': 200.,

    # The minimum DN count for a star to be detectable. These values are pretty
    # aggressively dim - there's no guarantee a star with this brightness can
    # actually be seen. But there's no point in looking at a star any dimmer.
    ('min_detectable_dn', 'LORRI'): 0,

    # The range of vmags to use when determining the dimmest star visible.
    'min_vmag': 5.,
    'max_vmag': 15.,
    'vmag_increment': 0.5,

    # The size of the box to analyze vs. the predicted integrated DN.
    # XXX These numbers are just made up, but seem to work.
    # More study should be done.
    'photometry_boxsize_1': (500,11),
    'photometry_boxsize_2': (100,9),
    'photometry_boxsize_default': 7,
    
    # How far (in pixels) a star has to be from a major body before it is no
    # longer considered to conflict.
    'star_body_conflict_margin': 3,
    
    # If star navigation fails, get rid of the brightest star if it's at least
    # this bright. None means don't apply this test.
    'too_bright_dn': 1000,
    
    # If star navigation fails, get rid of the brightest star if it's at least
    # this much brighter (factor) than the next dimmest star. None means don't
    # apply this test.
    'too_bright_factor': None
}

obs = lorri.from_file('j:/Temp/LOR_0034676524_0X630_SCI_1.FIT')

stars_metadata = stars_find_offset(obs, extend_fov=(0,0), stars_config=stars_config)

offset = stars_metadata['offset']

obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
bp = oops.Backplane(obs)

radii = bp.ring_radius('jupiter:ring').vals
plt.imshow(radii)
plt.show()

jrings = bp.border_atop('jupiter:ring', oops.JUPITER_MAIN_RING_LIMIT).vals.astype('float')

plt.imshow(jrings)
plt.show()
