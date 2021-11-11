################################################################################
# oops/obs_/insitu.py: Subclass InSitu of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation  import Observation
from oops.path_.path        import Path
from oops.frame_.frame      import Frame
from oops.cadence_.cadence  import Cadence
from oops.cadence_.instant  import Instant
from oops.fov_.nullfov      import NullFOV

class InSitu(Observation):
    """InSitu is a subclass of Observation that has timing and path information,
    but no attributes related to pointing or incoming photon direction. It can
    be useful for describing in situ measurements.

    It can also be used to obtain information from gridless backplanes, which
    do not require directional information.
    """

    PACKRAT_ARGS = ['cadence', 'path']

    def __init__(self, cadence, path):
        """Constructor for an InSitu observation.

        Input:
            cadence     a Cadence object defining the time and duration of
                        each "measurement". As a special case, a Scalar value
                        is converted to a Cadence of subclass Instant.
            path        the path waypoint co-located with the observer.
        """

        if isinstance(cadence, Cadence):
            self.cadence = cadence
        else:
            self.cadence = Instant(cadence)

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.fov = NullFOV()
        self.uv_shape = (1,1)
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False

        self.t_axis = -1
        self.shape = self.cadence.shape

        self.path = Path.as_waypoint(path)
        self.frame = Frame.J2000

        self.subfields = {}

        return

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_InSitu(unittest.TestCase):

    def runTest(self):

        # No tests here - this is just an abstract superclass

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
