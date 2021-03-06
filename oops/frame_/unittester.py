################################################################################
# oops/frame_/unittester.py
################################################################################

import unittest

from oops.frame_.frame           import Test_Frame
from oops.frame_.cmatrix         import Test_Cmatrix
from oops.frame_.inclinedframe   import Test_InclinedFrame
from oops.frame_.laplaceframe    import Test_LaplaceFrame
from oops.frame_.navigation      import Test_Navigation
from oops.frame_.poleframe       import Test_PoleFrame
from oops.frame_.postarg         import Test_PosTarg
from oops.frame_.ringframe       import Test_RingFrame
from oops.frame_.rotation        import Test_Rotation
from oops.frame_.spiceframe      import Test_SpiceFrame
from oops.frame_.spicetype1frame import Test_SpiceType1Frame
from oops.frame_.spinframe       import Test_SpinFrame
from oops.frame_.synchronous     import Test_Synchronous
from oops.frame_.tracker         import Test_Tracker
from oops.frame_.twovector       import Test_TwoVector

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
