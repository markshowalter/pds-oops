################################################################################
# oops_/frame/frame_.py: Abstract class Frame and its required subclasses
#
# 2/13/12 Modified (MRS) - implemented and tested QuickFrame.
# 3/1/12 Modified (MRS) - revised and implified connect() and connect_to();
#   added quick parameter dictionary and config file.
################################################################################

import numpy as np
import scipy.interpolate as interp

import oops_.registry as registry
from oops_.array.all import *
from oops_.config import QUICK, QUICK_DICTIONARY, LOGGING
from oops_.transform import Transform

class Frame(object):
    """A Frame is an abstract class that returns a Transform (rotation matrix
    and spin vector) given a Scalar time. A Transform converts from a reference
    coordinate frame to a target coordinate frame, either of which could be
    non-inertial. All coordinate frame are ultimately referenced to J2000.

    Every frame must have these attributes:
        frame_id        the ID of this frame, either a string or an integer.
        reference_id    the ID of the reference frame, either a string or an
                        integer. A frame rotates coordinates from the reference
                        frame into the given frame.
        origin_id       the ID of the origin relative to which the frame is
                        defined; None if the frame is inertial.
        shape           the shape of the Frame object, as a list of dimensions.
                        This is the shape of the Transform object returned by
                        frame.transform_at_time() for a single value of time.

    The primary definition of a frame will be assigned this attribute by the
    registry:

    ancestry            a list of Frame objects beginning with this frame and
                        ending with with J2000, where each Frame in sequence is
                        the reference of the previous frame:
                            self.ancestry[0] = self.
                            self.ancestry[1] = reference frame of self.
                            ...
                            self.ancestry[-1] = J2000.
                        Note that an immediate ancestor must either be an
                        inertial frame, or else use the same origin.

    wrt_j2000           a definition of the same frame relative to the J2000
                        coordinate frame.
    """

########################################
# Each subclass must override...
########################################

    def __init__(self):
        """Constructor for a Frame object."""

        pass

    def transform_at_time(self, time, quick=QUICK):
        """Returns a Transform object that rotates coordinates from the
        reference frame into this frame as a function of time. If the frame is
        rotating, then the coordinates must be given relative to the center of
        rotation.

        Input:
            time    a Scalar time.
            quick   True to consider using a QuickFrame, where warranted, to
                    improve computation speed.

        Return:     the corresponding Tranform applicable at the specified
                    time(s). The transform rotates vectors from the reference
                    frame to this frame.

        Note that the time and the Frame object are not required to have the
        same shape; standard rules of broadcasting apply.
        """

        pass

    # string operations
    def __str__(self):
        return "Frame(" + self.frame_id + "/" + self.reference_id + ")"

    def __repr__(self): return self.__str__()

################################################################################
# Registry Management
################################################################################

    # The global registry is keyed in two ways. If the key is a string, then
    # this constitutes the primary definition of the frame. The reference_id
    # of the frame must already be in the registry, and the new frame's ID
    # cannot be in the registry.

    # We also create secondary definitions of a frame, where it is defined
    # relative to a different reference frame. These are entered into the
    # registry keyed by a tuple: (frame_id, reference_id). This saves the
    # effort of re-creating a frame used repeatedly.

    @staticmethod
    def initialize_registry():
        """Initializes the registry. It is not generally necessary to call this
        function, but it can be used to reset the registry for purposes of
        debugging."""

        if registry.J2000 is None:
            registry.J2000 = NullFrame("J2000")
            registry.J2000.ancestry = [registry.J2000]

        registry.FRAME_REGISTRY = {"J2000": registry.J2000,
                                   ("J2000","J2000"): registry.J2000}

    def register(self, shortcut=None):
        """Registers a Frame definition. If the frame's ID is new, it is assumed
        to be the primary definition and is keyed by the ID alone. However, a
        primary definition must use a reference ID that is already registered.

        Otherwise or in addition, a secondary key is added to the registry if it
        is not already present:
            (frame_id, reference_id)
        This key also points to the same Frame object.

        If a shortcut name is given, then self is treated as a shortcut
        definition. The frame is registered under the shortcut name and also
        under the tuple (frame_id, reference_id), but other registered
        definitions of the path are not modified.
        """

        # Make sure the reference frame is registered; raise KeyError on failure
        reference = registry.FRAME_REGISTRY[self.reference_id]

        # Make sure the origins are compatible
#         if (reference.origin_id is not None and
#             reference.origin_id != self.origin_id):
#                 raise ValueError("cannot define frame " +
#                                  self.frame_id +
#                                  " relative to non-inertial frame " +
#                                  reference.frame_id)

        # Handle a shortcut
        if shortcut is not None:
            registry.FRAME_REGISTRY[shortcut] = self

            key = (self.frame_id, self.reference_id)
            registry.FRAME_REGISTRY[key] = self
            return

        # If the ID is unregistered, insert this as the primary definition
        key = self.frame_id
        if key not in registry.FRAME_REGISTRY.keys():
            registry.FRAME_REGISTRY[key] = self

            # Fill in the ancestry too
            self.ancestry = [self] + reference.ancestry

            # Also save the NullFrame Frame
            key = (self.frame_id, self.frame_id)
            registry.FRAME_REGISTRY[key] = NullFrame(self.frame_id)

            # Also define the path with respect to J2000 if possible
            if self.reference_id != "J2000":
              try:
                self.wrt_j2000 = self.connect_to("J2000")

                key = (wrt_j2000.frame_id, "J2000")
                registry.FRAME_REGISTRY[key] = self.wrt_j2000
              except: pass

        # Insert or replace the secondary definition for
        # (self.frame_id, self.reference_id) as a secondary definition
        key = (self.frame_id, self.reference_id)
        registry.FRAME_REGISTRY[key] = self

    def unregister(self):
        """Removes this frame from the registry."""

        # Note that we only delete the primary entry and any frame in which this
        # is one of the end points. If the frame is used as an intermediate step
        # between other frames, it will cease to be visible in the dictionary
        # but frames that use it will continue to function unchanged. However,
        # it is safest to remove all usage of a frame at the time it is
        # unregistered.

        frame_id = self.frame_id
        for key in registry.FRAME_REGISTRY.keys():
            if frame_id == key: del registry.FRAME_REGISTRY[key]

            if type(key) == type(()):
                if   frame_id == key[0]: del registry.FRAME_REGISTRY[key]
                elif frame_id == key[1]: del registry.FRAME_REGISTRY[key]

        # Also delete references in the Path registry...
        # Note that we only delete the primary entry and any path in which this
        # is one of the end points. If the path is used as an intermediate step
        # between other paths, it will cease to be visible in the dictionary
        # but paths that use it will continue to function unchange. However, it
        # is safest to remove all usage of a frame at the time it is
        # unregistered.

        for key in registry.PATH_REGISTRY.keys():
            if type(key) == type(()) and len(key) > 2:
                if frame_id == key[2]: del registry.PATH_REGISTRY[key]


    def reregister(self):
        """Adds this frame to the registry, replacing any definition of the same
        name."""

        self.unregister()
        self.register()

################################################################################
# Frame Generator
################################################################################

    @staticmethod
    def connect(target, reference):
        """Returns a Frame object that transforms from the given reference Frame
        to the given target Frame.

        Input:
            target      a Frame object or the registered ID of the destination
                        frame.
            reference   a Frame object or the registered ID of the starting
                        frame.

        The shape of the connected frame will will be the result of broadcasting
        the shapes of the target and reference.
        """

        # Convert to IDs
        target_id    = registry.as_frame_id(target)
        reference_id = registry.as_frame_id(reference)

        # If the frame already exists, just return it
        try:
            key = (target_id, reference_id)
            return registry.FRAME_REGISTRY[key]
        except KeyError: pass

        # Otherwise, construct it by other means
        target_frame = registry.FRAME_REGISTRY[target_id]
        return target_frame.connect_to(reference_id)

    # Can be overridden by some classes such as SpiceFrame, where it is easier
    # to make connections between frames.
    def connect_to(self, reference):
        """Returns a Frame object that transforms from an arbitrary reference
        frame to this frame. It is assumed that the desired frame does not
        already exist in the registry. This is not checked, and need not be
        checked by any methods that override this one.

        Input:
            reference       a reference Frame object or its registered name.
        """

        # Find the fundamental frame definitions
        target    = registry.as_primary_frame(self)
        reference = registry.as_primary_frame(reference)

        # Check for compatibility of rotating frames. The origins must match.
#         if target.origin_id is not None and reference.origin_id is not None:
#             assert target.origin_id == reference.origin_id:

        # If the target is an ancestor of the reference, reverse the direction
        # and try again
        if target in reference.ancestry:
            frame = Frame.connect(reference, target)
            return ReversedFrame(frame)

        # Otherwise, search from the parent frame and then link
        frame = Frame.connect(target.ancestry[1], reference)
        return LinkedFrame(target, frame)

    ############################################################################
    # 2/29/12: No longer needed but might still come in handy some day.
    # Associated unit tests have not been deleted from SpiceFrame.
    ############################################################################

    @staticmethod
    def common_ancestry(frame1, frame2):
        """Returns a pair of ancestry lists for the two given frames, where both
        lists end at Frames with the same name."""

        # Identify the first common ancestor of both frames
        for i in range(len(frame1.ancestry)):
            id1 = frame1.ancestry[i].frame_id

            for j in range(len(frame2.ancestry)):
                id2 = frame2.ancestry[j].frame_id

                if id1 == id2:
                    return (frame1.ancestry[:i+1], frame2.ancestry[:j+1])

        return (frame1.ancestry, frame2.ancestry)       # should never happen

    @staticmethod
    def str_ancestry(tuple):        # For debugging
        list = ["(["]

        for item in tuple:
            for frame in item:
                list += [frame.frame_id,"\", \""]

            list.pop()
            list += ["\"], [\""]

        list.pop()
        list += ["\"])"]

        return "".join(list)

########################################

    def quick_frame(self, time, quick=QUICK):
        """Returns a new QuickFrame object that provides accurate approximations
        to the transform returned by this frame. It can speed up performance
        substantially when the same frame must be evaluated repeatedly but
        within a narrow range of times.

        Input:
            time        a Scalar defining the set of times at which the frame is
                        to be evaluated.
            quick       if False, no QuickFrame is created and self is returned;
                        if True, the default dictionary QUICK_DICTIONARY is
                        used; if another dictionary, then the values provided
                        override the defaults and the merged dictionary is used.
        """

        OVERHEAD = 500      # Assume it takes the equivalent time of this many
                            # evaluations just to set up the QuickFrame.
        SPEEDUP = 5.        # Assume that evaluations are this much faster once
                            # the QuickFrame is set up.
        SAVINGS = 0.2       # Require at least a 20% savings in evaluation time.

        # Make sure a QuickFrame has been requested
        if quick is False: return self

        # A NullFrame frame is too easy
        if type(self) == NullFrame: return self

        # A QuickFrame would be redundant
        if type(self) == QuickFrame: return self

        # Obtain the local QuickFrame dictionary
        quickdict = QUICK_DICTIONARY
        if type(quick) == type({}):
            quickdict = dict(quickdict, **quick)

        if not quickdict["use_quickframes"]: return self

        # Determine the time interval and steps
        time = Scalar.as_scalar(time)
        vals = time.vals

        count = np.size(vals)
        if count < OVERHEAD: return self

        extension = quickdict["frame_time_extension"]
        dt = quickdict["frame_time_step"]
        extras = quickdict["frame_extra_steps"]

        tmin = np.min(vals) - extension
        tmax = np.max(vals) + extension
        steps = (tmax - tmin)/dt + 2*extras

        # Evaluate the effort using a QuickFrame compared to the effort without
        effort_using_quickframe = OVERHEAD + steps + count/SPEEDUP
        if count < (1. + SAVINGS) * effort_using_quickframe:
            return self

        if LOGGING.quickframe_creation:
            print "New QuickFrame: " + str(self), (tmin, tmax)

        return QuickFrame(self, (tmin, tmax), quickdict)

################################################################################
# Required Subclasses
################################################################################

class ReversedFrame(Frame):
    """A Frame that generates the inverse Transform of a given Frame.
    """

    def __init__(self, frame):

        self.oldframe = registry.as_frame(frame)

        # Required fields
        self.frame_id     = self.oldframe.reference_id
        self.reference_id = self.oldframe.frame_id
        self.origin_id    = self.oldframe.origin_id

        self.shape = self.oldframe.shape

    def transform_at_time(self, time, quick=QUICK):

        return self.oldframe.transform_at_time(time).invert()

################################################################################

class LinkedFrame(Frame):
    """LinkedFrame is a Frame subclass that returns the result of applying one
    frame's transform to another. The new frame describes coordinates in one
    frame relative to the reference of the second frame.
    """

    def __init__(self, frame, parent):
        """Constructor for a LinkedFrame.

        Input:
            frame       a frame, which must be define relative to the given
                        parent.
            parent      a frame to which the above will be linked.
        """

        self.frame  = registry.as_frame(frame)
        self.parent = registry.as_frame(parent)

        assert self.frame.reference_id == self.parent.frame_id

        # Required fields
        self.frame_id     = self.frame.frame_id
        self.reference_id = self.parent.reference_id
        self.shape = Array.broadcast_shape((self.frame, self.parent))

        if self.frame.origin_id is None:
            self.origin_id = self.parent.origin_id
        elif self.parent.origin_id is None:
            self.origin_id = self.frame.origin_id
        else:
            self.origin_id = self.parent.origin_id
            # assert self.frame.origin_id == self.parent.origin_id

    def transform_at_time(self, time, quick=QUICK):

        parent = self.parent.transform_at_time(time)
        transform = self.frame.transform_at_time(time)
        return transform.rotate_transform(parent)

        return transform

################################################################################

class NullFrame(Frame):
    """A Frame that always returns a null transform."""

    def __init__(self, frame_id, origin_id=None):

        # Required fields
        self.frame_id     = frame_id
        self.reference_id = frame_id

        if self.reference_id == "J2000":
            self.origin_id = None
        else:
            reference = registry.FRAME_REGISTRY[self.reference_id]
            self.origin_id = reference.origin_id

        self.shape = []

    def transform_at_time(self, time, quick=QUICK):

        return Transform.null_transform(self.frame_id, self.origin_id)

    def __str__(self): return "Frame(" + self.frame_id + ")"

################################################################################

class RelativeFrame(Frame):
    """A Frame that returns the one frame times the inverse of another. The
    two frames must have a common reference. The combined frame converts
    coordinates from the second frame to the first."""

    def __init__(self, frame1, frame2):

        self.frame1 = frame1
        self.frame2 = frame2

        assert self.frame1.reference_id == self.frame2.reference_id

        # Required fields
        self.frame_id     = self.frame1.frame_id
        self.reference_id = self.frame2.frame_id

        # Identify the origin; confirm compatibility
        self.origin_id = frame1.origin_id
        if self.origin_id is None:
            self.origin_id = frame2.origin_id
        elif frame2.origin_id is not None:
            self.origin_id = frame2.origin_id
            # assert frame2.origin_id == self.origin_id

        self.shape = Array.broadcast_shape((self.frame1, self.frame2))

    def transform_at_time(self, time, quick={}):

        return self.frame1.transform_at_time(time).rotate_transform(
               self.frame2.transform_at_time(time).invert())

################################################################################

class QuickFrame(Frame):
    """QuickFrame is a Frame subclass that returns Transform objects based on
    interpolation of another Frame within a specified time window."""

    def __init__(self, frame, interval, quickdict):
        """Constructor for a QuickFrame.

        Input:
            frame           the Frame object that this Frame will emulate.
            interval        a tuple containing the start and stop times to use,
                            in TDB seconds.
            quickdict       a dictionary containing all the QuickFrame
                            parameters.
        """

        self.frame = frame
        self.frame_id     = frame.frame_id
        self.reference_id = frame.reference_id
        self.origin_id    = frame.origin_id

        assert frame.shape == []
        self.shape = []

        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = quickdict["frame_time_step"]

        extras = quickdict["frame_extra_steps"]
        self.times = np.arange(self.t0 - extras * self.dt,
                               self.t1 + extras * self.dt + self.dt,
                               self.dt)
        self.t0 = self.times[0]
        self.t1 = self.times[-1]

        self.transforms = self.frame.transform_at_time(self.times)

        # This would be faster with quaternions but I'm lazy
        KIND = 3
        self.matrix = np.empty((3,3), dtype="object")
        for i in range(3):
          for j in range(3):
            self.matrix[i,j] = interp.UnivariateSpline(self.times,
                                    self.transforms.matrix.vals[...,i,j],
                                    k=KIND)

        # Don't interpolate omega if frame is inertial
        if self.transforms.omega == Vector3((0.,0.,0.)):
            self.omega = None
        else:
            self.omega = np.empty((3,), dtype="object")
            for i in range(3):
                self.omega[i] = interp.UnivariateSpline(self.times,
                                        self.transforms.omega.vals[...,i],
                                        k=KIND)

        # Test the precision
        precision_self_check = quickdict["frame_self_check"]
        if precision_self_check is not None:
            t = self.times[:-1] + self.dt/2.        # Halfway points

            true_transform = self.frame.transform_at_time(t)

            matrix = np.empty(t.shape + (3,3))
            omega  = np.zeros(t.shape + (3,))

            # Evaluate the matrix and rotation vector
            matrix[...,0,0] = self.matrix[0,0](t)
            matrix[...,0,1] = self.matrix[0,1](t)
            matrix[...,0,2] = self.matrix[0,2](t)
            matrix[...,1,0] = self.matrix[1,0](t)
            matrix[...,1,1] = self.matrix[1,1](t)
            matrix[...,1,2] = self.matrix[1,2](t)
            matrix[...,2,0] = self.matrix[2,0](t)
            matrix[...,2,1] = self.matrix[2,1](t)
            matrix[...,2,2] = self.matrix[2,2](t)

            if self.omega is not None:
                omega[...,0] = self.omega[0](t)
                omega[...,1] = self.omega[1](t)
                omega[...,2] = self.omega[2](t)

            # Normalize the matrix
            matrix[...,0,:] = utils.ucross3d(matrix[...,1,:], matrix[...,2,:])
            matrix[...,1,:] = utils.ucross3d(matrix[...,2,:], matrix[...,0,:])
            matrix[...,2,:] = utils.unit(matrix[...,2,:])

            dmatrix = abs(true_transform.matrix - matrix)

            domega = abs(true_transform.omega - omega)
            if abs(true_transform.omega) != 0.:
                domega /= abs(true_transform.omega)

            error = max(np.max(dmatrix.vals), np.max(domega.vals))
            if error > precision_self_check:
                raise ValueError("precision tolerance not achieved: " +
                                  str(error) + " > " +
                                  str(precision_self_check))

    def transform_at_time(self, time, quick=QUICK):

        # time can only be a 1-D array
        time   = Scalar.as_scalar(time)
        matrix = np.empty((np.size(time.vals), 3, 3))
        omega  = np.zeros((np.size(time.vals), 3))

        # Evaluate the matrix and rotation vector
        t = np.ravel(time.vals)

        matrix[...,0,0] = self.matrix[0,0](t)
        matrix[...,0,1] = self.matrix[0,1](t)
        matrix[...,0,2] = self.matrix[0,2](t)
        matrix[...,1,0] = self.matrix[1,0](t)
        matrix[...,1,1] = self.matrix[1,1](t)
        matrix[...,1,2] = self.matrix[1,2](t)
        matrix[...,2,0] = self.matrix[2,0](t)
        matrix[...,2,1] = self.matrix[2,1](t)
        matrix[...,2,2] = self.matrix[2,2](t)

        if self.omega is not None:
            omega[...,0] = self.omega[0](t)
            omega[...,1] = self.omega[1](t)
            omega[...,2] = self.omega[2](t)

        # Normalize the matrix
        matrix[...,0,:] = utils.ucross3d(matrix[...,1,:], matrix[...,2,:])
        matrix[...,1,:] = utils.ucross3d(matrix[...,2,:], matrix[...,0,:])
        matrix[...,2,:] = utils.unit(matrix[...,2,:])

        # Return the Event
        return Transform(matrix.reshape(np.shape(time.vals) + (3,3)),
                         omega.reshape(np.shape(time.vals) + (3,)),
                         self.frame_id, self.reference_id, self.origin_id)

    def __str__(self):
        return "QuickFrame(" + self.frame_id + "/" + self.reference_id + ")"

################################################################################
# Initialize the registry
################################################################################

registry.FRAME_CLASS = Frame
Frame.initialize_registry()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Frame(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        from spiceframe import SpiceFrame
        from oops_.path.spicepath import SpicePath

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

        # QuickFrame tests
        ignore = SpicePath("EARTH", "SSB")
        ignore = SpicePath("MOON", "SSB")
        ignore = SpiceFrame("IAU_EARTH", "J2000")
        moon  = SpiceFrame("IAU_MOON", "IAU_EARTH")
        quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK_DICTIONARY, **{"frame_self_check":3.e-14}))

        # Perfect precision is impossible
        try:
            quick = QuickFrame(moon, (-5.,5.),
                        dict(QUICK_DICTIONARY, **{"frame_self_check":0.}))
            self.assertTrue(False, "No ValueError raised for PRECISION = 0.")
        except ValueError: pass

        # Timing tests...
        test = np.zeros(200000)
        # ignore = moon.transform_at_time(test, quick=False)  # takes about 10 sec
        ignore = quick.transform_at_time(test) # takes way less than 1 sec

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################