################################################################################
# oops/config.py: General configuration parameters
################################################################################

################################################################################
# QuickPath and QuickFrame default parameters
#
# Disable the use of QuickPaths/Frames on an individual basis by calling the
# function with quick=False. The default set of parameters will be used whenever
# quick=True. If a function is called with quick as a dictionary, then any
# values in the dictionary override these defaults and the merged dictionary of
# parameters is used.
################################################################################

class QUICK(object):
  flag = True                   # Defines the default behavior as quick=True or
                                # quick=False.

  dictionary = {
    "use_quickpaths": True,
    "path_time_step": 0.05,     # time step in seconds.
    "path_time_extension": 5.,  # secs by which to extend interval at each end.
    "path_self_check": None,    # fractional precision for self-testing.
    "path_extra_steps": 4,      # number of extra time steps at each end.
    "quickpath_cache": 40,      # maximum number of non-overlapping quickpaths
                                # to cache for any given path.
    "quickpath_linear_interpolation_threshold": 3.,
                                # if a time span is less than this amount,
                                # perform linear interpolation instead of
                                # using InterpolatedUnivariateSpline; this
                                # improves performance

    "use_quickframes": True,
    "frame_time_step": 0.05,    # time interval in seconds.
    "frame_time_extension": 5., # secs by which to extend interval at each end.
    "frame_self_check": None,   # fractional precision for self-testing.
    "frame_extra_steps": 4,     # number of extra time steps at each end.
    "quickframe_cache": 40,     # maximum number of non-overlapping quickframes
                                # to cache for any given frame.
    "quickframe_linear_interpolation_threshold": 1.,
                                # if a time span is less than this amount,
                                # perform linear interpolation instead of
                                # using InterpolatedUnivariateSpline; this
                                # improves performance
    "quickframe_numerical_omega": False,
                                # True to derive the omega rotation vectors
                                # via numerical derivatives rather than via
                                # interpolation of the vector components.
    "ignore_quickframe_omega": False,
                                # True to derive the omega rotation vectors
                                # via numerical derivatives rather than via
                                # interpolation of the vector components.
    "use_superseded_quickframes": False,
                                # True to use the old frame interpolation method
                                # involving matrix components rather than
                                # quaternions. DEPRECATED by perserved for
                                # testing purposes.
}

################################################################################
# Photon solver parameters
################################################################################

# For Path._solve_photon()

class PATH_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-6       # Iterations stops when every change in light
                                # travel time from one iteration to the next
                                # drops below this threshold.
    dlt_limit = 10.             # The allowed range of variations in light
                                # travel time before they are truncated. This
                                # should be related to the physical scale of
                                # the system being studied.

# For Surface._solve_photon_by_los()

class SURFACE_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-6       # See PATH_PHOTONS for more info.
    dlt_limit = 10.             # See PATH_PHOTONS for more info.
    collapse_threshold = 3.     # When a surface intercept consists of a range
                                # of times smaller than this threshold, the
                                # times are converted to a single value.
                                # This approximation can speed up some
                                # calculations substantially.

################################################################################
# Event precision
################################################################################

class EVENT_CONFIG(object):
    collapse_threshold = 3.     # When an event returned by a calculation spans
                                # a range of times smaller than this threshold,
                                # the time field is converted to a single value.
                                # This approximation can speed up some
                                # calculations substantially.

################################################################################
# Logging and Monitoring
################################################################################

class LOGGING(object):
    prefix = ""                     # Prefix in front of a log message
    quickpath_creation = False      # Log the creation of QuickPaths.
    quickframe_creation = False     # Log the creation of QuickFrames.
    path_iterations = False         # Log iterations of Path._solve_photons().
    surface_iterations = False      # Log iterations of Surface._solve_photons()
    event_time_collapse = False     # Report event time collapse
    surface_time_collapse = False   # Report surface time collapse

    @staticmethod
    def all(flag):
        LOGGING.quickpath_creation = flag
        LOGGING.quickframe_creation = flag
        LOGGING.path_iterations = flag
        LOGGING.surface_iterations = flag
        LOGGING.event_time_collapse = flag
        LOGGING.surface_time_collapse = flag

    @staticmethod
    def off(): LOGGING.all(False)

    @staticmethod
    def on(prefix=""):
        LOGGING.all(True)
        LOGGING.prefix = prefix

################################################################################
# Aberration method (for backward compatibility)
################################################################################

class ABERRATION(object):
    old = False                 # Change to True for previous, incorrect
                                # interpretation of the C matrices.

################################################################################

