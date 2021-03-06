################################################################################
# oops/body.py: Body class
################################################################################

import numpy as np
import os
import numbers

import spicedb
import julian
import cspyce

from polymath import *

from oops.path_.path      import Path
from oops.path_.multipath import MultiPath
from oops.path_.spicepath import SpicePath

from oops.frame_.frame       import Frame, AliasFrame
from oops.frame_.ringframe   import RingFrame
from oops.frame_.poleframe   import PoleFrame
from oops.frame_.spiceframe  import SpiceFrame
from oops.frame_.synchronous import Synchronous
from oops.frame_.twovector   import TwoVector

from oops.surface_.nullsurface import NullSurface
from oops.surface_.ringplane   import RingPlane
from oops.surface_.orbitplane  import OrbitPlane
from oops.surface_.spice_shape import spice_shape

from oops.gravity_.gravity       import Gravity
from oops.gravity_.oblategravity import OblateGravity

import oops.constants     as constants
import oops.spice_support as spice_support

################################################################################
# A list of known changes in SPICE names and IDs
# This also standardizes the SPICE names of provisionally-named bodies.
################################################################################

JUPITER_ALIASES = [
    # Jupiter [new code, old code], [formal name, provisional name]
    [[     55060], [         'S2003_J_2' ]],
    [[     55061], [         'S2003_J_3' ]],
    [[     55062], [         'S2003_J_4' ]],
    [[557, 55063], [         'S2003_J_5' ]],
    [[     55064], [         'S2003_J_9' ]],
    [[     55065], [         'S2003_J_10']],
    [[     55066], [         'S2003_J_12']],
    [[558, 55067], [         'S2003_J_15']],
    [[     55068], [         'S2003_J_16']],
    [[550       ], ['HERSE', 'S2003_J_17']],
    [[555, 55069], [         'S2003_J_18']],
    [[     55070], [         'S2003_J_19']],
    [[     55071], [         'S2003_J_23']],
    [[551, 55072], [         'S2010_J_1' ]],
    [[552, 55073], [         'S2010_J_2' ]],
    [[     55074], [         'S2011_J_1' ]],
    [[556, 55075], [         'S2011_J_2' ]],
    [[554       ], [         'S2016_J_1' ]],
    [[553, 55076], ['DIA',   'S2000_J_11']],
]

SATURN_ALIASES = [
    # Saturn [new code, old code], [formal name, provisional name, NAIF name]
    [[     65035], [              'S2004_S_7' , 'S7_2004'  ]],
    [[642, 65036], ['FORNJOT'   , 'S2004_S8'  , 'S8_2004'  ]],
    [[640, 65037], ['FARBAUTI'  , 'S2004_S_9' , 'S9_2004'  ]],
    [[636, 65038], ['AEGIR'     , 'S2004_S_10', 'S10_2004' ]],
    [[637, 65039], ['BEBHIONN'  , 'S2004_S_11', 'S11_2004' ]],
    [[     65040], [              'S2004_S_12', 'S12_2004' ]],
    [[     65041], [              'S2004_S_13', 'S13_2004' ]],
    [[643, 65042], ['HATI'      , 'S2004_S_14', 'S14_2004' ]],
    [[638, 65043], ['BERGELMIR' , 'S2004_S_15', 'S15_2004' ]],
    [[641, 65044], ['FENRIR'    , 'S2004_S_16', 'S16_2004' ]],
    [[     65045], [              'S2004_S_17', 'S17_2004' ]],
    [[639, 65046], ['BESTLA'    , 'S2004_S_18', 'S18_2004' ]],
    [[644, 65047], ['HYRROKKIN' , 'S2004_S_19', 'S19_2004' ]],
    [[     65048], [              'S2006_S_1' , 'S01_2006' ]],
    [[645, 65049], ['KARI'      , 'S2006_S_2' , 'S02_2006' ]],
    [[     65050], [              'S2006_S_3' , 'S03_2006' ]],
    [[651, 65051], ['GREIP'     , 'S2006_S_4' , 'S04_2006' ]],
    [[646, 65052], ['LOGE'      , 'S2006_S_5' , 'S05_2006' ]],
    [[650, 65053], ['JARNSAXA'  , 'S2006_S_6' , 'S06_2006' ]],
    [[648, 65054], ['SURTUR'    , 'S2006_S_7' , 'S07_2006' ]],
    [[647       ], ['SKOLL'     , 'S2006_S_8' , 'S08_2006' ]],
    [[652       ], ['TARQEQ'    , 'S2007_S_1' , 'S01_2007' ]],
    [[     65055], [              'S2007_S_2' , 'S02_2007' ]],
    [[     65056], [              'S2007_S_3' , 'S03_2007' ]],
    [[653, 65060], ['AEGAEON'   , 'K07S4'                  ]],
    [[     65066], [              'S2004_S_29', 'S2004_S29']],
    [[     65067], [              'S2004_S_31', 'S2004_S31']],
    [[     65068], [              'S2004_S_26', 'S2004_S26']],
    [[     65069], [              'S2004_S_35', 'S2004_S35']],
    [[     65070], [              'S2004_S_24', 'S2004_S24']],
    [[     65071], [              'S2004_S_23', 'S2004_S23']],
    [[     65072], [              'S2004_S_25', 'S2004_S25']],
    [[     65073], [              'S2004_S_22', 'S2004_S22']],
    [[     65074], [              'S2004_S_32', 'S2004_S32']],
    [[     65075], [              'S2004_S_33', 'S2004_S33']],
    [[     65076], [              'S2004_S_34', 'S2004_S34']],
    [[     65077], [              'S2004_S_28', 'S2004_S28']],
    [[     65078], [              'S2004_S_30', 'S2004_S30']],
    [[     65079], [              'S2004_S_21', 'S2004_S21']],
    [[     65080], [              'S2004_S_20', 'S2004_S20']],
    [[     65081], [              'S2004_S_36', 'S2004_S36']],
    [[     65082], [              'S2004_S_37', 'S2004_S37']],
    [[     65083], [              'S2004_S_38', 'S2004_S38']],
    [[     65084], [              'S2004_S_39', 'S2004_S39']],
]

ALIASES = JUPITER_ALIASES + SATURN_ALIASES

# Define within cspyce
for (codes, names) in ALIASES:
    cspyce.define_body_aliases(*(names + codes))

################################################################################

class Body(object):
    """Defines the properties and relationships of solar system bodies.

    Bodies include planets, dwarf planets, satellites and rings. Each body has
    these attributes:
        name            the name of this body.
        spice_id        the ID from the SPICE toolkit, if the body found in
                        SPICE; otherwise, None.
        path            a Waypoint for the body's path.
        frame           a Wayframe for the body's frame.

        ring_frame      the Wayframe of a "despun" RingFrame relevant to a ring
                        that might orbit this body. None if not (yet) defined.
        ring_is_retrograde  True if the ring frame is retrograde relative to
                            IAU-defined north.
        ring_body       the Body object associated with an equatorial, unbounded
                        ring; None if not defined.

        parent          the physical body (not necessarily the barycenter) about
                        which this body orbits. If a string is given, the parent
                        is found by looking it up in the BODY_REGISTRY
                        dictionary.
        barycenter      the body defining the barycenter of motion and the
                        gravity field defining this body's motion. If a string
                        is given, the barycenter is found by looking it up in
                        the BODY_REGISTRY dictionary. If None, this is the
                        parent body.

        surface         the Surface object defining the body's surface. None if
                        the body is a point and has no surface.
        radius          a single value in km, defining the radius of a sphere
                        that encloses the entire body. Zero for bodies that have
                        no surface.
        inner_radius    a single value in km, defining the radius of a sphere
                        that is entirely enclosed by the body. Zero for bodies
                        that have no surface.
        gravity         the gravity field of the body; None if the gravity field
                        is undefined or negligible.
        keywords        a list of keywords associated with the body. Typical
                        values are "PLANET", "BARYCENTER", "SATELLITE",
                        "SPACECRAFT", "RING", and for satellites, "REGULAR",
                        "IRREGULAR", "CLASSICAL". The name of each body appears
                        as a keyword in its own keyword list. In addition, every
                        planet appears as a keyword for its system barycenter
                        and for each of its satellites and rings.

        children        a list of child bodies associated with this body. Every
                        Body object should appear on the list of the children of
                        its parent and also the children of its barycenter.
    """

    BODY_REGISTRY = {}          # global dictionary of body objects
    STANDARD_BODIES = set()     # Bodies that always have the same definition

    def __init__(self, name, path, frame, parent=None, barycenter=None,
                 spice_name=None):
        """Constructor for a Body object."""

        if not isinstance(name, str):
            raise TypeError('Body name must be a string: ' + str(name))

        self.name = name.upper()

        if spice_name is None:
            spice_name = self.name

        self.spice_name = spice_name

        try:
            self.spice_id = cspyce.bodn2c(spice_name)
        except (KeyError, ValueError):
            self.spice_id = None

        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        self.ring_frame = None
        self.ring_epoch = None
        self.ring_is_retrograde = False
        self.ring_pole = None
        self.ring_body = None
        self.is_ring = False

        self.invariable_pole = None
        self.invariable_frame = None

        if parent is None:
            self.parent = None
        else:
            self.parent = Body.as_body(parent)

        if barycenter is None:
            self.barycenter = self.parent
        else:
            self.barycenter = Body.as_body(barycenter)

        self.surface = NullSurface(self.path, self.frame)   # placeholder
        self.radius = 0.
        self.inner_radius = 0.

        self.gravity = None
        self.lightsource = None
        self.keywords = [self.name]

        self.children = []

        # Append this to the appropriate child lists
        if self.parent is not None:
            if self not in self.parent.children:
                self.parent.children.append(self)

        if self.barycenter is not None:
            if self not in self.barycenter.children:
                self.barycenter.children.append(self)

        # Save it in the Solar System dictionary
        Body.BODY_REGISTRY[self.name] = self

    def __str__(self):
        return 'Body(' + self.name + ')'

    def __repr__(self):
        return self.__str__()

    ############################################################################
    # For serialization, standard bodies are uniquely identified by name
    ############################################################################

    INIT_ARGS  = ['name', 'path', 'frame', 'parent', 'barycenter', 'spice_name']
    EXTRA_ARGS = ['surface', 'radius', 'inner_radius',
                  'ring_epoch', 'ring_is_retrograde', 'ring_pole',
                  'ring_body', 'gravity']

    def PACKRAT__args__(self):
        if self.name in Body.STANDARD_BODIES:
            return ['name'] + Body.EXTRA_ARGS

        return Body.INIT_ARGS + Body.EXTRA_ARGS

    @staticmethod
    def PACKRAT__init__(cls, **args):

        name = args['name']
        if name in Body.STANDARD_BODIES:
            obj = Body.lookup(name)
        else:
            obj = Body.__init__(name, args['path'], args['frame'],
                                      args['parent'], args['barycenter'],
                                      args['spice_name'])

        # Override parameters that might change
        obj.apply_surface(args['surface'], args['radius'], args['inner_radius'])

        obj.apply_ring_frame(args['ring_epoch'], args['ring_is_retrograde'],
                             args['ring_pole'])

        obj.apply_gravity(args['gravity'])

        return obj

    ############################################################################

    def apply_surface(self, surface, radius, inner_radius=0.):
        """Add the surface attribute to a Body."""

        self.surface = surface
        self.radius = radius
        self.inner_radius = inner_radius
        # assert self.surface.origin == self.path
        # This assertion is not strictly necessary

    def apply_ring_frame(self, epoch=None, retrograde=False, pole=None):
        """Add the ring and ring_frame attributes to a Body."""

        # On a repeat call, make sure the frames match
        if type(self.ring_frame) == RingFrame and pole is None:
            assert self.ring_frame.epoch == epoch
            assert self.ring_frame.retrograde == retrograde
            return

        if type(self.ring_frame) == PoleFrame and pole is not None:
            assert self.ring_frame.retrograde == retrograde
            assert self.ring_frame.invariable_pole == pole
            return

        if pole is not None:
            pole = Vector3.as_vector3(pole)
            self.ring_frame = PoleFrame(self.frame, pole=pole,
                                                    retrograde=retrograde)

            x_axis = Vector3.ZAXIS.ucross(pole)
            self.invariable_frame = TwoVector(Frame.J2000,
                                              pole, 'Z', x_axis, 'X',
                                              id=self.name + '_INVARIABLE')
            self.invariable_pole = pole

        else:
            self.ring_frame = RingFrame(self.frame, epoch=epoch,
                                                    retrograde=retrograde)
            self.invariable_frame = self.ring_frame

        self.ring_epoch = epoch
        self.ring_is_retrograde = retrograde

        if epoch is not None:
            xform = self.frame.wrt(Frame.J2000).transform_at_time(epoch)
            self.ring_pole = xform.matrix.row_vector(2, Vector3)
            if self.invariable_frame is self.ring_frame:
                self.invariable_pole = self.ring_pole

    def apply_gravity(self, gravity):
        """Add the gravity attribute to a Body."""

        self.gravity = gravity

    def add_keywords(self, keywords):
        """Add one or more keywords to the list associated with this Body."""

        if type(keywords) == type(""): keywords = [keywords]

        for keyword in keywords:
            # Avoid duplicates...
            if keyword not in self.keywords:
                self.keywords.append(keyword)

    ############################################################################
    # Tools for selecting the children of a body
    ############################################################################

    def select_children(self, include_all=None, include_any=None,
                              exclude=None, radius=None, recursive=False):
        """Return a list of body objects based on keywords and size."""

        if recursive:
            bodies = []
            self._recursive_children(bodies)
        else:
            bodies = self.children

        if include_all is not None:
            bodies = Body.keywords_include_all(bodies, include_all)

        if include_any is not None:
            bodies = Body.keywords_include_any(bodies, include_any)

        if exclude is not None:
            bodies = Body.keywords_do_not_include(bodies, exclude)

        if radius is not None:
            if type(radius) == type(0) or type(radius) == type(0.):
                radius = (radius, np.inf)
            elif len(radius) == 1:
                radius = (radius[0], np.inf)
            bodies = Body.radius_in_range(bodies, radius[0], radius[1])

        return bodies

    def _recursive_children(self, list):

        for child in self.children:
            if child not in list:
                list.append(child)
            child._recursive_children(list)

    @staticmethod
    def name_in(bodies, names):
        """Retain bodies if their names ARE found in the list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def name_not_in(bodies, names):
        """Retain bodies only if their names are NOT in the list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name not in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_in_range(bodies, min, max=np.inf):
        """Retain bodies if their radii fall INSIDE the range (min,max)."""

        list = []
        for body in bodies:
            if body.radius >= min and body.radius <= max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_not_in_range(bodies, min, max=np.inf):
        """Retain bodies if their radii fall OUTSIDE the range (min,max)."""

        list = []
        for body in bodies:
            if body.radius < min or body.radius > max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_in(bodies, class_names):
        """Retain bodies if the their surface class IS found in the list.

        Note that the name of the surface class is "NoneType" for cases where
        the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_not_in(bodies, class_names):
        """Retain bodies if their surface class is NOT found in the list.

        Note that the name of the surface class is "NoneType" for cases where
        the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name not in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_gravity(bodies):
        """Retain bodies on the list if they HAVE a defined gravity."""

        list = []
        for body in bodies:
            if body.gm is not None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_gravity(bodies):
        """Retain bodies on the list if they have NO gravity."""

        list = []
        for body in bodies:
            if body.gm is None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_children(bodies):
        """Retain bodies on the list if they HAVE children."""

        list = []
        for body in bodies:
            if body.children != [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_children(bodies):
        """Retain bodies on the list if they have NO children."""

        list = []
        for body in bodies:
            if body.children == [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def keywords_include_any(bodies, keywords):
        """Retain bodies that have at least one of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            for keyword in keywords:
                if keyword in body.keywords and body not in list:
                    list.append(body)
                    break
        return list

    @staticmethod
    def keywords_include_all(bodies, keywords):
        """Retain bodies if they have all of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            is_match = True
            for keyword in keywords:
                if keyword not in body.keywords:
                    is_match = False
                    break

            if is_match and body not in list:
                list.append(body)

        return list

    @staticmethod
    def keywords_do_not_include(bodies, keywords):
        """Retain bodies if they DO NOT have any of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            is_found = False
            for keyword in keywords:
                if keyword in body.keywords:
                    is_found = True
                    break

            if not is_found and body not in list:
                list.append(body)

        return list

    ########################################

    @staticmethod
    def define_multipath(bodies, origin="SSB", frame="J2000", id=None):
        """Construct a multipath for the centers of the given list of bodies.

        The default ID of the path returned is the name of the first body with
        a "+" appended."""

        paths = []
        for body in bodies:
            paths.append(body.path)

        return MultiPath(paths, origin, frame, id)

    ############################################################################
    # Body registry
    ############################################################################

    @staticmethod
    def lookup(key):
        """Return a body from the registry given its name."""

        return Body.BODY_REGISTRY[key.upper()]

    @staticmethod
    def exists(key):
        """Return True if the body's name exists in the registry."""

        return key.upper() in Body.BODY_REGISTRY

    @staticmethod
    def as_body(body):
        """Return a body object given the registered name or the object itself.
        """

        if isinstance(body, Body):
            return body

        return Body.lookup(body)

    @staticmethod
    def as_body_name(body):
        """Return a body name given the registered name or the object itself."""

        if isinstance(body, Body):
            return body.name

        return body

    @staticmethod
    def reset_registry():
        """Initialize the registry.
    
        It is not generally necessary to call this function, but it can be used
        to reset the registry for purposes of debugging.
        """

        Body.BODY_REGISTRY.clear()

        spice_support.initialize()

        Path.reset_registry()
        Frame.reset_registry()

    def as_path(self):
        """Path object for this body."""

        return Path.as_primary_path(self.path)

################################################################################
# General function to load Solar System components
################################################################################

def define_solar_system(start_time=None, stop_time=None, asof=None, **args):
    """Construct bodies, paths and frames for planets and their moons.

    Each planet is defined relative to the SSB. Each moon is defined relative to
    its planet. Names are as defined within the SPICE toolkit. Body associations
    are defined within the spicedb library.

    Input:
        start_time      start_time of the period to be convered, in ISO date
                        or date-time format.
        stop_time       stop_time of the period to be covered, in ISO date
                        or date-time format.
        asof            a UTC date such that only kernels released earlier
                        than that date will be included, in ISO format.

    Additional keyword-only parameters
        planets         1-9 to load kernels for a particular planet and its
                        moons. Omit to load the nine planets (including Pluto).
                        Use a tuple to list more than one planet number.
        irregulars      Omit this keyword or set to True to include the outer
                        irregular satellites. Specify irregulars=False to omit
                        these bodies.
        mst_pck         Omit this keyword or set to True to include the
                        MST PCKs, which define the rotation states of Saturn's
                        small moons during the Cassini era. Specify
                        mst_pck=False to omit these kernels.

    Return              an ordered list of SPICE kernel names
    """

    names = []

    # Interpret the keyword args
    irregulars = args.get('irregulars', True)

    planets = args.get('planets', None)
    if planets is None or planets == 0:
        planets = (1,2,3,4,5,6,7,8,9)
    if isinstance(planets, numbers.Integral):
        planets = (planets,)

    mst_pck = args.get('mst_pck', True)

    # Load the necessary SPICE kernels
    spicedb.open_db()

    names += spicedb.furnish_lsk(asof=asof)
    names += spicedb.furnish_pck(name='NAIF-PCK', asof=asof)

    # Special handling for Saturn
    if 6 in planets:
        names += spicedb.furnish_pck(name='CAS-FK-ROCKS', asof=asof)
        names += spicedb.furnish_pck(name='CAS-PCK', asof=asof)
        names += spicedb.furnish_pck(name='CAS-PCK-ROCKS', asof=asof)

        if mst_pck:
            names += spicedb.furnish_pck(name='SAT-PCK-MST', asof=asof)

    names += spicedb.furnish_spk(planets, time=(start_time, stop_time), asof=asof)

    # Define B1950 in addition to J2000
    ignore = SpiceFrame("B1950", "J2000")

    # SSB and Sun
    define_bodies(["SSB"], None, None, ["SUN", "BARYCENTER"])
    define_bodies(["SUN"], None, None, ["SUN"])

    # Mercury, Venus, Earth orbit the Sun
    define_bodies([199, 299, 399], "SUN", "SUN", ["PLANET"])

    # Add Earth's Moon
    define_bodies([301], "EARTH", "EARTH",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])

    # Define planetary systems
    if 4 in planets:
        names += _define_mars(start_time, stop_time, asof)
    if 5 in planets:
        names += _define_jupiter(start_time, stop_time, asof, irregulars)
    if 6 in planets:
        names += _define_saturn(start_time, stop_time, asof, irregulars)
    if 7 in planets:
        names += _define_uranus(start_time, stop_time, asof, irregulars)
    if 8 in planets:
        names += _define_neptune(start_time, stop_time, asof, irregulars)
    if 9 in planets:
        names += _define_pluto(start_time, stop_time, asof)

    spicedb.close_db()

    # Also define the solar disk as a light source. The import of the
    # LightSource class is local to this function because a file-level import
    # would result in recursive imports.

    from oops.lightsource import DiskSource
    _ = DiskSource('SOLAR_DISK', SpicePath(10), 695990., 11)

    return names

################################################################################
# Mars System
################################################################################

MARS_ALL_MOONS = range(401,403)
MARS_MOONS_LOADED = []

def _define_mars(start_time, stop_time, asof=None):
    """Define components of the Mars system."""

    global MARS_MOONS_LOADED

    MARS_MOONS_LOADED += MARS_ALL_MOONS
    names = spicedb.furnish_spk(MARS_MOONS_LOADED,
                                time=(start_time, stop_time),
                                asof=asof)

    # Mars and the Mars barycenter orbit the Sun
    define_bodies([499], "SUN", "SUN", ["PLANET"])
    define_bodies([4], "SUN", "SUN", ["BARYCENTER"])

    # Moons of Mars
    define_bodies(MARS_ALL_MOONS, "MARS", "MARS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])

    # Rings of Mars
    ring = define_ring("MARS", "MARS_RING_PLANE", None, [])
    ring.backplane_id = 'MARS:RING'
    ring.backplane_limits = None
    ring.unbounded_surface = ring

    Body.BODY_REGISTRY['MARS'].ring_body = ring

    return names

################################################################################
# Jupiter System
################################################################################

JUPITER_CLASSICAL = range(501,505)
JUPITER_REGULAR   = [505] + range(514,517)
JUPITER_IRREGULAR = range(506,514) + range(517,559) + [554] + \
                    [55060, 55061, 55062, 55064, 55065, 55066, 55068, 55070,
                     55071, 55074]
JUPITER_MOONS_LOADED = []

# See definition of JUPITER_ALIASES at the top of the file for the list of
# additional, ambiguous irregular moons

JUPITER_MAIN_RING_LIMIT = 128940.

def _define_jupiter(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Jupiter system."""

    global JUPITER_MOONS_LOADED

    # Load Jupiter system SPKs
    JUPITER_MOONS_LOADED += JUPITER_CLASSICAL + JUPITER_REGULAR
    if irregulars:
        JUPITER_MOONS_LOADED += JUPITER_IRREGULAR

    names = spicedb.furnish_spk(JUPITER_MOONS_LOADED,
                                time=(start_time, stop_time),
                                asof=asof)

    # Jupiter and the Jupiter barycenter orbit the Sun
    define_bodies([599], "SUN", "SUN", ["PLANET"])
    define_bodies([5], "SUN", "SUN", ["BARYCENTER"])

    # Moons and rings of Jupiter
    define_bodies(JUPITER_CLASSICAL, "JUPITER", "JUPITER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(JUPITER_REGULAR, "JUPITER", "JUPITER",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(JUPITER_IRREGULAR, "JUPITER", "JUPITER BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

        for (ids, names) in JUPITER_ALIASES:
            define_bodies([ids[0]], "JUPITER", "JUPITER BARYCENTER",
                                    ["SATELLITE", "IRREGULAR"])

    ring = define_ring("JUPITER", "JUPITER_RING_PLANE", None, [])
    ring.backplane_id = 'JUPITER:RING'
    ring.backplane_limits = None
    ring.unbounded_surface = ring
    unbounded_ring = ring
    Body.BODY_REGISTRY['JUPITER'].ring_body = ring

    ring = define_ring("JUPITER", "JUPITER_RING_SYSTEM",
                       JUPITER_MAIN_RING_LIMIT, [])
    ring.backplane_id = 'JUPITER:RING'
    ring.backplane_limits = (0., JUPITER_MAIN_RING_LIMIT)
    ring.unbounded_surface = unbounded_ring

    return names

################################################################################
# Saturn System
################################################################################

SATURN_CLASSICAL_INNER = range(601,607)     # Mimas through Titan orbit Saturn
SATURN_CLASSICAL_OUTER = range(607,609)     # Hyperion, Iapetus orbit barycenter
SATURN_CLASSICAL_IRREG = [609]              # Phoebe
SATURN_REGULAR   = range(610,619) + range(632,636) + [649,653]
SATURN_IRREGULAR = (range(619,632) + range(636,649) + range(650,653) +
                    [65035, 65040, 65041, 65045, 65048, 65050, 65056])
SATURN_MOONS_LOADED = []

SATURN_MAIN_RINGS = ( 74658., 136780.)
SATURN_D_RING =     ( 66900.,  74658.)
SATURN_C_RING =     ( 74658.,  91975.)
SATURN_B_RING =     ( 91975., 117507.)
SATURN_A_RING =     (122340., 136780.)
SATURN_F_RING_CORE  = 140220.
SATURN_F_RING_LIMIT = 140612.
SATURN_RINGS        = (SATURN_MAIN_RINGS[0], SATURN_F_RING_LIMIT)
SATURN_AB_RINGS     = (SATURN_B_RING[0], SATURN_A_RING[1])

def _define_saturn(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Saturn system."""

    global SATURN_MOONS_LOADED

    # Load Saturn system SPKs
    SATURN_MOONS_LOADED += (SATURN_CLASSICAL_INNER + SATURN_CLASSICAL_OUTER +
                            SATURN_CLASSICAL_IRREG + SATURN_REGULAR)
    if irregulars:
        SATURN_MOONS_LOADED += SATURN_IRREGULAR

    names = spicedb.furnish_spk(SATURN_MOONS_LOADED,
                                time=(start_time, stop_time), 
                                asof=asof)

    # Saturn and the Saturn barycenter orbit the SSB
    define_bodies([699], "SUN", "SSB", ["PLANET"])
    define_bodies([6], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Saturn
    define_bodies(SATURN_CLASSICAL_INNER, "SATURN", "SATURN",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_OUTER, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_IRREG, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(SATURN_REGULAR, "SATURN", "SATURN",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(SATURN_IRREGULAR, "SATURN", "SATURN BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

    ring = define_ring("SATURN", "SATURN_RING_PLANE", None, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = None
    ring.unbounded_surface = ring
    unbounded_ring = ring
    Body.BODY_REGISTRY['SATURN'].ring_body = ring

    ring = define_ring("SATURN", "SATURN_RING_SYSTEM", SATURN_F_RING_LIMIT, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = (0., SATURN_F_RING_LIMIT)
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_RINGS", SATURN_RINGS, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_RINGS
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_MAIN_RINGS", SATURN_MAIN_RINGS, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_MAIN_RINGS
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_A_RING", SATURN_A_RING, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_A_RING
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_B_RING", SATURN_B_RING, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_B_RING
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_C_RING", SATURN_C_RING, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_C_RING
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("SATURN", "SATURN_AB_RINGS", SATURN_AB_RINGS, [])
    ring.backplane_id = 'SATURN:RING'
    ring.backplane_limits = SATURN_AB_RINGS
    ring.unbounded_surface = unbounded_ring

    return names

################################################################################
# Uranus System
################################################################################

URANUS_CLASSICAL  = range(701,706)
URANUS_INNER      = range(706,716) + [725,726,727]
URANUS_IRREGULAR  = range(716,726)
URANUS_MOONS_LOADED = []

URANUS_EPSILON_LIMIT = 51604.
URANUS_MU_LIMIT = [97700. - 17000./2, 97700. + 17700./2]
URANUS_NU_LIMIT = [67300. -  3800./2, 67300. +  3800./2]

# Special definitions of Uranian eccentric/inclined rings
URANUS_OLD_GRAVITY = OblateGravity(5793939., [3.34343e-3, -2.885e-5], 26200.)

# Local function used to adapt the tabulated elements from French et al. 1991.
def _uranus_ring_elements(a, e, peri, i, node, da):
    n = URANUS_OLD_GRAVITY.n(a)
    prec = URANUS_OLD_GRAVITY.combo(a, (1,-1, 0))
    regr = URANUS_OLD_GRAVITY.combo(a, (1, 0,-1))
    peri *= constants.RPD
    node *= constants.RPD
    i *= constants.RPD
    return (a, 0., n, e, peri, prec, i, node, regr, (a+da/2))
    # The extra item returned is the outer radius in the orbit's frame

URANUS_SIX_ELEMENTS  = _uranus_ring_elements(
                        41837.15, 1.013e-3, 242.80, 0.0616,  12.12, 2.8)
URANUS_FIVE_ELEMENTS = _uranus_ring_elements(
                        42234.82, 1.899e-3, 170.31, 0.0536, 286.57, 2.8)
URANUS_FOUR_ELEMENTS = _uranus_ring_elements(
                        42570.91, 1.059e-3, 127.28, 0.0323,  89.26, 2.7)
URANUS_ALPHA_ELEMENTS = _uranus_ring_elements(
                        44718.45, 0.761e-3, 333.24, 0.0152,  63.08, 7.15+3.52)
URANUS_BETA_ELEMENTS = _uranus_ring_elements(
                        45661.03, 0.442e-3, 224.88, 0.0051, 310.05, 8.15+3.07)
URANUS_ETA_ELEMENTS = _uranus_ring_elements(
                        47175.91, 0.004e-3, 228.10, 0.0000,   0.00, 1.7)
URANUS_GAMMA_ELEMENTS = _uranus_ring_elements(
                        47626.87, 0.109e-3, 132.10, 0.0000,   0.00, 3.8)
URANUS_DELTA_ELEMENTS = _uranus_ring_elements(
                        48300.12, 0.004e-3, 216.70, 0.0011, 260.70, 7.0)
URANUS_LAMBDA_ELEMENTS = _uranus_ring_elements(
                        50023.94, 0.000e-3,   0.00, 0.0000,   0.00, 2.5)
URANUS_EPSILON_ELEMENTS = _uranus_ring_elements(
                        51149.32, 7.936e-3, 214.97, 0.0000,   0.00, 58.1+37.6)

def _define_uranus(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Uranus system."""

    global URANUS_MOONS_LOADED

    # Load Uranus system SPKs
    URANUS_MOONS_LOADED += URANUS_CLASSICAL + URANUS_INNER
    if irregulars:
        URANUS_MOONS_LOADED += URANUS_IRREGULAR

    names = spicedb.furnish_spk(URANUS_MOONS_LOADED,
                                time=(start_time, stop_time),
                                asof=asof)

    # Uranus and the Uranus barycenter orbit the SSB
    define_bodies([799], "SUN", "SSB", ["PLANET"])
    define_bodies([7], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Uranus
    define_bodies(URANUS_CLASSICAL, "URANUS", "URANUS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(URANUS_INNER, "URANUS", "URANUS",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(URANUS_IRREGULAR, "URANUS", "URANUS",
                      ["SATELLITE", "IRREGULAR"])

    ring = define_ring("URANUS", "URANUS_RING_PLANE", None,
                       [], retrograde=True)
    ring.backplane_id = 'URANUS:RING'
    ring.backplane_limits = None
    ring.unbounded_surface = ring
    unbounded_ring = ring
    Body.BODY_REGISTRY['URANUS'].ring_body = ring

    ring = define_ring("URANUS", "URANUS_RING_SYSTEM", URANUS_EPSILON_LIMIT,
                       [], retrograde=True)
    ring.backplane_id = 'URANUS:RING'
    ring.backplane_limits = (0., URANUS_EPSILON_LIMIT)
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("URANUS", "MU_RING", URANUS_MU_LIMIT,
                       [], retrograde=True)
    ring.backplane_id = 'URANUS:RING'
    ring.backplane_limits = URANUS_MU_LIMIT
    ring.unbounded_surface = unbounded_ring

    ring = define_ring("URANUS", "NU_RING", URANUS_NU_LIMIT,
                       [], retrograde=True)
    ring.backplane_id = 'URANUS:RING'
    ring.backplane_limits = URANUS_NU_LIMIT
    ring.unbounded_surface = unbounded_ring

    URANUS_EPOCH = cspyce.utc2et("1977-03-10T20:00:00")

    uranus_wrt_b1950 = AliasFrame("IAU_URANUS").wrt("B1950")
    ignore = RingFrame(uranus_wrt_b1950, URANUS_EPOCH, retrograde=True,
                       id="URANUS_RINGS_B1950")

    define_orbit("URANUS", "SIX_RING", URANUS_SIX_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "FIVE_RING", URANUS_FIVE_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "FOUR_RING", URANUS_FOUR_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "ALPHA_RING", URANUS_ALPHA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "BETA_RING", URANUS_BETA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "ETA_RING", URANUS_ETA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "GAMMA_RING", URANUS_GAMMA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "DELTA_RING", URANUS_DELTA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "LAMBDA_RING", URANUS_LAMBDA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "EPSILON_RING", URANUS_EPSILON_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])

    return names

################################################################################
# Neptune System
################################################################################

NEPTUNE_CLASSICAL_INNER = [801]             # Triton
NEPTUNE_CLASSICAL_OUTER = [802]             # Nereid orbits barycenter
NEPTUNE_REGULAR   = range(803,809)
NEPTUNE_IRREGULAR = range(809,814)
NEPTUNE_MOONS_LOADED = []

NEPTUNE_ADAMS_LIMIT = 62940.

# From Table 5 of Jacobson. The orbits of the Neptunian Satellites and the
# Orientation of the Pole of Neptune. Astron. J. 137, 4322-4329 (2009).
NEPTUNE_INVARIABLE_RA = 299.46086 * np.pi/180.
NEPTUNE_INVARIABLE_DEC = 43.40481 * np.pi/180.

def _define_neptune(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Neptune system."""

    global NEPTUNE_MOONS_LOADED

    # Load Neptune system SPKs
    NEPTUNE_MOONS_LOADED += (NEPTUNE_CLASSICAL_INNER + NEPTUNE_CLASSICAL_OUTER +
                             NEPTUNE_REGULAR)
    if irregulars:
        NEPTUNE_MOONS_LOADED += NEPTUNE_IRREGULAR

    names = spicedb.furnish_spk(NEPTUNE_MOONS_LOADED,
                                time=(start_time, stop_time),
                                asof=asof)

    # Neptune and the Neptune barycenter orbit the SSB
    define_bodies([899], "SUN", "SSB", ["PLANET"])
    define_bodies([8], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Neptune
    define_bodies(NEPTUNE_CLASSICAL_INNER, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(NEPTUNE_CLASSICAL_OUTER, "NEPTUNE", "NEPTUNE BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(NEPTUNE_REGULAR, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        _ = spicedb.furnish_spk(NEPTUNE_IRREGULAR,
                                time=(start_time, stop_time), asof=asof)

        define_bodies(NEPTUNE_IRREGULAR, "NEPTUNE", "NEPTUNE BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

#     ra  = cspyce.bodvrd('NEPTUNE', 'POLE_RA')[0]  * np.pi/180
#     dec = cspyce.bodvrd('NEPTUNE', 'POLE_DEC')[0] * np.pi/180
    ra  = NEPTUNE_INVARIABLE_RA
    dec = NEPTUNE_INVARIABLE_DEC
    pole = Vector3.from_ra_dec_length(ra,dec)

    ring = define_ring("NEPTUNE", "NEPTUNE_RING_PLANE",  None, [], pole=pole)
    ring.backplane_id = 'NEPTUNE:RING'
    ring.backplane_limits = None
    ring.unbounded_surface = ring
    unbounded_ring = ring
    Body.BODY_REGISTRY['NEPTUNE'].ring_body = ring

    ring = define_ring("NEPTUNE", "NEPTUNE_RING_SYSTEM", NEPTUNE_ADAMS_LIMIT,
                                                         [], pole=pole)
    ring.backplane_id = 'NEPTUNE:RING'
    ring.backplane_limits = (0., NEPTUNE_ADAMS_LIMIT)
    ring.unbounded_surface = unbounded_ring

    return names

################################################################################
# Pluto System
################################################################################

CHARON        = [901]
PLUTO_REGULAR = range(902,906)
PLUTO_MOONS_LOADED = []

PLUTO_RADIUS = 19591.
CHARON_RADIUS = 606.
PLUTO_CHARON_DISTANCE = 19591.

def _define_pluto(start_time, stop_time, asof=None):
    """Define components of the Pluto system."""

    global PLUTO_MOONS_LOADED

    PLUTO_MOONS_LOADED += CHARON + PLUTO_REGULAR
    names = spicedb.furnish_spk(PLUTO_MOONS_LOADED,
                                time=(start_time, stop_time),
                                asof=asof)

    # Pluto and the Pluto barycenter orbit the SSB
    define_bodies([999], "SUN", "SSB", ["PLANET"])
    define_bodies([9], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Pluto
    define_bodies(CHARON, "PLUTO", "PLUTO",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(PLUTO_REGULAR, "PLUTO", "PLUTO BARYCENTER",
                  ["SATELLITE", "REGULAR"])

    ring = define_ring("PLUTO", "PLUTO_RING_PLANE", None, [],
                       barycenter_name="PLUTO BARYCENTER")
    ring.backplane_id = 'PLUTO:RING'
    ring.backplane_limits = None
    Body.BODY_REGISTRY['PLUTO'].ring_body = ring

    ring = define_ring("PLUTO", "PLUTO_INNER_RING_PLANE",
                       PLUTO_CHARON_DISTANCE - CHARON_RADIUS, [],
                       barycenter_name="PLUTO")
    ring.backplane_id = 'PLUTO_INNER_RING_PLANE'
    ring.backplane_limits = (0., PLUTO_CHARON_DISTANCE)

    barycenter = Body.BODY_REGISTRY["PLUTO BARYCENTER"]
    barycenter.ring_frame = Body.BODY_REGISTRY["PLUTO"].ring_frame

    return names

################################################################################
# Define bodies and rings...
################################################################################

def define_bodies(spice_ids, parent, barycenter, keywords):
    """Define the path, frame, surface for bodies by name or SPICE ID.

    All must share a common parent and barycenter."""

    for spice_id in spice_ids:

        # Define the body's path
        path = SpicePath(spice_id, "SSB")

        # The name of the path is the name of the body
        name = path.path_id

        # If the body already exists, skip it
        if name in Body.BODY_REGISTRY: continue

        # Sometimes a frame is undefined for a new moon; in this case assume it
        # is synchronous
        try:
            frame = SpiceFrame(spice_id)
        except LookupError:
            if ('BARYCENTER' in keywords) or ('IRREGULAR' in keywords):
                frame = Frame.J2000
            else:
                frame = Synchronous(path, parent, id='SYNCHRONOUS_' + name)

        # Define the planet's body
        # Note that this will overwrite any registered body of the same name
        body = Body(name, name, frame.frame_id, parent, barycenter)
        body.add_keywords(keywords)

        # Add the gravity object if it exists
        try:
            body.apply_gravity(Gravity.lookup(name))
        except KeyError: pass

        # Add the surface object if shape information is available
        # RuntimeError was raised by old version of cspyce;
        # KeyError is raised during a name lookup if the body name is unknown;
        # ValueError is raised during a SPICE ID lookup if the ID is unknown.
        try:
            shape = spice_shape(spice_id, frame.frame_id, (1.,1.,1.))
        except (RuntimeError, ValueError, KeyError):
            shape = NullSurface(path, frame)
            body.apply_surface(shape, 0., 0.)
            shape.body = body
        else:
            body.apply_surface(shape, shape.req, shape.rpol)
            shape.body = body

        # Add a planet name to any satellite or barycenter
        if "SATELLITE" in body.keywords and parent is not None:
            body.add_keywords(parent)

        if "BARYCENTER" in body.keywords and parent is not None:
            body.add_keywords(parent)

        # Save solar system bodies as standard for serialization
        Body.STANDARD_BODIES.add(body.name)
        Path.STANDARD_PATHS.add(body.path.path_id)
        Frame.STANDARD_FRAMES.add(body.frame.frame_id)

    keys = Body.BODY_REGISTRY.keys()
    keys.sort()

def define_ring(parent_name, ring_name, radii, keywords, retrograde=False,
                barycenter_name=None, pole=None):
    """Define and return the body object associate with a ring around another
    body.

    A single radius value is used to define the outer limit of rings. Note that
    a ring has limits but a defined ring plane does not.

    Input:
        parent_name     the name of the central planet for the ring surface.
        ring_name       the name of the surface.
        radii           if this is a tuple with two values, these are the radial
                        limits of the ring; if it is a scalar, then the ring
                        plane has no defined radial limits, but the radius
                        attribute of the body will be set to this value; if
                        None, then the radius attribute of the body will be set
                        to zero.
        keywords        the list of keywords under which this surface is to be 
                        registered. Every ring is also registered under its own
                        name and under the keyword "RING".
        retrograde      True if the ring is retrograde relative to the central
                        planet's IAU-defined pole.
        barycenter_name the name of the ring's barycenter if this is not the
                        same as the name of the central planet.
        pole            if not None, this is the pole of the invariable plane.
                        It will be used to define the ring_frame as a
                        PoleFrame instead of a RingFrame.
    """

    # If the ring body already exists, skip it
    if ring_name in Body.BODY_REGISTRY: return Body.BODY_REGISTRY[ring_name]

    # Identify the parent
    parent = Body.lookup(parent_name)
    parent.apply_ring_frame(retrograde=retrograde, pole=pole)

    if barycenter_name is None:
        barycenter = parent
    else:
        barycenter = Body.lookup(barycenter_name)

    # Interpret the radii
    try:
        rmax = radii[1]
    except IndexError:
        rmax = radii[0]
        radii = None
    except TypeError:
        if radii is None:
            rmax = 0.
        else:
            rmax = radii
            radii = None

    # Create the ring body
    # Note that this will overwrite any registered ring of the same name
    body = Body(ring_name, barycenter.path, parent.ring_frame, parent)
    body.apply_gravity(barycenter.gravity)
    body.apply_ring_frame(retrograde=retrograde, pole=pole)

    ring = RingPlane(barycenter.path, parent.ring_frame, radii,
                     gravity=barycenter.gravity)
    body.apply_surface(ring, rmax, 0.)
    ring.body = body
    body.ring_body = body
    body.is_ring = True

    body.add_keywords([parent, "RING", ring_name])
    body.add_keywords(keywords)

    return body

def define_orbit(parent_name, ring_name, elements, epoch, reference, keywords):
    """Define the path, frame, surface and body for ring given orbital elements.

    The ring can be inclined or eccentric.
    """

    parent = Body.lookup(parent_name)

    orbit = OrbitPlane(elements, epoch, parent.path, reference, id=ring_name)

    body = Body(ring_name, orbit.internal_origin, orbit.internal_frame,
                parent, parent)
    body.apply_surface(orbit, elements[9], 0.)
    orbit.body = body

    body.add_keywords([parent, "RING", "ORBIT", ring_name])
    body.add_keywords(keywords)

def define_small_body(spice_id, name=None, spk=None, keywords=[],
                                parent='SUN', barycenter='SSB'):
    """Define the path, frame, surface for a body by SPICE ID.

    This body treats the Sun as its parent body and barycenter."""

    # Load the SPK if necessary
    if spk:
        cspyce.furnsh(spk)

    # Define the body's path
    path = SpicePath(spice_id, "SSB", id=name)

    # The name of the path is the name of the body
    name = name or path.path_id

    # If the body already exists, skip it
    if name in Body.BODY_REGISTRY: return

    # Sometimes a frame is undefined for a new moon; in this case assume it
    # is synchronous
    try:
        frame = SpiceFrame(spice_id)
    except LookupError:
        if ('BARYCENTER' in keywords) or ('IRREGULAR' in keywords):
            frame = Frame.J2000
        else:
            frame = Synchronous(path, parent, id='SYNCHRONOUS_' + name)

    # Define the planet's body
    # Note that this will overwrite any registered body of the same name
    body = Body(name, path.path_id, frame.frame_id,
                      parent=Body.lookup(parent),
                      barycenter=Body.lookup(barycenter))
    body.add_keywords(keywords)

    # Add the gravity object if it exists
    try:
        body.apply_gravity(Gravity.lookup(name))
    except KeyError:
        pass

    # Add the surface object if shape information is available
    try:
        shape = spice_shape(spice_id, frame.frame_id, (1.,1.,1.))
        body.apply_surface(shape, shape.req, shape.rpol)
    except RuntimeError:
        shape = NullSurface(path, frame)
        body.apply_surface(shape, 0., 0.)
    except LookupError:
        shape = NullSurface(path, frame)
        body.apply_surface(shape, 0., 0.)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Body(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

        define_solar_system("2000-01-01", "2020-01-01")

        self.assertEqual(Body.lookup("DAPHNIS").barycenter.name,
                         "SATURN")
        self.assertEqual(Body.lookup("PHOEBE").barycenter.name,
                         "SATURN BARYCENTER")

        mars = Body.lookup("MARS")
        moons = mars.select_children(include_all=["SATELLITE"])
        self.assertEqual(len(moons), 2)     # Phobos, Deimos

        saturn = Body.lookup("SATURN")
        moons = saturn.select_children(include_all=["CLASSICAL", "IRREGULAR"])
        self.assertEqual(len(moons), 1)     # Phoebe

        moons = saturn.select_children(exclude=["IRREGULAR","RING"], radius=160)
        self.assertEqual(len(moons), 8)     # Mimas-Iapetus

        rings = saturn.select_children(include_any=("RING"))
        self.assertEqual(len(rings), 8)     # A, B, C, AB, Main, all, plane,
                                            # system

        moons = saturn.select_children(include_all="SATELLITE",
                                       exclude=("IRREGULAR"), radius=1000)
        self.assertEqual(len(moons), 1)     # Titan only

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET"])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"],
                                      recursive=True)
        self.assertEqual(len(planets), 10)  # 9 planets plus Earth's moon

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "JUPITER"],
                                      exclude=["IRREGULAR", "BARYCENTER", "IO"],
                                      recursive=True)
        self.assertEqual(len(planets), 16)  # 9 planets + 7 Jovian moons

        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
