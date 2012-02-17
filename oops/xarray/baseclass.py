################################################################################
# Superclass of Empty, Scalar, Pair, Vector3, Matrix3 and Tuple. All represent
# N-dimensional arrays of intrinsically dimensional objects.
#
# Modified 12/12/2011 (BSW) - added help comments to most methods
#
# Modified 1/2/11 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
# Modified 2/8/12 (MRS) -- Supports array masks and units; uses a cleaner set of
#   definitions for math operators.
################################################################################

import numpy as np
import numpy.ma as ma

from oops.units import Units

class Array(object):
    """A class defining an arbitrary Array of possibly multidimensional items.
    Unlike numpy ndarrays, this class makes a clear distinction between the
    dimensions associated with the items and any additional, leading dimensions
    which describe the set of such items. The shape is defined by the leading
    axes only, so a 2x2 array of 3x3 matrices would have shape (2,2,3,3)
    according to numpy but has shape (2,2) according to Array.

    The Array object is designed as a lightweight "wrapper" on the numpy
    ndarray and numpy.ma.MaskedArray. All standard mathematical operators and
    indexing/slicing options are defined. One can mix Array arithmetic with
    scalars, numpy arrays, masked arrays, or anything array-like. The standard
    numpy rules of broadcasting apply to operations performed on Arrays of
    different shapes.

    Array objects have the following attributes:
        shape       a list (not tuple) representing the leading dimensions.
        rank        the number of trailing dimensions belonging to the
                    individual items.
        item        a list (not tuple) representing the shape of the
                    individual items.
        vals        the array's data as a numpy array. The shape of this array
                    is object.shape + object.item. It the object has units,
                    these values are in the specified units.
        mask        the array's mask as a numpy boolean array. The array value
                    is True if the Array value at the same location is masked.
                    A single value of False indicates that the array is not
                    masked.
        units       the units of the array, if any. None indicates no units.

        mvals       a read-only property that presents tha vals and the mask as
                    a masked array.
    """

    # A constant, defined for all Array subclasses, overridden by subclass Empty
    IS_EMPTY = False

    SCALAR_CLASS = None         # A reference to the Scalar subclass, filled in
                                # when that subclass is finished loading

    DUMMY = None                # A place-holder for indexing an Array with a
                                # boolean array that is entirely False.

    @property
    def mvals(self):
        # Construct something that behaves as a suitable mask
        if self.mask is False:
            newmask = ma.nomask
        elif self.mask is True:
            newmask = np.ones((len(self.shape) + self.rank) * [1], dtype="bool")
            (newmask, newvals) = np.broadcast_arrays(newmask, self.vals)
        elif self.rank > 0:
            newmask = self.mask.reshape(self.shape + self.rank * [1])
            (newmask, newvals) = np.broadcast_arrays(newmask, self.vals)
        else:
            newmask = self.mask

        # Return the masked array
        return ma.MaskedArray(self.vals, newmask)

    @staticmethod
    def is_empty(arg):
        """Returns True if the arg is of the Empty class. Carefully written to
        avoid the need to import the Empty subclass."""

        return isinstance(arg, Array) and arg.IS_EMPTY

    @staticmethod
    def as_scalar(arg):
        """Calls the function Scalar.as_scalar() using the given argument.
        Carefully written to avoid a circularity in the load order."""

        return Array.SCALAR_CLASS.as_scalar(arg)

    def __new__(subtype, *arguments, **keywords):
        obj = object.__new__(subtype)
        return obj

    def __repr__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""

        return self.__str__()

    def __str__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""

        suffix = ""

        is_masked = np.any(self.mask)
        if is_masked:
            suffix += ", mask"

        if self.units is not None:
            suffix += ", " + str(self.units)

        if np.shape(self.vals) == ():
            if is_masked:
                string = "--"
            else:
                string = str(self.vals)

        elif is_masked:
            masked_array = self.vals.view(ma.MaskedArray)

            if self.rank == 0:
                masked_array.mask = np.asarray(self.mask)
            else:
                temp_mask = np.asarray(self.mask)
                temp_mask = temp_mask.reshape(temp_mask.shape +
                                              self.rank * (1,))
                masked_array.mask = np.empty(self.vals.shape, dtype="bool")
                masked_array.mask[...] = temp_mask

            string = str(masked_array)
        else:
            string = str(self.vals)

        if string[0] == "[":
            return type(self).__name__ + string[:-1] + suffix + "]"
        else:
            return type(self).__name__ + "(" + string + suffix + ")"

    ####################################
    # Indexing operators
    ####################################

    def __getitem__(self, i):
        """returns the item value at a specific index. copies data to new
            location in memory. if self has no items, i.e. - no shape in the
            object-sense, then raise IndexError. called from x = obj[i]"""

        # Handle a shapeless Array
        if self.shape == []:
            if i is True: return self
            if i is False: return Array.DUMMY
            raise IndexError("too many indices")

        # Handle an index as a boolean
        if i is True: return self
        if i is False: return Array.DUMMY

        # Get the value and mask
        vals = self.vals[i]

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask[i]

        # Make sure we have not penetrated the components of a 1-D or 2-D item
        icount = 1
        if type(i) == type(()): icount = len(i)
        if icount > len(self.shape): raise IndexError("too many indices")

        # If the result is a single, unmasked, unitless value, return it as a
        # number
        if np.shape(vals) == () and not np.any(mask) and self.units is None:
            return vals

        # Construct the object and return it
        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def __getslice__(self, i, j):
        """returns slice of items. copies data to new location in memory. if
            self has no items, i.e. - no shape in the object-sense, then raise
            IndexError. called from x = obj[i:j]."""

        # Get the values and mask
        vals = self.vals[i:j]

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask[i:j]

        # Make sure we have not penetrated the components of a 1-D or 2-D item
        icount = 1
        if type(i) == type(()): icount = len(i)
        if icount > len(self.shape): raise IndexError("too many indices")

        # If the result is a single, unmasked, unitless value, return it as a
        # number
        if np.shape(vals) == () and not np.any(mask) and self.units is None:
            return vals

        # Construct the object and return it
        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def __setitem__(self, i, arg):
        """sets the item value at a specific index. if self has no items,
            i.e. - no shape in the object-sense, then raise IndexError. called
            from obj[i] = arg."""

        # Handle a single boolean index
        if i is False:
            return self
        if i is True:
            obj = Array.__new__(type(self))
            obj.__init__(arg)

            if np.shape(self.vals) == ():
                self.vals = obj.vals
                self.mask = obj.mask
            else:
                self.vals[...] = obj.vals
                if np.shape(self.mask) == ():
                    self.mask = obj.mask
                else:
                    self.mask[...] = obj.mask

            self.units = obj.units
            return self

        # Get the values and mask after converting arg to the same subclass
        (vals, mask, new_units) = self.get_array_mask_unit("[]", arg)

        # Replace the value(s)
        self.vals[i] = vals

        # If the mask is already an array, replace the mask value
        if np.shape(self.mask) != ():
            self.mask[i] = mask

        # Otherwise, if the mask values disagree...
        elif np.any(self.mask != mask):

            # Replace the mask with a boolean array, then fill in the new mask
            newmask = np.empty(self.shape, dtype="bool")
            newmask[...] = self.mask
            newmask[i] = mask
            self.mask = newmask

        return self

    def __setslice__(self, i, j, arg):
        """sets slice of items' values to values of arg. if self has no items,
            i.e. - no shape in the object-sense, then raise IndexError. called
            from obj[i:j] = arg."""

        # Get the values and mask after converting arg to the same subclass
        (vals, mask, new_units) = self.get_array_mask_unit("[]", arg)

        # Replace the value(s)
        self.vals[i:j] = vals

        # If the mask is already an array, replace the mask value
        if np.shape(self.mask) != ():
            self.mask[i:j] = mask

        # Otherwise, if the mask values disagree...
        elif np.any(self.mask != mask):

            # Replace the mask with a boolean array, then fill in the new mask
            newmask = np.empty(self.shape, dtype="bool")
            newmask[...] = self.mask
            newmask[i:j] = mask
            self.mask = newmask

        return self

    ####################################################
    # Default unary arithmetic operators
    ####################################################

    def __pos__(self):
        return self

    def __neg__(self):
        obj = Array.__new__(type(self))
        obj.__init__(-self.vals, self.mask, self.units)
        return obj

    def __abs__(self):
        obj = Array.__new__(type(self))
        obj.__init__(np.abs(self.vals), self.mask, self.units)
        return obj

    ####################################################
    # Arithmetic support methods
    ####################################################

    def get_array_mask_unit(self, op, arg):
        """This method converts the right operand to the same Array subclass
        as self, and then returns the array, mask, and new units, if any."""

        abbrev = op[0]

        # Addition, subtraction and replacement
        if abbrev in ("+", "-", "["):

            # Convert to the same subclass if necessary
            if not isinstance(arg, type(self)):
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg = obj

            arg_vals = arg.vals

            # Find the common units
            if self.units is None:
                new_units = arg.units
            elif arg.units is None:
                new_units = self.units
            elif arg.units.exponents == self.units.exponents:
                new_units = self.units
                arg_vals = arg.units.convert(arg_vals, self.units)
            else:
                # If the units are incompatible, raise an error
                raise self.raise_unit_mismatch(op, arg.units)

            return (arg_vals, arg.mask, new_units)

        # Multiplication, division and modulus

        # If the second operand is a unit...
        if isinstance(arg, Units):
            arg_vals = 1
            arg_mask = False
            arg_units = arg

        # If it's the same subclass...
        elif isinstance(arg, type(self)):
            arg_vals = arg.vals
            arg_mask = arg.mask
            arg_units = arg.units

        else:
            # Try casting to the same subclass...
            try:
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg_vals = obj.vals
                arg_mask = obj.mask
                arg_units = obj.units
            except:
                # On failure, try casting to a Scalar
                try:
                    arg = Array.as_scalar(arg)
                except:
                    # On failure, raise the previous error
                    obj = Array.__new__(type(self))
                    obj.__init__(arg)

                # Reshape the scalar for compatibility with self.vals
                arg_vals = np.reshape(arg.vals, np.shape(arg.vals) +
                                                self.rank * (1,))
                arg_mask = arg.mask
                arg_units = arg.units

        # Find the resulting units
        if self.units is None:
            if arg_units is None:
                new_units = None
            elif abbrev == "*":
                new_units = arg_units
            else:
                new_units = arg_unit**(-1)
        else:
            if arg_units is None:
                new_units = self.units
            elif abbrev == "*":
                new_units = self.units * arg_units
            else:
                new_units = self.units / arg_units

        return (arg_vals, arg_mask, new_units)

    def raise_type_mismatch(self, op, arg):
        """Raises a ValueError with text indicating that the operand types are
        unsupported."""

        raise ValueError("unsupported operand types for '" + op +
                         "': '"    + type(self).__name__ +
                         "' and '" + type(arg).__name__  + "'")

    def raise_shape_mismatch(self, op, vals):
        """Raises a ValueError with text indicating that the operand shapes are
        incompatible."""

        raise ValueError("incompatible operand shapes for '" + op +
                         "': "   + str(tuple(self.shape)) +
                         " and " + str(np.shape(vals)))

    def raise_unit_mismatch(self, op, units):
        """Raises a ValueError with text indicating that the operand units are
        incompatible."""

        self_units = self.units
        if self_units is None: self_units = Units.UNITLESS

        if units is None: units = Units.UNITLESS

        raise ValueError("incompatible units for '" + op +
                         "': '"   + self_units.name +
                         "' and '" + units.name + "'")

    ####################################################
    # Default binary arithmetic operators
    ####################################################

    def __add__(self, arg):
        if Array.is_empty(arg): return arg

        (vals, mask, new_units) = self.get_array_mask_unit("+", arg)

        try:
            obj = Array.__new__(type(self))
            obj.__init__(self.vals + vals, self.mask | mask, new_units)
            return obj
        except:
            self.raise_shape_mismatch("+", vals)

    def __radd__(self, arg): return self.__add__(arg)

    def __sub__(self, arg):
        if Array.is_empty(arg): return arg

        (vals, mask, new_units) = self.get_array_mask_unit("-", arg)

        try:
            obj = Array.__new__(type(self))
            obj.__init__(self.vals - vals, self.mask | mask, new_units)
            return obj
        except:
            self.raise_shape_mismatch("-", vals)

    def __rsub__(self, arg): return self.__sub__(arg).__neg__()

    def __mul__(self, arg):
        if Array.is_empty(arg): return arg

        try:
            (vals, mask, new_units) = self.get_array_mask_unit("*", arg)
        except:
            if isinstance(arg, Array) and self.rank == 0:
                return arg.__mul__(self)
            raise

        try:
            new_vals = self.vals * vals
        except:
            self.raise_shape_mismatch("*", vals)

        obj = Array.__new__(type(self))
        obj.__init__(new_vals, self.mask | mask, new_units)
        return obj

    # Reverse-multiply if forward multiply fails
    def __rmul__(self, arg):
        result = self.__mul__(arg)
        if result is not NotImplemented: return result

        # On failure, raise the original exception
        (vals, mask, new_units) = self.get_array_mask_unit("*", arg)
        obj = Array.__new__(type(self))
        obj.__init__(new_vals, self.mask | mask, new_units)

    def __div__(self, arg):
        if Array.is_empty(arg): return arg

        (vals, mask, new_units) = self.get_array_mask_unit("/", arg)

        # Mask any items divided by zero
        div_by_zero = (vals == 0)
        if np.any(div_by_zero):

            # Handle scalar case
            if np.shape(vals) == ():
                obj = Array.__new__(type(self))
                obj.__init__(self.vals, True, new_units)
                return obj

            # Prevent any warning
            if np.shape(vals) != ():
                vals = vals.copy()
                vals[div_by_zero] = 1

            # Collapse rightmost mask axes based on rank of object
            for i in range(self.rank):
                div_by_zero = np.any(div_by_zero, axis=-1)

        else:
            # Avoid converting a scalar mask to an array unless necessary
            div_by_zero = False

        try:
            obj = Array.__new__(type(self))
            obj.__init__(self.vals / vals,
                         self.mask | mask | div_by_zero, new_units)
            return obj
        except:
            self.raise_shape_mismatch("/", vals)

    def __mod__(self, arg):
        if Array.is_empty(arg): return arg

        (vals, mask, new_units) = self.get_array_mask_unit("%", arg)

        # Mask any items divided by zero
        div_by_zero = (vals == 0)
        if np.any(div_by_zero):

            # Prevent any warning
            vals[div_by_zero] = 1

            # Collapse rightmost mask axes based on rank of object
            for i in range(self.rank):
                div_by_zero = np.any(div_by_zero, axis=-1)

        else:
            # Avoid converting a scalar mask to an array unless necessary
            div_by_zero = False

        try:
            obj = Array.__new__(type(self))
            obj.__init__(self.vals % vals,
                         self.mask | mask | div_by_zero, new_units)
            return obj
        except:
            self.raise_shape_mismatch("%", obj.vals)

    ####################################################
    # Default in-place binary arithmetic operators
    ####################################################

    def __iadd__(self, arg):

        (vals, mask, new_units) = self.get_array_mask_unit("+=", arg)

        try:
            self.vals += vals
            self.mask |= mask
            self.units = new_units
            return self
        except:
            self.raise_shape_mismatch("+=", vals)

    def __isub__(self, arg):

        (vals, mask, new_units) = self.get_array_mask_unit("-=", arg)

        try:
            self.vals -= vals
            self.mask |= mask
            self.units = new_units
            return self
        except:
            self.raise_shape_mismatch("-=", vals)

    def __imul__(self, arg):

        (vals, mask, new_units) = self.get_array_mask_unit("*=", arg)

        try:
            self.vals *= vals
            self.mask |= mask
            self.units = new_units
            return self
        except:
            self.raise_shape_mismatch("*=", vals)

    def __idiv__(self, arg):

        (vals, mask, new_units) = self.get_array_mask_unit("/=", arg)

        div_by_zero = np.any(vals == 0, axis=-1)
        if not np.any(div_by_zero):
            div_by_zero = False

        try:
            self.vals /= vals
            self.mask |= (mask | div_by_zero)
            self.units = new_units
            return self
        except:
            self.raise_shape_mismatch("/=", vals)

    def __imod__(self, arg):

        (vals, mask, new_units) = self.get_array_mask_unit("%=", arg)

        div_by_zero = np.any(vals == 0, axis=-1)
        if not np.any(div_by_zero):
            div_by_zero = False

        try:
            self.vals %= vals
            self.mask |= (mask | div_by_zero)
            self.units = new_units
            return self
        except:
            self.raise_shape_mismatch("%=", vals)

    ####################################
    # Default comparison operators
    ####################################

    def __eq__(self, arg):

        # If the subclasses cannot be unified, the objects are unequal
        if not isinstance(arg, type(self)):
            try:
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return False

        # If the units are incompatible, the objects are unequal
        # If units are compatible, convert to the same units
        if self.units is not None and arg.units is not None:
            if self.units.exponents != arg.units.exponents: return False
            arg = arg.convert_units(self.units)
        else:
            if self.units != arg.units: return False

        # The comparison is easy if the shape is []
        if np.shape(self.vals) == () and np.shape(arg.vals) == ():
            if self.mask and arg.mask: return True
            if self.mask != arg.mask: return False
            return self.vals == arg.vals

        # Compare the values
        compare = (self.vals == arg.vals)

        # Collapse the rightmost axes based on rank
        for i in range(self.rank):
            compare = np.all(compare, axis=-1)

        # Quick test: If both masks are empty, just return the comparison
        if (not np.any(self.mask) and not np.any(arg.mask)):
            return Array.as_scalar(compare)

        # Otherwise, perform the detailed comparison
        compare[self.mask & arg.mask] = True
        compare[self.mask ^ arg.mask] = False
        return Array.as_scalar(compare)

    def __ne__(self, arg):

        # If the subclasses cannot be unified, the objects are unequal
        if not isinstance(arg, type(self)):
            try:
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return True

        # If the units are incompatible, the objects are unequal
        # If units are compatible, convert to the same units
        if self.units is not None and arg.units is not None:
            if self.units.exponents != arg.units.exponents: return True
            arg = arg.convert_units(self.units)
        else:
            if self.units != arg.units: return False

        # The comparison is easy if the shape is []
        if np.shape(self.vals) == () and np.shape(arg.vals) == ():
            if self.mask and arg.mask: return False
            if self.mask != arg.mask: return True
            return self.vals != arg.vals

        # Compare the values
        compare = (self.vals != arg.vals)

        # Collapse the rightmost axes based on rank
        for i in range(self.rank):
            compare = np.any(compare, axis=-1)

        # Quick test: If both masks are empty, just return the comparison
        if (not np.any(self.mask) and not np.any(arg.mask)):
            return Array.as_scalar(compare)

        # Otherwise, perform the detailed comparison
        compare[self.mask & arg.mask] = False
        compare[self.mask ^ arg.mask] = True
        return Array.as_scalar(compare)

    def __nonzero__(self):
        """This is the test performed by an if clause."""

        if self.mask is False:
            return bool(np.all(self.vals))
        return bool(np.all(self.vals[~self.mask]))

    ####################################
    # Default binary logical operators
    ####################################

    # (<) operator
    def __lt__(self, arg):
        return self.raise_type_mismatch("<", arg)

    # (>) operator
    def __gt__(self, arg):
        return self.raise_type_mismatch(">", arg)

    # (<=) operator
    def __le__(self, arg):
        return self.raise_type_mismatch("<=", arg)

    # (>=) operator
    def __ge__(self, arg):
        return self.raise_type_mismatch(">=", arg)

    # (~) operator
    def __invert__(self):
        return self.raise_type_mismatch("~", arg)

    # (&) operator
    def __and__(self, arg):
        return self.raise_type_mismatch("&", arg)

    # (|) operator
    def __or__(self, arg):
        return self.raise_type_mismatch("|", arg)

    # (^) operator
    def __xor__(self, arg):
        return self.raise_type_mismatch("^", arg)

    ####################################
    # Value Transformations
    ####################################

    def __copy__(self):
        """describes how python should handle copying of Array types. if vals
            are instance of type nparray then create new instance of Array with
            a copy of the values, else just use the memory location of the
            values.
            
            Return:     a new instance of type Array.
            """
        if isinstance(self.vals, np.ndarray):
            vals = self.vals.copy()
        else:
            vals = self.vals

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.copy()
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def copy(self): return self.__copy__()

    def astype(self, dtype):
        """converts dtype of elements of Array to said type. creates new
            instance of Array and returns it.
            
            Input:
            dtype:      type, such as int, float32, etc.
            
            Return:     new instance of Array with converted elements.
            """

        if isinstance(self.vals, np.ndarray):
            if self.vals.dtype == dtype: return self
            vals = self.vals.astype(dtype)
        else:
            vals = np.array([self.vals], dtype=dtype)[0]

        obj = Array.__new__(type(self))
        obj.__init__(vals, self.mask, self.units)
        return obj

    ####################################
    # Unit conversions
    ####################################
    # Note that these three methods have slightly different methods for how to
    # handle units of None.

    def convert_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, its units are assumed to be standard units
        of km, seconds and radians, which are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        if units is None:
            obj.__init__(self.units.to_standard(self.vals), self.mask, None)
        elif self.units is None:
            obj.__init__(units.to_units(self.vals), self.mask, units)
        else:
            obj.__init__(self.units.convert(self.vals, units), self.mask, units)

        return obj

    def attach_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, the units are assumed to be itarget units
        already and the values are returned unchanged. Arrays with different
        units are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        if self.units is None or units is None:
            obj.__init__(self.vals, self.mask, units)
        else:
            obj.__init__(self.units.convert(self.vals, units), self.mask, units)

        return obj

    def confirm_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, the values are assumed to be unitless. Arrays
        with different units are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        self_units = self.units
        if self_units is None: self_units = Units.UNITLESS

        obj.__init__(self_units.convert(self.vals, units), self.mask, units)
        return obj

    ####################################
    # Shaping functions
    ####################################

    def swapaxes(self, axis1, axis2):
        """returns a new instance of Array with the elements in axis1 and axis2
            swapped. use nparray.swapaxes on Array vals. if axis1 or axis2
            is greater than rank of self then raise ValueError. note that new
            data is not created, but a new Array shell is created that indexes
            its axes in a swapped fashion, therefore changing values of original
            Array will change values of newly created Array.
            
            Input:
            axis1       first axis to swap from/to.
            axis2       second axis to swap to/from.
            
            Return:     new instance of Array with axes swapped.
            """

        if axis1 >= len(self.shape):
            raise ValueError("bad axis1 argument to swapaxes")
        if axis2 >= len(self.shape):
            raise ValueError("bad axis2 argument to swapaxes")

        vals = self.vals.swapaxes(axis1, axis2)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.swapaxes(axis1, axis2)
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def reshape(self, shape):
        """returns a new instance of Array with the list of items reshaped to
            that of shape. note that the argument shape refers to the shape of
            Array, not that of the nparray. note that new data is not created,
            but a new Array shell is created that indexes its elemnts according
            to the new shape, therefore changing values of the original
            Array will change values of newly created Array.
            
            Input:
            shape       shape of Array
            
            Return:     new instance of Array with new shape.
            """

        vals = self.vals.reshape(list(shape) + self.item)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.reshape(shape)
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def flatten(self):
        """returns a new instance of Array with the items flattened into a one-
            dimensional nparray of Arrays. note that new data is not created,
            but a new Array shell is created that indexes its elemnts according
            to the new shape, therefore changing values of the original
            Array will change values of newly created Array.
            
            Return:     new instance of Array with 1D shape.
            """

        if len(self.shape) < 2: return self

        count = np.product(self.shape)
        vals = self.vals.reshape([count] + self.item)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.reshape((count))
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def ravel(self): return self.flatten()

    def reorder_axes(self, axes):
        """Puts the leading axes into the specified order. Item axes are
        unchanged."""

        allaxes = axes + range(len(self.shape), len(self.vals.shape))

        vals = self.vals.transpose(allaxes)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.transpose(allaxes[:len(self.shape)])
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def append_axes(self, axes):
        """Appends the specified number of unit axes to the end of the shape."""

        if axes == 0: return self

        vals = self.vals.reshape(self.shape + axes*[1] + self.item)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.reshape(self.shape + axes*[1])
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def prepend_axes(self, axes):
        """Prepends the specified number of unit axes to the end of the shape.
        """

        if axes == 0: return self

        vals = self.vals.reshape(axes*[1] + self.shape + self.item)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.reshape(axes*[1] + self.shape)
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def strip_axes(self, axes):
        """Removes unit axes from the beginning of an Array's shape."""

        newshape = self.shape
        while len(newshape) > 0 and newshape[0] == 1: newshape = newshape[1:]

        vals = self.vals.reshape(newshape + self.item)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.reshape(newshape)
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def rotate_axes(self, axis):
        """Rotates the axes until the specified axis comes first; leading axes
        are moved to the end but the item axes are unchanged."""

        allaxes = (range(axis, axis + len(self.shape)) +
                   range(axis) +
                   range(len(self.shape), len(self.vals.shape)))

        vals = self.vals.transpose(allaxes)

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.transpose(allaxes[:len(self.shape)])
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def prepend_rotate_strip(self, axes, rank):
        """This method prepends unit axes if necessary until the leading Array
        dimensions reach the specified rank. Then it rotates the axes to put the
        specified axis first. Then it strips away any leading unit axes.

        This procedure can be used to rotate the axes of multiple Arrays in a
        consistent way such that the same axes broadcast together both before
        and after the operation."""

        return self.prepend_axes(rank-len(self.shape)).rotate(axes).strip_axes()

    def rebroadcast(self, newshape):
        """Returns an Array broadcasted to the specified shape. It returns self
        if the shape already matches. Otherwise, the returned object shares data
        with the original and should be treated as read-only."""

        newshape = list(newshape)
        if newshape == self.shape: return self

        temp = np.empty(newshape + self.item, dtype="byte")
        vals = np.broadcast_arrays(self.vals, temp)[0]

        if isinstance(self.mask, np.ndarray):
            temp = np.empty(newshape, dtype="byte")
            mask = np.broadcast_arrays(self.mask, temp)[0]
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    @staticmethod
    def broadcast_shape(arrays, item=[]):
        """This static method returns the shape that would result from a
        broadcast across the provided set of Array objects. It raises a
        ValueError if the shapes cannot be broadcasted.

        Input:
            arrays      a list or tuple containing zero or more array objects.
                        Values of None and Empty() are ignored. Anything (such
                        as a numpy array) that has an intrinsic shape attribute
                        can be used. A list or tuple is treated as the
                        definition of an additional shape.

            item        a list or tuple to be appended to the shape. Default is
                        []. Makes it possible to use the returned shape in the
                        declaration of a numpy array containing items that are
                        not scalars.

        Return:         the broadcast shape, comprising the maximum value of
                        each corresponding axis, plus the item shape if any.
        """

        # Initialize the shape
        broadcast = []
        len_broadcast = 0

        # Loop through the arrays...
        for array in arrays:
            if array is None: continue

            # Get the next shape
            try:
                shape = list(array.shape)
            except AttributeError:
                shape = list(array)

            # Expand the shapes to the same rank
            len_shape = len(shape)

            if len_shape > len_broadcast:
                broadcast = [1] * (len_shape - len_broadcast) + broadcast
                len_broadcast = len_shape

            if len_broadcast > len_shape:
                shape = [1] * (len_broadcast - len_shape) + shape
                len_shape = len_broadcast

            # Update the broadcast shape and check for compatibility
            for i in range(len_shape):
                if broadcast[i] == 1:
                    broadcast[i] = shape[i]
                elif shape[i] == 1:
                    pass
                elif shape[i] != broadcast[i]:
                    raise ValueError("shape mismatch: two or more arrays " +
                        "have incompatible dimensions on axis " + str(i))

        return broadcast + item

    @staticmethod
    def broadcast_arrays(arrays):
        """This static method returns a list of Array objects all broadcasted to
        the same shape. It raises a ValueError if the shapes cannot be
        broadcasted.

        Input:
            arrays      a list or tuple containing zero or more array objects.
                        Values of None and Empty() are returned as is. Anything
                        scalar or array-like is first converted to a Scalar.

        Return:         A list of new Array objects, broadcasted to the same
                        shape. The array data should be treated as read-only.
                        In the returned array, the function arguments come
                        before the elements of the "arrays" list.
        """

        newshape = Array.broadcast_shape(arrays)

        results = []
        for array in arrays:
            if array is None:
                results.append(None)
            else:
                results.append(Array.rebroadcast(array, newshape))

        return results

    @staticmethod
    def shape(arg):
        """Returns the inferred shape of the given argument, regardless of its
        class, as a list."""

        if isinstance(arg, Array): return arg.shape
        return list(np.shape(arg))

    @staticmethod
    def item(arg):
        """Returns the inferred item shape of the given argument, regardless of
        its class, as a list."""

        if isinstance(arg, Array): return arg.item
        return []

    @staticmethod
    def rank(arg):
        """Returns the inferred dimensions of the items comprising the given
        argument, regardless of its class, as a list."""

        if isinstance(arg, Array): return arg.rank
        return 0

    @staticmethod
    def vals(arg):
        """Returns the value array from an Array object. For anything else it
        returns the object itself. In either case, the result is returned as a
        numpy ndarray."""

        if isinstance(Array): arg = arg.vals
        return np.asarray(arg)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Array(unittest.TestCase):

    # No tests here. Everything is tested by subclasses

    def runTest(self):
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
