'''IR module for the Assassyn compiler.'''

# Import core IR classes for convenience
from .array import Array, RegArray
from .block import Block
from .const import Const
from .dtype import DType, Int, UInt, Record, to_uint, to_int
from .value import Value
from .visitor import Visitor

# Import expr submodule
from . import expr

# Import module submodule
from . import module
