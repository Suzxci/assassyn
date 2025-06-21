'''The module provides the implementation of a class that is both IR builder and the system.'''

#pylint: disable=cyclic-import

from __future__ import annotations
import os
import typing
import site
import inspect
from decorator import decorator

if typing.TYPE_CHECKING:
    from .ir.module import Module
    from .ir.array import Array
    from .ir.dtype import DType
    from .ir.value import Value

@decorator
def ir_builder(func, *args, **kwargs):
    '''The decorator annotates the function whose return value will be inserted into the AST.'''
    res = func(*args, **kwargs)

    # This indicates this res is handled somewhere else, so we do not need to rehandle it
    if res is None:
        return res

    #pylint: disable=cyclic-import,import-outside-toplevel
    from .ir.const import Const
    from .utils import package_path
    from .ir.expr import Expr

    if not isinstance(res, Const):
        if isinstance(res, Expr):
            res.parent = Singleton.builder.current_block
            for i in res.operands:
                Singleton.builder.current_module.add_external(i)
        Singleton.builder.insert_point.append(res)

    package_dir = os.path.abspath(package_path())

    Singleton.initialize_dirs_to_exclude()
    for i in inspect.stack()[2:]:
        fname, lineno = i.filename, i.lineno
        fname_abs = os.path.abspath(fname)

        if not fname_abs.startswith(package_dir) \
            and not any(fname_abs.startswith(exclude_dir) \
                         for exclude_dir in Singleton.all_dirs_to_exclude):
            res.loc = f'{fname}:{lineno}'
            break
    assert hasattr(res, 'loc')
    return res


#pylint: disable=too-many-instance-attributes
class SysBuilder:
    '''The class serves as both the system and the IR builder.'''

    name: str  # Name of the system
    modules: typing.List[Module]  # List of modules
    downstreams: list  # List of downstream modules
    arrays: typing.List[Array]  # List of arrays
    _ctx_stack: dict  # Stack for context tracking
    _exposes: dict  # Dictionary of exposed nodes

    @property
    def current_module(self):
        '''Get the current module being built.'''
        return None if not self._ctx_stack['module'] else self._ctx_stack['module'][-1]

    @property
    def current_block(self):
        '''Get the current block being built.'''
        return None if not self._ctx_stack['block'] else self._ctx_stack['block'][-1]

    @property
    def insert_point(self):
        '''Get the insert point.'''
        return self.current_block.body

    def enter_context_of(self, ty, entry):
        '''Enter the context of the given type.'''
        #pylint: disable=import-outside-toplevel
        from .ir.block import CondBlock
        if isinstance(entry, CondBlock):
            self.current_module.add_external(entry.cond)
        self._ctx_stack[ty].append(entry)

    def exit_context_of(self, ty):
        '''Exit the context of the given type.'''
        self._ctx_stack[ty].pop()

    def has_driver(self):
        '''Check if the system has a driver module.'''
        for i in self.modules:
            if i.__class__.__name__ == 'Driver':
                return True
        return False

    def has_module(self, name):
        '''Check if a module with the given name exists.'''
        for i in self.modules:
            if i.name == name:
                return i
        return None

    def __init__(self, name):
        self.name = name
        self.modules = []
        self.downstreams = []
        self.arrays = []
        self._ctx_stack = {'module': [], 'block': []}
        self._exposes = {}

    def expose_on_top(self, node, kind=None):
        '''Expose the given node in the top function with the given kind.'''
        self._exposes[node] = kind

    @property
    def exposed_nodes(self):
        '''Get the exposed nodes.'''
        return self._exposes

    def __enter__(self):
        '''Designate the scope of this system builder.'''
        assert Singleton.builder is None
        Singleton.builder = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Leave the scope of this system builder.'''
        assert Singleton.builder is self
        Singleton.builder = None

    def __repr__(self):
        body = '\n\n'.join(map(repr, self.modules))
        body = body + '\n\n' + '\n\n'.join(map(repr, self.downstreams))
        array = '  ' + '\n  '.join(repr(elem) for elem in self.arrays)
        return f'system {self.name} {{\n{array}\n\n{body}\n}}'

class Singleton(type):
    '''The class maintains the global singleton instance of the system builder.'''
    builder: SysBuilder = None  # Global singleton instance of the system builder
    repr_ident: int = None  # Indentation level for string representation
    id_slice: slice = slice(-6, -1)  # Slice for identifiers
    with_py_loc: bool = False  # Whether to include Python location in string representation
    all_dirs_to_exclude: list = []  # Directories to exclude for stack inspection

    @classmethod
    def initialize_dirs_to_exclude(mcs):
        '''Initialize the directories to exclude if not already initialized.'''
        if not mcs.all_dirs_to_exclude:
            site_package_dirs = site.getsitepackages()
            user_site_package_dir = site.getusersitepackages()
            mcs.all_dirs_to_exclude = site_package_dirs + [user_site_package_dir]
