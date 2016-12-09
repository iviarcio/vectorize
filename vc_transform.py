# ==================================================================
# vc_transform.py
#
# Modify AST nodes trying to vectorize the OpenCL kernels.
# Look for inner loops that can be unrolled and assignments
# with array locations that can be replaced to vector statements.
#
# Copyright (C) 2016, Marcio Machado Pereira
# License: BSD
# ==================================================================

import copy
import vc_ast
from vc_ast import *


class DefUseChain(dict):
    """ Class representing ArrayRef locations. The class should
        provide functionality for adding and looking up nodes
        associated with identifiers. In its definition one can
        find the statements where locations are used.
    """
    def __init__(self):
        super().__init__()

    def add(self, name, value):
        # Insert the definition of a name
        self[name] = [value]

    def lookup(self, name):
        # Returns the definition of a name
        return self.get(name, None)


class vcTransform(object):
    """ Uses the same visitor pattern as vc_ast.NodeVisitor.
        While traversing the AST do:
        1) Mark some statements to be removed or to be refactored
        2) Construct defuse chains for ArrayRef locations
        3) Insert vector and temporary locations
        4) Modify AST to include vector statements.
    """

    def __init__(self):
        """
        param: success:
            Flag that indicates if vectorize was done successfully
        param: stack:
            Stack of defuselist. This is necessary because AST file
            may contain more than one kernel (with same loc names)
        param: defuselist:
            Symbol table for typedefs and array symbols in the kernel
        param: induction_var:
            Induction var object of loop candidate to refactoring
        param: refact_stmts:
            list of statements that will replace the loop statement
        param: block_stmt:
            Entry point in kernel function used to find position for
            insertion of vector statements
        param: result_stmt:
            statement that compute assignment associated to refactoring
        param: vectorid:
            Index of vector locations. These locations has the form
            _ftn where n equals the vectorid
        param: inside_region:
            Indicates that the next stmt (possible a compound stmt)
            are inside the 'selected loop' to be vectorized.
        param: vector_type:
            A dictionary that maps raw types to vector types.
        """
        self.success = False
        self.stack = []
        self.defuselist = DefUseChain()
        self.stack.append(self.defuselist)
        self.induction_var = None
        self.block_stmt = []
        self.refact_stmts = []
        self.result_stmt = []
        self.vectorid = 0
        self.inside_region = False
        self.vector_type = {'int': 'int4', 'long': 'long4', 'float': 'float4', 'double': 'double4'}

    def push(self, enclosure):
        self.stack.append(DefUseChain())

    def pop(self):
        self.stack.pop()

    def peek(self):
        return self.stack[-1]

    def add_definition(self, name, value):
        """ Add declare names and its object """
        self.peek().add(name, value)

    def lookup(self, name):
        for scope in reversed(self.stack):
            defuse = scope.lookup(name)
            if defuse is not None:
                return defuse
        return None

    def create_vector_location(self):
        _name = "_ft" + str(self.vectorid)
        self.vectorid += 1
        return _name

    def remove_induction_var(self, iname, n):
        if type(n) == ArrayRef:
            return vc_ast.ArrayRef(n.name,self.remove_induction_var(iname,n.subscript))
        elif type(n) == BinaryOp:
            _left = type(n.left)
            _op   = n.op
            _right = type (n.right)
            if (((_right == ID) and (n.right.name == iname)) or
                ((_right == Constant) and (n.right.value == '0'))):
                if _op == '+':
                    return n.left
                else:
                    return vc_ast.Constant('int', '0')
            else:
                n.right = self.remove_induction_var(iname, n.right)
            if (((_left == ID) and (n.left.name == iname)) or
                ((_left == Constant) and (n.left.value == '0'))):
                if _op == '+':
                    return n.right
                else:
                    return vc_ast.Constant('int', '0')
            else:
                n.left = self.remove_induction_var(iname, n.left)
            # cleanup generated "expr + 0" when remove iname
            if type(n.right) == Constant and (n.right.value == '0'):
                if n.op == '+':
                    return n.left
            return n
        else:
            return n

    def swap_induction_var_to(self, iname, ival, n):
        if type(n) == ArrayRef:
            return vc_ast.ArrayRef(n.name,self.swap_induction_var_to(iname, ival, n.subscript))
        elif type(n) == BinaryOp:
            _left = type(n.left)
            _right = type (n.right)
            if (_right == ID) and (n.right.name == iname):
                if ival == '1':
                    return n.left
                else:
                    n.right = vc_ast.Constant('int', ival)
                    return n
            else:
                n.right = self.swap_induction_var_to(iname, ival, n.right)
            if (_left == ID) and (n.left.name == iname):
                if ival == '1':
                    return n.right
                else:
                    n.left = vc_ast.Constant('int', ival)
                    return n
            else:
                n.left = self.swap_induction_var_to(iname, ival, n.left)
        return n

    def create_vload_for(self, n, name, iname):
        _expr = self.remove_induction_var(iname, n)
        _args = vc_ast.ExprList([vc_ast.Constant('int', '0'), vc_ast.UnaryOp('&',_expr)])
        _call = vc_ast.FuncCall(vc_ast.ID('vload4'), _args)
        self.refact_stmts.append(vc_ast.Assignment('=', vc_ast.ID(name), _call))

    def create_multiple_assignments_for(self, n, name, iname):
        _lid = vc_ast.ID(name)
        _x = vc_ast.StructRef(_lid,'.',vc_ast.ID('x'))
        _expr = self.remove_induction_var(iname, copy.deepcopy(n))
        self.refact_stmts.append(vc_ast.Assignment('=', _x, _expr))
        _y = vc_ast.StructRef(_lid,'.',vc_ast.ID('y'))
        _expr = self.swap_induction_var_to(iname, '1', copy.deepcopy(n))
        self.refact_stmts.append(vc_ast.Assignment('=', _y, _expr))
        _z = vc_ast.StructRef(_lid,'.',vc_ast.ID('z'))
        _expr = self.swap_induction_var_to(iname, '2', copy.deepcopy(n))
        self.refact_stmts.append(vc_ast.Assignment('=', _z, _expr))
        _w = vc_ast.StructRef(_lid,'.',vc_ast.ID('w'))
        _expr = self.swap_induction_var_to(iname, '3', copy.deepcopy(n))
        self.refact_stmts.append(vc_ast.Assignment('=', _w, _expr))

    def induction_var_dependence(self, n, iname):
        # return True if the Array var subscript depends on induction_var
        # Also return the kind of dependence (row or col) based on operator
        if type(n) == ID:
            return (n.name == iname), ''
        elif type(n) == UnaryOp:
            _dep, _ = self.induction_var_dependence(n.expr, iname)
            return _dep, n.op
        elif type(n) == BinaryOp:
            _left, _op = self.induction_var_dependence(n.left, iname)
            if _left:
                return (True, n.op) if (_op == '') else (True, _op)
            else:
                _right, _op = self.induction_var_dependence(n.right, iname)
                return (_right, n.op) if (_op == '') else (_right, _op)
        else:
            return False, ''

    def is_dyadic_operator(self, op):
        return (op == '+=') or (op == '-=') or (op == '*=') or (op == '/=')

    def get_raw_type(self, decl):
        if type(decl.type) == PtrDecl:
            return self.get_raw_type(decl.type)
        elif type(decl.type) == TypeDecl:
            return self.get_raw_type(decl.type)
        elif type(decl.type) == IdentifierType:
            _name = decl.type.names[0]
            _decl = self.lookup(_name)
            if _decl is not None:
                return self.get_raw_type(_decl[0])
            else:
                return _name
        # Otherwise
        return None

    def get_init_val(self, hostloc):
        # Normally, if Array location is initialized, it's the
        # first assign statement with some cte value. Otherwise
        # the default value is used, but in this case, dyadic
        # operator must be used in final assignment
        if len(hostloc) > 2:
            _loc = hostloc[1]
            if type(_loc) == Assignment:
                if _loc.op == '=':
                    if type(_loc.rvalue) == Constant:
                        return _loc.rvalue.value
        # signals that location must preserve initial value
        hostloc[0].has_initial_value = True
        return '0.'

    def get_typeid_for(self, decl):
        return vc_ast.IdentifierType(
            names=[self.vector_type[self.get_raw_type(decl)]])

    def get_typedecl_for(self, name, vdecl):
        return vc_ast.TypeDecl(
            declname=name,
            quals=None,
            type=self.get_typeid_for(vdecl))

    def insert_stmt(self, n):
        # insert the stmt into kernel body
        _block_items = self.block_stmt.block_items
        _index = 0
        while type(_block_items[_index]) == Decl:
            _index += 1
        _block_items.insert(_index, n)

    def insert_cte_decl(self, name, ctype, value):
        # create a constant vector declaration
        _cte = vc_ast.Constant(ctype, value)
        _initlist = vc_ast.InitList(
            exprs=[_cte, _cte, _cte, _cte])
        _declaration = vc_ast.Decl(
            name=name,
            quals=['const'],
            storage=None,
            funcspec=None,
            type=vc_ast.TypeDecl(
                declname=name,
                quals=['const'],
                type=vc_ast.IdentifierType([self.vector_type[ctype]])
            ),
            init=_initlist,
            bitsize=None,
            coord=None)
        # set visited flag equals true for this decl stmt
        _declaration.visited = True
        # insert the constant declaration into kernel body
        self.insert_stmt(_declaration)

    def insert_decl(self, name, hostloc, init=False):
        if init:
            _val=self.get_init_val(hostloc)
            _cte=vc_ast.Constant(
                type=self.get_raw_type(hostloc[0]),
                value=_val)
            _initlist=vc_ast.InitList(
                exprs=[_cte, _cte, _cte, _cte])
        else:
            _initlist=None
        _declaration = vc_ast.Decl(
            name=name,
            quals=None,
            storage=['__private'],
            funcspec=None,
            type=self.get_typedecl_for(name, hostloc[0]),
            init=_initlist,
            bitsize=None,
            coord=None)
        # set visited flag equals true for this decl stmt
        _declaration.visited=True
        self.insert_stmt(_declaration)

    def generate_expr(self, n):
        if type(n) == BinaryOp:
            self.generate_expr(n.left)
            self.generate_expr(n.right)
        elif type(n) == UnaryOp:
            self.generate_expr(n.expr)
        elif type(n) == ArrayRef:
            _ldefuse = self.lookup(self.visit(n))
            _ldefuse.append(n)
            _name = self.get_vector_location(n)
            if not _name:
                # Create a new vector location
                _name = self.create_vector_location()
                 # Associate the vector_location with ArrayRef expression
                _ldefuse[0].vector_locs.append((_name, n))
                # insert a declaration for the vector_location on AST
                self.insert_decl(_name, _ldefuse)
        else:
            return

    def refactoring(self, n):
        _loc = n.lvalue
        _rval = n.rvalue
        _op = n.op
        # get the decl stmt & raw type of location used in lvalue
        _ldefuse = self.lookup(self.visit(_loc.name))
        _typename = self.get_raw_type(_ldefuse[0])
        # get the vector_location associated with lvalue
        _name = self.get_vector_location(_loc)
        # There are two cases where assignment uses dyadic operator
        # 1. There are a previous assignment to initialize location
        #    In this case, a simple assignment will be used
        # 2. We need to use the default (previous) value
        if not _ldefuse[0].has_initial_value:
            _op = '='
        # note that the order of next two stmts are important
        # create refact stmts for variables present in rvalue
        # The refact_stmts will be inserted after 'for' stmt
        _rvalue = self.generate_r_stmt(_rval, _typename)
        # create refact stmt for lvalue (loc)
        _lvalue = self.generate_l_stmt(_loc, _name, _op)
        self.refact_stmts.append(vc_ast.Assignment(n.op, _lvalue, _rvalue))

    def generate_l_stmt(self, lvalue, rname, op):
        _rid = vc_ast.ID(rname)
        _x = vc_ast.StructRef (_rid,'.',vc_ast.ID('x'))
        _y = vc_ast.StructRef (_rid,'.',vc_ast.ID('y'))
        _z = vc_ast.StructRef (_rid,'.',vc_ast.ID('z'))
        _w = vc_ast.StructRef (_rid,'.',vc_ast.ID('w'))
        _rvalue = vc_ast.BinaryOp('+', vc_ast.BinaryOp('+', vc_ast.BinaryOp('+', _x, _y), _z), _w)
        self.result_stmt.append(vc_ast.Assignment(op, lvalue, _rvalue))
        return vc_ast.ID(rname)

    def generate_r_stmt(self, n, ltype):
        if type(n) == BinaryOp:
            _lnode = self.generate_r_stmt(n.left, ltype)
            _rnode = self.generate_r_stmt(n.right, ltype)
            return vc_ast.BinaryOp(n.op, _lnode, _rnode)
        elif type(n) == UnaryOp:
            return vc_ast.UnaryOp(n.op, self.generate_r_stmt(n.expr, ltype))
        elif type(n) == ArrayRef:
            # get the vector_loation associated with n stmt
            _name = self.get_vector_location(n)
            _iname = self.induction_var[0].name
            _depend, _op = self.induction_var_dependence(n.subscript, _iname)
            if (not _depend) or (_depend and (_op == '+')):
                self.create_vload_for(n, _name, _iname)
            else:
                self.create_multiple_assignments_for(n, _name, _iname)
            return vc_ast.ID(_name)
        elif type(n) == Constant:
            _name = self.get_vector_location(n)
            if not _name:
                _name = self.create_vector_location()
                # Sometimes the type of cte (e.g. int) is used in op with another type
                # So, the type of constant must be adjusted for type used during expr
                self.insert_cte_decl(_name, ltype, n.value)
                n.vector_locs.append(_name)
            return vc_ast.ID(_name)

    def get_vector_location(self, n):
        if type(n) == Constant:
            for _name in n.vector_locs:
                return _name
        else:
            _defuse = self.lookup(self.visit(n.name))
            for _name, _stmt in _defuse[0].vector_locs:
                if _stmt == n:
                    return _name
        return None

    def check_loop_nest(self, stmt, father):
        if type(stmt) == For:
            if stmt.remove:
                father.stmt = vc_ast.Compound(self.refact_stmts)
                self.refact_stmts = []
                return True
            else:
                return self.check_loop_nest(stmt.stmt, stmt)
        elif type(stmt) == Compound:
            return self.swap_for(stmt, 0)
        return False

    def swap_for(self, n, start):
        _stmts = n.block_items
        _index = start
        while (_index < len(_stmts)) and (type(_stmts[_index]) is not For):
            _index += 1
        if _index < len(_stmts):
            _for = _stmts[_index]
            if _for.remove:
                # replace the "for" with the refact_stmts list
                _stmts.pop(_index)
                for st in self.refact_stmts:
                    _stmts.insert(_index, st)
                    _index += 1
                # cleanup refactor stmts without destroy them
                self.refact_stmts = []
            else:
                _done = self.check_loop_nest(_for.stmt, _for)
                if not _done:
                    _done = self.swap_for(n, _index + 1)
                return _done
        return False

    def check_init_for_result_loc(self, n):
        if type(n) == Assignment:
            if type(n.lvalue) == ArrayRef and type(n.rvalue) == Constant:
                if self.is_dyadic_operator(n.op):
                    # In this case, we will preserve the stmt,
                    # copying to outside loop stmt
                    self.insert_stmt(copy.deepcopy(n))
                return True
        return False

    def remove_if_and_barrier_stmts(self, n, start):
        if type(n) == Compound:
            _stmts = n.block_items
            _index = start
            while True:
                if _index < len(_stmts):
                    if type(_stmts[_index]) == For:
                        self.remove_if_and_barrier_stmts(_stmts[_index].stmt, 0)
                    elif (type(_stmts[_index]) == If) or (type(_stmts[_index]) == FuncCall):
                        if _stmts[_index].remove:
                            # remove 'barrier' or if stmt (and all it's children)
                            _stmts.pop(_index)
                    _index += 1
                else:
                    return

    def append_result_stmt(self, n):
        # append result stmt after "refactored" for stmt
        for stmt in self.result_stmt:
            n.block_items.append(stmt)
        self.result_stmt = []
        
    def loop_invariant_code_motion(self, n):
        """
        Move to outside, stmts inside loops that are independent
        of the loop induction var, i.e., the expression is
        invariant inside the loop
        """
        _stmt_list = n.block_items
        # navigate to stmts until found a loop stmt. Also hold
        # the current index to move invariants at this place
        for _idx, _kst in enumerate(_stmt_list):
            if type(_kst) == For:
                self.induction_var = self.lookup(_kst.init.decls[0].name)
                _fst = _kst.stmt
                if type (_fst) == Compound:
                    for _pos, _ast in enumerate(_fst.block_items):
                        # Look for all ArrayRefs in this assignment
                        # and check if it's invariant
                        if self.is_invariant(_ast):
                            # Remove from 'for list' and insert into 'kernel list'
                            _stmt_list.insert(_idx, _fst.block_items.pop(_pos))
                            _idx += 1
                            _pos += 1

    def is_invariant(self, n):
        # After transformation, invariant ArrayRefs only appears at the right
        # Normally, inside a 'vload4' Call. Otherwise, there is nothing to do
        _invariant = False
        _iname = self.induction_var[0].name
        if type(n.rvalue) == FuncCall:
            _exp_list = n.rvalue.args.exprs
            if len(_exp_list) == 2:
                _exp = _exp_list[1].expr
                if type(_exp) == ArrayRef:
                    _depend, _ = self.induction_var_dependence(_exp.subscript, _iname)
                    _invariant = not _depend
        return _invariant

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)

    def generic_visit(self, node):
        if node is None: pass
        else:
            for c_name, c in node.children():
                self.visit(c)

    def visit_FileAST(self, n):
        _cl_khr_fp64 = False
        for ext in n.ext:
            if isinstance(ext, Typedef) or isinstance(ext, FuncDef):
                self.visit(ext)
            elif isinstance(ext, Pragma):
                _cl_khr_fp64 = 'cl_khr_fp64' in ext.string
            else:
                pass
        if self.success and not _cl_khr_fp64:
            _pragma = vc_ast.Pragma("OPENCL EXTENSION cl_khr_fp64 : enable")
            n.ext.insert(0,_pragma)

    def visit_Typedef(self, n):
        self.add_definition(n.type.declname, n.type)

    def visit_FuncDef(self, n):
        self.push(n)
        self.visit(n.decl)
        self.block_stmt = n.body
        self.visit(n.body)
        if self.success:
            # navigate through body to remove marked if stmts
            # and also unnecessary barriers
            self.remove_if_and_barrier_stmts(n.body, 0)
            # navigate through body to replace 'selected for' stmt
            self.swap_for(n.body, 0)
            # next, append result_stmt in the final of body of kernel
            self.append_result_stmt(n.body)
            # finally, execute loop invariant code motion
            self.loop_invariant_code_motion(n.body)
        self.pop()

    def visit_FuncCall(self,n):
        _name = self.visit(n.name)
        if _name == 'barrier':
            if not self.inside_region:
                n.remove = True
                n.visited = True

    def visit_DeclList(self, n):
        for dcl in n.decls:
            self.visit(dcl)

    def visit_Decl(self, n):
        if isinstance(n.type, FuncDecl):
            self.visit(n.type.args)
        else:
            self.add_definition(n.name, n)

    def visit_ParamList(self, n):
        for param in n.params: self.visit(param)

    def visit_Compound(self, n):
        _stmts = n.block_items
        _index = 0
        while _index < len(_stmts):
            _stmt = _stmts[_index]
            if hasattr(_stmt, "visited"):
                if not _stmt.visited:
                    if type(_stmt) == Decl:
                        self.add_definition(_stmt.name, _stmt)
                    else:
                        self.visit_stmt(_stmt)
                    _stmt.visited = True
            else:
                self.visit_stmt(_stmt)
            _index += 1

    def visit_stmt(self, n):
        """ This method exists as a wrapper for individual
            visit_* methods to handle different treatment
            of statements in the context of candidate loop.
        """
        if type(n) == For:
            # initialize the remove condition
            # If the loop satisfies the conditions to be unrolled.
            # then, the stmts will be replace by vectorized stmts
            self.visit_For(n)
            self.inside_region=False
        else:
            self.visit(n)

    def visit_For(self, n):
        # Include Induction var(s) on defuselist
        if n.init: self.visit(n.init)
        # Step must be (+= 1)
        if n.next:
            _single_step = self.visit_ForNext(n.next)
        # Check through condition if 'for stmt' is the
        # candidate to be vectorized
        if _single_step:
            if n.cond:
                self.inside_region=self.visit_ForCondition(n.cond)
        # set the remove condition flag
        n.remove = self.inside_region
        self.visit_stmt(n.stmt)
        if self.inside_region:
            # Visit stmts again to do refactoring
            self.visit_stmt(n.stmt)
            self.success = True

    def visit_ForCondition(self, n):
        if isinstance(n.right, Constant):
            _value = int(n.right.value)
            if ((_value == 3) and (n.op == '<=')) or \
               ((_value == 4) and (n.op == '<')):
                # Found the loop
                self.induction_var = self.lookup(n.left.name)
                return True
            else:
                return False
        return False

    def visit_ForNext(self, n):
        if type(n) == Assignment:
            if type(n.rvalue) == Constant:
                _value = int(n.rvalue.value)
                if (_value == 1) and (n.op == '+='):
                    return True
                else:
                    return False
        return False

    def visit_If(self, n):
        self.visit(n.cond)
        self.visit(n.iftrue)
        if n.iffalse is not None:
            self.visit(n.iffalse)
        elif type(n.cond) == BinaryOp:
            if type(n.cond.right) == Constant:
                if n.cond.right.value == '0':
                    # verify if is just a initialization of result value
                    # case of true, the 'if statement' will be removed
                    n.remove = self.check_init_for_result_loc(n.iftrue)

    def visit_Assignment(self, n):
        if n.refactor and self.inside_region:
            self.refactoring(n)
            n.refactor = False
        else:
            # This assignment will be refactored?
            n.refactor = self.inside_region
            _ldefuse = None
            _loc= n.lvalue
            _rval = n.rvalue
            _op = n.op
            if type(_loc) == ArrayRef:
                _ldefuse = self.lookup(self.visit(_loc))
                _ldefuse.append(n)
                if n.refactor:
                    _name = self.get_vector_location(_loc)
                    if not _name:
                        # Create a new vector location
                        _name = self.create_vector_location()
                        # Associate this vector_location with ArrayRef expression
                        _ldefuse[0].vector_locs.append((_name,_loc))
                        # Check if lvalue does not depend of induction_var
                        if self.induction_var is not None:
                            _iname = self.induction_var[0].name
                            _depend, _ = self.induction_var_dependence(_loc.subscript, _iname)
                        else:
                            _depend = False
                        # Check if assignment operation is dyadic
                        _dyadic = self.is_dyadic_operator(_op)
                        # look for some initialization
                        _init = _dyadic and not _depend
                        # Insert a Decl stmt for the vector location in AST
                        self.insert_decl(_name, _ldefuse, _init)
            if n.refactor:
                if (type(_rval) == Constant) and (_ldefuse is not None):
                    return
                self.generate_expr(_rval)

    def visit_ID(self, n):
        return n.name

    def visit_ArrayRef(self, n):
        return self.visit(n.name)

    def visit_StructRef(self, n):
        return self.visit(n.name)
