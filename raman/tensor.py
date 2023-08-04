import numpy as np
import sympy as sp
from copy import copy


def _full_expand(expr):
    while expr != expr.expand():
        expr = expr.expand()
    return expr


def sympify(a):
    r = a.copy().flatten()
    for i in range(len(r)):
        r[i] = sp.sympify(r[i])
    return r.reshape(a.shape)


def assert_real_symbols(a):
    for ai in a.flatten():
        if 'real' not in ai.assumptions0:
            raise TypeError('All symbols need to be real.')
        if ai.assumptions0['real'] is not True:
            raise TypeError('All symbols need to be real.')


def _make_symbol_real(s, suffix):
    assumptions = copy(s.assumptions0)
    assumptions['real'] = True
    return sp.Symbol(str(s)+suffix, **assumptions)


def array_subs(a, subs_dict):
    r = a.copy().flatten()
    for i in range(len(r)):
        r[i] = r[i].subs(subs_dict)
    return r.reshape(a.shape)


def real_and_imag(a):
    re_subs_dict = {}
    im_subs_dict = {}
    for i in range(len(a.flatten())):
        for fs in a.flatten()[i].free_symbols:
            re_subs_dict[fs] = _make_symbol_real(fs, '_real')
            im_subs_dict[fs] = _make_symbol_real(fs, '_imag')
    return array_subs(a, re_subs_dict), array_subs(a, im_subs_dict)


class RamanTensor:

    def __init__(self, tensor_real, tensor_imag):
        assert_real_symbols(tensor_real)
        assert_real_symbols(tensor_imag)
        self.tensor_real = sympify(tensor_real.copy())
        self.tensor_imag = sympify(tensor_imag.copy())
        self._original_tensor_real = self.tensor_real.copy()
        self._original_tensor_imag = self.tensor_imag.copy()

    @property
    def tensor(self):
        return self.tensor_real + sp.I*self.tensor_imag

    def copy(self):
        return RamanTensor(self.tensor_real, self.tensor_imag)

    @property
    def free_symbols(self):
        free_symbols = set()
        for a in self.tensor_real.flatten():
            free_symbols = free_symbols.union(sp.sympify(a).free_symbols)
        for a in self.tensor_imag.flatten():
            free_symbols = free_symbols.union(sp.sympify(a).free_symbols)
        return free_symbols

    def _rotate_dispatcher(self, **kwargs):
        'orientation_matrix'
        'v_initial'
        'v_final'
        if (
            kwargs['v_initial'] is None
            and kwargs['v_final'] is None
            and kwargs['orientation_matrix'] is not None
        ):
            return self._rotate_by_orientation_matrix(
                kwargs['orientation_matrix'],
            )
        elif (
            kwargs['v_initial'] is not None
            and kwargs['v_final'] is not None
            and kwargs['orientation_matrix'] is None
        ):
            return self._rotate_by_initial_final_vector(
                kwargs['v_initial'],
                kwargs['v_final'],
            )
        else:
            raise TypeError('Invalid argument to rotate')

    def _transform(self, R):
        self.tensor_real = R @ self.tensor_real @ R.transpose()
        self.tensor_imag = R @ self.tensor_imag @ R.transpose()

    def _rotate_by_orientation_matrix(self, orientation_matrix):
        self._transform(orientation_matrix)

    # TODO
    def _rotate_by_initial_final_vector(self, v_initial, v_final):
        raise NotImplementedError

    def rotate(self, orientation_matrix=None, v_initial=None, v_final=None):
        return self._rotate_dispatcher(
            orientation_matrix=None,
            v_initial=None,
            v_final=None,
        )

    def get_model_func(self):
        p = sp.Symbol('p_angle')
        a = sp.Symbol('a_angle')
        expr = self.get_model_expr()
        args = [p, a] + list(sorted(self.free_symbols, key=str))
        return sp.lambdify(args, expr, modules='numpy')

    def get_model_expr(self):
        p = sp.Symbol('p_angle')
        a = sp.Symbol('a_angle')
        Ein = np.array([sp.cos(p/180*sp.pi), sp.sin(p/180*sp.pi), 0])
        Eout = np.array([sp.cos(a/180*sp.pi), sp.sin(a/180*sp.pi), 0])
        return _full_expand(
            (Eout @ self.tensor_real @ Ein)**2
            + (Eout @ self.tensor_imag @ Ein)**2,
        )
