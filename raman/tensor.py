import numpy as np
import sympy as sp


def _full_expand(expr):
    while expr != expr.expand():
        expr = expr.expand()
    return expr

def sympify(a):
    r = a.copy().flatten()
    for i in range(len(r)):
        r[i] = sp.sympify(r[i])
    return r.reshape(a.shape)


class RamanTensor:

    def __init__(self, tensor):
        self.tensor = sympify(tensor.copy())
        self._original_tensor = self.tensor.copy()

    def copy(self):
        return RamanTensor(self.tensor)

    @property
    def free_symbols(self):
        free_symbols = set()
        for a in self.tensor.flatten():
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
        self.tensor = R @ self.tensor @ R.transpose()

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
        return _full_expand((Eout @ self.tensor @ Ein)**2)
