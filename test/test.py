import numpy as np
import sympy as sp


A = np.array(
    [
        [sp.Symbol(f'a_{i}_{j}') for j in range(3)]
        for i in range(3)
    ],
)
B = np.array(
    [
        [sp.Symbol(f'b_{i}_{j}') for j in range(3)]
        for i in range(3)
    ],
)
C = np.array(
    [
        [sp.Symbol(f'c_{i}_{j}') for j in range(3)]
        for i in range(3)
    ],
)
Bp = A @ B @ C.transpose()
